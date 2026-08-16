"""AIRE V201.1 — Alpha World Core A1.5 runtime.

This is the stripped execution path for the pre-agent world. It contains only
the world layers that can affect an inhabitant or support future biological
evolution. NASA/CFD, historical solvers, discovery campaigns, benchmarks,
validation suites and research data are deliberately outside this runtime.

The Alpha agent itself is not embedded here: the runtime exposes a body,
sensors, actuators and an evolving environment. Cognition, memory, goals and
learning remain external/future layers.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import pickle
from pathlib import Path
import numpy as np

from .universe import (
    WorldCore, WorldState, WorldIdentity, Space3D, SimulationClock,
    PlanetConfig, create_environment,
    initialize_u2_dynamics, step_u2_dynamics,
    ChemicalSpecies, ChemistryNetwork, create_chemistry, step_chemistry,
    BiologicalCompartment, create_biology,
    EcologicalResource, EcologicalPopulation, create_ecology,
    BiogeochemicalPools, create_u5_consolidated, step_u5_consolidated,
    create_organism, sense_environment, step_organism, MotorCommand,
    transfer_environment_to_biology, step_nutrient_mineral_uptake,
    step_atmospheric_chemistry,
    attach_organism,
    AutonomyConfig, AutonomyState, ACTION_NAMES, action_command, choose_action, learn, autonomy_digest,
)


@dataclass(frozen=True)
class AlphaStepReport:
    dt: float
    time: float
    environment_certified: bool
    atmosphere_certified: bool
    chemistry_certified: bool
    biology_certified: bool
    ecology_certified: bool
    organism_certified: bool
    nutrient_uptake_certified: bool
    alive: bool
    digest: str
    autonomy_certified: bool = False
    action: str = "external"
    reward: float = 0.0


class AlphaWorld:
    """Single authoritative runtime for Alpha's physical world."""

    def __init__(self, core, environment, u2, chemistry, biology, ecology, organism,
                 *, autonomy_config=None, autonomous=True, agent_seed=0):

        self.core = core
        self.environment = environment
        self.u2 = u2
        self.chemistry = chemistry
        self.biology = biology
        self.ecology = ecology
        self.organism = organism
        self.autonomy_config = autonomy_config or AutonomyConfig()
        self.autonomous = bool(autonomous)
        self.agent_seed = int(agent_seed)
        self.autonomy = AutonomyState(epsilon=self.autonomy_config.exploration_initial)
        self.last_report: AlphaStepReport | None = None

    @property
    def time(self) -> float:
        return float(self.core.state.clock.time)

    @property
    def state(self):
        return self.core.state

    def _cell_for_position(self):
        """Map the organism's horizontal world position to the local terrain cell."""
        c = self.environment.config
        x = float(self.organism.position[0])
        y = float(self.organism.position[1])
        ix = int(np.clip(np.floor(x / c.cell_size + c.nx / 2), 0, c.nx - 1))
        iy = int(np.clip(np.floor(y / c.cell_size + c.ny / 2), 0, c.ny - 1))
        return iy, ix

    def _local_environment(self):
        iy, ix = self._cell_for_position()
        o2 = float(self.environment.atmospheric_oxygen_mass_kg[iy, ix])
        n2 = float(self.environment.atmospheric_nitrogen_mass_kg[iy, ix])
        co2 = float(self.environment.atmospheric_co2_mass_kg[iy, ix])
        ar = float(self.environment.atmospheric_argon_mass_kg[iy, ix])
        co = float(self.environment.atmospheric_co_mass_kg[iy, ix])
        gas_total = max(o2 + n2 + co2 + ar + co, 1e-30)
        oxygen_fraction = o2 / gas_total
        return iy, ix, oxygen_fraction

    def sense(self):
        """Refresh Alpha's raw receptors from the local world state only."""
        iy, ix, oxygen_fraction = self._local_environment()
        ground = float(self.environment.elevation[iy, ix])
        contact = 1.0 if self.organism.position[2] <= ground + 0.75 else 0.0
        self.organism = sense_environment(
            self.organism,
            temperature=float(self.environment.temperature[iy, ix]),
            pressure=float(self.environment.pressure[iy, ix]),
            light=float(self.environment.sunlight[iy, ix]),
            oxygen=oxygen_fraction,
            water=float(self.environment.water_depth[iy, ix]),
            nutrient=float(self.environment.resources[iy, ix]),
            contact=contact,
            vision_rgb=(float(self.environment.sunlight[iy, ix]), oxygen_fraction, float(self.environment.humidity[iy, ix])),
        )
        return self.organism.sensors

    def step(self, dt: float | None = None, command: MotorCommand | None = None, *, compute_digest: bool = True) -> AlphaStepReport:
        h = self.core.state.clock.dt if dt is None else float(dt)
        if not np.isfinite(h) or h <= 0:
            raise ValueError("dt must be finite and > 0")

        # U2 is the sole authority for environmental evolution.
        self.u2, u2_report = step_u2_dynamics(self.u2, h)
        self.environment = self.u2.base

        # V201 atmospheric chemistry uses U2 gas reservoirs as its authority.
        self.environment, atm_report = step_atmospheric_chemistry(self.environment, h)
        self.u2 = replace(self.u2, base=self.environment)

        # U3 chemical workspace.
        self.chemistry, chemistry_report = step_chemistry(self.chemistry, h)

        # The organism perceives the current world before choosing an action.
        self.organism = replace(self.organism, physiology=self.biology)
        self.sense()
        learning_before = self.organism
        chosen_action = None
        if command is None and self.autonomous and self.organism.alive:
            chosen_action, self.autonomy = choose_action(
                self.autonomy, self.organism, seed=self.agent_seed,
                config=self.autonomy_config
            )
            active_command = action_command(chosen_action, self.autonomy_config)
        else:
            active_command = command or MotorCommand()

        # Explicit SI environmental -> organism transfer.
        cell = self._cell_for_position()
        self.environment, self.biology, transfer_report = transfer_environment_to_biology(
            self.environment, self.biology, cell=cell, compartment_id="core",
            water_kg=0.05 * active_command.intake_effort,
            oxygen_kg=0.005 * active_command.intake_effort
        )
        self.u2 = replace(self.u2, base=self.environment)

        # U5 ecology and population-level evolution.
        self.ecology, ecology_report = step_u5_consolidated(self.ecology, h)

        # Explicit U2/U5 -> U4 nutrient/mineral transfer.
        self.u2, self.biology, self.ecology, uptake_report = step_nutrient_mineral_uptake(
            self.u2, self.biology, self.ecology,
            cell=cell, compartment_id="core",
            requested_kg={"C": 1.0e-4 * active_command.intake_effort,
                          "N": 5.0e-5 * active_command.intake_effort,
                          "P": 1.0e-5 * active_command.intake_effort,
                          "S": 1.0e-5 * active_command.intake_effort,
                          "Ca": 1.0e-6 * active_command.intake_effort,
                          "Mg": 5.0e-7 * active_command.intake_effort,
                          "K": 5.0e-7 * active_command.intake_effort,
                          "Na": 5.0e-7 * active_command.intake_effort,
                          "Fe": 2.0e-7 * active_command.intake_effort,
                          "Si": 1.0e-6 * active_command.intake_effort}
        )
        self.environment = self.u2.base

        # U6 is the physical body boundary of Alpha. The action chosen above is
        # the only action for this step. Environmental uptake performed above is
        # committed to the organism physiology before the body is advanced.
        self.organism = replace(self.organism, physiology=self.biology)
        iy, ix = self._cell_for_position()
        ground = float(self.environment.elevation[iy, ix])
        gravity = np.array([0.0, 0.0, -float(self.environment.config.gravity)])
        self.organism, organism_report = step_organism(
            self.organism, h, active_command, external_acceleration=gravity
        )
        # Solid terrain is an actual world boundary: no free fall through the ground.
        body_radius = max(0.25, 0.5 * self.organism.anatomy.body_volume ** (1.0 / 3.0))
        pos = self.organism.position.copy()
        vel = self.organism.velocity.copy()
        ground_contact = False
        if pos[2] < ground + body_radius:
            pos[2] = ground + body_radius
            if vel[2] < 0.0: vel[2] = 0.0
            ground_contact = True
        # The generated terrain is a finite local World, not an infinite plane.
        # Alpha cannot leave the represented terrain domain; hitting an edge is a
        # physical boundary, with the outward velocity component removed.
        half_x = 0.5 * self.environment.config.nx * self.environment.config.cell_size
        half_y = 0.5 * self.environment.config.ny * self.environment.config.cell_size
        x_limit = max(0.0, half_x - body_radius)
        y_limit = max(0.0, half_y - body_radius)
        if pos[0] < -x_limit:
            pos[0] = -x_limit
            if vel[0] < 0.0: vel[0] = 0.0
        elif pos[0] > x_limit:
            pos[0] = x_limit
            if vel[0] > 0.0: vel[0] = 0.0
        if pos[1] < -y_limit:
            pos[1] = -y_limit
            if vel[1] < 0.0: vel[1] = 0.0
        elif pos[1] > y_limit:
            pos[1] = y_limit
            if vel[1] > 0.0: vel[1] = 0.0
        self.organism = replace(self.organism, position=pos, velocity=vel)
        organism_report = dict(organism_report, ground_contact=ground_contact,
                               world_boundary_contact=bool(
                                   abs(pos[0]) >= x_limit - 1e-12 or abs(pos[1]) >= y_limit - 1e-12))
        self.biology = self.organism.physiology

        # U4 water loss is routed back into U2 after the single biological step.
        water_loss = float(organism_report.get("water_loss_to_environment", 0.0))
        if water_loss > 0:
            vapor = self.u2.vapor.copy()
            delta_depth = water_loss / (self.environment.config.cell_size**2 * 1000.0)
            vapor[cell[0], cell[1]] += delta_depth
            self.u2 = replace(self.u2, vapor=vapor)
            self.environment = self.u2.base

        # U0 is the sole owner of global time. The organism is attached to the
        # world state so its position/velocity are part of the causal World.
        # U0 advances the authoritative clock and attaches Alpha to the causal World.
        # Per-step telemetry is intentionally not appended to the immutable U0 event
        # ledger: doing so creates O(n²) copying/validation overhead over long runs
        # without changing the physical state seen by Alpha. Action/learning history
        # remains in the bounded autonomy memory and the final World digest.
        self.core.step(h, updater=lambda state, _h: attach_organism(state, self.organism))

        reward = 0.0
        autonomy_certified = True
        if chosen_action is not None:
            self.autonomy = learn(
                self.autonomy, learning_before, chosen_action, self.organism,
                dt=h, config=self.autonomy_config
            )
            reward = self.autonomy.last_reward
            autonomy_certified = bool(
                np.isfinite(reward) and np.isfinite(self.autonomy.cumulative_reward)
                and self.autonomy.steps >= 1
            )

        # Full World hashing is intentionally optional: it is diagnostic telemetry,
        # not part of the physical update. Long autonomous runs can disable it and
        # compute a complete digest only at checkpoints/finalization.
        digest = self.digest() if compute_digest else ""
        self.last_report = AlphaStepReport(
            dt=h, time=self.time,
            environment_certified=bool(u2_report.certified_step),
            atmosphere_certified=bool(atm_report.certified),
            chemistry_certified=bool(chemistry_report["certified_step"]),
            biology_certified=bool(organism_report["certified_step"]),
            ecology_certified=bool(ecology_report["certified_step"]),
            organism_certified=bool(organism_report["certified_step"]),
            nutrient_uptake_certified=bool(uptake_report.certified),
            alive=bool(self.organism.alive),
            digest=digest,
            autonomy_certified=bool(autonomy_certified and chosen_action is not None),
            action=("external" if chosen_action is None else ACTION_NAMES[chosen_action]),
            reward=reward,
        )
        return self.last_report

    def __getstate__(self):
        from .universe.persistence import serialize_world
        state = dict(self.__dict__)
        core = state.pop("core")
        state["_core_snapshot"] = serialize_world(core.state)
        return state

    def __setstate__(self, state):
        from .universe.persistence import deserialize_world
        from .universe.core import WorldCore
        core_snapshot = state.pop("_core_snapshot")
        self.__dict__.update(state)
        self.core = WorldCore(deserialize_world(core_snapshot))
        # V203 memory migration: old checkpoints retain only the short episodic
        # buffer. Preserve those raw experiences as the initial autobiography.
        if hasattr(self, "autonomy") and not getattr(self.autonomy, "autobiography", ()) and getattr(self.autonomy, "memory", ()):
            a=self.autonomy
            self.autonomy=AutonomyState(
                q_values=a.q_values, memory=a.memory, autobiography=list(a.memory),
                epsilon=a.epsilon, steps=a.steps, cumulative_reward=a.cumulative_reward,
                last_action=a.last_action, last_state_key=a.last_state_key,
                last_reward=a.last_reward, visit_counts=getattr(a,"visit_counts",{})
            )
        # Normalize checkpoints created before A1.7: their AutonomyState did
        # not yet persist uncertainty visit counts. The dataclass reconstructs
        # them from retained episodic memory on replacement.
        if not hasattr(self.autonomy, "visit_counts"):
            self.autonomy = AutonomyState(
                q_values=self.autonomy.q_values,
                memory=self.autonomy.memory,
                autobiography=getattr(self.autonomy, "autobiography", ()),
                epsilon=self.autonomy.epsilon,
                steps=self.autonomy.steps,
                cumulative_reward=self.autonomy.cumulative_reward,
                last_action=self.autonomy.last_action,
                last_state_key=self.autonomy.last_state_key,
                last_reward=self.autonomy.last_reward,
                visit_counts={}
            )

    def save_checkpoint(self, path: str | Path) -> dict:
        """Persist the complete live Alpha instance for exact later continuation.

        The checkpoint contains the World, Alpha physiology, autonomy memory,
        Q-values, clocks and all deterministic state required to resume without
        resetting the individual. It is versioned and integrity-checked.
        Loading is intentionally explicit because checkpoint files are trusted
        local artifacts; pickle must never be loaded from an untrusted source.
        """
        path = Path(path)
        payload = pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)
        digest = hashlib.sha256(payload).hexdigest()
        envelope = {
            "schema": "AIRE-ALPHA-CHECKPOINT-1",
            "runtime_version": "V201.1-A1.7",
            "time": self.time,
            "step_index": int(self.core.state.clock.step_index),
            "alpha_digest": self.digest(),
            "payload_sha256": digest,
            "payload": payload,
        }
        path.write_bytes(pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL))
        return {
            "path": str(path), "schema": envelope["schema"],
            "time": envelope["time"], "step_index": envelope["step_index"],
            "alpha_digest": envelope["alpha_digest"], "payload_sha256": digest,
        }

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> "AlphaWorld":
        """Restore an Alpha checkpoint and verify its integrity before use."""
        path = Path(path)
        envelope = pickle.loads(path.read_bytes())
        if envelope.get("schema") != "AIRE-ALPHA-CHECKPOINT-1":
            raise ValueError("unsupported Alpha checkpoint schema")
        payload = envelope.get("payload")
        if not isinstance(payload, (bytes, bytearray)):
            raise ValueError("invalid Alpha checkpoint payload")
        actual = hashlib.sha256(payload).hexdigest()
        if actual != envelope.get("payload_sha256"):
            raise ValueError("Alpha checkpoint integrity mismatch")
        world = pickle.loads(payload)
        if not isinstance(world, cls):
            raise TypeError("checkpoint does not contain an AlphaWorld")
        if world.time != float(envelope["time"]):
            raise ValueError("checkpoint clock mismatch")
        if int(world.core.state.clock.step_index) != int(envelope["step_index"]):
            raise ValueError("checkpoint step-index mismatch")
        runtime_version = str(envelope.get("runtime_version", ""))
        # A1.7 changes the autonomy digest because it persists uncertainty
        # visit counts. Older checkpoints remain loadable after their payload
        # integrity is verified; their visit counts are reconstructed from the
        # retained memory by AutonomyState.__post_init__/normalization.
        if runtime_version == "V201.1-A1.7":
            if world.digest() != envelope["alpha_digest"]:
                raise ValueError("Alpha checkpoint semantic digest mismatch")
        elif not runtime_version.startswith("V201.1-A"):
            raise ValueError("unsupported Alpha checkpoint runtime version")
        return world

    def digest(self) -> str:
        h = hashlib.sha256(b"AIRE-V201.1-ALPHA-WORLD-V2")
        for value in (
            self.environment.digest(), self.u2.digest(), self.chemistry.digest(),
            self.biology.digest(), self.ecology.digest(), self.organism.digest(),
            autonomy_digest(self.autonomy), str(self.core.state.clock.time),
        ):
            h.update(value.encode())
        return h.hexdigest()


def create_alpha_world(*, seed: int = 2011, dt: float = 1.0,
                       grid: tuple[int, int] = (8, 8)) -> AlphaWorld:
    """Create a deterministic, inhabited-ready V201.1 Alpha World V2."""
    nx, ny = map(int, grid)
    if nx < 1 or ny < 1:
        raise ValueError("grid dimensions must be positive")

    core = WorldCore(WorldState(
        WorldIdentity("alpha-world", seed=seed),
        Space3D(bounds_min=[-1e5] * 3, bounds_max=[1e5] * 3),
        SimulationClock(dt=float(dt)),
    ))

    env = create_environment(PlanetConfig(nx=nx, ny=ny, seed=seed))
    u2 = initialize_u2_dynamics(env)

    # U3 contains atmospheric-relevant species plus a substrate. The World
    # keeps the chemistry state explicit; atmospheric gases remain U2-authoritative.
    species = {
        "glucose": ChemicalSpecies("glucose", 180.156, 0, {"C": 6, "H": 12, "O": 6}),
        "O2": ChemicalSpecies("O2", 31.998, 0, {"O": 2}),
        "CO2": ChemicalSpecies("CO2", 44.009, 0, {"C": 1, "O": 2}),
        "H2O": ChemicalSpecies("H2O", 18.015, 0, {"H": 2, "O": 1}),
    }
    concentrations = {name: np.zeros((ny, nx), dtype=float) for name in species}
    concentrations["O2"][:] = 1.0
    concentrations["CO2"][:] = 0.04
    concentrations["glucose"][:] = 0.01
    chemistry = create_chemistry(
        ChemistryNetwork(species), shape=(ny, nx),
        temperature=env.temperature, concentrations=concentrations
    )

    biology = create_biology({
        "core": BiologicalCompartment(
            "core", volume=0.07, water=42.0, oxygen=0.25, nutrient=0.5,
            atp=0.1, biomass=1.0, integrity=1.0
        )
    })

    ecology = create_ecology(
        {"food": EcologicalResource("food", 20.0, 0.1, 50.0)},
        {"pop": EcologicalPopulation(
            "pop", 5.0, biomass=5.0, resource="food",
            consumption_rate=0.02, reproduction_rate=0.05,
            carrying_capacity=50.0
        )}
    )
    u5 = create_u5_consolidated(
        ecology,
        cycles=BiogeochemicalPools(
            organic_carbon=10, inorganic_carbon=10,
            organic_nitrogen=5, inorganic_nitrogen=5,
            oxygen=2, detritus_carbon=1, detritus_nitrogen=1
        )
    )

    organism = create_organism(
        "alpha",
        position=(0.0, 0.0, 0.0),
        physiology=biology,
    )
    world = AlphaWorld(core, env, u2, chemistry, biology, u5, organism, agent_seed=seed)
    world.sense()
    return world
