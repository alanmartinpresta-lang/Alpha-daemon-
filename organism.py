"""AIRE U6 — deterministic complex organism layer.

U6 turns the mesoscopic U4 physiology into an individual organism with a
minimal body/anatomy model, explicit organ-system capacities, sensors and
motor actuators. It deliberately contains no cognition, planning, memory or
agent-learning logic; actions are externally supplied commands.
"""
from __future__ import annotations

from dataclasses import dataclass, replace, field
import hashlib, math
from typing import Any, Mapping
import numpy as np

from .core import WorldCore, WorldState, WorldObject
from .biology import BiologicalParameters, BiologicalCompartment, BiologicalState, step_biology

U6_SCHEMA = "U6.2"
U6_CONSOLIDATED_SCHEMA = "U6.2"


def _finite(x: float, name: str) -> float:
    x = float(x)
    if not math.isfinite(x):
        raise ValueError(f"{name} must be finite")
    return x


def _nonneg(x: float, name: str) -> float:
    x = _finite(x, name)
    if x < 0:
        raise ValueError(f"{name} must be >= 0")
    return x


def _bounded(x: float, lo: float, hi: float, name: str) -> float:
    x = _finite(x, name)
    if x < lo or x > hi:
        raise ValueError(f"{name} must be in [{lo}, {hi}]")
    return x


def _vec3(v: Any, name: str) -> np.ndarray:
    a = np.asarray(v, dtype=float).reshape(-1)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError(f"{name} must be a finite 3-vector")
    return a.copy()


@dataclass(frozen=True)
class Anatomy:
    """Compact anatomy envelope used by U6."""
    mass: float = 70.0
    body_volume: float = 0.07
    surface_area: float = 1.8
    muscle_fraction: float = 0.40
    bone_fraction: float = 0.15
    lung_capacity: float = 6.0
    heart_capacity: float = 5.0
    digestive_capacity: float = 2.0
    inertia_diagonal: tuple[float, float, float] = (10.0, 10.0, 10.0)

    def __post_init__(self):
        for n in ("mass", "body_volume", "surface_area", "lung_capacity", "heart_capacity", "digestive_capacity"):
            _nonneg(getattr(self, n), n)
        if self.mass <= 0 or self.body_volume <= 0 or self.surface_area <= 0:
            raise ValueError("mass, body_volume and surface_area must be > 0")
        for value in self.inertia_diagonal:
            _nonneg(value, "inertia_diagonal")
            if value <= 0:
                raise ValueError("inertia_diagonal values must be > 0")
        for n in ("muscle_fraction", "bone_fraction"):
            _bounded(getattr(self, n), 0.0, 1.0, n)
        if self.muscle_fraction + self.bone_fraction > 1.0:
            raise ValueError("muscle_fraction + bone_fraction must be <= 1")


@dataclass(frozen=True)
class OrganSystems:
    """Normalized functional capacities; no cognitive state is included."""
    respiratory: float = 1.0
    circulatory: float = 1.0
    digestive: float = 1.0
    renal: float = 1.0
    thermoregulatory: float = 1.0
    neural: float = 1.0

    def __post_init__(self):
        for n in ("respiratory", "circulatory", "digestive", "renal", "thermoregulatory", "neural"):
            _bounded(getattr(self, n), 0.0, 1.0, n)


@dataclass(frozen=True)
class SensorState:
    """Raw/low-level sensor channels. Interpretation belongs to future U7."""
    temperature: float = 293.15
    pressure: float = 101325.0
    light: float = 0.0
    oxygen: float = 0.0
    water: float = 0.0
    nutrient: float = 0.0
    contact: float = 0.0
    sound_pressure: float = 0.0
    odor: float = 0.0
    taste: float = 0.0
    nociception: float = 0.0
    vision_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0)
    vestibular_linear_accel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    vestibular_angular_rate: tuple[float, float, float] = (0.0, 0.0, 0.0)
    proprioception_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self):
        for n in ("temperature", "pressure", "light", "oxygen", "water", "nutrient", "contact", "sound_pressure", "odor", "taste", "nociception"):
            _nonneg(getattr(self, n), n)
        for n in ("vision_rgb", "vestibular_linear_accel", "vestibular_angular_rate", "proprioception_velocity"):
            _vec3(getattr(self, n), n)


@dataclass(frozen=True)
class FunctionalState:
    """Dynamic non-cognitive organ-system state."""
    heart_rate: float = 70.0
    ventilation: float = 1.0
    hydration: float = 1.0
    core_temperature: float = 310.15
    fatigue: float = 0.0
    pain: float = 0.0
    balance: float = 1.0

    def __post_init__(self):
        _bounded(self.heart_rate, 0.0, 400.0, "heart_rate")
        _nonneg(self.ventilation, "ventilation")
        for n in ("hydration", "fatigue", "pain", "balance"):
            _bounded(getattr(self, n), 0.0, 1.0, n)
        _bounded(self.core_temperature, 150.0, 500.0, "core_temperature")


@dataclass(frozen=True)
class MotorCommand:
    """Externally supplied motor command; it is not chosen by the organism."""
    force: tuple[float, float, float] = (0.0, 0.0, 0.0)
    torque: tuple[float, float, float] = (0.0, 0.0, 0.0)
    muscle_activation: float = 0.0
    joint_torque_scale: float = 1.0
    intake_effort: float = 0.0

    def __post_init__(self):
        _vec3(self.force, "force"); _vec3(self.torque, "torque")
        _bounded(self.muscle_activation, 0.0, 1.0, "muscle_activation")
        _bounded(self.joint_torque_scale, 0.0, 1.0, "joint_torque_scale")
        _bounded(self.intake_effort, 0.0, 1.0, "intake_effort")


@dataclass(frozen=True)
class Genome:
    """Minimal heritable biological parameter set; no knowledge or goals."""
    mass_scale: float = 1.0
    muscle_fraction: float = 0.40
    thermal_tolerance: float = 1.0
    sensory_gain: float = 1.0

    def __post_init__(self):
        _bounded(self.mass_scale, 0.5, 2.0, "mass_scale")
        _bounded(self.muscle_fraction, 0.05, 0.80, "muscle_fraction")
        _bounded(self.thermal_tolerance, 0.5, 1.5, "thermal_tolerance")
        _bounded(self.sensory_gain, 0.25, 2.0, "sensory_gain")

    def digest(self) -> str:
        return hashlib.sha256(repr((self.mass_scale, self.muscle_fraction,
                                    self.thermal_tolerance, self.sensory_gain)).encode()).hexdigest()


def mutate_genome(genome: Genome, *, seed: int, mutation_rate: float = 0.01,
                  mutation_scale: float = 0.02) -> Genome:
    """Deterministically mutate inherited body parameters."""
    if not (0.0 <= mutation_rate <= 1.0) or mutation_scale < 0:
        raise ValueError("invalid mutation parameters")
    rng = np.random.default_rng(int(seed))
    def mutate(x, lo, hi):
        if rng.random() > mutation_rate:
            return x
        return float(np.clip(x + rng.normal(0.0, mutation_scale), lo, hi))
    return Genome(mutate(genome.mass_scale, 0.5, 2.0),
                  mutate(genome.muscle_fraction, 0.05, 0.80),
                  mutate(genome.thermal_tolerance, 0.5, 1.5),
                  mutate(genome.sensory_gain, 0.25, 2.0))


@dataclass(frozen=True)
class OrganismState:
    organism_id: str
    anatomy: Anatomy
    systems: OrganSystems = OrganSystems()
    physiology: BiologicalState = BiologicalState("U4.1", {})
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    orientation: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    genome: Genome = Genome()
    sensors: SensorState = SensorState()
    functional: FunctionalState = FunctionalState()
    alive: bool = True
    age: float = 0.0

    def __post_init__(self):
        if not self.organism_id:
            raise ValueError("organism_id must be non-empty")
        object.__setattr__(self, "position", _vec3(self.position, "position"))
        object.__setattr__(self, "velocity", _vec3(self.velocity, "velocity"))
        q = np.asarray(self.orientation, dtype=float).reshape(-1)
        if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) <= 0:
            raise ValueError("orientation must be a finite non-zero quaternion")
        q = q / np.linalg.norm(q); q.setflags(write=False); object.__setattr__(self, "orientation", q)
        object.__setattr__(self, "angular_velocity", _vec3(self.angular_velocity, "angular_velocity"))
        _nonneg(self.age, "age")
        if not isinstance(self.alive, bool):
            raise ValueError("alive must be bool")
        if self.physiology.schema != "U4.1":
            raise ValueError("U6 physiology must use U4.1")

    def digest(self) -> str:
        h = hashlib.sha256()
        h.update(U6_SCHEMA.encode()); h.update(self.organism_id.encode())
        h.update(repr((self.anatomy, self.systems, self.genome, self.physiology.digest(),
                       tuple(np.round(self.position, 14)), tuple(np.round(self.velocity, 14)),
                       tuple(np.round(self.orientation, 14)), tuple(np.round(self.angular_velocity, 14)),
                       self.sensors, self.functional, self.alive, self.age)).encode())
        return h.hexdigest()


def create_organism(organism_id: str = "organism-1", *, anatomy: Anatomy | None = None,
                    systems: OrganSystems | None = None, physiology: BiologicalState | None = None,
                    genome: Genome | None = None,
                    position=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0)) -> OrganismState:
    physiology = physiology or BiologicalState("U4.1", {
        "core": BiologicalCompartment("core", volume=0.07, water=42.0, oxygen=0.25,
                                      nutrient=0.5, atp=0.1, biomass=1.0, integrity=1.0)
    })
    genome = genome or Genome()
    if anatomy is None:
        anatomy = Anatomy(mass=70.0*genome.mass_scale, muscle_fraction=genome.muscle_fraction)
    return OrganismState(organism_id, anatomy, systems or OrganSystems(),
                         physiology, position=np.asarray(position, float), velocity=np.asarray(velocity, float), genome=genome)


def sense_environment(org: OrganismState, *, temperature=293.15, pressure=101325.0,
                      light=0.0, oxygen=0.0, water=0.0, nutrient=0.0, contact=0.0,
                      sound_pressure=0.0, odor=0.0, taste=0.0, nociception=0.0,
                      vision_rgb=(0.0, 0.0, 0.0)) -> OrganismState:
    """Update raw receptor channels only; no interpretation, memory or learning."""
    return replace(org, sensors=SensorState(temperature=temperature, pressure=pressure,
                                             light=light, oxygen=oxygen, water=water,
                                             nutrient=nutrient, contact=contact,
                                             sound_pressure=sound_pressure, odor=odor,
                                             taste=taste, nociception=nociception,
                                             vision_rgb=tuple(float(x) for x in vision_rgb),
                                             vestibular_linear_accel=tuple(org.velocity * 0.0),
                                             vestibular_angular_rate=tuple(org.angular_velocity),
                                             proprioception_velocity=tuple(org.velocity)))


def _integrate_orientation(q: np.ndarray, omega: np.ndarray, h: float) -> np.ndarray:
    """Deterministic quaternion update using an exponential-map increment."""
    wnorm = float(np.linalg.norm(omega))
    theta = wnorm * h
    if wnorm < 1e-15:
        return q / np.linalg.norm(q)
    half = 0.5 * theta
    s = math.sin(half) / wnorm
    dq = np.array([math.cos(half), *(omega * s)], dtype=float)
    # Hamilton product dq * q (world-frame angular rate).
    a,b,c,d = dq; e,f,g,k = q
    out = np.array([a*e-b*f-c*g-d*k, a*f+b*e+c*k-d*g,
                    a*g-b*k+c*e+d*f, a*k+b*g-c*f+d*e], dtype=float)
    return out / np.linalg.norm(out)


def step_organism(org: OrganismState, dt: float, command: MotorCommand | None = None,
                   *, parameters: BiologicalParameters | None = None,
                   external_acceleration=(0.0, 0.0, 0.0)) -> tuple[OrganismState, Mapping[str, Any]]:
    h = _finite(dt, "dt")
    if h <= 0:
        raise ValueError("dt must be > 0")
    if not org.alive:
        return org, {"dt": h, "alive": False, "certified_step": True, "digest": org.digest()}
    cmd = command or MotorCommand()
    acc_ext = _vec3(external_acceleration, "external_acceleration")
    # Muscle capacity is bounded by anatomy and activation. This is a body-level
    # actuator, not an agent decision system.
    max_acc = 25.0 * org.anatomy.muscle_fraction * org.systems.neural * org.systems.circulatory
    force_acc = _vec3(cmd.force, "force") / org.anatomy.mass
    force_acc = np.clip(force_acc, -max_acc, max_acc)
    acc = acc_ext + force_acc
    velocity = org.velocity + acc * h
    position = org.position + velocity * h
    muscle_cost = 0.02 * cmd.muscle_activation * h
    phys_params = parameters or BiologicalParameters()
    # Activity increases metabolic demand while reduced organ capacity lowers it.
    scaled = replace(phys_params, metabolic_rate=phys_params.metabolic_rate * (1.0 + 4.0 * cmd.muscle_activation))
    physiology, report = step_biology(org.physiology, h, parameters=scaled)
    core_comp = physiology.compartments.get("core")
    if core_comp is not None:
        atp = max(0.0, core_comp.atp - muscle_cost)
        integrity = core_comp.integrity
        core_comp = replace(core_comp, atp=atp, integrity=integrity)
        physiology = BiologicalState("U4.1", {**physiology.compartments, "core": core_comp}, physiology.time)
    inertia = np.asarray(org.anatomy.inertia_diagonal, dtype=float)
    angular_acc = (np.asarray(cmd.torque, float) * cmd.joint_torque_scale) / inertia
    omega = org.angular_velocity + angular_acc * h
    orientation = _integrate_orientation(org.orientation, omega, h)
    core = physiology.compartments.get("core")
    hydration = 0.0 if core is None else min(1.0, core.water / 42.0)
    fatigue = min(1.0, max(0.0, org.functional.fatigue + 0.18 * cmd.muscle_activation * h - 0.04 * h))
    pain = min(1.0, max(0.0, (0.0 if core is None else 1.0 - core.integrity) * 2.0))
    balance = max(0.0, min(1.0, 1.0 - 0.2 * np.linalg.norm(omega) - pain * 0.5))
    heart = min(400.0, max(0.0, 70.0 + 80.0 * cmd.muscle_activation + 30.0 * fatigue))
    ventilation = max(0.0, 1.0 + 2.0 * cmd.muscle_activation)
    temp = 310.15 if core is None else min(500.0, max(150.0, core.temperature))

    # Broad thermal safety limits: ordinary regulation is unharmed, while
    # sufficiently extreme cold/heat gradually damages integrity.
    thermal_excess = max(0.0, 250.0-temp) + max(0.0, temp-390.0)
    if core is not None and thermal_excess > 0.0:
        thermal_damage = min(0.02 * thermal_excess * h / 10.0, 0.5)
        core = replace(core, integrity=max(0.0, core.integrity-thermal_damage))
        physiology = BiologicalState("U4.1", {**physiology.compartments, "core": core}, physiology.time)

    functional = FunctionalState(heart_rate=heart, ventilation=ventilation, hydration=hydration,
                                 core_temperature=temp, fatigue=fatigue, pain=pain, balance=balance)
    # U6.2 derives additional raw receptor channels directly from body state.
    sensors = replace(org.sensors,
                       vestibular_linear_accel=tuple(acc),
                       vestibular_angular_rate=tuple(omega),
                       proprioception_velocity=tuple(velocity),
                       nociception=pain,
                       sound_pressure=org.sensors.sound_pressure)
    alive = all(c.integrity > 0.0 for c in physiology.compartments.values()) and any(c.biomass > 0 for c in physiology.compartments.values())
    new = replace(org, physiology=physiology, position=position, velocity=velocity,
                  orientation=orientation, angular_velocity=omega, functional=functional, sensors=sensors, age=org.age+h, alive=alive)
    out = dict(report)
    out.update({"dt": h, "position": tuple(position), "velocity": tuple(velocity),
                "alive": alive, "motor_cost": muscle_cost, "digest": new.digest()})
    out["finite"] = bool(np.isfinite(position).all() and np.isfinite(velocity).all() and np.isfinite(orientation).all())
    out["physiology_valid"] = bool(report.get("certified_step", False))
    out["certified_step"] = bool(out["finite"] and out["physiology_valid"])
    return new, out


def create_offspring(parent: OrganismState, organism_id: str, *, seed: int,
                    mutation_rate: float = 0.01, mutation_scale: float = 0.02,
                    position=None) -> OrganismState:
    """Create a newborn body from inherited genome plus bounded mutation."""
    genome = mutate_genome(parent.genome, seed=seed, mutation_rate=mutation_rate, mutation_scale=mutation_scale)
    anatomy = Anatomy(mass=70.0*genome.mass_scale, body_volume=parent.anatomy.body_volume*genome.mass_scale,
                      surface_area=parent.anatomy.surface_area*genome.mass_scale**(2/3),
                      muscle_fraction=genome.muscle_fraction, bone_fraction=parent.anatomy.bone_fraction,
                      lung_capacity=parent.anatomy.lung_capacity*genome.mass_scale,
                      heart_capacity=parent.anatomy.heart_capacity*genome.mass_scale,
                      digestive_capacity=parent.anatomy.digestive_capacity*genome.mass_scale,
                      inertia_diagonal=tuple(x*genome.mass_scale for x in parent.anatomy.inertia_diagonal))
    birth_pos = parent.position.copy() if position is None else np.asarray(position, dtype=float)
    return create_organism(organism_id, anatomy=anatomy, systems=parent.systems, genome=genome,
                           position=birth_pos, velocity=(0.0, 0.0, 0.0))


def attach_organism(state: WorldState, org: OrganismState) -> WorldState:
    obj = WorldObject(object_id=org.organism_id, kind="organism", position=org.position,
                      velocity=org.velocity,
                      state={"mass": org.anatomy.mass, "alive": org.alive,
                             "age": org.age, "digest": org.digest()},
                      tags=("u6", "organism", "u6.2"))
    objects = dict(state.objects); objects[org.organism_id] = obj
    meta = dict(state.metadata)
    meta["u6_organism"] = {"schema": U6_CONSOLIDATED_SCHEMA, "organism_id": org.organism_id,
                            "digest": org.digest(), "alive": org.alive, "age": org.age}
    return replace(state, objects=objects, metadata=meta)


class OrganismWorld:
    """World-Core handoff for one deterministic complex organism."""
    def __init__(self, core: WorldCore, organism: OrganismState):
        if abs(organism.physiology.time - core.state.clock.time) > 1e-9:
            raise ValueError("organism physiology time must match world clock")
        self.core, self.organism, self.last_report = core, organism, None
        self.core._state = attach_organism(core.state, organism)

    @property
    def state(self):
        return self.core.state

    def step(self, dt=None, command=None, *, external_acceleration=(0.0, 0.0, 0.0), parameters=None):
        h = self.state.clock.dt if dt is None else float(dt)
        new, report = step_organism(self.organism, h, command,
                                    external_acceleration=external_acceleration, parameters=parameters)
        self.core.step(h, updater=lambda s, _h: attach_organism(s, new))
        self.organism, self.last_report = new, report
        self.core.emit("organism_step", target_ids=(new.organism_id,),
                       payload={"dt": h, "certified": report["certified_step"], "organism_digest": new.digest()})
        return report




