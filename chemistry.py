"""AIRE U3 — deterministic chemistry layer.

U3 provides a conservative, solver-independent chemistry model suitable for
world-scale experiments. Concentrations are mol/m^3. Species carry molar mass
and charge metadata; reactions use integer stoichiometry and mass-action
kinetics. The implementation is intentionally not a quantum-chemistry or
high-fidelity molecular-dynamics solver.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import hashlib, math
from typing import Any, Mapping
import numpy as np

from .core import WorldCore, WorldState

U3_SCHEMA = "U3.1"
R_GAS = 8.31446261815324


def _field(a: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    x = np.asarray(a, dtype=float)
    if x.shape != shape or not np.all(np.isfinite(x)) or np.any(x < 0):
        raise ValueError(f"{name} must be finite and non-negative with shape {shape}")
    return x.copy()


@dataclass(frozen=True)
class ChemicalSpecies:
    name: str
    molar_mass: float
    charge: int = 0
    elements: Mapping[str, int] = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("species name must be non-empty")
        if not math.isfinite(self.molar_mass) or self.molar_mass <= 0:
            raise ValueError("molar_mass must be finite and > 0")
        if int(self.charge) != self.charge:
            raise ValueError("charge must be an integer")
        elems = {} if self.elements is None else {str(k): int(v) for k, v in self.elements.items()}
        if any(v <= 0 for v in elems.values()):
            raise ValueError("element counts must be positive integers")
        object.__setattr__(self, "charge", int(self.charge))
        object.__setattr__(self, "elements", elems)


@dataclass(frozen=True)
class ChemicalReaction:
    reaction_id: str
    reactants: Mapping[str, int]
    products: Mapping[str, int]
    rate_constant: float
    activation_energy: float = 0.0  # J/mol
    enthalpy: float = 0.0            # J/mol reaction extent; positive=endothermic
    reverse_rate_constant: float = 0.0
    catalysts: Mapping[str, int] = None
    equilibrium_constant: float | None = None

    def __post_init__(self):
        if not self.reaction_id:
            raise ValueError("reaction_id must be non-empty")
        r = {str(k): int(v) for k, v in self.reactants.items()}
        p = {str(k): int(v) for k, v in self.products.items()}
        if not r or not p or any(v <= 0 for v in (*r.values(), *p.values())):
            raise ValueError("reaction stoichiometry must use positive integers")
        if set(r) & set(p):
            raise ValueError("species cannot appear on both sides; use net stoichiometry")
        if not math.isfinite(self.rate_constant) or self.rate_constant < 0:
            raise ValueError("rate_constant must be finite and >= 0")
        if not math.isfinite(self.activation_energy) or self.activation_energy < 0:
            raise ValueError("activation_energy must be finite and >= 0")
        if not math.isfinite(self.enthalpy):
            raise ValueError("enthalpy must be finite")
        if not math.isfinite(self.reverse_rate_constant) or self.reverse_rate_constant < 0:
            raise ValueError("reverse_rate_constant must be finite and >= 0")
        cats = {} if self.catalysts is None else {str(k): int(v) for k, v in self.catalysts.items()}
        if any(v < 0 for v in cats.values()):
            raise ValueError("catalyst orders must be >= 0")
        if self.equilibrium_constant is not None and (not math.isfinite(self.equilibrium_constant) or self.equilibrium_constant <= 0):
            raise ValueError("equilibrium_constant must be finite and > 0")
        object.__setattr__(self, "reactants", r)
        object.__setattr__(self, "products", p)
        object.__setattr__(self, "catalysts", cats)

    def stoich(self, species: str) -> int:
        return self.products.get(species, 0) - self.reactants.get(species, 0)


@dataclass(frozen=True)
class ChemistryNetwork:
    species: Mapping[str, ChemicalSpecies]
    reactions: tuple[ChemicalReaction, ...] = ()

    def __post_init__(self):
        clean = {str(k): v for k, v in self.species.items()}
        if len(clean) != len(self.species):
            raise ValueError("duplicate species")
        for key, sp in clean.items():
            if key != sp.name:
                raise ValueError("species mapping key must equal species name")
        for rxn in self.reactions:
            unknown = (set(rxn.reactants) | set(rxn.products)) - set(clean)
            if unknown:
                raise ValueError(f"reaction references unknown species: {sorted(unknown)}")
        object.__setattr__(self, "species", clean)
        object.__setattr__(self, "reactions", tuple(self.reactions))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.species))

    def mass_balance(self, reaction: ChemicalReaction) -> float:
        return sum(reaction.stoich(n) * self.species[n].molar_mass for n in self.species)

    def charge_balance(self, reaction: ChemicalReaction) -> int:
        return sum(reaction.stoich(n) * self.species[n].charge for n in self.species)


@dataclass(frozen=True)
class ChemistryState:
    schema: str
    network: ChemistryNetwork
    concentrations: Mapping[str, np.ndarray]
    temperature: np.ndarray
    volume: float = 1.0
    time: float = 0.0
    heat_capacity: float = 1000.0

    def __post_init__(self):
        if self.schema != U3_SCHEMA:
            raise ValueError("unsupported chemistry schema")
        shape = np.asarray(self.temperature, dtype=float).shape
        if len(shape) != 2 or min(shape) < 1:
            raise ValueError("temperature must be a 2-D field")
        temp = _field(self.temperature, shape, "temperature")
        if np.any(temp <= 0):
            raise ValueError("temperature must be > 0")
        conc = {}
        for name in self.network.names:
            if name not in self.concentrations:
                raise ValueError(f"missing concentration field for {name}")
            conc[name] = _field(self.concentrations[name], shape, f"concentration[{name}]")
            conc[name].setflags(write=False)
        object.__setattr__(self, "concentrations", conc)
        temp.setflags(write=False); object.__setattr__(self, "temperature", temp)
        if not math.isfinite(self.volume) or self.volume <= 0:
            raise ValueError("volume must be finite and > 0")
        if not math.isfinite(self.time):
            raise ValueError("time must be finite")
        if not math.isfinite(self.heat_capacity) or self.heat_capacity <= 0:
            raise ValueError("heat_capacity must be finite and > 0")

    @property
    def shape(self): return self.temperature.shape

    def total_moles(self, species: str) -> float:
        return float(np.sum(self.concentrations[species]) * self.volume)

    def total_mass(self) -> float:
        return float(sum(self.total_moles(n) * self.network.species[n].molar_mass for n in self.network.names))

    def total_charge(self) -> float:
        return float(sum(self.total_moles(n) * self.network.species[n].charge for n in self.network.names))

    def digest(self) -> str:
        h = hashlib.sha256(); h.update(self.schema.encode()); h.update(repr(self.network).encode())
        h.update(np.ascontiguousarray(self.temperature).tobytes())
        for n in self.network.names: h.update(n.encode()); h.update(np.ascontiguousarray(self.concentrations[n]).tobytes())
        h.update(repr((self.volume, self.time, self.heat_capacity)).encode()); return h.hexdigest()


def _arrhenius(k0: float, activation_energy: float, temperature: np.ndarray) -> np.ndarray:
    if activation_energy == 0: return np.full_like(temperature, k0, dtype=float)
    exponent = np.clip(-activation_energy / (R_GAS * temperature), -700.0, 0.0)
    return k0 * np.exp(exponent)


def create_chemistry(network: ChemistryNetwork, temperature: Any = 298.15, *, shape=(8, 8), volume=1.0, time=0.0, concentrations=None) -> ChemistryState:
    temp = np.full(shape, float(temperature)) if np.isscalar(temperature) else np.asarray(temperature, dtype=float)
    concentrations = concentrations or {n: np.zeros(temp.shape, dtype=float) for n in network.names}
    return ChemistryState(U3_SCHEMA, network, concentrations, temp, volume, time, 1000.0)


def reaction_rates(state: ChemistryState) -> dict[str, np.ndarray]:
    rates = {}
    for rxn in state.network.reactions:
        rate = _arrhenius(rxn.rate_constant, rxn.activation_energy, state.temperature)
        for name, order in rxn.reactants.items(): rate = rate * np.power(state.concentrations[name], order)
        for name, order in rxn.catalysts.items(): rate = rate * np.power(state.concentrations[name], order)
        if rxn.reverse_rate_constant > 0:
            reverse = np.full(state.shape, rxn.reverse_rate_constant, dtype=float)
            for name, order in rxn.products.items(): reverse *= np.power(state.concentrations[name], order)
            for name, order in rxn.catalysts.items(): reverse *= np.power(state.concentrations[name], order)
            rate = rate - reverse
        rates[rxn.reaction_id] = rate
    return rates


def step_chemistry(state: ChemistryState, dt: float, *, diffusion: float = 0.0) -> tuple[ChemistryState, Mapping[str, Any]]:
    h = float(dt)
    if not math.isfinite(h) or h <= 0: raise ValueError("dt must be finite and > 0")
    if not 0 <= diffusion <= 1: raise ValueError("diffusion must be in [0,1]")
    conc = {n: state.concentrations[n].copy() for n in state.network.names}
    temp = state.temperature.copy(); rates = reaction_rates(state)
    reaction_extents = {}
    for rxn in state.network.reactions:
        r = rates[rxn.reaction_id]
        # Explicit Euler with a positivity-preserving extent limiter per cell.
        max_forward = np.full(state.shape, np.inf)
        for name, coeff in rxn.reactants.items():
            max_forward = np.minimum(max_forward, conc[name] / coeff)
        max_reverse = np.full(state.shape, np.inf)
        for name, coeff in rxn.products.items():
            max_reverse = np.minimum(max_reverse, conc[name] / coeff)
        raw = h * r
        extent = np.where(raw >= 0, np.minimum(raw, max_forward), -np.minimum(-raw, max_reverse))
        for name in rxn.reactants: conc[name] -= rxn.reactants[name] * extent
        for name in rxn.products: conc[name] += rxn.products[name] * extent
        if rxn.enthalpy != 0:
            # Approximate local thermal coupling. Positive enthalpy consumes heat.
            temp -= (rxn.enthalpy * extent) / state.heat_capacity
        reaction_extents[rxn.reaction_id] = float(np.sum(extent) * state.volume)
    if diffusion > 0:
        alpha = min(0.25, float(diffusion) * h)
        for n in conc:
            a = conc[n]
            conc[n] = np.maximum(0.0, (1-4*alpha)*a + alpha*(np.roll(a,1,0)+np.roll(a,-1,0)+np.roll(a,1,1)+np.roll(a,-1,1)))
    temp = np.maximum(1.0, temp)
    out = ChemistryState(U3_SCHEMA, state.network, conc, temp, state.volume, state.time+h)
    report = {
        "dt": h, "reaction_extents": reaction_extents,
        "mass_before": state.total_mass(), "mass_after": out.total_mass(),
        "charge_before": state.total_charge(), "charge_after": out.total_charge(),
        "temperature_min": float(np.min(out.temperature)), "digest": out.digest(),
    }
    report["mass_residual"] = report["mass_after"] - report["mass_before"]
    report["charge_residual"] = report["charge_after"] - report["charge_before"]
    report["certified_step"] = bool(abs(report["mass_residual"]) <= 1e-10 * max(1.0, abs(report["mass_before"])) and
                                      abs(report["charge_residual"]) <= 1e-10 * max(1.0, abs(report["charge_before"])))
    return out, report


def attach_chemistry(state: WorldState, chemistry: ChemistryState) -> WorldState:
    meta = dict(state.metadata)
    meta["u3_chemistry"] = {"schema": chemistry.schema, "species": list(chemistry.network.names),
                             "shape": list(chemistry.shape), "digest": chemistry.digest(), "time": chemistry.time}
    return replace(state, metadata=meta)


class ChemistryWorld:
    def __init__(self, core: WorldCore, chemistry: ChemistryState):
        if abs(chemistry.time - core.state.clock.time) > 1e-9: raise ValueError("chemistry time must match world clock")
        self.core, self.chemistry, self.last_report = core, chemistry, None
        self.core._state = attach_chemistry(core.state, chemistry)

    @property
    def state(self): return self.core.state

    def step(self, dt=None, *, diffusion=0.0):
        h = self.state.clock.dt if dt is None else float(dt)
        new, report = step_chemistry(self.chemistry, h, diffusion=diffusion)
        self.core.step(h, updater=lambda s, _h: attach_chemistry(s, new))
        self.chemistry, self.last_report = new, report
        return report




