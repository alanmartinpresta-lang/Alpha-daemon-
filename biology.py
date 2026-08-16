"""AIRE U4 — deterministic biological foundation.

U4 introduces a solver-independent physiological layer above U3 chemistry.
It models bounded biological compartments with water, oxygen, nutrient,
energy carrier (ATP), carbon dioxide, waste, biomass and integrity. The model
is intentionally mesoscopic: it is not a molecular/cellular simulator and
contains no cognition or agent logic.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from typing import Any, Mapping

from .core import WorldCore, WorldState, WorldObject

U4_SCHEMA = "U4.1"


def _finite_nonnegative(x: float, name: str) -> float:
    x = float(x)
    if not math.isfinite(x) or x < 0:
        raise ValueError(f"{name} must be finite and >= 0")
    return x


def _bounded(x: float, lo: float, hi: float, name: str) -> float:
    x = float(x)
    if not math.isfinite(x) or x < lo or x > hi:
        raise ValueError(f"{name} must be finite and in [{lo}, {hi}]")
    return x


@dataclass(frozen=True)
class BiologicalParameters:
    """Conservative mesoscopic physiology constants."""
    metabolic_rate: float = 5.0e-5     # kg nutrient / s at saturation
    oxygen_per_nutrient: float = 1.0    # kg O2 / kg nutrient
    atp_yield: float = 3.0e7            # J / kg nutrient
    atp_capacity: float = 5.0e5          # bounded intracellular ATP pool (J)
    co2_yield: float = 1.0              # kg CO2 / kg nutrient (model substrate)
    heat_yield: float = 2.0e6           # J heat / kg nutrient
    biomass_yield: float = 0.08
    repair_cost: float = 0.01          # J / s of basal repair
    basal_atp_rate: float = 80.0        # J / s baseline cellular energy demand
    waste_fraction: float = 0.05        # kg waste / kg nutrient
    water_loss_rate: float = 2.0e-4    # kg water / s
    thermal_relaxation: float = 0.002
    target_temperature: float = 310.15
    target_ph: float = 7.4
    ph_relaxation: float = 0.01

    def __post_init__(self):
        for name in ("metabolic_rate", "oxygen_per_nutrient", "atp_yield", "atp_capacity",
                     "co2_yield", "heat_yield", "biomass_yield", "repair_cost", "basal_atp_rate",
                     "waste_fraction", "water_loss_rate", "thermal_relaxation",
                     "ph_relaxation"):
            _finite_nonnegative(getattr(self, name), name)
        if self.biomass_yield > 1 or self.waste_fraction > 1:
            raise ValueError("fractions must be <= 1")
        _bounded(self.target_temperature, 150.0, 500.0, "target_temperature")
        _bounded(self.target_ph, 0.0, 14.0, "target_ph")


@dataclass(frozen=True)
class BiologicalCompartment:
    """A bounded physiological compartment."""
    compartment_id: str
    volume: float = 1.0
    temperature: float = 310.15
    ph: float = 7.4
    water: float = 1.0       # kg H2O
    oxygen: float = 0.2      # kg O2
    nutrient: float = 0.2    # kg metabolizable substrate
    atp: float = 1000.0      # J chemical energy carrier
    co2: float = 0.0         # kg CO2
    waste: float = 0.0       # kg waste matter
    biomass: float = 1.0     # kg biological dry/wet mass envelope
    integrity: float = 1.0
    heat_j: float = 0.0      # J stored/dissipated biological heat ledger
    elemental_mass_kg: Mapping[str, float] | None = None  # declared SI dry-biomass basis

    def __post_init__(self):
        if not isinstance(self.compartment_id, str) or not self.compartment_id:
            raise ValueError("compartment_id must be non-empty")
        _finite_nonnegative(self.volume, "volume")
        if self.volume <= 0:
            raise ValueError("volume must be > 0")
        for name in ("temperature", "water", "oxygen", "nutrient", "atp",
                     "co2", "waste", "biomass", "heat_j"):
            _finite_nonnegative(getattr(self, name), name)
        _bounded(self.ph, 0.0, 14.0, "ph")
        _bounded(self.integrity, 0.0, 1.0, "integrity")
        fractions = {"C":0.50,"H":0.08,"O":0.20,"N":0.12,"P":0.01,"S":0.005,
                     "Na":0.005,"K":0.01,"Ca":0.02,"Cl":0.01,"Mg":0.01,"Fe":0.005,"Si":0.01,"trace":0.015}
        raw_comp = ({k: v * float(self.biomass) for k, v in fractions.items()}
                    if self.elemental_mass_kg is None else dict(self.elemental_mass_kg))
        if set(raw_comp) != set(fractions):
            raise ValueError("elemental_mass_kg must contain the declared SI composition basis")
        comp={k:_finite_nonnegative(v,f"elemental_mass_kg[{k}]") for k,v in raw_comp.items()}
        total=sum(comp.values())
        if abs(total-float(self.biomass)) > 1e-9*max(1.0,float(self.biomass)):
            raise ValueError("elemental mass must close exactly to biomass mass")
        object.__setattr__(self,"elemental_mass_kg",comp)

    @property
    def elemental_total_mass_kg(self) -> float:
        return float(sum(self.elemental_mass_kg.values()))

    def digest(self) -> str:
        raw = repr((self.compartment_id, self.volume, self.temperature, self.ph,
                    self.water, self.oxygen, self.nutrient, self.atp, self.co2,
                    self.waste, self.biomass, self.integrity, self.heat_j,
                    tuple(sorted(self.elemental_mass_kg.items())))).encode()
        return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class PhysiologicalRegulation:
    """Non-cognitive whole-organism regulation state for U4 consolidation.

    These signals are deliberately low-dimensional: they represent endocrine,
    immune and aging pressure without introducing cognition or agent policy.
    """
    sodium: float = 140.0
    potassium: float = 4.0
    calcium: float = 2.4
    immune_load: float = 0.0
    immune_capacity: float = 1.0
    hormone_signal: float = 1.0
    age: float = 0.0

    def __post_init__(self):
        for name in ("sodium", "potassium", "calcium", "age"):
            _finite_nonnegative(getattr(self, name), name)
        _bounded(self.immune_load, 0.0, 1.0, "immune_load")
        _bounded(self.immune_capacity, 0.0, 1.0, "immune_capacity")
        _bounded(self.hormone_signal, 0.0, 2.0, "hormone_signal")


@dataclass(frozen=True)
class BiologicalState:
    schema: str
    compartments: Mapping[str, BiologicalCompartment]
    time: float = 0.0
    regulation: PhysiologicalRegulation = PhysiologicalRegulation()

    def __post_init__(self):
        if self.schema != U4_SCHEMA:
            raise ValueError("unsupported biology schema")
        clean = {str(k): v for k, v in self.compartments.items()}
        if any(k != v.compartment_id for k, v in clean.items()):
            raise ValueError("compartment mapping key must equal compartment_id")
        if not math.isfinite(self.time) or self.time < 0:
            raise ValueError("time must be finite and >= 0")
        if not isinstance(self.regulation, PhysiologicalRegulation):
            raise ValueError("regulation must be PhysiologicalRegulation")
        object.__setattr__(self, "compartments", clean)
        object.__setattr__(self, "time", float(self.time))

    @property
    def total_biomass(self) -> float:
        return float(sum(c.biomass for c in self.compartments.values()))

    @property
    def total_water(self) -> float:
        return float(sum(c.water for c in self.compartments.values()))

    @property
    def total_atp(self) -> float:
        """Total ATP energy in joules."""
        return float(sum(c.atp for c in self.compartments.values()))

    def digest(self) -> str:
        h = hashlib.sha256(); h.update(self.schema.encode()); h.update(repr(self.time).encode())
        for cid in sorted(self.compartments):
            h.update(cid.encode()); h.update(self.compartments[cid].digest().encode())
        h.update(repr(self.regulation).encode())
        return h.hexdigest()


def create_biology(compartments: Mapping[str, BiologicalCompartment] | None = None,
                   *, time: float = 0.0) -> BiologicalState:
    return BiologicalState(U4_SCHEMA, compartments or {}, time)


def _metabolism(c: BiologicalCompartment, h: float, p: BiologicalParameters):
    # Limiter keeps nutrient and oxygen non-negative and makes the step
    # deterministic even when dt is large.
    demand = p.metabolic_rate * h * c.integrity
    available = min(c.nutrient, c.oxygen / max(p.oxygen_per_nutrient, 1e-15))
    used = min(demand, max(0.0, available))
    nutrient = c.nutrient - used
    oxygen = c.oxygen - p.oxygen_per_nutrient * used
    atp = c.atp + p.atp_yield * used
    co2 = c.co2 + p.co2_yield * used
    heat = p.heat_yield * used
    biomass_gain = p.biomass_yield * used
    waste = c.waste + p.waste_fraction * used
    # ATP is consumed for basal repair. Damage is reduced only when energy is
    # available; this prevents "free healing".
    repair = min(atp, p.repair_cost * h * (0.5 + c.integrity))
    atp -= repair
    # Basal cellular work consumes ATP even with no motor command.
    basal_demand = p.basal_atp_rate * h
    basal_used = min(atp, basal_demand)
    atp -= basal_used
    # ATP is a finite intracellular reservoir; it cannot accumulate without bound.
    atp = min(p.atp_capacity, max(0.0, atp))
    energy_deficit = max(0.0, basal_demand - basal_used)
    damage = p.repair_cost * h * (1.0 - c.integrity) + 1.0e-6 * energy_deficit
    integrity = max(0.0, min(1.0, c.integrity + repair * 0.02 - damage))
    biomass = max(0.0, c.biomass + biomass_gain - damage * 0.002)
    biomass_loss = max(0.0, c.biomass - biomass)
    return nutrient, oxygen, atp, co2, waste, biomass, integrity, heat, used, repair, biomass_loss


def exchange_compartments(state: BiologicalState, dt: float, *, permeability: float = 0.1) -> BiologicalState:
    """Conservative pairwise exchange of mobile pools between compartments."""
    h = float(dt)
    k = float(permeability)
    if not math.isfinite(h) or h <= 0:
        raise ValueError("dt must be finite and > 0")
    if not math.isfinite(k) or k < 0 or k > 1:
        raise ValueError("permeability must be in [0,1]")
    ids = sorted(state.compartments)
    data = {cid: state.compartments[cid] for cid in ids}
    for i, left_id in enumerate(ids):
        for right_id in ids[i + 1:]:
            left, right = data[left_id], data[right_id]
            # The transfer coefficient is bounded so large dt cannot move more
            # than the donor pool. Equal and opposite transfers preserve totals.
            alpha = min(0.5, k * h)
            updates = {}
            for field in ("water", "oxygen", "nutrient", "co2"):
                a = getattr(left, field); b = getattr(right, field)
                delta = alpha * (a - b)
                if delta > 0:
                    delta = min(delta, a)
                else:
                    delta = -min(-delta, b)
                updates[field] = delta
            data[left_id] = replace(left, **{f: getattr(left, f) - d for f, d in updates.items()})
            data[right_id] = replace(right, **{f: getattr(right, f) + d for f, d in updates.items()})
    return BiologicalState(U4_SCHEMA, data, state.time)


def step_biology(state: BiologicalState, dt: float, *,
                 parameters: BiologicalParameters | None = None,
                 permeability: float = 0.0) -> tuple[BiologicalState, Mapping[str, Any]]:
    h = float(dt)
    if not math.isfinite(h) or h <= 0:
        raise ValueError("dt must be finite and > 0")
    p = parameters or BiologicalParameters()
    exchanged = exchange_compartments(state, h, permeability=permeability) if permeability > 0 else state
    out = {}
    total_used = total_heat = total_atp_generated = 0.0
    # Whole-organism regulation is deliberately slow and bounded. It provides
    # endocrine/immune/aging pressure without creating cognition.
    reg = exchanged.regulation
    immune_stress = 0.0
    for cid in sorted(exchanged.compartments):
        c = exchanged.compartments[cid]
        nutrient, oxygen, atp, co2, waste, biomass, integrity, heat, used, repair, biomass_loss = _metabolism(c, h, p)
        waste += biomass_loss
        water = max(0.0, c.water - p.water_loss_rate * h)
        # Homeostasis is a bounded relaxation toward target conditions.
        temp = c.temperature + p.thermal_relaxation * h * (p.target_temperature - c.temperature) + heat * 0.001
        ph = c.ph + p.ph_relaxation * h * (p.target_ph - c.ph) - 0.001 * (co2 - c.co2)
        ph = min(14.0, max(0.0, ph))
        # Excess waste/CO2 acts as a non-specific physiological stressor.
        local_stress = min(1.0, max(0.0, 0.02 * c.waste + 0.01 * c.co2))
        immune_stress += local_stress
        scale = biomass / c.biomass if c.biomass > 0 else 0.0
        composition = {k: v * scale for k, v in c.elemental_mass_kg.items()}
        out[cid] = BiologicalCompartment(cid, c.volume, temp, ph, water, oxygen,
                                         nutrient, atp, co2, waste, biomass, integrity,
                                         c.heat_j + heat + repair, composition)
        total_used += used; total_heat += heat; total_atp_generated += p.atp_yield * used
    mean_stress = immune_stress / max(1, len(out))
    age = reg.age + h
    # Ageing is a slow, resource-independent pressure; repair can offset it
    # only through the existing metabolism, never by creating energy.
    age_pressure = min(0.02, 1e-6 * age * h)
    immune_load = min(1.0, max(0.0, reg.immune_load + h * (mean_stress - 0.15 * reg.immune_load)))
    immune_capacity = max(0.0, min(1.0, reg.immune_capacity + h * (0.002 - 0.001 * immune_load) - age_pressure))
    hormone_target = max(0.0, min(2.0, 1.0 + 0.5 * (1.0 - immune_load) - 0.25 * min(1.0, age / 100000.0)))
    hormone_signal = max(0.0, min(2.0, reg.hormone_signal + 0.01 * h * (hormone_target - reg.hormone_signal)))
    new_reg = PhysiologicalRegulation(
        sodium=reg.sodium, potassium=reg.potassium, calcium=reg.calcium,
        immune_load=immune_load, immune_capacity=immune_capacity,
        hormone_signal=hormone_signal, age=age)
    # Ageing/damage may reduce biomass, but biomass is matter: it is routed to
    # the compartment waste pool instead of silently disappearing.
    if age_pressure > 0 and out:
        adjusted = {}
        for cid, c in out.items():
            aged_biomass = max(0.0, c.biomass - age_pressure * 0.02)
            loss = max(0.0, c.biomass - aged_biomass)
            scale = aged_biomass / c.biomass if c.biomass > 0 else 0.0
            aged_comp = {k: v * scale for k, v in c.elemental_mass_kg.items()}
            adjusted[cid] = replace(c, integrity=max(0.0, c.integrity - age_pressure),
                                     biomass=aged_biomass, waste=c.waste + loss,
                                     elemental_mass_kg=aged_comp)
        out = adjusted
    new = BiologicalState(U4_SCHEMA, out, state.time + h, new_reg)
    total_water_loss = max(0.0, state.total_water - new.total_water)
    total_biomass_loss = max(0.0, state.total_biomass - new.total_biomass)
    report = {
        "dt": h,
        "nutrient_consumed": total_used,
        "heat_generated": total_heat,
        "atp_generated": total_atp_generated,
        "biomass_before": state.total_biomass,
        "biomass_after": new.total_biomass,
        "water_before": state.total_water,
        "water_after": new.total_water,
        "water_loss_to_environment": total_water_loss,
        "biomass_loss_routed_to_waste": total_biomass_loss,
        "exchange_enabled": bool(permeability > 0),
        "exchange_water_residual": new.total_water - exchanged.total_water,
        "exchange_oxygen_residual": sum(c.oxygen for c in new.compartments.values()) - sum(c.oxygen for c in exchanged.compartments.values()),
        "digest": new.digest(),
        "immune_load": new.regulation.immune_load,
        "immune_capacity": new.regulation.immune_capacity,
        "hormone_signal": new.regulation.hormone_signal,
        "age": new.regulation.age,
        "age_pressure": age_pressure,
        "positive_pools": all(min(c.water, c.oxygen, c.nutrient, c.atp, c.co2, c.waste, c.biomass) >= 0
                               for c in new.compartments.values()),
        "homeostasis_finite": all(math.isfinite(c.temperature) and 0 <= c.ph <= 14
                                   for c in new.compartments.values()),
    }
    report["certified_step"] = bool(report["positive_pools"] and report["homeostasis_finite"])
    return new, report


def attach_biology(state: WorldState, biology: BiologicalState) -> WorldState:
    meta = dict(state.metadata)
    meta["u4_biology"] = {
        "schema": biology.schema,
        "compartments": sorted(biology.compartments),
        "time": biology.time,
        "digest": biology.digest(),
    }
    return replace(state, metadata=meta)


class BiologyWorld:
    """World-Core handoff for deterministic physiology."""
    def __init__(self, core: WorldCore, biology: BiologicalState):
        if abs(biology.time - core.state.clock.time) > 1e-9:
            raise ValueError("biology time must match world clock")
        self.core, self.biology, self.last_report = core, biology, None
        self.core._state = attach_biology(core.state, biology)

    @property
    def state(self):
        return self.core.state

    def step(self, dt=None, *, parameters=None):
        h = self.state.clock.dt if dt is None else float(dt)
        new, report = step_biology(self.biology, h, parameters=parameters)
        self.core.step(h, updater=lambda s, _h: attach_biology(s, new))
        self.biology, self.last_report = new, report
        self.core.emit("biology_step", payload={"dt": h, "certified": report["certified_step"],
                                                 "biology_digest": new.digest()})
        return report






