"""AIRE U5 — deterministic mesoscopic ecology.

U5 adds population/resource dynamics above U4 physiology. It models bounded
resource pools, organism populations, reproduction, mortality, competition and
predation using deterministic rate equations. It intentionally does not model
individual cognition, genetics or agent behaviour; those belong to U6-U9.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping

from .core import WorldCore, WorldState

U5_SCHEMA = "U5.1"


def _nn(x: float, name: str) -> float:
    x = float(x)
    if not math.isfinite(x) or x < 0:
        raise ValueError(f"{name} must be finite and >= 0")
    return x


def _frac(x: float, name: str) -> float:
    x = float(x)
    if not math.isfinite(x) or x < 0 or x > 1:
        raise ValueError(f"{name} must be finite and in [0,1]")
    return x


@dataclass(frozen=True)
class EcologicalResource:
    resource_id: str
    amount: float
    regeneration_rate: float = 0.0
    carrying_capacity: float = 1.0

    def __post_init__(self):
        if not self.resource_id:
            raise ValueError("resource_id must be non-empty")
        _nn(self.amount, "amount")
        _nn(self.regeneration_rate, "regeneration_rate")
        _nn(self.carrying_capacity, "carrying_capacity")
        if self.carrying_capacity <= 0:
            raise ValueError("carrying_capacity must be > 0")

    def digest(self) -> str:
        return hashlib.sha256(repr((self.resource_id, self.amount,
                                    self.regeneration_rate,
                                    self.carrying_capacity)).encode()).hexdigest()


@dataclass(frozen=True)
class EcologicalPopulation:
    population_id: str
    size: float
    biomass: float = 1.0
    resource: str | None = None
    consumption_rate: float = 0.0
    reproduction_rate: float = 0.0
    mortality_rate: float = 0.0
    carrying_capacity: float = 1e9
    resource_efficiency: float = 1.0
    predation_loss_rate: float = 0.0
    thermal_tolerance: tuple[float, float] = (250.0, 330.0)

    def __post_init__(self):
        if not self.population_id:
            raise ValueError("population_id must be non-empty")
        _nn(self.size, "size")
        _nn(self.biomass, "biomass")
        for n in ("consumption_rate", "reproduction_rate", "mortality_rate",
                  "resource_efficiency", "predation_loss_rate"):
            _nn(getattr(self, n), n)
        _nn(self.carrying_capacity, "carrying_capacity")
        if self.carrying_capacity <= 0:
            raise ValueError("carrying_capacity must be > 0")
        lo, hi = self.thermal_tolerance
        if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
            raise ValueError("thermal_tolerance must be finite and ordered")

    def digest(self) -> str:
        return hashlib.sha256(repr((self.population_id, self.size, self.biomass,
                                    self.resource, self.consumption_rate,
                                    self.reproduction_rate, self.mortality_rate,
                                    self.carrying_capacity, self.resource_efficiency,
                                    self.predation_loss_rate,
                                    self.thermal_tolerance)).encode()).hexdigest()


@dataclass(frozen=True)
class PredationLink:
    predator_id: str
    prey_id: str
    rate: float
    efficiency: float = 0.5

    def __post_init__(self):
        if not self.predator_id or not self.prey_id or self.predator_id == self.prey_id:
            raise ValueError("predator and prey ids must be distinct and non-empty")
        _nn(self.rate, "rate")
        _frac(self.efficiency, "efficiency")


@dataclass(frozen=True)
class EcologyState:
    schema: str
    resources: Mapping[str, EcologicalResource]
    populations: Mapping[str, EcologicalPopulation]
    predation: tuple[PredationLink, ...] = ()
    time: float = 0.0

    def __post_init__(self):
        if self.schema != U5_SCHEMA:
            raise ValueError("unsupported ecology schema")
        r = {str(k): v for k, v in self.resources.items()}
        p = {str(k): v for k, v in self.populations.items()}
        if any(k != v.resource_id for k, v in r.items()):
            raise ValueError("resource mapping key must equal resource_id")
        if any(k != v.population_id for k, v in p.items()):
            raise ValueError("population mapping key must equal population_id")
        if not math.isfinite(float(self.time)) or self.time < 0:
            raise ValueError("time must be finite and >= 0")
        object.__setattr__(self, "resources", r)
        object.__setattr__(self, "populations", p)
        object.__setattr__(self, "predation", tuple(self.predation))
        object.__setattr__(self, "time", float(self.time))

    @property
    def total_population(self) -> float:
        return float(sum(p.size for p in self.populations.values()))

    @property
    def total_resource(self) -> float:
        return float(sum(r.amount for r in self.resources.values()))

    def digest(self) -> str:
        h = hashlib.sha256(); h.update(self.schema.encode()); h.update(repr(self.time).encode())
        for rid in sorted(self.resources):
            h.update(rid.encode()); h.update(self.resources[rid].digest().encode())
        for pid in sorted(self.populations):
            h.update(pid.encode()); h.update(self.populations[pid].digest().encode())
        for link in sorted(self.predation, key=lambda x: (x.predator_id, x.prey_id)):
            h.update(repr(link).encode())
        return h.hexdigest()


def create_ecology(resources: Mapping[str, EcologicalResource] | None = None,
                   populations: Mapping[str, EcologicalPopulation] | None = None,
                   predation: tuple[PredationLink, ...] = (), *, time: float = 0.0) -> EcologyState:
    return EcologyState(U5_SCHEMA, resources or {}, populations or {}, predation, time)


def step_ecology(state: EcologyState, dt: float) -> tuple[EcologyState, Mapping[str, Any]]:
    h = float(dt)
    if not math.isfinite(h) or h <= 0:
        raise ValueError("dt must be finite and > 0")
    # Deterministic sub-stepping protects explicit rate equations from very large dt.
    n = max(1, int(math.ceil(h / 1.0)))
    sh = h / n
    resources = dict(state.resources)
    populations = dict(state.populations)
    total_consumed = total_regenerated = total_births = total_deaths = total_predation = 0.0

    for _ in range(n):
        # Resource regeneration is logistic and bounded by carrying capacity.
        for rid, r in list(resources.items()):
            growth = r.regeneration_rate * r.amount * max(0.0, 1.0 - r.amount / r.carrying_capacity)
            amount = min(r.carrying_capacity, max(0.0, r.amount + growth * sh))
            total_regenerated += max(0.0, amount - r.amount)
            resources[rid] = EcologicalResource(rid, amount, r.regeneration_rate, r.carrying_capacity)

        consumption_by_pop: dict[str, float] = {rid: 0.0 for rid in resources}
        births: dict[str, float] = {}
        deaths: dict[str, float] = {}
        predation_loss: dict[str, float] = {pid: 0.0 for pid in populations}

        for link in state.predation:
            predator = populations.get(link.predator_id)
            prey = populations.get(link.prey_id)
            if predator is None or prey is None:
                continue
            eaten = min(prey.size, link.rate * predator.size * sh)
            predation_loss[prey.population_id] += eaten
            births[link.predator_id] = births.get(link.predator_id, 0.0) + eaten * link.efficiency
            total_predation += eaten

        for pid, p in populations.items():
            resource_amount = resources[p.resource].amount if p.resource in resources else 0.0
            desired = p.consumption_rate * p.size * sh
            consumed = min(resource_amount, desired)
            if p.resource in consumption_by_pop:
                consumption_by_pop[p.resource] += consumed
            else:
                consumed = 0.0
            # Resource availability controls reproduction; carrying capacity adds density dependence.
            food_factor = 0.0 if desired <= 0 else consumed / desired
            density_factor = max(0.0, 1.0 - p.size / p.carrying_capacity)
            temp_factor = 1.0
            # Ecology state is deliberately environment-agnostic; tolerance is used only
            # if a future environment handoff provides temperature in metadata.
            births[pid] = births.get(pid, 0.0) + p.reproduction_rate * p.size * food_factor * density_factor * sh
            deaths[pid] = deaths.get(pid, 0.0) + p.mortality_rate * p.size * sh + predation_loss.get(pid, 0.0)

        for rid, consumed in consumption_by_pop.items():
            r = resources[rid]
            actual = min(r.amount, max(0.0, consumed))
            resources[rid] = EcologicalResource(rid, r.amount - actual, r.regeneration_rate, r.carrying_capacity)
            total_consumed += actual

        for pid, p in populations.items():
            b = births.get(pid, 0.0)
            d = min(p.size + b, deaths.get(pid, 0.0))
            size = max(0.0, min(p.carrying_capacity, p.size + b - d))
            # Biomass tracks population size and cannot become negative.
            biomass = max(0.0, p.biomass * (size / p.size) if p.size > 0 else size)
            populations[pid] = EcologicalPopulation(
                pid, size, biomass, p.resource, p.consumption_rate,
                p.reproduction_rate, p.mortality_rate, p.carrying_capacity,
                p.resource_efficiency, p.predation_loss_rate, p.thermal_tolerance)
            total_births += b
            total_deaths += d

    new = EcologyState(U5_SCHEMA, resources, populations, state.predation, state.time + h)
    report = {
        "dt": h,
        "resource_before": state.total_resource,
        "resource_after": new.total_resource,
        "population_before": state.total_population,
        "population_after": new.total_population,
        "resource_consumed": total_consumed,
        "resource_regenerated": total_regenerated,
        "births": total_births,
        "deaths": total_deaths,
        "predation": total_predation,
        "nonnegative": all(r.amount >= 0 for r in new.resources.values()) and all(p.size >= 0 for p in new.populations.values()),
        "bounded_resources": all(r.amount <= r.carrying_capacity + 1e-12 for r in new.resources.values()),
        "bounded_populations": all(p.size <= p.carrying_capacity + 1e-12 for p in new.populations.values()),
        "finite": all(math.isfinite(r.amount) for r in new.resources.values()) and all(math.isfinite(p.size) for p in new.populations.values()),
        "digest": new.digest(),
    }
    report["certified_step"] = bool(report["nonnegative"] and report["bounded_resources"] and report["bounded_populations"] and report["finite"])
    return new, report


def attach_ecology(state: WorldState, ecology: EcologyState) -> WorldState:
    meta = dict(state.metadata)
    meta["u5_ecology"] = {
        "schema": ecology.schema,
        "resources": sorted(ecology.resources),
        "populations": sorted(ecology.populations),
        "time": ecology.time,
        "digest": ecology.digest(),
    }
    return state.__class__(state.identity, state.space, state.clock, state.objects, state.laws,
                           state.events, state.causality, meta)


class EcologyWorld:
    """World-Core handoff for deterministic population ecology."""
    def __init__(self, core: WorldCore, ecology: EcologyState):
        if abs(ecology.time - core.state.clock.time) > 1e-9:
            raise ValueError("ecology time must match world clock")
        self.core, self.ecology, self.last_report = core, ecology, None
        self.core._state = attach_ecology(core.state, ecology)

    @property
    def state(self) -> WorldState:
        return self.core.state

    def step(self, dt: float | None = None):
        h = self.state.clock.dt if dt is None else float(dt)
        new, report = step_ecology(self.ecology, h)
        self.core.step(h, updater=lambda s, _h: attach_ecology(s, new))
        self.ecology, self.last_report = new, report
        self.core.emit("ecology_step", payload={"dt": h, "certified": report["certified_step"], "ecology_digest": new.digest()})
        return report




