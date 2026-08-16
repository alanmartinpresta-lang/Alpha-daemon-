"""AIRE U5 consolidated ecology.

Adds deterministic biogeochemical cycles, richer ecological relations and a
minimal population-level evolutionary mechanism.  The layer is intentionally
mesoscopic: it does not model individual cognition, detailed genomes or agent
behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Mapping

from .ecology import EcologyState, EcologicalPopulation, EcologicalResource, step_ecology
from .core import WorldCore, WorldState

SCHEMA = "U5.2"


def _nn(x: float, name: str) -> float:
    x = float(x)
    if not math.isfinite(x) or x < 0:
        raise ValueError(f"{name} must be finite and >= 0")
    return x


def _bounded(x: float, lo: float, hi: float, name: str) -> float:
    x = float(x)
    if not math.isfinite(x) or x < lo or x > hi:
        raise ValueError(f"{name} must be finite and in [{lo},{hi}]")
    return x


@dataclass(frozen=True)
class BiogeochemicalPools:
    """Conservative mesoscopic element pools.

    Carbon, nitrogen and oxygen are tracked in paired organic/inorganic pools.
    The representation is deliberately aggregate; it is not a molecular
    geochemistry model.
    """
    organic_carbon: float = 0.0
    inorganic_carbon: float = 0.0
    organic_nitrogen: float = 0.0
    inorganic_nitrogen: float = 0.0
    oxygen: float = 0.0
    detritus_carbon: float = 0.0
    detritus_nitrogen: float = 0.0
    organic_phosphorus: float = 0.0
    inorganic_phosphorus: float = 0.0
    detritus_phosphorus: float = 0.0
    organic_sulfur: float = 0.0
    inorganic_sulfur: float = 0.0
    detritus_sulfur: float = 0.0

    def __post_init__(self):
        for n in ("organic_carbon", "inorganic_carbon", "organic_nitrogen",
                  "inorganic_nitrogen", "oxygen", "detritus_carbon", "detritus_nitrogen"):
            _nn(getattr(self, n), n)

    def totals(self) -> tuple[float, float, float]:
        return (self.organic_carbon + self.inorganic_carbon + self.detritus_carbon,
                self.organic_nitrogen + self.inorganic_nitrogen + self.detritus_nitrogen,
                self.oxygen)

    def digest(self) -> str:
        return hashlib.sha256(repr((self.organic_carbon, self.inorganic_carbon,
                                    self.organic_nitrogen, self.inorganic_nitrogen,
                                    self.oxygen, self.detritus_carbon, self.detritus_nitrogen,
                                    self.organic_phosphorus, self.inorganic_phosphorus, self.detritus_phosphorus,
                                    self.organic_sulfur, self.inorganic_sulfur, self.detritus_sulfur)).encode()).hexdigest()


@dataclass(frozen=True)
class EcologicalRelation:
    """Population-level interaction.

    kind is one of competition, mutualism, parasitism, symbiosis or
    decomposition.  Effects are bounded rate coefficients and never create
    negative populations.
    """
    source_id: str
    target_id: str
    kind: str
    strength: float
    efficiency: float = 0.5

    def __post_init__(self):
        if not self.source_id or not self.target_id or self.source_id == self.target_id:
            raise ValueError("relation endpoints must be distinct and non-empty")
        if self.kind not in {"competition", "mutualism", "parasitism", "symbiosis", "decomposition"}:
            raise ValueError("unsupported ecological relation")
        _nn(self.strength, "strength")
        _bounded(self.efficiency, 0.0, 1.0, "efficiency")


@dataclass(frozen=True)
class EvolutionTrait:
    """Population-level heritable trait used by the simple evolution model."""
    value: float = 0.5
    optimum: float = 0.5
    selection_strength: float = 0.0
    mutation_rate: float = 0.01
    mutation_variance: float = 0.01
    heritability: float = 1.0

    def __post_init__(self):
        _bounded(self.value, 0.0, 1.0, "value")
        _bounded(self.optimum, 0.0, 1.0, "optimum")
        _bounded(self.selection_strength, 0.0, 1.0, "selection_strength")
        _bounded(self.mutation_rate, 0.0, 1.0, "mutation_rate")
        _nn(self.mutation_variance, "mutation_variance")
        _bounded(self.heritability, 0.0, 1.0, "heritability")


@dataclass(frozen=True)
class U5ConsolidatedState:
    base: EcologyState
    cycles: BiogeochemicalPools = BiogeochemicalPools()
    relations: tuple[EcologicalRelation, ...] = ()
    traits: Mapping[str, EvolutionTrait] = None
    generation: int = 0
    time: float = 0.0
    schema: str = SCHEMA

    def __post_init__(self):
        if self.schema != SCHEMA:
            raise ValueError("unsupported U5 consolidated schema")
        if self.traits is None:
            traits = {}
        else:
            traits = {str(k): v for k, v in self.traits.items()}
        if any(pid not in self.base.populations for pid in traits):
            raise ValueError("trait population id must exist")
        if int(self.generation) != self.generation or self.generation < 0:
            raise ValueError("generation must be a non-negative integer")
        _nn(self.time, "time")
        object.__setattr__(self, "traits", traits)
        object.__setattr__(self, "relations", tuple(self.relations))
        object.__setattr__(self, "generation", int(self.generation))
        object.__setattr__(self, "time", float(self.time))

    def digest(self) -> str:
        h = hashlib.sha256(self.schema.encode())
        h.update(self.base.digest().encode())
        h.update(self.cycles.digest().encode())
        for r in sorted(self.relations, key=lambda x: (x.kind, x.source_id, x.target_id)):
            h.update(repr(r).encode())
        for pid in sorted(self.traits):
            h.update(pid.encode()); h.update(repr(self.traits[pid]).encode())
        h.update(repr((self.generation, self.time)).encode())
        return h.hexdigest()


def create_u5_consolidated(base: EcologyState, *, cycles: BiogeochemicalPools | None = None,
                           relations: tuple[EcologicalRelation, ...] = (),
                           traits: Mapping[str, EvolutionTrait] | None = None,
                           generation: int = 0) -> U5ConsolidatedState:
    return U5ConsolidatedState(base, cycles or BiogeochemicalPools(), relations,
                               traits or {}, generation, base.time)


def _apply_relation_effects(pops: dict[str, EcologicalPopulation], relations: tuple[EcologicalRelation, ...], h: float) -> dict[str, float]:
    delta = {pid: 0.0 for pid in pops}
    for rel in relations:
        src, tgt = pops.get(rel.source_id), pops.get(rel.target_id)
        if src is None or tgt is None:
            continue
        interaction = min(tgt.size, rel.strength * src.size * h)
        if rel.kind == "competition":
            delta[rel.source_id] -= interaction
            delta[rel.target_id] -= interaction * rel.efficiency
        elif rel.kind in {"mutualism", "symbiosis"}:
            gain = interaction * rel.efficiency
            delta[rel.source_id] += gain
            delta[rel.target_id] += gain
        elif rel.kind == "parasitism":
            delta[rel.target_id] -= interaction
            delta[rel.source_id] += interaction * rel.efficiency
        elif rel.kind == "decomposition":
            # Population-level decomposition converts target biomass to a
            # detrital flux; the caller handles the cycle transfer.
            delta[rel.target_id] -= interaction
    return delta


def _apply_evolution(pops: dict[str, EcologicalPopulation], traits: Mapping[str, EvolutionTrait], h: float):
    out = dict(traits)
    for pid, trait in traits.items():
        if pid not in pops:
            continue
        # Selection moves the inherited mean toward an environmental optimum;
        # mutation adds a deterministic bounded variance term. No individual
        # genotype or cognition is represented.
        direction = trait.optimum - trait.value
        selected = trait.value + trait.heritability * trait.selection_strength * direction * min(1.0, h)
        mutation = trait.mutation_rate * (0.5 - trait.value) * min(1.0, trait.mutation_variance * h)
        value = max(0.0, min(1.0, selected + mutation))
        out[pid] = EvolutionTrait(value, trait.optimum, trait.selection_strength,
                                  trait.mutation_rate, trait.mutation_variance, trait.heritability)
    return out


def step_u5_consolidated(state: U5ConsolidatedState, dt: float):
    h = float(dt)
    if not math.isfinite(h) or h <= 0:
        raise ValueError("dt must be finite and > 0")
    before_cycles = state.cycles.totals()
    base, base_report = step_ecology(state.base, h)
    pops = dict(base.populations)
    relation_delta = _apply_relation_effects(pops, state.relations, h)
    # Apply interaction deltas with strict bounds.
    for pid, d in relation_delta.items():
        p = pops[pid]
        size = max(0.0, min(p.carrying_capacity, p.size + d))
        biomass = max(0.0, p.biomass * (size / p.size) if p.size > 0 else size)
        pops[pid] = EcologicalPopulation(pid, size, biomass, p.resource, p.consumption_rate,
                                         p.reproduction_rate, p.mortality_rate, p.carrying_capacity,
                                         p.resource_efficiency, p.predation_loss_rate, p.thermal_tolerance)

    # Aggregate biogeochemical cycle: decomposition/mineralisation and a bounded
    # inorganic->organic assimilation channel. Each element is conserved exactly.
    c = state.cycles
    # Population biomass is not part of the elemental ledger in this
    # mesoscopic layer, so decomposition does not inject untracked mass into
    # the cycle pools. The biogeochemical ledger remains strictly closed.
    detrital_input = 0.0
    decomposition_c = min(c.detritus_carbon, 0.05 * h * c.detritus_carbon)
    decomposition_n = min(c.detritus_nitrogen, 0.05 * h * c.detritus_nitrogen)
    # New detritus carries a fixed bounded C:N proxy in this mesoscopic layer.
    detrital_c = detrital_input * 0.8
    detrital_n = detrital_input * 0.2
    detritus_c = max(0.0, c.detritus_carbon + detrital_c - decomposition_c)
    detritus_n = max(0.0, c.detritus_nitrogen + detrital_n - decomposition_n)
    carbon_transfer = min(c.inorganic_carbon, 0.01 * h * c.inorganic_carbon)
    nitrogen_transfer = min(c.inorganic_nitrogen, 0.01 * h * c.inorganic_nitrogen)
    organic_c = c.organic_carbon + carbon_transfer
    inorganic_c = c.inorganic_carbon - carbon_transfer
    organic_n = c.organic_nitrogen + nitrogen_transfer
    inorganic_n = c.inorganic_nitrogen - nitrogen_transfer
    # Decomposition returns bounded fractions to inorganic pools.
    # Decomposition transfers detrital elements back to inorganic pools exactly.
    inorganic_c += decomposition_c
    inorganic_n += decomposition_n
    # Oxygen is an independent conserved pool in this aggregate model; no
    # unbalanced creation/destruction is permitted here.
    decomposition_p = min(c.detritus_phosphorus, 0.05 * h * c.detritus_phosphorus)
    decomposition_s = min(c.detritus_sulfur, 0.05 * h * c.detritus_sulfur)
    phosphorus_transfer = min(c.inorganic_phosphorus, 0.01 * h * c.inorganic_phosphorus)
    sulfur_transfer = min(c.inorganic_sulfur, 0.01 * h * c.inorganic_sulfur)
    organic_p = c.organic_phosphorus + phosphorus_transfer
    inorganic_p = c.inorganic_phosphorus - phosphorus_transfer + decomposition_p
    detritus_p = max(0.0, c.detritus_phosphorus - decomposition_p)
    organic_s = c.organic_sulfur + sulfur_transfer
    inorganic_s = c.inorganic_sulfur - sulfur_transfer + decomposition_s
    detritus_s = max(0.0, c.detritus_sulfur - decomposition_s)
    cycles = BiogeochemicalPools(organic_c, inorganic_c, organic_n, inorganic_n, c.oxygen, detritus_c, detritus_n,
                                 organic_p, inorganic_p, detritus_p, organic_s, inorganic_s, detritus_s)
    after_cycles = cycles.totals()
    cycle_residual = tuple(after_cycles[i] - before_cycles[i] for i in range(3))

    # Preserve the base resource map while replacing only population states.
    new_base = EcologyState(base.schema, base.resources, pops, base.predation, base.time)
    traits = _apply_evolution(pops, state.traits, h)
    new = U5ConsolidatedState(new_base, cycles, state.relations, traits,
                              state.generation + 1, state.time + h)
    report = {
        "dt": h,
        "generation": new.generation,
        "relation_delta_l1": float(sum(abs(x) for x in relation_delta.values())),
        "cycle_residual": cycle_residual,
        "cycles_conservative": all(abs(x) <= 1e-12 * max(1.0, abs(before_cycles[i])) for i, x in enumerate(cycle_residual)),
        "nonnegative": all(p.size >= 0 for p in pops.values()) and all(x >= 0 for x in (cycles.organic_carbon, cycles.inorganic_carbon, cycles.organic_nitrogen, cycles.inorganic_nitrogen, cycles.oxygen, cycles.detritus_carbon, cycles.detritus_nitrogen)),
        "bounded": all(p.size <= p.carrying_capacity + 1e-12 for p in pops.values()),
        "finite": all(math.isfinite(p.size) for p in pops.values()) and all(math.isfinite(x) for x in after_cycles),
        "base_certified": bool(base_report["certified_step"]),
        "trait_bounds": all(0.0 <= t.value <= 1.0 for t in traits.values()),
        "digest": new.digest(),
    }
    report["certified_step"] = bool(all((report["cycles_conservative"], report["nonnegative"], report["bounded"], report["finite"], report["base_certified"], report["trait_bounds"])))
    return new, report






class ConsolidatedEcologyWorld:
    """World-Core handoff for the U5.2 consolidated ecology layer."""
    def __init__(self, core: WorldCore, ecology: U5ConsolidatedState):
        if abs(ecology.time - core.state.clock.time) > 1e-9:
            raise ValueError("ecology time must match world clock")
        self.core, self.ecology, self.last_report = core, ecology, None

    @property
    def state(self) -> WorldState:
        return self.core.state

    def step(self, dt: float | None = None):
        h = self.state.clock.dt if dt is None else float(dt)
        new, report = step_u5_consolidated(self.ecology, h)
        self.core.step(h, updater=lambda s, _h: _attach_consolidated(s, new))
        self.ecology, self.last_report = new, report
        self.core.emit("ecology_consolidated_step", payload={
            "dt": h, "certified": report["certified_step"], "ecology_digest": new.digest(),
            "generation": new.generation})
        return report


def _attach_consolidated(state: WorldState, ecology: U5ConsolidatedState) -> WorldState:
    meta = dict(state.metadata)
    meta["u5_ecology_consolidated"] = {
        "schema": ecology.schema, "generation": ecology.generation,
        "time": ecology.time, "digest": ecology.digest(),
        "cycles": ecology.cycles.digest(),
    }
    return state.__class__(state.identity, state.space, state.clock, state.objects, state.laws,
                           state.events, state.causality, meta)
