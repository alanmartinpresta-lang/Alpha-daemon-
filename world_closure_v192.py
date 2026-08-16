"""AIRE V192 — explicit SI biomass/waste and global respiration energy closure.

This layer closes the remaining post-UW bookkeeping gaps without inventing
molecular detail. Biomass and waste are partitioned by an explicit mass
composition into C/N/O and an unresolved-but-mass-accounted remainder.
Respiration chemical free energy is represented by an explicit U3 chemical
energy reservoir and transferred to U4 ATP plus heat. No implicit conversion
is allowed.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import hashlib, math
from typing import Mapping

from .biology import BiologicalState
from .u5_consolidated import U5ConsolidatedState, BiogeochemicalPools
from .interdomain_metabolism import RespirationContract

_EPS = 1e-12

def _nn(x: float, name: str) -> float:
    x=float(x)
    if not math.isfinite(x) or x < 0:
        raise ValueError(f"{name} must be finite and >= 0")
    return x

@dataclass(frozen=True)
class MatterComposition:
    """Mass fractions for a tracked biological material.

    C/N/O are routed into U5 element pools. ``other_fraction`` is explicitly
    retained as a mass-only remainder, so total transferred mass remains closed
    even though those elements are not yet chemically resolved.
    """
    carbon_fraction: float
    nitrogen_fraction: float
    oxygen_fraction: float
    other_fraction: float

    def __post_init__(self):
        vals=(self.carbon_fraction,self.nitrogen_fraction,self.oxygen_fraction,self.other_fraction)
        if any(not math.isfinite(float(v)) or float(v)<0 for v in vals):
            raise ValueError("composition fractions must be finite and >= 0")
        if abs(sum(vals)-1.0) > 1e-12:
            raise ValueError("composition fractions must sum to 1")

    def mass_split(self, mass_kg: float) -> Mapping[str,float]:
        m=_nn(mass_kg,"mass_kg")
        return {
            "C":m*self.carbon_fraction,
            "N":m*self.nitrogen_fraction,
            "O":m*self.oxygen_fraction,
            "other":m*self.other_fraction,
        }

@dataclass(frozen=True)
class MatterClosureState:
    """Mass-accounting state spanning U4 and U5."""
    unresolved_mass_kg: float = 0.0
    digest_seed: str = "V192"

    def __post_init__(self):
        _nn(self.unresolved_mass_kg,"unresolved_mass_kg")
        if not self.digest_seed:
            raise ValueError("digest_seed must be non-empty")

    def digest(self)->str:
        return hashlib.sha256(repr((self.unresolved_mass_kg,self.digest_seed)).encode()).hexdigest()

@dataclass(frozen=True)
class EnergyClosureState:
    """Explicit energy reservoirs in joules.

    U3 chemical free energy is debited when respiration occurs. U4 ATP and
    heat receive exactly the declared yield. Internal transfers conserve the
    sum of these reservoirs.
    """
    chemical_free_energy_j: float
    atp_j: float
    heat_j: float

    def __post_init__(self):
        for n in ("chemical_free_energy_j","atp_j","heat_j"):
            _nn(getattr(self,n),n)

    @property
    def total_j(self)->float:
        return self.chemical_free_energy_j+self.atp_j+self.heat_j

    def digest(self)->str:
        return hashlib.sha256(repr((self.chemical_free_energy_j,self.atp_j,self.heat_j)).encode()).hexdigest()

    def apply_respiration(self, contract: RespirationContract, substrate_kg: float) -> "EnergyClosureState":
        m=_nn(substrate_kg,"substrate_kg")
        energy=m*contract.free_energy_j_per_kg_substrate
        if energy > self.chemical_free_energy_j + 1e-12*max(1.0,self.chemical_free_energy_j):
            raise ValueError("insufficient chemical free energy reservoir")
        atp=energy*contract.atp_fraction
        heat=energy-atp
        return EnergyClosureState(self.chemical_free_energy_j-energy,self.atp_j+atp,self.heat_j+heat)

def _transfer_material(
    biology: BiologicalState,
    ecology: U5ConsolidatedState,
    closure: MatterClosureState,
    *,
    compartment_id: str,
    mass_kg: float,
    composition: MatterComposition,
    source_field: str,
    target: str,
):
    if compartment_id not in biology.compartments:
        raise KeyError("unknown biological compartment")
    if target not in {"biomass","waste"}:
        raise ValueError("target must be biomass or waste")
    m=_nn(mass_kg,"mass_kg")
    c=biology.compartments[compartment_id]
    available=float(getattr(c,source_field))
    moved=min(m,available)
    split=composition.mass_split(moved)
    changes={source_field:available-moved}
    if source_field == "biomass":
        new_biomass=available-moved
        scale=new_biomass/available if available>0 else 0.0
        changes["elemental_mass_kg"]={k:v*scale for k,v in c.elemental_mass_kg.items()}
    c2=replace(c, **changes)
    b2=replace(biology, compartments={**biology.compartments,compartment_id:c2})
    cyc=ecology.cycles
    if target=="biomass":
        cyc2=BiogeochemicalPools(
            cyc.organic_carbon+split["C"],cyc.inorganic_carbon,
            cyc.organic_nitrogen+split["N"],cyc.inorganic_nitrogen,
            cyc.oxygen+split["O"],cyc.detritus_carbon,cyc.detritus_nitrogen,
            cyc.organic_phosphorus,cyc.inorganic_phosphorus,cyc.detritus_phosphorus,
            cyc.organic_sulfur,cyc.inorganic_sulfur,cyc.detritus_sulfur)
    else:
        cyc2=BiogeochemicalPools(
            cyc.organic_carbon,cyc.inorganic_carbon,
            cyc.organic_nitrogen,cyc.inorganic_nitrogen,
            cyc.oxygen+split["O"],cyc.detritus_carbon+split["C"],cyc.detritus_nitrogen+split["N"],
            cyc.organic_phosphorus,cyc.inorganic_phosphorus,cyc.detritus_phosphorus,
            cyc.organic_sulfur,cyc.inorganic_sulfur,cyc.detritus_sulfur)
    e2=U5ConsolidatedState(ecology.base,cyc2,ecology.relations,ecology.traits,ecology.generation,ecology.time)
    cl2=MatterClosureState(closure.unresolved_mass_kg+split["other"],closure.digest_seed)
    return b2,e2,cl2,split

def transfer_biomass_to_ecology(biology, ecology, closure, *, compartment_id, mass_kg, composition):
    return _transfer_material(biology,ecology,closure,compartment_id=compartment_id,mass_kg=mass_kg,
                               composition=composition,source_field="biomass",target="biomass")

def transfer_waste_to_ecology(biology, ecology, closure, *, compartment_id, mass_kg, composition):
    return _transfer_material(biology,ecology,closure,compartment_id=compartment_id,mass_kg=mass_kg,
                               composition=composition,source_field="waste",target="waste")

@dataclass(frozen=True)
class V192ClosureReport:
    mass_before_kg: float
    mass_after_kg: float
    energy_before_j: float
    energy_after_j: float
    mass_residual_kg: float
    energy_residual_j: float
    certified: bool
    digest: str


@dataclass(frozen=True)
class FullClosureReport:
    certified: bool
    mass_residual_kg: float
    energy_residual_j: float
    digest: str

def step_full_material_energy_closure(
    environment, chemistry, biology, ecology, closure: MatterClosureState, energy: EnergyClosureState,
    *, cell=(0,0), compartment_id="core", substrate_kg=0.0,
    biomass_to_ecology_kg=0.0, waste_to_ecology_kg=0.0,
    biomass_composition: MatterComposition,
    waste_composition: MatterComposition,
    contract: RespirationContract | None = None,
):
    """One immutable U2→U5 transaction with explicit mass and energy closure."""
    from .interdomain_metabolism import step_coupled_respiration
    env2, chem2, bio2, rr = step_coupled_respiration(
        environment, chemistry, biology, cell=cell, compartment_id=compartment_id,
        substrate_kg_requested=substrate_kg, contract=contract)
    # The energy bridge is tied to the exact substrate actually consumed.
    energy2 = energy.apply_respiration(contract or RespirationContract(), rr.substrate_kg)
    if abs(energy2.atp_j - (energy.atp_j + rr.atp_j)) > 1e-9:
        raise RuntimeError("energy ledger ATP does not match U4 ATP")
    b3,e3,c3,_ = transfer_biomass_to_ecology(
        bio2, ecology, closure, compartment_id=compartment_id,
        mass_kg=biomass_to_ecology_kg, composition=biomass_composition)
    b4,e4,c4,_ = transfer_waste_to_ecology(
        b3, e3, c3, compartment_id=compartment_id,
        mass_kg=waste_to_ecology_kg, composition=waste_composition)
    before_mass = biology.compartments[compartment_id].biomass + biology.compartments[compartment_id].waste + sum(ecology.cycles.totals()) + closure.unresolved_mass_kg
    after_mass = b4.compartments[compartment_id].biomass + b4.compartments[compartment_id].waste + sum(e4.cycles.totals()) + c4.unresolved_mass_kg
    # Respiration transfers chemical free energy internally, so total closure is exact.
    before_energy = energy.total_j
    after_energy = energy2.total_j
    mass_res = after_mass - before_mass
    energy_res = after_energy - before_energy
    digest=hashlib.sha256(repr((env2.digest(),chem2.digest(),b4.digest(),e4.digest(),c4.digest(),energy2.digest())).encode()).hexdigest()
    certified=bool(rr.certified and abs(mass_res)<=1e-9*max(1.0,before_mass) and abs(energy_res)<=1e-9*max(1.0,before_energy))
    return env2,chem2,b4,e4,c4,energy2,FullClosureReport(certified,mass_res,energy_res,digest)
