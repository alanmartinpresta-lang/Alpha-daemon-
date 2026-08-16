"""AIRE U2/U4/U5 nutrient and mineral uptake bridge.

All transfers are explicit SI kg transactions. Carbon/nitrogen/phosphorus/sulfur
come from U5 biogeochemical pools; mineral elements come from U2 mobile soil
pools. Uptake removes exactly the transferred mass from the authoritative source.
Carbon is stored first in the U4 metabolizable substrate pool; mineral elements
are incorporated into structural mass. No energy or biomass is created by uptake.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import hashlib, math
from typing import Mapping

from .u2_consolidated import U2DynamicsState
from .biology import BiologicalState, BiologicalCompartment
from .u5_consolidated import U5ConsolidatedState, BiogeochemicalPools

_EPS = 1e-12
MINERAL_ELEMENTS = ("Ca", "Mg", "K", "Na", "Fe", "Si")
CYCLE_ELEMENTS = ("C", "N", "P", "S")


def _nn(x: float, name: str) -> float:
    x=float(x)
    if not math.isfinite(x) or x < 0:
        raise ValueError(f"{name} must be finite and >= 0")
    return x

@dataclass(frozen=True)
class UptakeReport:
    requested_kg: Mapping[str,float]
    transferred_kg: Mapping[str,float]
    source_residual_kg: Mapping[str,float]
    biomass_before_kg: float
    biomass_after_kg: float
    mass_residual_kg: float
    certified: bool
    digest: str


def _take_soil_batch(u2: U2DynamicsState, cell: tuple[int,int], requested: Mapping[str, float]):
    """Take all requested mineral elements in one immutable soil-state rebuild.

    The previous implementation rebuilt and validated the complete SoilMineralState
    once per element. Alpha only interacts with one cell per step, so the six
    independent transactions can be applied to copies and committed once while
    preserving the same SI mass-transfer semantics.
    """
    iy, ix = map(int, cell)
    s = u2.soil_minerals
    mobile = dict(s.mobile_mineral_mass_kg)
    transferred = {}
    changed = False
    for element, req in requested.items():
        arr = s.mobile_mineral_mass_kg[element]
        available = float(arr[iy, ix])
        taken = min(available, req)
        transferred[element] = taken
        if taken != 0.0:
            out = arr.copy()
            out[iy, ix] -= taken
            mobile[element] = out
            changed = True
    if not changed:
        return u2, transferred
    s2 = type(s)(s.schema, s.dry_soil_mass_kg, s.mineral_mass_kg,
                 mobile, s.exported_mass_kg, s.time)
    return replace(u2, soil_minerals=s2), transferred


def _take_cycle(eco: U5ConsolidatedState, element: str, requested: float):
    c=eco.cycles
    if element=="C":
        avail=c.organic_carbon; taken=min(avail,requested)
        c2=replace(c, organic_carbon=avail-taken)
    elif element=="N":
        avail=c.organic_nitrogen; taken=min(avail,requested)
        c2=replace(c, organic_nitrogen=avail-taken)
    elif element=="P":
        avail=c.organic_phosphorus; taken=min(avail,requested)
        c2=replace(c, organic_phosphorus=avail-taken)
    elif element=="S":
        avail=c.organic_sulfur; taken=min(avail,requested)
        c2=replace(c, organic_sulfur=avail-taken)
    else:
        raise ValueError(element)
    return replace(eco, cycles=c2), taken


def step_nutrient_mineral_uptake(
    u2: U2DynamicsState, biology: BiologicalState, ecology: U5ConsolidatedState,
    *, cell=(0,0), compartment_id="core", requested_kg: Mapping[str,float] | None=None
):
    if compartment_id not in biology.compartments:
        raise KeyError(f"unknown biological compartment: {compartment_id}")
    req={e:_nn((requested_kg or {}).get(e,0.0),f"requested_kg[{e}]") for e in (*CYCLE_ELEMENTS,*MINERAL_ELEMENTS)}
    u2c=u2; ecoc=ecology; transferred={e:0.0 for e in req}
    for e in CYCLE_ELEMENTS:
        ecoc,taken=_take_cycle(ecoc,e,req[e]); transferred[e]=taken
    u2c, mineral_transferred = _take_soil_batch(
        u2c, cell, {e: req[e] for e in MINERAL_ELEMENTS}
    )
    transferred.update(mineral_transferred)
    b=biology.compartments[compartment_id]
    comp=dict(b.elemental_mass_kg)
    before=b.biomass
    # Carbon enters the free metabolizable substrate pool first. It is not
    # structural biomass until U4 metabolism assimilates it. The other
    # requested elements are incorporated into structural mass immediately,
    # preserving the existing mesoscopic convention for mineral uptake.
    carbon_substrate = float(transferred.get("C", 0.0))
    structural_added = sum(float(taken) for e, taken in transferred.items() if e != "C")
    for e,taken in transferred.items():
        if e != "C":
            comp[e]=comp.get(e,0.0)+taken
    after=before+structural_added
    nutrient_after = b.nutrient + carbon_substrate
    bio2=replace(biology, compartments={**biology.compartments,
        compartment_id: replace(b, biomass=after, nutrient=nutrient_after,
                                 elemental_mass_kg=comp)})
    residual={e:req[e]-transferred[e] for e in req}
    mass_res=sum(transferred.values())-(structural_added+carbon_substrate)
    certified=abs(mass_res)<=1e-12*max(1.0,after) and all(v>=-1e-15 for v in residual.values())
    digest=hashlib.sha256(repr((u2c.digest(),bio2.digest(),ecoc.digest(),tuple(sorted(transferred.items())),mass_res)).encode()).hexdigest()
    return u2c,bio2,ecoc,UptakeReport(req,transferred,residual,before,after,mass_res,certified,digest)
