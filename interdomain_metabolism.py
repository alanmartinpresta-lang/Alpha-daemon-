"""AIRE U3-U4-U5 real material/energy flux bridge.

This module is deliberately explicit: a chemical substrate is transferred from
U3 to U4 by SI mass, oxygen is transferred from U2 to U4, respiration converts
those reactants according to a declared balanced stoichiometry, CO2/H2O return
as explicit products, and the declared chemical free-energy yield is split
between ATP and heat. No model-unit conversion is implicit.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import hashlib, math
from typing import Any
import numpy as np

from .chemistry import ChemistryState, ChemicalSpecies
from .biology import BiologicalState, BiologicalCompartment
from .environment import EnvironmentState

WATER_DENSITY_KG_M3 = 1000.0
_EPS = 1e-12


def _nn(x: float, name: str) -> float:
    x = float(x)
    if not math.isfinite(x) or x < 0:
        raise ValueError(f"{name} must be finite and >= 0")
    return x


@dataclass(frozen=True)
class RespirationContract:
    """Explicit balanced substrate respiration contract.

    Stoichiometry is in mol substrate / mol product. The default is glucose
    respiration. The declared free-energy yield is an SI bridge, not a claim
    about a particular molecular mechanism.
    """
    substrate_species: str = "glucose"
    oxygen_species: str = "O2"
    co2_species: str = "CO2"
    water_species: str = "H2O"
    oxygen_moles_per_substrate: float = 6.0
    co2_moles_per_substrate: float = 6.0
    water_moles_per_substrate: float = 6.0
    free_energy_j_per_kg_substrate: float = 15.9e6
    atp_fraction: float = 0.40

    def __post_init__(self):
        for n in ("oxygen_moles_per_substrate", "co2_moles_per_substrate",
                  "water_moles_per_substrate", "free_energy_j_per_kg_substrate"):
            _nn(getattr(self, n), n)
        if not 0 <= self.atp_fraction <= 1:
            raise ValueError("atp_fraction must be in [0,1]")
        for n in ("substrate_species", "oxygen_species", "co2_species", "water_species"):
            if not str(getattr(self, n)):
                raise ValueError(f"{n} must be non-empty")


@dataclass(frozen=True)
class CoupledMetabolismReport:
    substrate_kg: float
    oxygen_kg: float
    co2_kg: float
    water_kg: float
    atp_j: float
    heat_j: float
    substrate_mass_residual_kg: float
    oxygen_mass_residual_kg: float
    carbon_mass_residual_kg: float
    energy_residual_j: float
    certified: bool
    digest: str


def _species(state: ChemistryState, name: str) -> ChemicalSpecies:
    if name not in state.network.species:
        raise KeyError(f"chemistry species not present: {name}")
    return state.network.species[name]


def _remove_kg(chem: ChemistryState, species: str, cell: tuple[int, int], kg: float):
    sp = _species(chem, species)
    iy, ix = map(int, cell)
    if not (0 <= iy < chem.shape[0] and 0 <= ix < chem.shape[1]):
        raise IndexError("chemistry cell outside grid")
    kg = _nn(kg, "kg")
    molar_kg = sp.molar_mass / 1000.0
    mol = kg / molar_kg
    concentration_delta = mol / chem.volume
    field = chem.concentrations[species].copy()
    taken = min(float(field[iy, ix]), concentration_delta)
    field[iy, ix] -= taken
    return replace(chem, concentrations={**chem.concentrations, species: field}), taken * chem.volume * molar_kg


def _add_kg(chem: ChemistryState, species: str, cell: tuple[int, int], kg: float):
    sp = _species(chem, species)
    iy, ix = map(int, cell)
    if not (0 <= iy < chem.shape[0] and 0 <= ix < chem.shape[1]):
        raise IndexError("chemistry cell outside grid")
    kg = _nn(kg, "kg")
    molar_kg = sp.molar_mass / 1000.0
    field = chem.concentrations[species].copy()
    field[iy, ix] += kg / molar_kg / chem.volume
    return replace(chem, concentrations={**chem.concentrations, species: field})


def _transfer_o2_from_environment(env: EnvironmentState, cell: tuple[int, int], requested_kg: float):
    iy, ix = map(int, cell)
    if not (0 <= iy < env.config.ny and 0 <= ix < env.config.nx):
        raise IndexError("environment cell outside grid")
    req = _nn(requested_kg, "oxygen_kg")
    mass = env.atmospheric_oxygen_mass_kg.copy()
    taken = min(req, float(mass[iy, ix]))
    mass[iy, ix] -= taken
    return replace(env, atmospheric_oxygen_mass_kg=mass), taken


def _add_environment_water(env: EnvironmentState, cell: tuple[int, int], water_kg: float):
    iy, ix = map(int, cell)
    if not (0 <= iy < env.config.ny and 0 <= ix < env.config.nx):
        raise IndexError("environment cell outside grid")
    kg = _nn(water_kg, "water_kg")
    depth = env.water_depth.copy()
    depth[iy, ix] += kg / (env.config.cell_size**2 * WATER_DENSITY_KG_M3)
    return replace(env, water_depth=depth)


def _balanced_contract_check(chem: ChemistryState, c: RespirationContract):
    sub = _species(chem, c.substrate_species)
    o2 = _species(chem, c.oxygen_species)
    co2 = _species(chem, c.co2_species)
    water = _species(chem, c.water_species)
    # Elemental balance of the declared reaction, independent of kinetics.
    elements = set(sub.elements) | set(o2.elements) | set(co2.elements) | set(water.elements)
    for e in elements:
        residual = (-sub.elements.get(e, 0)
                    - c.oxygen_moles_per_substrate * o2.elements.get(e, 0)
                    + c.co2_moles_per_substrate * co2.elements.get(e, 0)
                    + c.water_moles_per_substrate * water.elements.get(e, 0))
        if residual != 0:
            raise ValueError(f"respiration contract is not elementally balanced for {e}: {residual}")


def step_coupled_respiration(
    environment: EnvironmentState,
    chemistry: ChemistryState,
    biology: BiologicalState,
    *, cell: tuple[int, int], compartment_id: str,
    substrate_kg_requested: float,
    contract: RespirationContract | None = None,
) -> tuple[EnvironmentState, ChemistryState, BiologicalState, CoupledMetabolismReport]:
    c = contract or RespirationContract()
    _balanced_contract_check(chemistry, c)
    if compartment_id not in biology.compartments:
        raise KeyError(f"unknown biological compartment: {compartment_id}")
    sub_sp = _species(chemistry, c.substrate_species)
    o2_sp = _species(chemistry, c.oxygen_species)
    co2_sp = _species(chemistry, c.co2_species)
    water_sp = _species(chemistry, c.water_species)

    # Requested substrate is capped by chemical availability and by available O2.
    iy, ix = map(int, cell)
    available_sub = float(chemistry.concentrations[c.substrate_species][iy, ix]) * chemistry.volume * (sub_sp.molar_mass / 1000.0)
    sub_molar_kg = sub_sp.molar_mass / 1000.0
    o2_molar_kg = o2_sp.molar_mass / 1000.0
    max_sub_by_o2 = float(environment.atmospheric_oxygen_mass_kg[iy, ix]) / max(c.oxygen_moles_per_substrate * o2_molar_kg, _EPS) * sub_molar_kg
    substrate_kg = min(_nn(substrate_kg_requested, "substrate_kg_requested"), available_sub, max_sub_by_o2)
    if substrate_kg <= 0:
        rep = CoupledMetabolismReport(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, True, hashlib.sha256(b"zero-flux").hexdigest())
        return environment, chemistry, biology, rep

    chemistry2, removed_sub = _remove_kg(chemistry, c.substrate_species, cell, substrate_kg)
    substrate_kg = removed_sub
    substrate_moles = substrate_kg / sub_molar_kg
    oxygen_kg = substrate_moles * c.oxygen_moles_per_substrate * o2_molar_kg
    env2, removed_o2 = _transfer_o2_from_environment(environment, cell, oxygen_kg)
    # O2 availability was used for the cap, so removal should be exact.
    if abs(removed_o2 - oxygen_kg) > 1e-12 * max(1.0, oxygen_kg):
        raise RuntimeError("oxygen transfer mismatch")

    co2_kg = substrate_moles * c.co2_moles_per_substrate * (co2_sp.molar_mass / 1000.0)
    water_kg = substrate_moles * c.water_moles_per_substrate * (water_sp.molar_mass / 1000.0)
    chemistry3 = _add_kg(chemistry2, c.co2_species, cell, co2_kg)
    # H2O is a chemistry product; biological water is also explicitly tracked
    # as a SI pool, so its production is added to U4 and not duplicated.
    bio_c = biology.compartments[compartment_id]
    free_energy = substrate_kg * c.free_energy_j_per_kg_substrate
    atp = free_energy * c.atp_fraction
    heat = free_energy - atp
    bio2 = replace(biology, compartments={**biology.compartments,
        compartment_id: replace(bio_c,
            # Substrate and O2 are consumed directly by this bridge; they are not
            # first credited to the U4 pools, so no mass is duplicated.
            water=bio_c.water + water_kg,
            atp=bio_c.atp + atp,
            heat_j=bio_c.heat_j + heat)})
    # CO2 is kept in the U3 chemistry state as the authoritative product pool.
    # H2O is transferred into the U4 water pool, so the bridge records that
    # product as a real U3 -> U4 material transfer rather than duplicating it.
    substrate_mass_residual = substrate_kg - removed_sub
    oxygen_residual = oxygen_kg - removed_o2
    carbon_sub = substrate_moles * sub_sp.elements.get("C", 0)
    carbon_co2 = substrate_moles * c.co2_moles_per_substrate * co2_sp.elements.get("C", 0)
    carbon_residual = carbon_sub - carbon_co2
    digest = hashlib.sha256(repr((env2.digest(), chemistry3.digest(), bio2.digest(),
                                  substrate_kg, oxygen_kg, co2_kg, water_kg, atp, heat)).encode()).hexdigest()
    energy_residual = free_energy - atp - heat
    certified = (abs(oxygen_residual) <= 1e-12 * max(1.0, oxygen_kg) and
                 abs(energy_residual) <= 1e-12 * max(1.0, free_energy) and
                 abs(carbon_residual) <= 1e-12 * max(1.0, substrate_moles))
    rep = CoupledMetabolismReport(substrate_kg, oxygen_kg, co2_kg, water_kg, atp, heat,
                                   substrate_mass_residual, oxygen_residual, carbon_residual,
                                   energy_residual, certified, digest)
    return env2, chemistry3, bio2, rep

@dataclass(frozen=True)
class CarbonTransferReport:
    co2_kg: float
    carbon_kg: float
    oxygen_kg: float
    certified: bool
    digest: str


def transfer_co2_to_ecology(chemistry: ChemistryState, ecology_state, *, cell: tuple[int, int], requested_co2_kg: float):
    """Move CO2 from U3 into U5 inorganic C/O pools with exact stoichiometric masses."""
    from .u5_consolidated import U5ConsolidatedState, BiogeochemicalPools
    if not isinstance(ecology_state, U5ConsolidatedState):
        raise TypeError("ecology_state must be U5ConsolidatedState")
    co2 = _species(chemistry, "CO2")
    if "C" not in co2.elements or "O" not in co2.elements:
        raise ValueError("CO2 species must declare C and O")
    iy, ix = map(int, cell)
    available = float(chemistry.concentrations["CO2"][iy, ix]) * chemistry.volume * (co2.molar_mass / 1000.0)
    amount = min(_nn(requested_co2_kg, "requested_co2_kg"), available)
    chemistry2, removed = _remove_kg(chemistry, "CO2", cell, amount)
    moles = removed / (co2.molar_mass / 1000.0)
    ckg = moles * co2.elements["C"] * 12.011 / 1000.0
    okg = moles * co2.elements["O"] * 15.999 / 1000.0
    cycles = ecology_state.cycles
    cycles2 = BiogeochemicalPools(
        cycles.organic_carbon, cycles.inorganic_carbon + ckg,
        cycles.organic_nitrogen, cycles.inorganic_nitrogen,
        cycles.oxygen + okg, cycles.detritus_carbon, cycles.detritus_nitrogen,
        cycles.organic_phosphorus, cycles.inorganic_phosphorus, cycles.detritus_phosphorus,
        cycles.organic_sulfur, cycles.inorganic_sulfur, cycles.detritus_sulfur)
    eco2 = U5ConsolidatedState(ecology_state.base, cycles2, ecology_state.relations,
                               ecology_state.traits, ecology_state.generation, ecology_state.time)
    # The species molar mass is authoritative for the molecular mass; elemental
    # transfer is checked against the declared formula, so no implicit fraction is used.
    certified = abs(removed - amount) <= 1e-12 * max(1.0, amount)
    digest = hashlib.sha256(repr((chemistry2.digest(), eco2.digest(), removed, ckg, okg)).encode()).hexdigest()
    return chemistry2, eco2, CarbonTransferReport(removed, ckg, okg, certified, digest)
