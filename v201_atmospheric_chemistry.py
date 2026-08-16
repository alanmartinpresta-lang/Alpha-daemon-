"""AIRE V201 — conservative atmospheric gas/chemistry coupling.

U2 atmospheric gas reservoirs remain the authoritative SI state. The U3
reaction engine is used as a transient chemistry workspace, then the tracked
species are mapped back to U2. This avoids a second persistent reservoir and
therefore avoids double-counting atmospheric mass in the global ledger.

Scope: deterministic mesoscopic atmospheric gas chemistry. The current
reaction set contains the balanced CO2 <-> CO + O2 redox proxy. It is not a
photochemical, cloud-microphysical, aerosol, or molecular-dynamics model.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import hashlib
import math
import numpy as np

from .environment import EnvironmentState
from .chemistry import ChemicalSpecies, ChemicalReaction, ChemistryNetwork, create_chemistry, step_chemistry

ATM_SPECIES = {
    "N2": (28.0134, {"N": 2}),
    "O2": (31.9988, {"O": 2}),
    "CO2": (44.0095, {"C": 1, "O": 2}),
    "Ar": (39.948, {"Ar": 1}),
    "CO": (28.0101, {"C": 1, "O": 1}),
}

@dataclass(frozen=True)
class AtmosphericChemistryReport:
    dt: float
    extent_mol: float
    mass_before_kg: float
    mass_after_kg: float
    mass_residual_kg: float
    carbon_before_kg: float
    carbon_after_kg: float
    oxygen_before_kg: float
    oxygen_after_kg: float
    certified: bool
    digest: str


def _network() -> ChemistryNetwork:
    species = {n: ChemicalSpecies(n, mm, 0, elems) for n, (mm, elems) in ATM_SPECIES.items()}
    reaction = ChemicalReaction(
        "co2_co_redox",
        {"CO2": 2}, {"CO": 2, "O2": 1},
        rate_constant=1.0e-10,
        reverse_rate_constant=1.0e-10,
        enthalpy=0.0,
    )
    return ChemistryNetwork(species, (reaction,))


def _gas_volume(env: EnvironmentState) -> float:
    return env.config.cell_size**2 * max(env.config.scale_height, 1.0)


def _masses(env: EnvironmentState) -> dict[str, np.ndarray]:
    return {
        "N2": env.atmospheric_nitrogen_mass_kg.copy(),
        "O2": env.atmospheric_oxygen_mass_kg.copy(),
        "CO2": env.atmospheric_co2_mass_kg.copy(),
        "Ar": env.atmospheric_argon_mass_kg.copy(),
        "CO": env.atmospheric_co_mass_kg.copy(),
    }


def _element_mass_kg(masses: dict[str, np.ndarray], element: str) -> float:
    total = 0.0
    for name, field in masses.items():
        mm_g, elems = ATM_SPECIES[name]
        count = elems.get(element, 0)
        if count:
            total += float(np.sum(field)) * (count * ({"C": 12.011, "O": 15.999, "N": 14.007, "Ar": 39.948}[element]) / mm_g)
    return total


def step_atmospheric_chemistry(env: EnvironmentState, dt: float) -> tuple[EnvironmentState, AtmosphericChemistryReport]:
    h = float(dt)
    if not math.isfinite(h) or h <= 0:
        raise ValueError("dt must be finite and > 0")
    masses = _masses(env)
    volume = _gas_volume(env)
    shape = env.shape
    concentrations = {}
    for name, (mm_g, _) in ATM_SPECIES.items():
        concentrations[name] = masses[name] / ((mm_g / 1000.0) * volume)
    chemistry = create_chemistry(_network(), temperature=env.temperature, shape=shape,
                                 volume=volume, concentrations=concentrations)
    mass_before = sum(float(np.sum(v)) for v in masses.values())
    carbon_before = _element_mass_kg(masses, "C")
    oxygen_before = _element_mass_kg(masses, "O")
    out, report = step_chemistry(chemistry, h)
    new = {name: np.maximum(0.0, out.concentrations[name] * volume * (ATM_SPECIES[name][0] / 1000.0))
           for name in ATM_SPECIES}
    mass_after = sum(float(np.sum(v)) for v in new.values())
    carbon_after = _element_mass_kg(new, "C")
    oxygen_after = _element_mass_kg(new, "O")
    env2 = replace(env,
        atmospheric_nitrogen_mass_kg=new["N2"],
        atmospheric_oxygen_mass_kg=new["O2"],
        atmospheric_co2_mass_kg=new["CO2"],
        atmospheric_argon_mass_kg=new["Ar"],
        atmospheric_co_mass_kg=new["CO"],
    )
    mass_residual = mass_after - mass_before
    digest = hashlib.sha256((env2.digest() + report["digest"]).encode()).hexdigest()
    certified = bool(
        abs(mass_residual) <= 1e-10 * max(1.0, abs(mass_before)) and
        abs(carbon_after-carbon_before) <= 1e-10 * max(1.0, abs(carbon_before)) and
        abs(oxygen_after-oxygen_before) <= 1e-10 * max(1.0, abs(oxygen_before)) and
        all(np.isfinite(v).all() and np.all(v >= 0) for v in new.values())
    )
    return env2, AtmosphericChemistryReport(h, float(report["reaction_extents"]["co2_co_redox"]),
        mass_before, mass_after, mass_residual, carbon_before, carbon_after,
        oxygen_before, oxygen_after, certified, digest)


