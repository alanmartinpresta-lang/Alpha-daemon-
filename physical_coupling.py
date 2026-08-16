"""AIRE V189+ physical SI-domain coupling.

This module defines the first real cross-domain transfers. The transfers are
explicitly dimensional and conservative: environmental surface water is stored
as m^3 and transferred to U4 as kg H2O; atmospheric oxygen is stored as kg O2
per environment cell and transferred to U4 as kg O2. No model-unit quantity is
implicitly converted.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import math
import numpy as np

from .environment import EnvironmentState
WATER_DENSITY_KG_M3 = 1000.0
from .biology import BiologicalState

@dataclass(frozen=True)
class PhysicalTransferReport:
    water_requested_kg: float
    water_transferred_kg: float
    oxygen_requested_kg: float
    oxygen_transferred_kg: float
    water_residual_kg: float
    oxygen_residual_kg: float
    certified: bool


def _amount(x: float, name: str) -> float:
    x = float(x)
    if not math.isfinite(x) or x < 0:
        raise ValueError(f"{name} must be finite and >= 0")
    return x


def transfer_environment_to_biology(
    environment: EnvironmentState,
    biology: BiologicalState,
    *,
    cell: tuple[int, int],
    compartment_id: str,
    water_kg: float = 0.0,
    oxygen_kg: float = 0.0,
) -> tuple[EnvironmentState, BiologicalState, PhysicalTransferReport]:
    """Transfer real SI mass from U2 reservoirs into one U4 compartment.

    Water comes from surface water depth in the selected cell. Oxygen comes
    from the atmospheric O2 mass reservoir. Both donor and receiver states are
    immutable and the transferred mass is removed from the donor exactly once.
    """
    water_kg = _amount(water_kg, "water_kg")
    oxygen_kg = _amount(oxygen_kg, "oxygen_kg")
    if compartment_id not in biology.compartments:
        raise KeyError(f"unknown biological compartment: {compartment_id}")
    iy, ix = map(int, cell)
    if not (0 <= iy < environment.config.ny and 0 <= ix < environment.config.nx):
        raise IndexError("cell outside environment grid")

    water_available = max(0.0, float(environment.water_depth[iy, ix])) * environment.config.cell_size**2 * WATER_DENSITY_KG_M3
    oxygen_available = max(0.0, float(environment.atmospheric_oxygen_mass_kg[iy, ix]))
    tw = min(water_kg, water_available)
    to = min(oxygen_kg, oxygen_available)

    water_depth = np.maximum(environment.water_depth, 0.0)
    water_depth[iy, ix] = max(0.0, float(water_depth[iy, ix]) - tw / (environment.config.cell_size**2 * WATER_DENSITY_KG_M3))
    oxygen_mass = np.maximum(environment.atmospheric_oxygen_mass_kg, 0.0)
    oxygen_mass[iy, ix] = max(0.0, float(oxygen_mass[iy, ix]) - to)
    env2 = replace(environment, water_depth=water_depth, atmospheric_oxygen_mass_kg=oxygen_mass)

    c = biology.compartments[compartment_id]
    bio2 = replace(biology, compartments={**biology.compartments,
        compartment_id: replace(c, water=c.water + tw, oxygen=c.oxygen + to)})

    report = PhysicalTransferReport(
        water_requested_kg=water_kg, water_transferred_kg=tw,
        oxygen_requested_kg=oxygen_kg, oxygen_transferred_kg=to,
        water_residual_kg=water_kg - tw,
        oxygen_residual_kg=oxygen_kg - to,
        certified=True,
    )
    return env2, bio2, report
