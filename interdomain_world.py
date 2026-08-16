"""Integrated U2 -> U3 -> U4 -> U5 material/energy transaction.

This is a deterministic world-level handoff, not an agent or cognition layer.
It proves that a real SI substrate can leave chemistry, O2 can leave the
atmosphere, respiration can create explicit energy, CO2 can enter chemistry and
then be transferred to ecological elemental pools, with the combined ledger
remaining closed.
"""
from __future__ import annotations
from dataclasses import dataclass
from .environment import EnvironmentState
from .chemistry import ChemistryState
from .biology import BiologicalState
from .u5_consolidated import U5ConsolidatedState
from .interdomain_metabolism import RespirationContract, step_coupled_respiration, transfer_co2_to_ecology

@dataclass(frozen=True)
class IntegratedFluxReport:
    respiration: object
    carbon_transfer: object
    certified: bool
    digest: str


def step_integrated_flux(environment: EnvironmentState, chemistry: ChemistryState,
                         biology: BiologicalState, ecology: U5ConsolidatedState,
                         *, cell=(0,0), compartment_id="core", substrate_kg=0.0,
                         contract: RespirationContract | None = None):
    env2, chem2, bio2, respiration = step_coupled_respiration(
        environment, chemistry, biology, cell=cell, compartment_id=compartment_id,
        substrate_kg_requested=substrate_kg, contract=contract)
    chem3, eco2, carbon = transfer_co2_to_ecology(
        chem2, ecology, cell=cell, requested_co2_kg=respiration.co2_kg)
    import hashlib
    digest = hashlib.sha256(repr((env2.digest(), chem3.digest(), bio2.digest(), eco2.digest(),
                                  respiration.digest, carbon.digest)).encode()).hexdigest()
    certified = bool(respiration.certified and carbon.certified)
    return env2, chem3, bio2, eco2, IntegratedFluxReport(respiration, carbon, certified, digest)
