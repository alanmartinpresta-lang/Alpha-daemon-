"""AIRE cross-domain energy contract.

The world modules historically used different energy representations (Joules,
chemical enthalpy proxies, ATP units, ecological biomass units).  This module
introduces an explicit, auditable ledger without pretending those proxies are
physically equivalent until a conversion factor is declared.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib, math
from typing import Mapping

_EPS = 1e-12

@dataclass(frozen=True)
class EnergyReservoir:
    reservoir_id: str
    amount_j: float
    capacity_j: float | None = None
    domain: str = "generic"
    def __post_init__(self):
        if not self.reservoir_id or not isinstance(self.reservoir_id, str):
            raise ValueError("reservoir_id must be non-empty")
        if not math.isfinite(self.amount_j) or self.amount_j < 0:
            raise ValueError("amount_j must be finite and >= 0")
        if self.capacity_j is not None and (not math.isfinite(self.capacity_j) or self.capacity_j <= 0):
            raise ValueError("capacity_j must be finite and > 0")
        if self.capacity_j is not None and self.amount_j > self.capacity_j + _EPS:
            raise ValueError("reservoir exceeds capacity")
        object.__setattr__(self, "domain", str(self.domain))

@dataclass(frozen=True)
class EnergyTransfer:
    transfer_id: str
    source: str
    target: str
    amount_j: float
    external_work_j: float = 0.0
    heat_loss_j: float = 0.0
    reason: str = "transfer"
    def __post_init__(self):
        if not self.transfer_id or not self.source or not self.target:
            raise ValueError("transfer/source/target identifiers must be non-empty")
        if self.source == self.target:
            raise ValueError("source and target must differ")
        for name in ("amount_j", "external_work_j", "heat_loss_j"):
            x = float(getattr(self, name))
            if not math.isfinite(x) or x < 0:
                raise ValueError(f"{name} must be finite and >= 0")

@dataclass(frozen=True)
class EnergyLedger:
    reservoirs: Mapping[str, EnergyReservoir]
    transfers: tuple[EnergyTransfer, ...] = ()
    external_input_j: float = 0.0
    external_output_j: float = 0.0

    def __post_init__(self):
        clean = {str(k): v for k, v in self.reservoirs.items()}
        if any(k != v.reservoir_id for k, v in clean.items()):
            raise ValueError("reservoir mapping key must equal reservoir_id")
        for name in ("external_input_j", "external_output_j"):
            x = float(getattr(self, name))
            if not math.isfinite(x) or x < 0:
                raise ValueError(f"{name} must be finite and >= 0")
        ids = set(clean)
        for t in self.transfers:
            if t.source not in ids or t.target not in ids:
                raise ValueError("transfer references unknown reservoir")
        object.__setattr__(self, "reservoirs", clean)
        object.__setattr__(self, "transfers", tuple(self.transfers))

    @property
    def total_stored_j(self) -> float:
        return float(sum(r.amount_j for r in self.reservoirs.values()))

    @property
    def total_heat_loss_j(self) -> float:
        return float(sum(t.heat_loss_j for t in self.transfers))

    @property
    def total_external_work_j(self) -> float:
        return float(sum(t.external_work_j for t in self.transfers))

    @property
    def balance_residual_j(self) -> float:
        """Stored change against declared external inputs/outputs and losses.

        A closed ledger has residual zero.  Transfers between internal
        reservoirs cancel exactly and therefore cannot create energy.
        """
        return self.total_stored_j + self.external_output_j + self.total_heat_loss_j + self.total_external_work_j - self.external_input_j

    def digest(self) -> str:
        h = hashlib.sha256(); h.update(b"AIRE-ENERGY-1")
        for rid in sorted(self.reservoirs):
            r = self.reservoirs[rid]
            h.update(repr((r.reservoir_id, r.amount_j, r.capacity_j, r.domain)).encode())
        for t in self.transfers:
            h.update(repr((t.transfer_id, t.source, t.target, t.amount_j, t.external_work_j, t.heat_loss_j, t.reason)).encode())
        h.update(repr((self.external_input_j, self.external_output_j)).encode())
        return h.hexdigest()

    def assert_balanced(self, before_stored_j: float, *, external_input_j: float = 0.0,
                        external_output_j: float = 0.0, heat_loss_j: float = 0.0,
                        external_work_j: float = 0.0, atol: float = 1e-9) -> float:
        before = float(before_stored_j)
        for x in (before, external_input_j, external_output_j, heat_loss_j, external_work_j):
            if not math.isfinite(x) or x < 0:
                raise ValueError("energy balance inputs must be finite and >= 0")
        residual = self.total_stored_j - before - external_input_j + external_output_j + heat_loss_j + external_work_j
        scale = max(1.0, abs(before), self.total_stored_j, abs(external_input_j))
        if abs(residual) > atol * scale:
            raise ValueError(f"energy ledger not balanced: residual={residual}")
        return float(residual)
