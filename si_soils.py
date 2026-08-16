"""AIRE SI soil/mineral reservoirs.

Tracks declared mineral-element pools in kg per environment cell. Weathering
moves mass from mineral reserve to a mobile pool; leaching moves mobile mass to
an explicit export ledger. No mineral mass is silently deleted.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib, math
from typing import Mapping
import numpy as np

DEFAULT_MINERAL_MASS_FRACTIONS = {
    "Ca": 0.12, "Mg": 0.06, "K": 0.035, "Na": 0.025,
    "Fe": 0.045, "P": 0.012, "S": 0.010, "Si": 0.55,
    "Al": 0.143,
}

def _field(a, shape, name):
    x=np.asarray(a,dtype=float)
    if x.shape!=shape or not np.isfinite(x).all() or np.any(x<0):
        raise ValueError(f"{name} must be finite, non-negative and match shape")
    return x.copy()

def _maps_equal(a,b): return set(a)==set(b)==set(DEFAULT_MINERAL_MASS_FRACTIONS)

@dataclass(frozen=True)
class SoilMineralState:
    schema: str
    dry_soil_mass_kg: np.ndarray
    mineral_mass_kg: Mapping[str, np.ndarray]
    mobile_mineral_mass_kg: Mapping[str, np.ndarray] | None = None
    exported_mass_kg: Mapping[str, float] | None = None
    time: float = 0.0

    def __post_init__(self):
        if self.schema != "U2-SI-SOIL.1": raise ValueError("unsupported soil/mineral schema")
        shape=self.dry_soil_mass_kg.shape
        dry=_field(self.dry_soil_mass_kg,shape,"dry_soil_mass_kg")
        if not np.isfinite(self.time) or self.time<0: raise ValueError("time must be finite >=0")
        dry.setflags(write=False); object.__setattr__(self,"dry_soil_mass_kg",dry)
        clean={}
        for name,a in self.mineral_mass_kg.items():
            if name not in DEFAULT_MINERAL_MASS_FRACTIONS: raise ValueError(f"unsupported mineral element: {name}")
            x=_field(a,shape,f"mineral_mass_kg[{name}]"); x.setflags(write=False); clean[str(name)]=x
        if set(clean)!=set(DEFAULT_MINERAL_MASS_FRACTIONS): raise ValueError("all tracked mineral elements must be present")
        raw_mobile=self.mobile_mineral_mass_kg or {k:np.zeros(shape) for k in clean}
        mobile={}
        for k in clean:
            x=_field(raw_mobile[k],shape,f"mobile_mineral_mass_kg[{k}]"); x.setflags(write=False); mobile[k]=x
        raw_export=self.exported_mass_kg or {k:0.0 for k in clean}
        exports={k:float(raw_export[k]) for k in clean}
        if any(not math.isfinite(v) or v<0 for v in exports.values()): raise ValueError("export ledger invalid")
        object.__setattr__(self,"mineral_mass_kg",clean); object.__setattr__(self,"mobile_mineral_mass_kg",mobile); object.__setattr__(self,"exported_mass_kg",exports); object.__setattr__(self,"time",float(self.time))
        reserve=np.zeros(shape); mobile_total=np.zeros(shape)
        for x in clean.values(): reserve += x
        for x in mobile.values(): mobile_total += x
        if np.any(reserve+mobile_total > dry + 1e-9*np.maximum(1.0,dry)):
            raise ValueError("tracked mineral mass cannot exceed dry soil mass")

    @property
    def total_mineral_mass_kg(self): return float(sum(np.sum(x)+np.sum(self.mobile_mineral_mass_kg[k]) for k,x in self.mineral_mass_kg.items()))
    @property
    def total_exported_mass_kg(self): return float(sum(self.exported_mass_kg.values()))
    @property
    def initial_tracked_mass_kg(self): return self.total_mineral_mass_kg+self.total_exported_mass_kg

    def digest(self):
        h=hashlib.sha256(); h.update(self.schema.encode()); h.update(np.ascontiguousarray(self.dry_soil_mass_kg).tobytes())
        for k in sorted(self.mineral_mass_kg):
            h.update(k.encode()); h.update(np.ascontiguousarray(self.mineral_mass_kg[k]).tobytes()); h.update(np.ascontiguousarray(self.mobile_mineral_mass_kg[k]).tobytes()); h.update(repr(self.exported_mass_kg[k]).encode())
        h.update(repr(self.time).encode()); return h.hexdigest()

def create_soil_minerals(shape, *, dry_soil_mass_kg_per_cell=1.5e6, fractions: Mapping[str,float]|None=None, time=0.0):
    fractions=dict(fractions or DEFAULT_MINERAL_MASS_FRACTIONS)
    if set(fractions)!=set(DEFAULT_MINERAL_MASS_FRACTIONS): raise ValueError("fraction basis mismatch")
    s=sum(float(v) for v in fractions.values())
    if not math.isfinite(s) or s<=0 or s>1: raise ValueError("mineral fractions must sum to (0,1]")
    dry=np.full(shape,float(dry_soil_mass_kg_per_cell))
    pools={k:dry*float(v) for k,v in fractions.items()}
    return SoilMineralState("U2-SI-SOIL.1",dry,pools,time=time)

def step_soil_minerals(state: SoilMineralState, dt: float, *, weathering_rate=1e-9, leaching_rate=1e-7):
    h=float(dt); wr=float(weathering_rate); lr=float(leaching_rate)
    if not math.isfinite(h) or h<=0 or not math.isfinite(wr) or not math.isfinite(lr) or wr<0 or lr<0: raise ValueError("invalid soil/mineral step")
    reserve={}; mobile={}; exports=dict(state.exported_mass_kg)
    before=state.total_mineral_mass_kg+state.total_exported_mass_kg
    for k,v in state.mineral_mass_kg.items():
        w=np.minimum(v,v*wr*h); reserve[k]=v-w
        m=state.mobile_mineral_mass_kg[k]+w
        ex=np.minimum(m,m*lr*h); mobile[k]=m-ex; exports[k]+=float(np.sum(ex))
    new=SoilMineralState(state.schema,state.dry_soil_mass_kg,reserve,mobile,exports,state.time+h)
    after=new.total_mineral_mass_kg+new.total_exported_mass_kg
    report={"weathered_kg":float(sum(np.sum(state.mineral_mass_kg[k]-reserve[k]) for k in reserve)),"exported_increment_kg":new.total_exported_mass_kg-state.total_exported_mass_kg,"tracked_before_kg":before,"tracked_after_kg":after,"mass_residual_kg":after-before,"digest":new.digest()}
    report["certified_step"]=abs(report["mass_residual_kg"])<=1e-10*max(1.0,before)
    return new,report
