"""AIRE U2 — deterministic planetary environment layer.

U2 provides a solver-independent environmental world above U0/U1:
- deterministic terrain/elevation fields;
- surface water and soil moisture;
- global atmosphere state and local temperature/pressure fields;
- solar/day-night forcing and axial seasons;
- deterministic resource fields;
- bounded environmental evolution and conservation audits.

This is an environment model, not a replacement for specialized CFD,
weather, ocean or climate solvers. Its contract is intentionally explicit.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import hashlib
import math
from typing import Any, Mapping
import numpy as np

import collections
from .core import WorldCore, WorldState

U2_SCHEMA = "U2.1"


def _finite_nonnegative(a: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    x = np.asarray(a, dtype=float)
    if x.shape != shape or not np.all(np.isfinite(x)) or np.any(x < 0):
        raise ValueError(f"{name} must be finite and non-negative with shape {shape}")
    return x.copy()


@dataclass(frozen=True)
class PlanetConfig:
    """Global deterministic environmental parameters."""
    nx: int = 32
    ny: int = 32
    cell_size: float = 1000.0
    planet_radius: float = 6_371_000.0
    axial_tilt: float = math.radians(23.43928)
    day_length: float = 86_400.0
    year_length: float = 31_557_600.0
    gravity: float = 9.80665
    sea_level: float = 0.0
    base_temperature: float = 288.15
    lapse_rate: float = 0.0065
    base_pressure: float = 101325.0
    scale_height: float = 8434.5
    albedo: float = 0.30
    solar_constant: float = 1361.0
    rain_rate: float = 1.0e-6
    evaporation_rate: float = 2.0e-7
    runoff_rate: float = 0.15
    soil_capacity: float = 0.25
    diffusion: float = 0.08
    resource_capacity: float = 1.0
    seed: int = 0

    def __post_init__(self):
        if self.nx < 2 or self.ny < 2 or self.cell_size <= 0:
            raise ValueError("invalid grid configuration")
        vals = (self.planet_radius, self.day_length, self.year_length, self.gravity,
                self.base_temperature, self.lapse_rate, self.base_pressure,
                self.scale_height, self.albedo, self.solar_constant,
                self.rain_rate, self.evaporation_rate, self.runoff_rate,
                self.soil_capacity, self.diffusion, self.resource_capacity)
        if not all(math.isfinite(float(v)) for v in vals):
            raise ValueError("planet parameters must be finite")
        if not 0 <= self.albedo <= 1 or not 0 <= self.runoff_rate <= 1 or not 0 <= self.diffusion <= 1:
            raise ValueError("albedo, runoff_rate and diffusion must be in [0,1]")


@dataclass(frozen=True)
class EnvironmentState:
    """Canonical environmental fields. Arrays are copied on construction."""
    schema: str
    config: PlanetConfig
    elevation: np.ndarray
    water_depth: np.ndarray
    soil_moisture: np.ndarray
    resources: np.ndarray
    temperature: np.ndarray
    pressure: np.ndarray
    sunlight: np.ndarray
    humidity: np.ndarray
    atmospheric_time: float = 0.0
    atmospheric_water: np.ndarray | None = None
    atmospheric_oxygen_mass_kg: np.ndarray | None = None
    atmospheric_nitrogen_mass_kg: np.ndarray | None = None
    atmospheric_co2_mass_kg: np.ndarray | None = None
    atmospheric_argon_mass_kg: np.ndarray | None = None
    atmospheric_co_mass_kg: np.ndarray | None = None
    biome: np.ndarray | None = None
    vegetation: np.ndarray | None = None

    def __setstate__(self, state):
        self.__dict__.update(state)
        if getattr(self, "biome", None) is None:
            object.__setattr__(self, "biome", _classify_biomes(self.elevation, self.water_depth, self.temperature, self.humidity, self.soil_moisture, self.sunlight))
        if getattr(self, "vegetation", None) is None:
            object.__setattr__(self, "vegetation", np.clip(0.15 * (self.humidity + np.clip(self.resources / max(self.config.resource_capacity,1e-12),0,1)), 0, 1))

    def __setstate__(self, state):
        self.__dict__.update(state)
        if getattr(self, "biome", None) is None:
            object.__setattr__(self, "biome", _classify_biomes(self.elevation, self.water_depth, self.temperature, self.humidity, self.soil_moisture, self.sunlight))
        if getattr(self, "vegetation", None) is None:
            object.__setattr__(self, "vegetation", np.clip(0.15 * (self.humidity + np.clip(self.resources / max(self.config.resource_capacity,1e-12),0,1)), 0, 1))

    def __post_init__(self):
        shape = (self.config.ny, self.config.nx)
        fields = {
            "elevation": np.asarray(self.elevation, dtype=float),
            "water_depth": _finite_nonnegative(self.water_depth, shape, "water_depth"),
            "soil_moisture": _finite_nonnegative(self.soil_moisture, shape, "soil_moisture"),
            "resources": _finite_nonnegative(self.resources, shape, "resources"),
            "temperature": np.asarray(self.temperature, dtype=float),
            "pressure": np.asarray(self.pressure, dtype=float),
            "sunlight": _finite_nonnegative(self.sunlight, shape, "sunlight"),
            "humidity": _finite_nonnegative(self.humidity, shape, "humidity"),
        }
        for name, arr in fields.items():
            if arr.shape != shape or not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} must be finite with shape {shape}")
            arr = arr.copy(); arr.setflags(write=False); object.__setattr__(self, name, arr)
        if np.any(self.temperature <= 0) or np.any(self.pressure <= 0):
            raise ValueError("temperature and pressure must be positive")
        if np.any(self.humidity > 1):
            raise ValueError("humidity must be <= 1")
        if not math.isfinite(float(self.atmospheric_time)):
            raise ValueError("atmospheric_time must be finite")
        object.__setattr__(self, "atmospheric_time", float(self.atmospheric_time))
        vapor = np.zeros(shape, dtype=float) if self.atmospheric_water is None else np.asarray(self.atmospheric_water, dtype=float)
        if vapor.shape != shape or not np.isfinite(vapor).all() or np.any(vapor < 0):
            raise ValueError("atmospheric_water must be finite, non-negative and match the environment shape")
        vapor = vapor.copy(); vapor.setflags(write=False); object.__setattr__(self, "atmospheric_water", vapor)
        gas_defaults = {
            "atmospheric_oxygen_mass_kg": self.atmospheric_oxygen_mass_kg,
            "atmospheric_nitrogen_mass_kg": self.atmospheric_nitrogen_mass_kg,
            "atmospheric_co2_mass_kg": self.atmospheric_co2_mass_kg,
            "atmospheric_argon_mass_kg": self.atmospheric_argon_mass_kg,
            "atmospheric_co_mass_kg": self.atmospheric_co_mass_kg,
        }
        for name, raw in gas_defaults.items():
            gas = np.zeros(shape, dtype=float) if raw is None else np.asarray(raw, dtype=float)
            if gas.shape != shape or not np.isfinite(gas).all() or np.any(gas < 0):
                raise ValueError(f"{name} must be finite, non-negative and match the environment shape")
            gas = gas.copy(); gas.setflags(write=False); object.__setattr__(self, name, gas)
        if self.biome is None:
            biome = _classify_biomes(self.elevation, self.water_depth, self.temperature, self.humidity, self.soil_moisture, self.sunlight)
        else:
            biome = np.asarray(self.biome, dtype=np.int8)
        if self.vegetation is None:
            vegetation = np.clip(0.15 * (self.humidity + np.clip(self.resources / max(self.config.resource_capacity,1e-12),0,1)), 0, 1)
        else:
            vegetation = np.asarray(self.vegetation, dtype=float)
        if biome.shape != shape or not np.isfinite(biome).all():
            raise ValueError("biome must be finite and match the environment shape")
        if vegetation.shape != shape or not np.isfinite(vegetation).all() or np.any(vegetation < 0):
            raise ValueError("vegetation must be finite, non-negative and match the environment shape")
        vegetation = np.clip(vegetation, 0.0, 1.0)
        biome = biome.copy(); biome.setflags(write=False); object.__setattr__(self, "biome", biome)
        vegetation = vegetation.copy(); vegetation.setflags(write=False); object.__setattr__(self, "vegetation", vegetation)

    @property
    def shape(self):
        return self.elevation.shape

    @property
    def water_volume(self) -> float:
        """Surface water volume in m^3."""
        return float(np.sum(self.water_depth) * self.config.cell_size**2)

    @property
    def water_mass_kg(self) -> float:
        """Surface-water mass in kg using the declared water density."""
        return self.water_volume * 1000.0

    @property
    def mean_temperature(self) -> float:
        return float(np.mean(self.temperature))

    def digest(self) -> str:
        h = hashlib.sha256()
        h.update(self.schema.encode())
        h.update(repr(self.config).encode())
        for arr in (self.elevation, self.water_depth, self.soil_moisture,
                    self.resources, self.temperature, self.pressure,
                    self.sunlight, self.humidity, self.atmospheric_water,
                    self.atmospheric_oxygen_mass_kg, self.atmospheric_nitrogen_mass_kg,
                    self.atmospheric_co2_mass_kg, self.atmospheric_argon_mass_kg,
                    self.atmospheric_co_mass_kg):
            h.update(np.ascontiguousarray(arr).tobytes())
        h.update(repr(self.atmospheric_time).encode())
        return h.hexdigest()


@dataclass(frozen=True)
class EnvironmentStepReport:
    dt: float
    rainfall_volume: float
    evaporation_volume: float
    water_before: float
    water_after: float
    runoff_volume: float
    water_balance_residual: float
    mean_temperature_before: float
    mean_temperature_after: float
    deterministic_digest: str

    @property
    def certified_step(self) -> bool:
        scale = max(1.0, abs(self.water_before), abs(self.water_after))
        return bool(abs(self.water_balance_residual) <= 1e-10 * scale and
                    math.isfinite(self.mean_temperature_after) and
                    self.mean_temperature_after > 0)


def _terrain(config: PlanetConfig) -> np.ndarray:
    """Generate deterministic multi-scale relief with explicit geographic features.

    This remains a 2-D surface world: mountains, valleys, basins/lakes and
    canyon-like depressions are represented by elevation. It does not yet
    introduce a 3-D subsurface/cave volume.
    """
    rng = np.random.default_rng(config.seed)
    coarse = rng.normal(
        0.0, 1.0, (max(2, config.ny // 4), max(2, config.nx // 4))
    )
    iy = np.linspace(0, coarse.shape[0] - 1, config.ny).astype(int)
    ix = np.linspace(0, coarse.shape[1] - 1, config.nx).astype(int)
    noise = coarse[np.ix_(iy, ix)]

    y = np.linspace(-1, 1, config.ny)[:, None]
    x = np.linspace(-1, 1, config.nx)[None, :]

    # Multi-scale continental relief.
    relief = (
        450.0 * np.sin(2.7 * x) * np.cos(2.1 * y)
        + 220.0 * np.sin(5.1 * (x + y))
        + 180.0 * noise
    )

    # Deterministic mountain chains: elongated Gaussian ridges.
    for x0, amp, width, slope in [(-0.48, 900.0, 0.13, 0.55),
                                   (0.20, 700.0, 0.10, -0.35)]:
        ridge_axis = y * 0.55 + slope * x
        ridge = np.exp(-((x - x0) ** 2) / (2 * width**2))
        relief += amp * ridge * (0.65 + 0.35 * np.cos(5.0 * ridge_axis))

    # Valleys/basins: broad negative Gaussian depressions.
    for x0, y0, amp, sx, sy in [(-0.25, 0.38, -650.0, 0.22, 0.18),
                                 (0.55, -0.35, -500.0, 0.18, 0.28)]:
        relief += amp * np.exp(
            -(((x - x0) ** 2) / (2 * sx**2) + ((y - y0) ** 2) / (2 * sy**2))
        )

    # A narrow winding canyon/river corridor expressed as a surface depression.
    river_axis = y - (0.18 * np.sin(4.0 * x) - 0.18 * x)
    river = np.exp(-(river_axis**2) / (2 * 0.028**2))
    relief -= 260.0 * river

    # A few deep lake basins; surface water is derived from sea level below.
    for x0, y0, depth, sx, sy in [(-0.62, -0.25, 520.0, 0.10, 0.07),
                                  (0.34, 0.55, 420.0, 0.08, 0.10)]:
        relief -= depth * np.exp(
            -(((x - x0) ** 2) / (2 * sx**2) + ((y - y0) ** 2) / (2 * sy**2))
        )

    relief -= np.mean(relief)
    return relief.astype(float)


def _latitude(config: PlanetConfig) -> np.ndarray:
    return np.linspace(-math.pi / 2, math.pi / 2, config.ny)[:, None] * np.ones((1, config.nx))


def _solar_field(config: PlanetConfig, time: float) -> np.ndarray:
    lat = _latitude(config)
    day_phase = 2 * math.pi * ((time % config.day_length) / config.day_length)
    year_phase = 2 * math.pi * ((time % config.year_length) / config.year_length)
    declination = config.axial_tilt * math.sin(year_phase)
    longitude = np.linspace(-math.pi, math.pi, config.nx)[None, :]
    hour_angle = day_phase - math.pi + longitude
    cos_zenith = np.sin(lat) * math.sin(declination) + np.cos(lat) * math.cos(declination) * np.cos(hour_angle)
    return np.maximum(0.0, config.solar_constant * cos_zenith) * (1.0 - config.albedo)



def _hydrograph(elevation: np.ndarray, sea_level: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive local downhill direction, flow accumulation and basin classes.

    This is a diagnostic geographic layer: it does not alter mass by itself.
    Classes: 0 dry upland, 1 drainage/river corridor, 2 closed basin/lake candidate,
    3 sea-connected lowland candidate.
    """
    z = np.asarray(elevation, dtype=float)
    ny, nx = z.shape
    dirs = np.full((ny, nx), -1, dtype=np.int8)
    flow = np.ones((ny, nx), dtype=float)
    # 4-neighbour steepest descent; boundaries are outlets.
    for y in range(ny):
        for x in range(nx):
            best = z[y, x]; bi = -1
            for i,(dy,dx) in enumerate(((-1,0),(1,0),(0,-1),(0,1))):
                yy,xx=y+dy,x+dx
                if 0<=yy<ny and 0<=xx<nx and z[yy,xx] < best:
                    best=z[yy,xx]; bi=i
            dirs[y,x]=bi
    # Accumulation: process cells high -> low so upstream contributions arrive first.
    order=np.argsort(z.ravel())[::-1]
    for flat in order:
        y,x=np.unravel_index(flat,z.shape)
        d=dirs[y,x]
        if d>=0:
            dy,dx=(( -1,0),(1,0),(0,-1),(0,1))[d]
            yy,xx=y+dy,x+dx
            flow[yy,xx]+=flow[y,x]
    # Sea-connected candidate: below sea level and connected to boundary through
    # below-sea cells. Other low areas are closed-basin/lake candidates.
    below=z<=sea_level
    sea=np.zeros_like(below,dtype=bool)
    q=collections.deque()
    for x in range(nx):
        for y in (0,ny-1):
            if below[y,x] and not sea[y,x]: sea[y,x]=True;q.append((y,x))
    for y in range(ny):
        for x in (0,nx-1):
            if below[y,x] and not sea[y,x]: sea[y,x]=True;q.append((y,x))
    while q:
        y,x=q.popleft()
        for dy,dx in ((-1,0),(1,0),(0,-1),(0,1)):
            yy,xx=y+dy,x+dx
            if 0<=yy<ny and 0<=xx<nx and below[yy,xx] and not sea[yy,xx]:
                sea[yy,xx]=True;q.append((yy,xx))
    cls=np.zeros_like(z,dtype=np.int8)
    cls[below & ~sea]=2
    cls[sea]=3
    cls[(flow>=4) & ~sea & ~below]=1
    return dirs,flow,cls

def create_environment(config: PlanetConfig | None = None, *, time: float = 0.0) -> EnvironmentState:
    config = config or PlanetConfig()
    elevation = _terrain(config)
    water = np.maximum(0.0, config.sea_level - elevation) * 0.02
    soil = np.minimum(config.soil_capacity, water * 0.5)
    rng = np.random.default_rng(config.seed + 1)
    resources = rng.random((config.ny, config.nx)) * config.resource_capacity
    sunlight = _solar_field(config, time)
    temperature = config.base_temperature - config.lapse_rate * np.maximum(elevation, 0.0)
    pressure = config.base_pressure * np.exp(-np.maximum(elevation, 0.0) / config.scale_height)
    humidity = np.clip(0.45 + 0.25 * soil / max(config.soil_capacity, 1e-12), 0, 1)
    vapor_capacity = max(1e-9, 10.0 * config.soil_capacity)
    atmospheric_water = humidity * vapor_capacity
    # Column atmospheric mass from p*A/g, with a declared dry-air O2 mass fraction.
    # Dry-air mass fractions are explicit model parameters for the tracked gas ledger.
    # The remaining trace gas fraction is deliberately not inferred into a hidden pool.
    nitrogen_mass_fraction = 0.7552
    oxygen_mass_fraction = 0.2315
    co2_mass_fraction = 0.00065
    argon_mass_fraction = 0.01265
    gas_fraction_sum = nitrogen_mass_fraction + oxygen_mass_fraction + co2_mass_fraction + argon_mass_fraction
    if abs(gas_fraction_sum - 1.0) > 1e-12:
        raise RuntimeError("tracked atmospheric gas fractions must close to one")
    atmospheric_column_mass = pressure * (config.cell_size ** 2) / config.gravity
    atmospheric_nitrogen_mass_kg = atmospheric_column_mass * nitrogen_mass_fraction
    atmospheric_oxygen_mass_kg = atmospheric_column_mass * oxygen_mass_fraction
    atmospheric_co2_mass_kg = atmospheric_column_mass * co2_mass_fraction
    atmospheric_argon_mass_kg = atmospheric_column_mass * argon_mass_fraction
    biome = _classify_biomes(elevation, water, temperature, humidity, soil, sunlight)
    vegetation = np.clip(0.15 * (humidity + np.clip(resources / max(config.resource_capacity,1e-12),0,1)), 0, 1)
    return EnvironmentState(U2_SCHEMA, config, elevation, water, soil, resources,
                            temperature, pressure, sunlight, humidity, time, atmospheric_water,
                            atmospheric_oxygen_mass_kg, atmospheric_nitrogen_mass_kg, atmospheric_co2_mass_kg,
                            atmospheric_argon_mass_kg, np.zeros_like(atmospheric_argon_mass_kg), biome, vegetation)


def _laplacian(field: np.ndarray) -> np.ndarray:
    return (np.roll(field, 1, 0) + np.roll(field, -1, 0) +
            np.roll(field, 1, 1) + np.roll(field, -1, 1) - 4 * field)


def _classify_biomes(elevation, water_depth, temperature, humidity, soil_moisture=None, sunlight=None):
    """Deterministic physical biome classification; no Alpha behavior is encoded."""
    z=np.asarray(elevation,float); w=np.asarray(water_depth,float)
    t=np.asarray(temperature,float); h=np.asarray(humidity,float)
    sm=np.zeros_like(h) if soil_moisture is None else np.asarray(soil_moisture,float)
    sun=np.zeros_like(h) if sunlight is None else np.asarray(sunlight,float)
    moisture=np.clip(0.65*h + 0.35*np.clip(sm/0.5,0,1),0,1)
    sunfrac=np.clip(sun/950.0,0,1)
    b=np.full(z.shape,7,dtype=np.int8)
    # IDs: 0 ocean, 1 freshwater/wetland, 2 desert, 3 grassland,
    # 4 forest, 5 alpine, 6 tundra, 7 shrubland.
    sea=(z<0.0)&(w>0.02); b[sea]=0
    wet=(w>0.35)&~sea; b[wet]=1
    cold=(t<274.0)&~sea; b[cold]=6
    alpine=(z>700.0)&~sea; b[alpine]=5
    dry=(moisture<0.30)&(sunfrac>0.30)&~sea; b[dry]=2
    forest=(moisture>=0.52)&(sunfrac>0.15)&(t>=276.0)&~sea; b[forest]=4
    grass=(moisture>=0.30)&(moisture<0.52)&~sea; b[grass]=3
    return b


def step_environment(env: EnvironmentState, dt: float) -> tuple[EnvironmentState, EnvironmentStepReport]:
    h = float(dt)
    if not math.isfinite(h) or h <= 0:
        raise ValueError("dt must be finite and > 0")
    c = env.config
    area = c.cell_size ** 2
    water0 = env.water_volume
    vapor0 = float(np.sum(env.atmospheric_water) * area)
    soil0 = float(np.sum(env.soil_moisture) * area)
    # Closed atmospheric-water reservoir: evaporation enters vapor and rainfall
    # leaves vapor. Rain is limited by the vapor available in each cell.
    potential_evap = np.minimum(env.water_depth, c.evaporation_rate * h * (0.5 + env.sunlight / max(c.solar_constant, 1e-12)))
    vapor_capacity = max(1e-9, 10.0 * c.soil_capacity)
    rain_potential = c.rain_rate * h * np.clip(env.humidity, 0, 1)
    rain_depth = np.minimum(env.atmospheric_water + potential_evap, rain_potential)
    evap_depth = potential_evap
    vapor = np.maximum(0.0, env.atmospheric_water + evap_depth - rain_depth)
    water = np.maximum(0.0, env.water_depth + rain_depth - evap_depth)
    # Conservative infiltration/drainage: soil water is an internal reservoir,
    # so it cannot be created from nothing or disappear from the water ledger.
    infiltration = np.minimum(water, np.minimum(c.soil_capacity - env.soil_moisture,
                                                0.4 * rain_depth + 0.02 * water))
    infiltration = np.maximum(0.0, infiltration)
    water = water - infiltration
    soil = env.soil_moisture + infiltration
    drainage = np.minimum(soil, 0.001 * h / 86400.0 * soil)
    soil = soil - drainage
    water = water + drainage
    # Conservative surface runoff: water preferentially moves from higher
    # elevation cells toward lower neighboring cells. This creates actual
    # drainage paths instead of treating all below-sea cells as independent
    # static water patches. The transfer is mass-conserving.
    runoff_depth = np.zeros_like(water)
    flow_strength = min(0.20, max(0.0, h) * 0.02 / max(c.day_length, 1.0))
    if flow_strength > 0:
        for axis in (0, 1):
            for direction in (-1, 1):
                neigh_z = np.roll(env.elevation, direction, axis=axis)
                neigh_w = np.roll(water, direction, axis=axis)
                dz = np.maximum(env.elevation - neigh_z, 0.0)
                transfer = np.minimum(water, flow_strength * dz / max(1.0, float(np.ptp(env.elevation))) * water)
                water = water - transfer
                runoff_depth = runoff_depth + transfer
                incoming = np.roll(transfer, -direction, axis=axis)
                water = water + incoming
        water = np.maximum(0.0, water)

    # Small conservative redistribution; total water is preserved by diffusion.
    redistribution = min(0.25, c.diffusion * h / max(c.day_length, 1.0))
    if redistribution > 0:
        water = ((1.0 - 4.0 * redistribution) * water +
                 redistribution * (np.roll(water, 1, 0) + np.roll(water, -1, 0) +
                                   np.roll(water, 1, 1) + np.roll(water, -1, 1)))
    # Deterministic energy proxy: sunlight warms; elevation and radiative loss cool.
    forcing = env.sunlight / max(c.solar_constant, 1e-12)
    radiative_loss = 0.02 * ((env.temperature - 255.0) / 100.0)
    temp = env.temperature + h * (0.35 * forcing - 0.08 * radiative_loss) / 86400.0
    temp -= 0.000002 * np.maximum(env.elevation, 0.0)
    temp = np.maximum(150.0, temp)
    pressure = c.base_pressure * np.exp(-np.maximum(env.elevation, 0.0) / c.scale_height) * (temp / c.base_temperature)
    humidity = np.clip(vapor / vapor_capacity + 0.10 * soil / max(c.soil_capacity, 1e-12), 0.0, 1.0)
    sunlight = _solar_field(c, env.atmospheric_time + h)
    resources = np.clip(env.resources + h / 86400.0 * (0.02 * sunlight / max(c.solar_constant, 1e-12) - 0.01 * (1 - humidity)), 0, c.resource_capacity)
    biome = _classify_biomes(env.elevation, water, temp, humidity, soil, sunlight)
    growth = np.clip((humidity - 0.18) / 0.65, 0, 1) * np.clip((temp - 265.0) / 35.0, 0, 1) * (sunlight / max(c.solar_constant,1e-12))
    vegetation = np.clip(env.vegetation + h / 86400.0 * (0.18 * growth - 0.06 * (1 - humidity)), 0, 1)
    # Vegetation is a physical ecological reservoir, not a direct reward or policy.
    resources = np.clip(resources + h / 86400.0 * 0.015 * vegetation, 0, c.resource_capacity)
    out = EnvironmentState(U2_SCHEMA, c, env.elevation, water, soil, resources,
                           temp, pressure, sunlight, humidity, env.atmospheric_time + h, vapor,
                           env.atmospheric_oxygen_mass_kg, env.atmospheric_nitrogen_mass_kg,
                           env.atmospheric_co2_mass_kg, env.atmospheric_argon_mass_kg, env.atmospheric_co_mass_kg)
    rainfall_volume = float(np.sum(rain_depth) * area)
    evaporation_volume = float(np.sum(evap_depth) * area)
    runoff_volume = float(np.sum(runoff_depth) * area)
    vapor_volume = float(np.sum(out.atmospheric_water) * area)
    soil_volume = float(np.sum(out.soil_moisture) * area)
    total_water_before = water0 + soil0 + vapor0
    total_water_after = out.water_volume + soil_volume + vapor_volume
    # Closed surface+atmospheric reservoir: runoff is an internal transfer, not
    # a sink. Rainfall and evaporation exchange mass with vapor exactly.
    residual = total_water_after - total_water_before
    report = EnvironmentStepReport(h, rainfall_volume, evaporation_volume, water0,
                                   out.water_volume, runoff_volume, residual, env.mean_temperature,
                                   out.mean_temperature, out.digest())
    return out, report


def attach_environment(state: WorldState, env: EnvironmentState) -> WorldState:
    """Attach a JSON-safe environment snapshot to the U0 metadata contract."""
    meta = dict(state.metadata)
    meta["u2_environment"] = {
        "schema": env.schema,
        "config": {k: (float(v) if isinstance(v, (float, np.floating)) else int(v) if isinstance(v, (int, np.integer)) else v)
                   for k, v in env.config.__dict__.items()},
        "shape": list(env.shape),
        "digest": env.digest(),
        "atmospheric_time": env.atmospheric_time,
    }
    return replace(state, metadata=meta)


class EnvironmentWorld:
    """Deterministic U2 environment coupled to the U0 world clock."""
    def __init__(self, core: WorldCore, environment: EnvironmentState):
        if abs(environment.atmospheric_time - core.state.clock.time) > 1e-9:
            raise ValueError("environment time must match world clock")
        self.core = core
        self.environment = environment
        self.last_report: EnvironmentStepReport | None = None
        self.core._state = attach_environment(self.core.state, environment)

    @property
    def state(self) -> WorldState:
        return self.core.state

    def step(self, dt: float | None = None) -> EnvironmentStepReport:
        h = self.state.clock.dt if dt is None else float(dt)
        new_env, report = step_environment(self.environment, h)
        self.core.step(h, updater=lambda s, _h: attach_environment(s, new_env))
        self.environment = new_env
        self.last_report = report
        return report




