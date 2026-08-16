"""AIRE U2 consolidated environment dynamics.

Adds a deterministic, bounded environmental model above the certified U2 base:
closed water-cycle transfers, simple atmospheric wind/cloud dynamics, surface
thermal balance, and slow geomorphic evolution.  This is intentionally a
world-scale surrogate, not a replacement for CFD, weather, ocean or climate
solvers.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import hashlib
import math
import numpy as np

from .core import WorldCore, WorldState
from .environment import EnvironmentState, PlanetConfig, _solar_field, attach_environment
from .si_soils import SoilMineralState, create_soil_minerals, step_soil_minerals

SCHEMA = "U2.2"
_EPS = 1e-12


def _field(x, shape, name, nonnegative=False):
    a = np.asarray(x, dtype=float)
    if a.shape != shape or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    if nonnegative and np.any(a < 0):
        raise ValueError(f"{name} must be non-negative")
    return a.copy()


@dataclass(frozen=True)
class U2DynamicsConfig:
    # Atmospheric column heat capacity is derived from the hydrostatic column
    # mass; this parameter is the specific heat of dry air (J kg^-1 K^-1).
    air_specific_heat_j_kg_k: float = 1005.0
    longwave_emissivity: float = 0.95
    stefan_boltzmann: float = 5.670374419e-8
    latent_heat_vaporization_j_kg: float = 2.5e6
    latent_heat_fusion_j_kg: float = 3.34e5
    heat_capacity: float = 2.0e6
    longwave_coeff: float = 0.000000003
    latent_heat_coeff: float = 0.04
    atmospheric_relaxation: float = 0.02
    wind_relaxation: float = 0.10
    max_wind: float = 80.0
    evaporation_rate: float = 1.0e-7
    condensation_rate: float = 2.0e-7
    infiltration_rate: float = 0.10
    groundwater_drainage: float = 0.002
    runoff_rate: float = 0.04
    erosion_rate: float = 2.0e-7
    deposition_rate: float = 0.15
    uplift_rate: float = 1.0e-10
    max_substeps: int = 256

    def __post_init__(self):
        vals = (self.air_specific_heat_j_kg_k, self.longwave_emissivity, self.stefan_boltzmann,
                self.latent_heat_vaporization_j_kg, self.latent_heat_fusion_j_kg,
                self.heat_capacity, self.longwave_coeff, self.latent_heat_coeff,
                self.atmospheric_relaxation, self.wind_relaxation, self.max_wind,
                self.evaporation_rate, self.condensation_rate, self.infiltration_rate,
                self.groundwater_drainage, self.runoff_rate, self.erosion_rate,
                self.deposition_rate, self.uplift_rate)
        if not all(math.isfinite(float(v)) for v in vals):
            raise ValueError("U2 dynamics parameters must be finite")
        if (self.air_specific_heat_j_kg_k <= 0 or self.longwave_emissivity < 0 or self.longwave_emissivity > 1 or
                self.stefan_boltzmann <= 0 or self.latent_heat_vaporization_j_kg <= 0 or
                self.latent_heat_fusion_j_kg <= 0 or self.heat_capacity <= 0 or
                self.max_wind <= 0 or self.max_substeps < 1):
            raise ValueError("invalid U2 dynamics configuration")
        if not (0 <= self.infiltration_rate <= 1 and 0 <= self.runoff_rate <= 1 and 0 <= self.deposition_rate <= 1):
            raise ValueError("fraction parameters must be in [0,1]")


@dataclass(frozen=True)
class U2DynamicsState:
    base: EnvironmentState
    groundwater: np.ndarray
    snow: np.ndarray
    vapor: np.ndarray
    cloud_fraction: np.ndarray
    wind_u: np.ndarray
    wind_v: np.ndarray
    sediment: np.ndarray
    geologic_time: float = 0.0
    schema: str = SCHEMA
    thermal_energy_j: np.ndarray | None = None
    radiative_input_j: float = 0.0
    radiative_output_j: float = 0.0
    latent_heat_j: float = 0.0
    soil_minerals: SoilMineralState | None = None

    def __post_init__(self):
        shape = self.base.shape
        names = ("groundwater", "snow", "vapor", "cloud_fraction", "wind_u", "wind_v", "sediment")
        for name in names:
            a = _field(getattr(self, name), shape, name, nonnegative=name not in ("wind_u", "wind_v"))
            if name == "cloud_fraction" and np.any(a > 1):
                raise ValueError("cloud_fraction must be <= 1")
            a.setflags(write=False)
            object.__setattr__(self, name, a)
        if not math.isfinite(float(self.geologic_time)) or self.geologic_time < 0:
            raise ValueError("geologic_time must be finite and non-negative")
        object.__setattr__(self, "geologic_time", float(self.geologic_time))
        thermal = self.thermal_energy_j
        if thermal is None:
            area = self.base.config.cell_size ** 2
            column_mass = (self.base.config.base_pressure *
                           np.exp(-np.maximum(self.base.elevation, 0.0) / self.base.config.scale_height) *
                           area / self.base.config.gravity)
            thermal = column_mass * 1005.0 * self.base.temperature
        thermal = np.asarray(thermal, dtype=float)
        if thermal.shape != shape or not np.isfinite(thermal).all() or np.any(thermal <= 0):
            raise ValueError("thermal_energy_j must be finite, positive and match the environment shape")
        thermal = thermal.copy(); thermal.setflags(write=False); object.__setattr__(self, "thermal_energy_j", thermal)
        for name in ("radiative_input_j", "radiative_output_j"):
            x = float(getattr(self, name))
            if not math.isfinite(x) or x < 0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, x)
        if self.soil_minerals is None:
            object.__setattr__(self, "soil_minerals", create_soil_minerals(shape))
        elif self.soil_minerals.dry_soil_mass_kg.shape != shape:
            raise ValueError("soil_minerals shape must match environment")
        x = float(self.latent_heat_j)
        if not math.isfinite(x):
            raise ValueError("latent_heat_j must be finite")
        object.__setattr__(self, "latent_heat_j", x)

    def digest(self) -> str:
        h = hashlib.sha256(self.schema.encode())
        h.update(self.base.digest().encode())
        for a in (self.groundwater, self.snow, self.vapor, self.cloud_fraction,
                  self.wind_u, self.wind_v, self.sediment, self.thermal_energy_j):
            h.update(np.ascontiguousarray(a).tobytes())
        h.update(self.soil_minerals.digest().encode())
        h.update(repr((self.geologic_time, self.radiative_input_j, self.radiative_output_j, self.latent_heat_j)).encode())
        return h.hexdigest()

    @property
    def total_water_volume(self) -> float:
        area = self.base.config.cell_size ** 2
        # all water pools are expressed as equivalent depth in metres
        return float(np.sum(self.base.water_depth + self.base.soil_moisture +
                            self.groundwater + self.snow + self.vapor) * area)


@dataclass(frozen=True)
class U2ConsolidatedReport:
    dt: float
    substeps: int
    water_before: float
    water_after: float
    water_residual: float
    thermal_energy_before_j: float
    thermal_energy_after_j: float
    absorbed_solar_j: float
    outgoing_longwave_j: float
    latent_heat_net_j: float
    thermal_balance_residual_j: float
    max_wind: float
    max_slope: float
    terrain_change_l1: float
    digest: str

    @property
    def certified_step(self) -> bool:
        wscale = max(1.0, abs(self.water_before), abs(self.water_after))
        return bool(abs(self.water_residual) <= 1e-9 * wscale and
                    math.isfinite(self.thermal_energy_after_j) and
                    self.thermal_energy_after_j > 0 and
                    abs(self.thermal_balance_residual_j) <= 1e-9 * max(1.0, abs(self.thermal_energy_before_j), abs(self.thermal_energy_after_j), abs(self.absorbed_solar_j)) and
                    math.isfinite(self.max_wind) and self.max_wind >= 0 and
                    math.isfinite(self.terrain_change_l1))


def initialize_u2_dynamics(env: EnvironmentState, config: U2DynamicsConfig | None = None) -> U2DynamicsState:
    c = env.config
    z = np.zeros(env.shape)
    # Start with the existing surface water as the accessible water reservoir.
    # Reuse the canonical EnvironmentState vapor reservoir.
    vapor = np.asarray(env.atmospheric_water, dtype=float).copy()
    clouds = np.clip(env.humidity ** 2, 0, 1)
    sediment = np.zeros(env.shape)
    area = c.cell_size ** 2
    column_mass = c.base_pressure * np.exp(-np.maximum(env.elevation, 0.0) / c.scale_height) * area / c.gravity
    thermal = column_mass * 1005.0 * env.temperature
    return U2DynamicsState(env, z, z, vapor, clouds, z, z, sediment, env.atmospheric_time, SCHEMA, thermal, 0.0, 0.0, 0.0, create_soil_minerals(env.shape, time=env.atmospheric_time))


def _gradient(a, cell):
    gy, gx = np.gradient(a, cell, cell, edge_order=1)
    return gx, gy


def _lap(a):
    return (np.roll(a, 1, 0) + np.roll(a, -1, 0) + np.roll(a, 1, 1) + np.roll(a, -1, 1) - 4*a)


def _conservative_diffuse(a, alpha):
    alpha = max(0.0, min(0.249, float(alpha)))
    return (1-4*alpha)*a + alpha*(np.roll(a,1,0)+np.roll(a,-1,0)+np.roll(a,1,1)+np.roll(a,-1,1))


def step_u2_dynamics(state: U2DynamicsState, dt: float, config: U2DynamicsConfig | None = None):
    h_total = float(dt)
    if not math.isfinite(h_total) or h_total <= 0:
        raise ValueError("dt must be finite and > 0")
    c = state.base.config
    k = config or U2DynamicsConfig()
    # A deterministic bounded substep policy keeps transfer fractions stable.
    n = max(1, min(k.max_substeps, int(math.ceil(h_total / 1800.0))))
    h = h_total / n
    s = state
    soil_minerals, soil_report = step_soil_minerals(state.soil_minerals, h_total)
    water0 = state.total_water_volume
    thermal0 = float(np.sum(state.thermal_energy_j))
    absorbed_total = 0.0
    outgoing_total = 0.0
    latent_net_total = 0.0
    for _ in range(n):
        b = s.base
        area = c.cell_size ** 2
        elev = b.elevation
        # --- atmosphere / wind ---
        pgx, pgy = _gradient(b.pressure, c.cell_size)
        wind_u = np.clip((1-k.wind_relaxation)*s.wind_u - k.wind_relaxation * 0.02 * pgx * h, -k.max_wind, k.max_wind)
        wind_v = np.clip((1-k.wind_relaxation)*s.wind_v - k.wind_relaxation * 0.02 * pgy * h, -k.max_wind, k.max_wind)
        wind_speed = np.sqrt(wind_u**2 + wind_v**2)
        scale = np.maximum(1.0, wind_speed / k.max_wind)
        wind_u = wind_u / scale
        wind_v = wind_v / scale
        wind_speed = np.sqrt(wind_u**2 + wind_v**2)

        # --- closed water cycle: evaporation -> vapor -> cloud -> rain ---
        warm = np.clip((b.temperature - 273.15) / 40.0, 0, 2)
        evap_surface = np.minimum(b.water_depth, k.evaporation_rate * h * (0.2 + 0.8*b.sunlight/max(c.solar_constant, 1e-12)) * (0.5 + warm))
        evap_soil = np.minimum(b.soil_moisture, 0.25 * evap_surface)
        vapor = s.vapor + evap_surface + evap_soil
        # wind-driven mixing and condensation; condensation is bounded by vapor.
        cloud_target = np.clip((vapor / max(c.soil_capacity, 1e-9)) + 0.3*b.humidity, 0, 1)
        clouds = np.clip(s.cloud_fraction + k.atmospheric_relaxation*(cloud_target-s.cloud_fraction)*h, 0, 1)
        rain = np.minimum(vapor, k.condensation_rate * h * clouds * (1 + 0.5*np.clip(1-b.temperature/273.15, -1, 1)))
        vapor = np.maximum(0, vapor-rain)
        snow = s.snow + np.where(b.temperature < 273.15, rain, 0.0)
        liquid_rain = np.where(b.temperature >= 273.15, rain, 0.0)
        melt = np.minimum(snow, np.where(b.temperature > 273.15, (b.temperature-273.15)*1e-5*h, 0.0))
        snow = np.maximum(0, snow-melt)
        liquid = liquid_rain + melt

        # infiltration / runoff transfer between water reservoirs
        infiltration = np.minimum(liquid, k.infiltration_rate * h/3600.0 * (1+0.5*np.clip(b.soil_moisture/max(c.soil_capacity,1e-12),0,1)))
        soil_pre = b.soil_moisture + infiltration - evap_soil
        overflow = np.maximum(0, soil_pre-c.soil_capacity)
        soil = np.minimum(c.soil_capacity, soil_pre)
        percolation = 0.20 * overflow
        runoff = overflow - percolation
        surface = b.water_depth + liquid - infiltration - evap_surface + runoff
        surface = np.maximum(0, surface)
        groundwater = s.groundwater + percolation - k.groundwater_drainage*h/3600.0*s.groundwater
        drainage = k.groundwater_drainage*h/3600.0*s.groundwater
        surface += drainage
        groundwater = np.maximum(0, groundwater)
        # conservative groundwater smoothing
        groundwater = _conservative_diffuse(groundwater, min(0.02, 0.01*h/3600.0))

        # --- conservative atmospheric-gas mixing + dimensioned thermal/radiative balance ---
        # Tracked N2/O2/CO2/Ar are mixed between cells without changing global mass.
        gas_alpha = min(0.249, max(0.0, 0.01 * h / 3600.0))
        gas_fields = {}
        for gas_name in ("atmospheric_nitrogen_mass_kg", "atmospheric_oxygen_mass_kg",
                          "atmospheric_co2_mass_kg", "atmospheric_argon_mass_kg", "atmospheric_co_mass_kg"):
            gas_fields[gas_name] = np.maximum(0.0, _conservative_diffuse(getattr(b, gas_name), gas_alpha))
        # b.sunlight is albedo-adjusted clear-sky forcing; clouds attenuate it.
        area = c.cell_size ** 2
        column_mass = c.base_pressure * np.exp(-np.maximum(elev, 0.0) / c.scale_height) * area / c.gravity
        column_heat_capacity = column_mass * k.air_specific_heat_j_kg_k
        cloud_transmission = 1.0 - 0.35 * np.clip(clouds, 0.0, 1.0)
        incoming_j = b.sunlight * cloud_transmission * area * h
        co2_column = gas_fields["atmospheric_co2_mass_kg"] / area
        vapor_column = vapor * 1000.0
        effective_emissivity = np.clip(0.35
            + 0.55 * (1.0 - np.exp(-np.maximum(co2_column, 0.0) / 10.0))
            + 0.20 * (1.0 - np.exp(-np.maximum(vapor_column, 0.0) / 20.0)), 0.05, 0.99)
        outgoing_j = effective_emissivity * k.stefan_boltzmann * np.maximum(b.temperature, 150.0)**4 * area * h
        evap_mass_kg = (evap_surface + evap_soil) * area * 1000.0
        rain_mass_kg = rain * area * 1000.0
        melt_mass_kg = melt * area * 1000.0
        # Net latent term: evaporation consumes latent heat; condensation
        # releases it; melting consumes fusion heat.
        latent_j = (float(np.sum(evap_mass_kg)) * k.latent_heat_vaporization_j_kg
                    - float(np.sum(rain_mass_kg)) * k.latent_heat_vaporization_j_kg
                    + float(np.sum(melt_mass_kg)) * k.latent_heat_fusion_j_kg)
        net_j = incoming_j - outgoing_j - (
            (evap_mass_kg - rain_mass_kg) * k.latent_heat_vaporization_j_kg
            + melt_mass_kg * k.latent_heat_fusion_j_kg)
        # Advective redistribution is internal; enforce zero global integral.
        advective_flux = 0.00005 * wind_speed * (b.temperature - np.mean(b.temperature))
        advective_flux = advective_flux - np.mean(advective_flux)
        net_j = net_j - advective_flux * area * h
        thermal_prev = s.thermal_energy_j
        thermal_new = thermal_prev + net_j
        thermal_new = np.maximum(1.0, thermal_new)
        temp = thermal_new / column_heat_capacity
        absorbed_total += float(np.sum(incoming_j))
        outgoing_total += float(np.sum(outgoing_j))
        latent_net_total += latent_j
        pressure = c.base_pressure * np.exp(-np.maximum(elev,0)/c.scale_height) * (temp/c.base_temperature)
        humidity = np.clip((soil/max(c.soil_capacity,1e-12))*0.7 + vapor/(vapor+1.0)*0.3, 0, 1)
        sunlight = _solar_field(c, b.atmospheric_time+h)
        resources = np.clip(b.resources + h/86400.0*(0.02*sunlight/max(c.solar_constant,1e-12)-0.01*(1-humidity)), 0, c.resource_capacity)

        # --- slow geology: water erosion + conservative sediment redistribution + uplift ---
        gx, gy = _gradient(elev, c.cell_size)
        slope = np.sqrt(gx*gx+gy*gy)
        erosion = np.minimum(np.maximum(0, elev-elev.min()), k.erosion_rate*h*slope*(runoff+surface+1e-12))
        uplift = k.uplift_rate*h
        sediment = s.sediment + erosion
        # Deposit a bounded fraction uniformly through conservative diffusion.
        dep = np.minimum(sediment, k.deposition_rate*h/86400.0*sediment)
        sediment = np.maximum(0, sediment-dep)
        terrain = elev - erosion + uplift + dep - np.mean(dep) + np.mean(erosion)
        # Preserve mean terrain except for explicit uplift; sediment remains local.
        terrain += (np.mean(elev) + uplift) - np.mean(terrain)

        # Canonicalize atmospheric vapor: U2 dynamics and EnvironmentState
        # must expose the same reservoir, otherwise the global ledger can
        # double-count or silently lose atmospheric water.
        base = EnvironmentState("U2.2", c, terrain, surface, soil, resources,
                                temp, pressure, sunlight, humidity, b.atmospheric_time+h,
                                atmospheric_water=vapor,
                                atmospheric_oxygen_mass_kg=gas_fields["atmospheric_oxygen_mass_kg"],
                                atmospheric_nitrogen_mass_kg=gas_fields["atmospheric_nitrogen_mass_kg"],
                                atmospheric_co2_mass_kg=gas_fields["atmospheric_co2_mass_kg"],
                                atmospheric_argon_mass_kg=gas_fields["atmospheric_argon_mass_kg"],
                                atmospheric_co_mass_kg=gas_fields["atmospheric_co_mass_kg"])
        s = U2DynamicsState(base, groundwater, snow, vapor, clouds, wind_u, wind_v,
                            sediment, s.geologic_time+h, SCHEMA, thermal_new,
                            s.radiative_input_j + float(np.sum(incoming_j)),
                            s.radiative_output_j + float(np.sum(outgoing_j)),
                            s.latent_heat_j + latent_j, soil_minerals)
    water1 = s.total_water_volume
    thermal1 = float(np.sum(s.thermal_energy_j))
    thermal_residual = thermal1 - thermal0 - absorbed_total + outgoing_total + latent_net_total
    report = U2ConsolidatedReport(
        h_total, n, water0, water1, water1-water0,
        thermal0, thermal1, absorbed_total, outgoing_total, latent_net_total, thermal_residual,
        float(np.max(np.sqrt(s.wind_u**2+s.wind_v**2))),
        float(np.max(np.sqrt(_gradient(s.base.elevation,c.cell_size)[0]**2 + _gradient(s.base.elevation,c.cell_size)[1]**2))),
        float(np.sum(np.abs(s.base.elevation-state.base.elevation))), s.digest())
    return s, report


def attach_u2_dynamics(state: WorldState, env: U2DynamicsState) -> WorldState:
    meta = dict(state.metadata)
    meta["u2_environment_dynamics"] = {
        "schema": env.schema,
        "digest": env.digest(),
        "shape": list(env.base.shape),
        "geologic_time": env.geologic_time,
        "total_water_volume": env.total_water_volume,
    }
    return replace(state, metadata=meta)


class ConsolidatedEnvironmentWorld:
    def __init__(self, core: WorldCore, environment: U2DynamicsState):
        if abs(environment.base.atmospheric_time-core.state.clock.time) > 1e-9:
            raise ValueError("environment time must match world clock")
        self.core = core
        self.environment = environment
        self.last_report = None
        self.core._state = attach_u2_dynamics(self.core.state, environment)

    @property
    def state(self):
        return self.core.state

    def step(self, dt=None, config=None):
        h = self.state.clock.dt if dt is None else float(dt)
        new_env, report = step_u2_dynamics(self.environment, h, config)
        self.core.step(h, updater=lambda s, _h: attach_u2_dynamics(s, new_env))
        self.environment = new_env
        self.last_report = report
        return report
