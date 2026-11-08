from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pvlib


def as_utc_index(timestamp: datetime) -> pd.DatetimeIndex:
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return pd.DatetimeIndex([timestamp])


def solar_position(latitude: float, longitude: float, timestamp: datetime) -> dict:
    times = as_utc_index(timestamp)
    solpos = pvlib.solarposition.get_solarposition(times, latitude, longitude)
    row = solpos.iloc[0]
    return {
        "apparent_zenith_deg": float(row["apparent_zenith"]),
        "zenith_deg": float(row["zenith"]),
        "apparent_elevation_deg": float(row["apparent_elevation"]),
        "elevation_deg": float(row["elevation"]),
        "azimuth_deg": float(row["azimuth"]),
    }


def decompose_ghi(ghi_w_m2: float, apparent_zenith_deg: float, timestamp: datetime) -> dict:
    if apparent_zenith_deg >= 90.0:
        return {"dni_w_m2": 0.0, "dhi_w_m2": 0.0, "clearness_index_kt": 0.0}

    times = as_utc_index(timestamp)
    out = pvlib.irradiance.erbs(
        ghi=pd.Series([ghi_w_m2], index=times),
        zenith=pd.Series([apparent_zenith_deg], index=times),
        datetime_or_doy=times,
    )
    row = out.iloc[0]
    dni_extra = float(pvlib.irradiance.get_extra_radiation(times).iloc[0])
    return {
        "dni_w_m2": float(min(max(row["dni"], 0.0), dni_extra)),
        "dhi_w_m2": float(max(row["dhi"], 0.0)),
        "clearness_index_kt": float(row["kt"]),
    }


def poa_at_orientation(
    tilt_deg: float,
    azimuth_deg: float,
    *,
    solar_zenith_deg: float,
    solar_azimuth_deg: float,
    dni_w_m2: float,
    ghi_w_m2: float,
    dhi_w_m2: float,
) -> float:
    if solar_zenith_deg >= 90.0:
        return 0.0
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt_deg,
        surface_azimuth=azimuth_deg,
        solar_zenith=solar_zenith_deg,
        solar_azimuth=solar_azimuth_deg,
        dni=dni_w_m2,
        ghi=ghi_w_m2,
        dhi=dhi_w_m2,
        model="isotropic",
    )
    return float(poa["poa_global"])


def orientation_grid(
    tilt_step_deg: float = 1.0, azimuth_step_deg: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    tilts = np.arange(0.0, 90.0 + tilt_step_deg, tilt_step_deg)
    azimuths = np.arange(0.0, 360.0, azimuth_step_deg)
    tilt_grid, azimuth_grid = np.meshgrid(tilts, azimuths, indexing="ij")
    return tilt_grid.ravel(), azimuth_grid.ravel()


def poa_grid_w_m2(
    tilt_flat: np.ndarray,
    azimuth_flat: np.ndarray,
    *,
    solar_zenith_deg: float,
    solar_azimuth_deg: float,
    dni_w_m2: float,
    ghi_w_m2: float,
    dhi_w_m2: float,
) -> np.ndarray:
    if solar_zenith_deg >= 90.0:
        return np.zeros_like(tilt_flat, dtype=float)
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt_flat,
        surface_azimuth=azimuth_flat,
        solar_zenith=solar_zenith_deg,
        solar_azimuth=solar_azimuth_deg,
        dni=dni_w_m2,
        ghi=ghi_w_m2,
        dhi=dhi_w_m2,
        model="isotropic",
    )
    return np.asarray(poa["poa_global"], dtype=float)
