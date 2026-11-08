from __future__ import annotations

from datetime import date as date_type, datetime, time as time_type, timezone
from zoneinfo import ZoneInfo

import numpy as np

from . import inference, solar
from .locations import LOCATIONS
from .schemas import (
    HourPoint,
    IrradianceDecomposition,
    OrientationResult,
    PredictRequest,
    PredictResponse,
    SolarPosition,
)

LOCAL_TZ = ZoneInfo("Asia/Kuala_Lumpur")


def run_daily_prediction(req: PredictRequest) -> PredictResponse:
    has_user = req.user_tilt_deg is not None and req.user_azimuth_deg is not None
    target_date: date_type = req.date or datetime.now(LOCAL_TZ).date()

    model_hourly_ghi = None
    model_id_used = None
    is_replay = False
    replay_basis_date = None
    ghi_source = "clearsky_placeholder"

    loc = LOCATIONS.get(req.location) if req.location else None
    if loc is not None:
        if target_date == inference.MODEL_PREDICTED_DAY:
            try:
                model_hourly_ghi, replay_basis_date, target_date, ghi_source = inference.predict_ghi_day_w_m2(
                    loc.dataset_state, req.model_id
                )
                model_id_used = req.model_id
                is_replay = True
            except inference.UnknownLocationForModel:
                pass
        elif inference.DATASET_FIRST_DAY <= target_date <= inference.DATASET_LAST_REAL_DAY:
            historical = inference.get_historical_ghi_day_w_m2(loc.dataset_state, target_date)
            if historical is not None:
                model_hourly_ghi = historical
                ghi_source = "historical"

    tilt_flat, azimuth_flat = solar.orientation_grid()
    total_grid = np.zeros_like(tilt_flat, dtype=float)
    hour_grids: list[np.ndarray] = []

    hourly_points: list[HourPoint] = []
    ghi_w_m2_total = 0.0
    user_daily_total_w_m2 = 0.0

    for hour in range(24):
        local_ts = datetime.combine(target_date, time_type(hour=hour), tzinfo=LOCAL_TZ)
        utc_ts = local_ts.astimezone(timezone.utc)

        if model_hourly_ghi is not None:
            ghi_w_m2 = model_hourly_ghi[hour]
        else:
            ghi_w_m2, ghi_source = inference.predict_ghi_w_m2(req.latitude, req.longitude, utc_ts)
        ghi_w_m2_total += ghi_w_m2

        solpos = solar.solar_position(req.latitude, req.longitude, utc_ts)
        decomp = solar.decompose_ghi(ghi_w_m2, solpos["apparent_zenith_deg"], utc_ts)

        hour_grid = solar.poa_grid_w_m2(
            tilt_flat,
            azimuth_flat,
            solar_zenith_deg=solpos["apparent_zenith_deg"],
            solar_azimuth_deg=solpos["azimuth_deg"],
            dni_w_m2=decomp["dni_w_m2"],
            ghi_w_m2=ghi_w_m2,
            dhi_w_m2=decomp["dhi_w_m2"],
        )
        total_grid += hour_grid
        hour_grids.append(hour_grid)

        user_poa = None
        if has_user:
            user_poa = solar.poa_at_orientation(
                req.user_tilt_deg,
                req.user_azimuth_deg,
                solar_zenith_deg=solpos["apparent_zenith_deg"],
                solar_azimuth_deg=solpos["azimuth_deg"],
                dni_w_m2=decomp["dni_w_m2"],
                ghi_w_m2=ghi_w_m2,
                dhi_w_m2=decomp["dhi_w_m2"],
            )
            user_daily_total_w_m2 += user_poa

        hourly_points.append(
            HourPoint(
                timestamp=local_ts,
                solar_position=SolarPosition(**solpos),
                irradiance=IrradianceDecomposition(
                    ghi_w_m2=ghi_w_m2,
                    dni_w_m2=decomp["dni_w_m2"],
                    dhi_w_m2=decomp["dhi_w_m2"],
                    clearness_index_kt=decomp["clearness_index_kt"],
                ),
                recommended_poa_w_m2=0.0,
                user_poa_w_m2=user_poa,
            )
        )

    best_idx = int(np.argmax(total_grid))
    for point, hour_grid in zip(hourly_points, hour_grids):
        point.recommended_poa_w_m2 = float(hour_grid[best_idx])

    recommended = OrientationResult(
        tilt_deg=float(tilt_flat[best_idx]),
        azimuth_deg=float(azimuth_flat[best_idx]),
        poa_daily_total_wh_m2=float(total_grid[best_idx]),
    )

    ghi_kwh_m2_total = ghi_w_m2_total / 1000.0

    user_result = None
    gain_over_user_pct = None
    if has_user:
        user_result = OrientationResult(
            tilt_deg=req.user_tilt_deg,
            azimuth_deg=req.user_azimuth_deg,
            poa_daily_total_wh_m2=user_daily_total_w_m2,
        )
        if user_daily_total_w_m2 > 0:
            gain_over_user_pct = (
                (recommended.poa_daily_total_wh_m2 - user_daily_total_w_m2)
                / user_daily_total_w_m2
                * 100.0
            )

    return PredictResponse(
        location=req.location,
        latitude=req.latitude,
        longitude=req.longitude,
        date=target_date,
        ghi_source=ghi_source,
        model_id=model_id_used,
        is_replay=is_replay,
        replay_basis_date=replay_basis_date,
        predicted_ghi_kwh_m2_total=ghi_kwh_m2_total,
        recommended=recommended,
        user=user_result,
        gain_over_user_pct=gain_over_user_pct,
        hourly=hourly_points,
    )
