from __future__ import annotations

from datetime import date as date_type, datetime
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator

from .inference import AVAILABLE_MODEL_IDS
from .locations import LOCATIONS


class LocationOut(BaseModel):
    id: str
    state: str
    city: str
    latitude: float
    longitude: float


class PredictRequest(BaseModel):
    location: Optional[str] = Field(
        default=None,
        description="A location id from GET /api/locations, e.g. 'kuala_lumpur'. Overrides latitude/longitude when given.",
    )
    latitude: Optional[float] = Field(default=None, ge=-90, le=90)
    longitude: Optional[float] = Field(default=None, ge=-180, le=180)
    date: Optional[date_type] = Field(
        default=None,
        description="Calendar day to predict, in the location's local date (Malaysia, UTC+8). Defaults to today if omitted.",
    )
    user_tilt_deg: Optional[float] = Field(
        default=None,
        ge=0,
        le=90,
        description="The user's own panel tilt, for comparison against the recommendation.",
    )
    user_azimuth_deg: Optional[float] = Field(
        default=None,
        ge=0,
        le=360,
        description="The user's own panel azimuth (0=N, 90=E, 180=S, 270=W).",
    )
    model_id: str = Field(
        default="ensemble",
        description=(
            "Which trained model predicts GHI: 'ensemble' (default, validation-optimized "
            "weighted combination of all three -- the project's best result), 'lstm', "
            "'transformer', or 'linreg'. Only affects locations from GET /api/locations -- "
            "raw latitude/longitude requests always use the clear-sky fallback regardless "
            "of this field, since there's no historical window to run a trained model on."
        ),
    )

    @model_validator(mode="after")
    def _resolve_location(self) -> "PredictRequest":
        if self.location is not None:
            loc = LOCATIONS.get(self.location)
            if loc is None:
                valid = ", ".join(sorted(LOCATIONS))
                raise ValueError(f"unknown location id {self.location!r}; valid ids: {valid}")
            self.latitude = loc.latitude
            self.longitude = loc.longitude
        elif self.latitude is None or self.longitude is None:
            raise ValueError("either `location` or both `latitude` and `longitude` are required")
        if self.model_id not in AVAILABLE_MODEL_IDS:
            valid = ", ".join(AVAILABLE_MODEL_IDS)
            raise ValueError(f"unknown model_id {self.model_id!r}; valid ids: {valid}")
        return self


class SolarPosition(BaseModel):
    apparent_zenith_deg: float
    zenith_deg: float
    apparent_elevation_deg: float
    elevation_deg: float
    azimuth_deg: float


class IrradianceDecomposition(BaseModel):
    ghi_w_m2: float
    dni_w_m2: float
    dhi_w_m2: float
    clearness_index_kt: float


class HourPoint(BaseModel):
    timestamp: datetime
    solar_position: SolarPosition
    irradiance: IrradianceDecomposition
    recommended_poa_w_m2: float
    user_poa_w_m2: Optional[float] = None


class OrientationResult(BaseModel):
    tilt_deg: float
    azimuth_deg: float
    poa_daily_total_wh_m2: float


class PredictResponse(BaseModel):
    location: Optional[str] = None
    latitude: float
    longitude: float
    date: date_type 
    ghi_source: str  
    model_id: Optional[str] = None  
    is_replay: bool = False 
    replay_basis_date: Optional[date_type] = None 
    predicted_ghi_kwh_m2_total: float 
    recommended: OrientationResult
    user: Optional[OrientationResult] = None
    gain_over_user_pct: Optional[float] = None
    hourly: List[HourPoint]


class PanelConfig(BaseModel):
    tilt_deg: float = Field(ge=0, le=90, description="Panel tilt from flat ground, in degrees.")
    azimuth_deg: float = Field(
        ge=0, le=360, description="Panel compass azimuth (0=N, 90=E, 180=S, 270=W)."
    )
