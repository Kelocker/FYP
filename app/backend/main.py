from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config_store import load_panel_config, save_panel_config
from app.locations import LOCATIONS
from app.metrics_data import MODEL_METRICS
from app.pipeline import run_daily_prediction
from app.schemas import LocationOut, PanelConfig, PredictRequest, PredictResponse

app = FastAPI(title="Apolius Backend")

# Dev-only CORS: tighten to the real frontend origin before deploying.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/locations", response_model=List[LocationOut])
def get_locations() -> List[LocationOut]:
    return [
        LocationOut(id=loc.id, state=loc.state, city=loc.city, latitude=loc.latitude, longitude=loc.longitude)
        for loc in sorted(LOCATIONS.values(), key=lambda loc: loc.state)
    ]


@app.get("/api/metrics")
def get_metrics():
    return MODEL_METRICS


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """Feeds dashboard.html's live prediction + pvlib orientation recommendation.

    Day-level: recommends one fixed tilt/azimuth that maximizes total
    energy across `req.date` (today if omitted), not just one instant.
    `ghi_source` reports whether the number came from the model, real
    history, or the clear-sky fallback.
    """
    return run_daily_prediction(req)


@app.get("/api/config", response_model=PanelConfig)
def get_config() -> PanelConfig:
    """The persisted panel config from the frontend's first-run setup
    wizard (tilt + azimuth only -- see PanelConfig's docstring for why
    location isn't part of it). 404 means it hasn't been set up yet; the
    frontend takes that as the signal to show the wizard rather than
    trusting a placeholder."""
    config = load_panel_config()
    if config is None:
        raise HTTPException(status_code=404, detail="panel not configured yet")
    return config


@app.put("/api/config", response_model=PanelConfig)
def put_config(config: PanelConfig) -> PanelConfig:
    """Saves the panel config from the setup wizard, or a later edit via
    the brand menu's "Panel setup" item. Single config, no auth/multi-user
    (see backend/README.md) -- a PUT here always overwrites the one file."""
    return save_panel_config(config)


@app.get("/health")
def health():
    return {"status": "ok"}
