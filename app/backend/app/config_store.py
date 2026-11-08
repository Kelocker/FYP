from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .schemas import PanelConfig

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "panel_config.json"


def load_panel_config() -> Optional[PanelConfig]:
    if not CONFIG_PATH.exists():
        return None
    try:
        data = json.loads(CONFIG_PATH.read_text())
        return PanelConfig.model_validate(data)
    except (json.JSONDecodeError, ValueError):
        return None


def save_panel_config(config: PanelConfig) -> PanelConfig:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(config.model_dump_json(indent=2))
    return config
