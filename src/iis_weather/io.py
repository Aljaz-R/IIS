from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import CAPITALS, CITIES_FILE


def city_slug(city: str) -> str:
    return city.lower().replace(" ", "_")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_cities(path: Path = CITIES_FILE) -> pd.DataFrame:
    frame = pd.DataFrame(CAPITALS)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def load_cities() -> pd.DataFrame:
    if CITIES_FILE.exists():
        return pd.read_csv(CITIES_FILE)
    return save_cities()

