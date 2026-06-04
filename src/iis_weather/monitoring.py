from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import math
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error

from .config import MODELS_DIR, REPORTS_DIR, ROOT_DIR, SUPERVISED_DATASET
from .io import read_json, write_json
from .models import (
    MODEL_METADATA_FILE,
    PRECIPITATION_MODEL_FILE,
    TEMPERATURE_MODEL_FILE,
    _load_bundle,
)
from .preprocessing import build_supervised_dataset


def evaluate_production_window(
    dataset_path: Path = SUPERVISED_DATASET,
    report_path: Path = REPORTS_DIR / "production_monitoring.json",
    recent_rows: int = 800,
) -> dict[str, Any]:
    if not TEMPERATURE_MODEL_FILE.exists() or not PRECIPITATION_MODEL_FILE.exists():
        raise FileNotFoundError("Train the weather models before production monitoring.")

    if dataset_path.exists():
        dataset = pd.read_csv(dataset_path, parse_dates=["time"])
    else:
        dataset = build_supervised_dataset(dataset_path)

    temperature_bundle = _load_bundle(TEMPERATURE_MODEL_FILE)
    precipitation_bundle = _load_bundle(PRECIPITATION_MODEL_FILE)
    feature_columns = temperature_bundle["feature_columns"]
    window = dataset.sort_values(["time", "city"]).tail(recent_rows).copy()
    X_window = window[feature_columns]
    temperature_pred = temperature_bundle["model"].predict(X_window)
    precipitation_pred = precipitation_bundle["model"].predict(X_window)

    temp_mae = float(mean_absolute_error(window["target_temperature_next"], temperature_pred))
    temp_rmse = float(math.sqrt(mean_squared_error(window["target_temperature_next"], temperature_pred)))
    precip_accuracy = float(accuracy_score(window["target_precipitation_next"].astype(int), precipitation_pred))
    precip_f1 = float(f1_score(window["target_precipitation_next"].astype(int), precipitation_pred, zero_division=0))

    training_report = read_json(MODEL_METADATA_FILE, default={}) or {}
    baseline_mae = training_report.get("metrics", {}).get("temperature_regression", {}).get("mae")
    status = "unknown"
    if baseline_mae:
        status = "warn" if temp_mae > baseline_mae * 1.5 else "pass"

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "window": {
            "rows": int(len(window)),
            "time_min": str(window["time"].min()),
            "time_max": str(window["time"].max()),
            "cities": int(window["city"].nunique()),
        },
        "metrics": {
            "temperature_mae": temp_mae,
            "temperature_rmse": temp_rmse,
            "precipitation_accuracy": precip_accuracy,
            "precipitation_f1": precip_f1,
        },
        "baseline_training_metrics": training_report.get("metrics", {}),
        "models_dir": str(MODELS_DIR.relative_to(ROOT_DIR)),
    }
    write_json(report_path, report)
    return report
