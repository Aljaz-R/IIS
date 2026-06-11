from __future__ import annotations

import json
import math
import pickle
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    EXPERIMENTS_DIR,
    LATEST_FEATURES,
    MODELS_DIR,
    PRECIPITATION_THRESHOLD_MM,
    RANDOM_STATE,
    RAW_FORECAST_DIR,
    REPORTS_DIR,
    ROOT_DIR,
    SUPERVISED_DATASET,
)
from .io import city_slug, load_cities, write_json
from .preprocessing import build_latest_features, build_supervised_dataset

TEMPERATURE_MODEL_FILE = MODELS_DIR / "temperature_regressor.pkl"
PRECIPITATION_MODEL_FILE = MODELS_DIR / "precipitation_classifier.pkl"
MODEL_METADATA_FILE = MODELS_DIR / "model_metadata.json"
MODEL_REGISTRY_FILE = MODELS_DIR / "model_registry.json"


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover
        return OneHotEncoder(handle_unknown="ignore")


def _feature_columns(dataset: pd.DataFrame) -> list[str]:
    excluded = {"time", "target_temperature_next", "target_precipitation_next"}
    return [column for column in dataset.columns if column not in excluded]


def _build_pipeline(task: str, feature_columns: list[str]) -> Pipeline:
    categorical_features = ["city", "country"]
    numeric_features = [column for column in feature_columns if column not in categorical_features]
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", _one_hot_encoder(), categorical_features),
            ("numeric", StandardScaler(), numeric_features),
        ]
    )
    if task == "regression":
        estimator = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.0008,
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            random_state=RANDOM_STATE,
        )
    else:
        estimator = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.0008,
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            random_state=RANDOM_STATE,
        )
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


def _time_split(dataset: pd.DataFrame, test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = dataset.sort_values(["time", "city"]).reset_index(drop=True)
    split_index = max(1, int(len(ordered) * (1 - test_fraction)))
    if split_index >= len(ordered):
        split_index = len(ordered) - 1
    return ordered.iloc[:split_index].copy(), ordered.iloc[split_index:].copy()


def _classification_metrics(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    except ValueError:
        metrics["roc_auc"] = None
    return metrics


def train_models(
    dataset_path: Path = SUPERVISED_DATASET,
    models_dir: Path = MODELS_DIR,
    reports_dir: Path = REPORTS_DIR,
) -> dict[str, Any]:
    if dataset_path.exists():
        dataset = pd.read_csv(dataset_path, parse_dates=["time"])
    else:
        dataset = build_supervised_dataset(dataset_path)

    if len(dataset) < 500:
        raise ValueError("Not enough supervised rows for weather model training.")

    feature_columns = _feature_columns(dataset)
    train_data, test_data = _time_split(dataset)
    X_train = train_data[feature_columns]
    X_test = test_data[feature_columns]

    temperature_model = _build_pipeline("regression", feature_columns)
    precipitation_model = _build_pipeline("classification", feature_columns)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        temperature_model.fit(X_train, train_data["target_temperature_next"])
        precipitation_model.fit(X_train, train_data["target_precipitation_next"].astype(int))

    temperature_pred = temperature_model.predict(X_test)
    precipitation_pred = precipitation_model.predict(X_test)
    precipitation_proba = precipitation_model.predict_proba(X_test)[:, 1]

    regression_metrics = {
        "mae": float(mean_absolute_error(test_data["target_temperature_next"], temperature_pred)),
        "rmse": float(math.sqrt(mean_squared_error(test_data["target_temperature_next"], temperature_pred))),
        "r2": float(r2_score(test_data["target_temperature_next"], temperature_pred)),
    }
    classification_metrics = _classification_metrics(
        test_data["target_precipitation_next"].astype(int),
        precipitation_pred,
        precipitation_proba,
    )

    created_at = datetime.now(timezone.utc).isoformat()
    experiment_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    with TEMPERATURE_MODEL_FILE.open("wb") as file:
        pickle.dump(
            {
                "model": temperature_model,
                "feature_columns": feature_columns,
                "task": "next_hour_temperature_regression",
                "created_at": created_at,
            },
            file,
        )
    with PRECIPITATION_MODEL_FILE.open("wb") as file:
        pickle.dump(
            {
                "model": precipitation_model,
                "feature_columns": feature_columns,
                "task": "next_hour_precipitation_classification",
                "threshold_mm": PRECIPITATION_THRESHOLD_MM,
                "created_at": created_at,
            },
            file,
        )

    if LATEST_FEATURES.exists():
        latest_features = pd.read_csv(LATEST_FEATURES)
    else:
        latest_features = build_latest_features(LATEST_FEATURES)
    metadata = {
        "created_at": created_at,
        "experiment_id": experiment_id,
        "feature_columns": feature_columns,
        "data": {
            "rows": int(len(dataset)),
            "cities": int(dataset["city"].nunique()),
            "train_rows": int(len(train_data)),
            "test_rows": int(len(test_data)),
            "time_min": str(dataset["time"].min()),
            "time_max": str(dataset["time"].max()),
            "latest_feature_rows": int(len(latest_features)),
        },
        "models": {
            "temperature_regressor": str(TEMPERATURE_MODEL_FILE.relative_to(ROOT_DIR)),
            "precipitation_classifier": str(PRECIPITATION_MODEL_FILE.relative_to(ROOT_DIR)),
        },
        "metrics": {
            "temperature_regression": regression_metrics,
            "precipitation_classification": classification_metrics,
        },
    }
    write_json(MODEL_METADATA_FILE, metadata)
    write_json(reports_dir / "model_evaluation.json", metadata)
    registry = {
        "created_at": created_at,
        "active_version": experiment_id,
        "models": [
            {
                "name": "temperature_regressor",
                "version": experiment_id,
                "task": "regression",
                "stage": "production",
                "path": str(TEMPERATURE_MODEL_FILE.relative_to(ROOT_DIR)),
                "metrics": regression_metrics,
            },
            {
                "name": "precipitation_classifier",
                "version": experiment_id,
                "task": "classification",
                "stage": "production",
                "path": str(PRECIPITATION_MODEL_FILE.relative_to(ROOT_DIR)),
                "metrics": classification_metrics,
            },
        ],
    }
    write_json(MODEL_REGISTRY_FILE, registry)
    write_json(EXPERIMENTS_DIR / f"{experiment_id}.json", metadata)
    write_json(EXPERIMENTS_DIR / "index.json", registry)
    return metadata


def _load_bundle(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing trained model: {path}")
    with path.open("rb") as file:
        return pickle.load(file)


def _future_time_features(row: pd.DataFrame, future_time: pd.Timestamp) -> pd.DataFrame:
    updated = row.copy()
    updated["time"] = future_time
    updated["hour"] = future_time.hour
    updated["day_of_week"] = future_time.dayofweek
    updated["month"] = future_time.month
    updated["hour_sin"] = np.sin(2 * np.pi * future_time.hour / 24)
    updated["hour_cos"] = np.cos(2 * np.pi * future_time.hour / 24)
    updated["dow_sin"] = np.sin(2 * np.pi * future_time.dayofweek / 7)
    updated["dow_cos"] = np.cos(2 * np.pi * future_time.dayofweek / 7)
    updated["month_sin"] = np.sin(2 * np.pi * future_time.month / 12)
    updated["month_cos"] = np.cos(2 * np.pi * future_time.month / 12)
    return updated


def _load_forecast_guidance(city: str) -> pd.DataFrame:
    path = RAW_FORECAST_DIR / f"{city_slug(city)}.json"
    if not path.exists():
        return pd.DataFrame()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return pd.DataFrame()
    hourly = payload.get("hourly") or {}
    if not hourly:
        return pd.DataFrame()
    frame = pd.DataFrame(hourly)
    if frame.empty or "time" not in frame.columns:
        return pd.DataFrame()
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    for column in ["precipitation", "precipitation_probability"]:
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)


def _forecast_row_for_time(forecast: pd.DataFrame, future_time: pd.Timestamp) -> pd.Series | None:
    if forecast.empty:
        return None
    target = future_time.round("h")
    delta = (forecast["time"] - target).abs()
    if delta.empty:
        return None
    index = delta.idxmin()
    if pd.isna(index) or delta.loc[index] > pd.Timedelta(minutes=45):
        return None
    return forecast.loc[index]


def predict_next_24h(city: str, latest_path: Path = LATEST_FEATURES) -> dict[str, Any]:
    temperature_bundle = _load_bundle(TEMPERATURE_MODEL_FILE)
    precipitation_bundle = _load_bundle(PRECIPITATION_MODEL_FILE)
    if latest_path.exists():
        latest = pd.read_csv(latest_path, parse_dates=["time"])
    else:
        latest = build_latest_features(latest_path)

    row = latest.loc[latest["city"].str.lower() == city.lower()]
    if row.empty:
        raise ValueError(f"Unknown city: {city}")
    row = row.tail(1).copy()
    last_time = pd.Timestamp(row.iloc[0]["time"])
    feature_columns = temperature_bundle["feature_columns"]
    temperature_model = temperature_bundle["model"]
    precipitation_model = precipitation_bundle["model"]
    forecast_guidance = _load_forecast_guidance(str(row.iloc[0]["city"]))
    predictions = []

    for hour in range(1, 25):
        future_time = last_time + pd.Timedelta(hours=hour)
        future_row = _future_time_features(row, future_time)
        X_future = future_row[feature_columns]
        temperature = float(temperature_model.predict(X_future)[0])
        model_precipitation_probability = float(precipitation_model.predict_proba(X_future)[0, 1])
        precipitation_probability = model_precipitation_probability
        recent_amounts = [
            float(row.iloc[0].get("precipitation_lag_1", 0.0)),
            float(row.iloc[0].get("precipitation_lag_2", 0.0)),
            float(row.iloc[0].get("precipitation_lag_3", 0.0)),
            float(row.iloc[0].get("precipitation_lag_6", 0.0)),
            float(row.iloc[0].get("precipitation_lag_12", 0.0)),
            float(row.iloc[0].get("precipitation_lag_24", 0.0)),
            float(row.iloc[0].get("precipitation_roll_sum_24", 0.0)) / 24,
        ]
        recent_positive = [value for value in recent_amounts if value > 0]
        typical_intensity = float(np.mean(recent_positive)) if recent_positive else 0.8
        precipitation_mm = precipitation_probability * max(0.5, typical_intensity)
        forecast_row = _forecast_row_for_time(forecast_guidance, future_time)
        if forecast_row is not None:
            forecast_probability = forecast_row.get("precipitation_probability")
            forecast_precipitation = forecast_row.get("precipitation")
            if pd.notna(forecast_probability):
                precipitation_probability = max(precipitation_probability, float(forecast_probability) / 100)
            if pd.notna(forecast_precipitation):
                precipitation_mm = max(precipitation_mm, float(forecast_precipitation))
        precipitation_expected = precipitation_probability >= 0.5 or precipitation_mm > PRECIPITATION_THRESHOLD_MM
        predictions.append(
            {
                "time": future_time.isoformat(),
                "temperature_c": round(temperature, 2),
                "precipitation_mm": round(max(precipitation_mm, 0.0), 2),
                "precipitation_probability": round(precipitation_probability, 4),
                "model_precipitation_probability": round(model_precipitation_probability, 4),
                "precipitation_expected": bool(precipitation_expected),
            }
        )
        row["temperature_c"] = temperature
        row["precipitation_event"] = 1.0 if precipitation_expected else 0.0
        row["precipitation_mm"] = precipitation_mm
        row["temperature_lag_1"] = temperature
        row["precipitation_event_lag_1"] = row["precipitation_event"]
        row["precipitation_lag_1"] = row["precipitation_mm"]

    cities = load_cities()
    city_meta = cities.loc[cities["city"].str.lower() == city.lower()]
    city_payload = {"city": city}
    if not city_meta.empty:
        city_payload.update(city_meta.iloc[0].to_dict())

    return {
        "city": city_payload,
        "last_measurement_time": last_time.isoformat(),
        "precipitation_threshold_mm": PRECIPITATION_THRESHOLD_MM,
        "predictions": predictions,
    }
