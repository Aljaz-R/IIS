from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import (
    LAG_HOURS,
    LATEST_FEATURES,
    PRECIPITATION_THRESHOLD_MM,
    PREPROCESSED_WEATHER_DIR,
    PROCESSED_WEATHER_DIR,
    RAW_FORECAST_DIR,
    ROLLING_WINDOWS,
    SUPERVISED_DATASET,
)
from .io import city_slug, load_cities


def load_preprocessed_weather_data() -> pd.DataFrame:
    frames = []
    for path in sorted(PREPROCESSED_WEATHER_DIR.glob("*.csv")):
        if path.name == "cities.csv":
            continue
        frame = pd.read_csv(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    data["time"] = pd.to_datetime(data["time"], errors="coerce")
    for column in ["temperature_c", "precipitation_mm", "relative_humidity", "pressure_msl", "wind_speed_kmh"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data.sort_values(["city", "time"]).reset_index(drop=True)


def _normalise_weather_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.rename(
        columns={
            "temperature_2m": "temperature_c",
            "precipitation": "precipitation_mm",
            "relative_humidity_2m": "relative_humidity",
            "wind_speed_10m": "wind_speed_kmh",
        }
    )
    result["time"] = pd.to_datetime(result["time"], errors="coerce")
    for column in ["temperature_c", "precipitation_mm", "relative_humidity", "pressure_msl", "wind_speed_kmh"]:
        if column not in result.columns:
            result[column] = np.nan
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def load_recent_forecast_weather_data() -> pd.DataFrame:
    frames = []
    cities = load_cities()
    if cities.empty:
        return pd.DataFrame()

    for _, city in cities.iterrows():
        path = RAW_FORECAST_DIR / f"{city_slug(city['city'])}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "hourly" not in payload:
            continue

        hourly = pd.DataFrame(payload["hourly"])
        if not hourly.empty:
            hourly = _normalise_weather_frame(hourly)
        current = payload.get("current") or {}
        current_frame = pd.DataFrame([current]) if current.get("time") else pd.DataFrame()
        if not current_frame.empty:
            current_frame = _normalise_weather_frame(current_frame)

        frame = pd.concat([hourly, current_frame], ignore_index=True)
        if frame.empty:
            continue
        current_time = pd.to_datetime(current.get("time"), errors="coerce") if current else pd.NaT
        if pd.notna(current_time):
            frame = frame.loc[frame["time"] <= current_time]
        frame["city"] = city["city"]
        frame["country"] = city["country"]
        frame["latitude"] = city["latitude"]
        frame["longitude"] = city["longitude"]
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    return (
        data.dropna(subset=["time", "temperature_c", "precipitation_mm"])
        .drop_duplicates(subset=["city", "time"], keep="last")
        .sort_values(["city", "time"])
        .reset_index(drop=True)
    )


def _add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["hour"] = result["time"].dt.hour
    result["day_of_week"] = result["time"].dt.dayofweek
    result["month"] = result["time"].dt.month
    result["hour_sin"] = np.sin(2 * np.pi * result["hour"] / 24)
    result["hour_cos"] = np.cos(2 * np.pi * result["hour"] / 24)
    result["dow_sin"] = np.sin(2 * np.pi * result["day_of_week"] / 7)
    result["dow_cos"] = np.cos(2 * np.pi * result["day_of_week"] / 7)
    result["month_sin"] = np.sin(2 * np.pi * result["month"] / 12)
    result["month_cos"] = np.cos(2 * np.pi * result["month"] / 12)
    return result


def _city_features(city_frame: pd.DataFrame, include_targets: bool) -> pd.DataFrame:
    frame = city_frame.sort_values("time").copy()
    weather_columns = ["temperature_c", "precipitation_mm", "relative_humidity", "pressure_msl", "wind_speed_kmh"]
    frame[weather_columns] = frame[weather_columns].interpolate(method="linear", limit_direction="both")
    frame["precipitation_event"] = (frame["precipitation_mm"] > PRECIPITATION_THRESHOLD_MM).astype(float)

    for lag in LAG_HOURS:
        frame[f"temperature_lag_{lag}"] = frame["temperature_c"].shift(lag)
        frame[f"precipitation_lag_{lag}"] = frame["precipitation_mm"].shift(lag)
        frame[f"precipitation_event_lag_{lag}"] = frame["precipitation_event"].shift(lag)

    shifted_temperature = frame["temperature_c"].shift(1)
    shifted_precipitation = frame["precipitation_mm"].shift(1)
    for window in ROLLING_WINDOWS:
        frame[f"temperature_roll_mean_{window}"] = shifted_temperature.rolling(window, min_periods=2).mean()
        frame[f"precipitation_roll_sum_{window}"] = shifted_precipitation.rolling(window, min_periods=2).sum()

    frame = _add_time_features(frame)
    if include_targets:
        frame["target_temperature_next"] = frame["temperature_c"].shift(-1)
        next_precipitation = frame["precipitation_mm"].shift(-1)
        frame["target_precipitation_next"] = (next_precipitation > PRECIPITATION_THRESHOLD_MM).astype(float)
        frame.loc[next_precipitation.isna(), "target_precipitation_next"] = np.nan
    return frame


def _build_feature_frame_from_data(data: pd.DataFrame, include_targets: bool) -> pd.DataFrame:
    if data.empty:
        raise ValueError("No preprocessed weather data found. Run src/data/fetch_weather_data.py and preprocess first.")

    frames = [_city_features(city_frame, include_targets) for _, city_frame in data.groupby("city", sort=True)]
    features = pd.concat(frames, ignore_index=True)
    required = [
        "temperature_c",
        "precipitation_mm",
        "relative_humidity",
        "pressure_msl",
        "wind_speed_kmh",
        "precipitation_event",
        *[f"temperature_lag_{lag}" for lag in LAG_HOURS],
        *[f"precipitation_lag_{lag}" for lag in LAG_HOURS],
        *[f"precipitation_event_lag_{lag}" for lag in LAG_HOURS],
        *[f"temperature_roll_mean_{window}" for window in ROLLING_WINDOWS],
        *[f"precipitation_roll_sum_{window}" for window in ROLLING_WINDOWS],
        "hour",
        "day_of_week",
        "month",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
    ]
    if include_targets:
        required.extend(["target_temperature_next", "target_precipitation_next"])
    return features.dropna(subset=required).reset_index(drop=True)


def build_feature_frame(include_targets: bool = True) -> pd.DataFrame:
    data = load_preprocessed_weather_data()
    return _build_feature_frame_from_data(data, include_targets)


def build_supervised_dataset(output_path=SUPERVISED_DATASET) -> pd.DataFrame:
    dataset = build_feature_frame(include_targets=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    return dataset


def build_latest_features(output_path=LATEST_FEATURES) -> pd.DataFrame:
    historical = load_preprocessed_weather_data()
    recent = load_recent_forecast_weather_data()
    data = pd.concat([historical, recent], ignore_index=True) if not recent.empty else historical
    data = (
        data.drop_duplicates(subset=["city", "time"], keep="last")
        .sort_values(["city", "time"])
        .reset_index(drop=True)
    )
    features = _build_feature_frame_from_data(data, include_targets=False)
    latest = (
        features.sort_values("time")
        .groupby("city", as_index=False, sort=True)
        .tail(1)
        .sort_values("city")
        .reset_index(drop=True)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    latest.to_csv(output_path, index=False)
    return latest


def build_all_processed_outputs() -> dict[str, int]:
    PROCESSED_WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    dataset = build_supervised_dataset()
    latest = build_latest_features()
    return {"rows": int(len(dataset)), "latest_rows": int(len(latest)), "cities": int(dataset["city"].nunique())}
