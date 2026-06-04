from __future__ import annotations

import math
import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    CAPITALS,
    FORECAST_API_URL,
    HISTORICAL_DAYS,
    HISTORY_API_URL,
    HOURLY_FORECAST_VARIABLES,
    HOURLY_HISTORY_VARIABLES,
    PREPROCESSED_WEATHER_DIR,
    RAW_FORECAST_DIR,
    RAW_HISTORY_DIR,
)
from .io import city_slug, save_cities, write_json


def _request_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    import requests

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=45)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
    raise last_error or RuntimeError("Open-Meteo request failed.")


def fetch_weather_data(days: int = HISTORICAL_DAYS, include_forecast: bool = True) -> dict[str, Any]:
    end_date = date.today() - timedelta(days=2)
    start_date = end_date - timedelta(days=days)
    RAW_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    RAW_FORECAST_DIR.mkdir(parents=True, exist_ok=True)
    save_cities()

    results = []
    for capital in CAPITALS:
        slug = city_slug(capital["city"])
        base_params = {
            "latitude": capital["latitude"],
            "longitude": capital["longitude"],
            "timezone": capital["timezone"],
        }
        history = _request_json(
            HISTORY_API_URL,
            {
                **base_params,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "hourly": ",".join(HOURLY_HISTORY_VARIABLES),
            },
        )
        write_json(RAW_HISTORY_DIR / f"{slug}.json", history)
        if include_forecast:
            try:
                forecast = _request_json(
                    FORECAST_API_URL,
                    {
                        **base_params,
                        "forecast_days": 2,
                        "hourly": ",".join(HOURLY_FORECAST_VARIABLES),
                    },
                )
            except Exception as exc:
                forecast = {
                    "warning": "Forecast endpoint was unavailable during fetch. Historical data was saved successfully.",
                    "error": str(exc),
                    "city": capital["city"],
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
        else:
            forecast = {
                "warning": "Forecast fetch was skipped. Historical data was saved successfully.",
                "city": capital["city"],
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        write_json(RAW_FORECAST_DIR / f"{slug}.json", forecast)
        results.append({"city": capital["city"], "history_hours": len(history.get("hourly", {}).get("time", []))})
        print(f"Fetched Open-Meteo data for {capital['city']}")

    return {"cities": len(results), "start_date": start_date.isoformat(), "end_date": end_date.isoformat(), "results": results}


def generate_demo_weather_data(days: int = 120) -> dict[str, Any]:
    """Create deterministic local data for offline development only."""
    rng = np.random.default_rng(42)
    end_time = pd.Timestamp.now().floor("h") - pd.Timedelta(days=1)
    times = pd.date_range(end=end_time, periods=days * 24, freq="h")
    RAW_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    RAW_FORECAST_DIR.mkdir(parents=True, exist_ok=True)
    save_cities()

    for city_index, capital in enumerate(CAPITALS):
        seasonal_offset = 9 + city_index * 0.6
        hour_angle = 2 * math.pi * times.hour / 24
        day_angle = 2 * math.pi * np.arange(len(times)) / (24 * 32)
        temperature = seasonal_offset + 7 * np.sin(hour_angle - 1.0) + 4 * np.sin(day_angle) + rng.normal(0, 1.4, len(times))
        rain_signal = 0.32 + 0.17 * np.sin(day_angle + city_index) + 0.10 * np.cos(hour_angle)
        rain_event = rng.random(len(times)) < np.clip(rain_signal, 0.05, 0.72)
        precipitation = np.where(rain_event, rng.gamma(1.4, 1.2, len(times)), 0.0)
        humidity = np.clip(68 + 18 * rain_event - 9 * np.sin(hour_angle) + rng.normal(0, 5, len(times)), 25, 100)
        pressure = 1015 + 7 * np.cos(day_angle + city_index / 2) + rng.normal(0, 2.2, len(times))
        wind = np.clip(8 + 5 * rng.random(len(times)) + 3 * np.sin(day_angle), 0, None)
        payload = {
            "latitude": capital["latitude"],
            "longitude": capital["longitude"],
            "timezone": capital["timezone"],
            "hourly": {
                "time": [str(t).replace(" ", "T")[:16] for t in times],
                "temperature_2m": np.round(temperature, 2).tolist(),
                "precipitation": np.round(precipitation, 2).tolist(),
                "relative_humidity_2m": np.round(humidity, 2).tolist(),
                "pressure_msl": np.round(pressure, 2).tolist(),
                "wind_speed_10m": np.round(wind, 2).tolist(),
            },
        }
        write_json(RAW_HISTORY_DIR / f"{city_slug(capital['city'])}.json", payload)
        write_json(RAW_FORECAST_DIR / f"{city_slug(capital['city'])}.json", payload)
        print(f"Generated demo data for {capital['city']}")

    return {"cities": len(CAPITALS), "mode": "demo", "created_at": datetime.now(timezone.utc).isoformat()}


def preprocess_weather_data() -> dict[str, Any]:
    PREPROCESSED_WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    cities = save_cities()
    written = []
    for capital in CAPITALS:
        path = RAW_HISTORY_DIR / f"{city_slug(capital['city'])}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing raw weather file for {capital['city']}: {path}")

        payload = json.loads(path.read_text(encoding="utf-8"))
        hourly = payload["hourly"]
        frame = pd.DataFrame(hourly)
        frame = frame.rename(
            columns={
                "time": "time",
                "temperature_2m": "temperature_c",
                "precipitation": "precipitation_mm",
                "relative_humidity_2m": "relative_humidity",
                "pressure_msl": "pressure_msl",
                "wind_speed_10m": "wind_speed_kmh",
            }
        )
        frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
        frame["city"] = capital["city"]
        frame["country"] = capital["country"]
        frame["latitude"] = capital["latitude"]
        frame["longitude"] = capital["longitude"]
        for column in ["temperature_c", "precipitation_mm", "relative_humidity", "pressure_msl", "wind_speed_kmh"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = (
            frame.dropna(subset=["time", "temperature_c", "precipitation_mm"])
            .drop_duplicates(subset=["time"])
            .sort_values("time")
        )
        out_path = PREPROCESSED_WEATHER_DIR / f"{city_slug(capital['city'])}.csv"
        frame.to_csv(out_path, index=False)
        written.append({"city": capital["city"], "rows": int(len(frame)), "path": str(out_path)})
        print(f"Saved {out_path}")

    return {"cities": int(len(cities)), "files": written}
