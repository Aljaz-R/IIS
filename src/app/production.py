from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iis_weather.config import HISTORICAL_DAYS
from iis_weather.ingestion import fetch_weather_data, preprocess_weather_data
from iis_weather.models import PRECIPITATION_MODEL_FILE, TEMPERATURE_MODEL_FILE, train_models
from iis_weather.monitoring import evaluate_production_window
from iis_weather.preprocessing import build_all_processed_outputs, build_latest_features
from iis_weather.web import serve


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _models_exist() -> bool:
    return TEMPERATURE_MODEL_FILE.exists() and PRECIPITATION_MODEL_FILE.exists()


def bootstrap_artifacts(days: int, force_train: bool = False) -> None:
    print(f"Bootstrapping production weather artifacts for {days} days.", flush=True)
    fetch_weather_data(days=days, include_forecast=True)
    preprocess_weather_data()
    build_all_processed_outputs()
    if force_train or not _models_exist():
        train_models()
    evaluate_production_window()


def refresh_runtime_weather(days: int) -> None:
    print(f"Refreshing runtime weather data for {days} days.", flush=True)
    fetch_weather_data(days=days, include_forecast=True)
    preprocess_weather_data()
    build_latest_features()


def run_server() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    refresh_on_start = _env_bool("IIS_REFRESH_ON_START", True)
    refresh_days = int(os.getenv("IIS_RUNTIME_REFRESH_DAYS", "14"))

    if refresh_on_start:
        try:
            if _models_exist():
                refresh_runtime_weather(refresh_days)
            else:
                bootstrap_days = max(refresh_days, int(os.getenv("IIS_BOOTSTRAP_DAYS", str(HISTORICAL_DAYS))))
                bootstrap_artifacts(bootstrap_days, force_train=True)
        except Exception as exc:
            if not _models_exist():
                raise
            print(f"Runtime weather refresh failed; starting with packaged artifacts. Error: {exc}", flush=True)

    serve(host=host, port=port)


def main() -> None:
    parser = argparse.ArgumentParser(description="Production bootstrap and server entrypoint.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap = subparsers.add_parser("bootstrap", help="Fetch data, build datasets, train models and monitoring reports.")
    bootstrap.add_argument("--days", type=int, default=HISTORICAL_DAYS)
    bootstrap.add_argument("--force-train", action="store_true")

    subparsers.add_parser("serve", help="Refresh runtime data if enabled and serve the web app.")

    args = parser.parse_args()
    if args.command == "bootstrap":
        bootstrap_artifacts(days=args.days, force_train=args.force_train)
    elif args.command == "serve":
        run_server()


if __name__ == "__main__":
    main()
