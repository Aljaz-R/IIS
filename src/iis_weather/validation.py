from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import PREPROCESSED_WEATHER_DIR, REPORTS_DIR
from .io import write_json

REQUIRED_COLUMNS = {
    "time",
    "city",
    "country",
    "latitude",
    "longitude",
    "temperature_c",
    "precipitation_mm",
    "relative_humidity",
    "pressure_msl",
    "wind_speed_kmh",
}


def validate_preprocessed_weather_data(
    data_dir: Path = PREPROCESSED_WEATHER_DIR,
    report_path: Path = REPORTS_DIR / "data_validation.json",
) -> dict[str, Any]:
    city_reports = []
    paths = sorted(path for path in data_dir.glob("*.csv") if path.name != "cities.csv")
    if not paths:
        report = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "fail",
            "summary": {"cities": 0, "rows": 0},
            "checks": [{"name": "weather_files_present", "status": "fail"}],
            "cities": [],
        }
        write_json(report_path, report)
        return report

    total_rows = 0
    total_missing = 0
    total_duplicates = 0
    for path in paths:
        frame = pd.read_csv(path)
        total_rows += int(len(frame))
        missing_columns = sorted(REQUIRED_COLUMNS - set(frame.columns))
        city_report: dict[str, Any] = {"city": path.stem, "rows": int(len(frame)), "missing_columns": missing_columns}
        if missing_columns:
            city_report["status"] = "fail"
            city_reports.append(city_report)
            continue

        parsed_time = pd.to_datetime(frame["time"], errors="coerce")
        numeric = frame[["temperature_c", "precipitation_mm", "relative_humidity", "pressure_msl", "wind_speed_kmh"]].apply(
            pd.to_numeric,
            errors="coerce",
        )
        invalid_ranges = int(
            (
                (numeric["temperature_c"] < -60)
                | (numeric["temperature_c"] > 60)
                | (numeric["precipitation_mm"] < 0)
                | (numeric["relative_humidity"] < 0)
                | (numeric["relative_humidity"] > 100)
                | (numeric["pressure_msl"] < 850)
                | (numeric["pressure_msl"] > 1100)
                | (numeric["wind_speed_kmh"] < 0)
            ).sum()
        )
        missing = int(parsed_time.isna().sum() + numeric.isna().sum().sum())
        duplicates = int(parsed_time.duplicated().sum())
        total_missing += missing
        total_duplicates += duplicates
        chronological = bool(parsed_time.dropna().is_monotonic_increasing)
        status = "pass"
        if invalid_ranges or missing or len(frame) < 72:
            status = "fail"
        elif duplicates or not chronological:
            status = "warn"
        city_report.update(
            {
                "status": status,
                "missing_values": missing,
                "duplicate_times": duplicates,
                "invalid_ranges": invalid_ranges,
                "is_chronological": chronological,
                "time_min": str(parsed_time.min()),
                "time_max": str(parsed_time.max()),
            }
        )
        city_reports.append(city_report)

    failing = [item for item in city_reports if item.get("status") == "fail"]
    warning = [item for item in city_reports if item.get("status") == "warn"]
    status = "fail" if failing else "warn" if warning else "pass"
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "summary": {
            "cities": len(paths),
            "rows": total_rows,
            "missing_values": total_missing,
            "duplicate_times": total_duplicates,
        },
        "checks": [
            {"name": "weather_files_present", "status": "pass", "value": len(paths)},
            {"name": "required_columns", "status": "fail" if failing else "pass"},
            {"name": "numeric_ranges", "status": "fail" if failing else "pass"},
            {"name": "missing_values", "status": "fail" if total_missing else "pass", "value": total_missing},
            {"name": "duplicate_times", "status": "warn" if total_duplicates else "pass", "value": total_duplicates},
        ],
        "cities": city_reports,
    }
    write_json(report_path, report)
    return report

