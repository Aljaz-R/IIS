from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iis_weather.ingestion import fetch_weather_data, generate_demo_weather_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Open-Meteo weather data for European capitals.")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--demo", action="store_true", help="Generate deterministic local demo data instead of calling Open-Meteo.")
    parser.add_argument("--skip-forecast", action="store_true", help="Fetch only historical data and skip the optional forecast snapshot.")
    args = parser.parse_args()
    if args.demo:
        summary = generate_demo_weather_data(days=args.days)
    else:
        summary = fetch_weather_data(days=args.days, include_forecast=not args.skip_forecast)
    print(summary)


if __name__ == "__main__":
    main()
