from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iis_weather.validation import validate_preprocessed_weather_data


if __name__ == "__main__":
    report = validate_preprocessed_weather_data()
    print(f"Weather data validation status: {report['status']}")
    print("Report written to reports/data_validation.json")

