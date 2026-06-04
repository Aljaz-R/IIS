from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iis_weather.ingestion import preprocess_weather_data


if __name__ == "__main__":
    summary = preprocess_weather_data()
    print(summary)

