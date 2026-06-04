from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iis_weather.preprocessing import build_all_processed_outputs


if __name__ == "__main__":
    summary = build_all_processed_outputs()
    print(f"Built supervised dataset: {summary}")
