from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from iis_weather.models import predict_next_24h


def main():
    prediction = predict_next_24h("Ljubljana")
    print(prediction)


if __name__ == "__main__":
    main()
