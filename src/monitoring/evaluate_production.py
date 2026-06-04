from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iis_weather.monitoring import evaluate_production_window


if __name__ == "__main__":
    report = evaluate_production_window()
    print(f"Production monitoring status: {report['status']}")
    print("Monitoring report written to reports/production_monitoring.json")
