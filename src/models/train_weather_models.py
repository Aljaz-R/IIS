from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iis_weather.models import train_models


if __name__ == "__main__":
    report = train_models()
    print(f"Training complete. Experiment: {report['experiment_id']}")
    print("Evaluation report written to reports/model_evaluation.json")

