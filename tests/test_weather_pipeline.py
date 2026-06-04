import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iis_weather.preprocessing import build_feature_frame
from iis_weather.validation import validate_preprocessed_weather_data


class WeatherPipelineTest(unittest.TestCase):
    def test_validation_report_can_be_created(self):
        report = validate_preprocessed_weather_data()
        self.assertIn(report["status"], {"pass", "warn"})
        self.assertGreater(report["summary"]["cities"], 0)
        self.assertGreater(report["summary"]["rows"], 0)

    def test_feature_engineering_builds_supervised_rows(self):
        dataset = build_feature_frame(include_targets=True)
        self.assertGreater(len(dataset), 500)
        self.assertIn("target_temperature_next", dataset.columns)
        self.assertIn("target_precipitation_next", dataset.columns)
        self.assertGreater(dataset["city"].nunique(), 1)


if __name__ == "__main__":
    unittest.main()
