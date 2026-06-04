from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
RAW_WEATHER_DIR = DATA_DIR / "raw" / "weather"
RAW_HISTORY_DIR = RAW_WEATHER_DIR / "history"
RAW_FORECAST_DIR = RAW_WEATHER_DIR / "forecast"
PREPROCESSED_WEATHER_DIR = DATA_DIR / "preprocessed" / "weather"
PROCESSED_WEATHER_DIR = DATA_DIR / "processed" / "weather"
SUPERVISED_DATASET = PROCESSED_WEATHER_DIR / "weather_supervised.csv"
LATEST_FEATURES = PROCESSED_WEATHER_DIR / "latest_features.csv"
CITIES_FILE = PREPROCESSED_WEATHER_DIR / "cities.csv"

REPORTS_DIR = ROOT_DIR / "reports"
EXPERIMENTS_DIR = REPORTS_DIR / "experiments"
MODELS_DIR = ROOT_DIR / "models" / "weather"
PREDICTIONS_DIR = ROOT_DIR / "predictions"

HISTORY_API_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"

RANDOM_STATE = 42
PRECIPITATION_THRESHOLD_MM = 0.1
LAG_HOURS = (1, 2, 3, 6, 12, 24)
ROLLING_WINDOWS = (3, 6, 12, 24)

CAPITALS = [
    {"city": "Ljubljana", "country": "Slovenia", "latitude": 46.0569, "longitude": 14.5058, "timezone": "Europe/Ljubljana"},
    {"city": "Zagreb", "country": "Croatia", "latitude": 45.8150, "longitude": 15.9819, "timezone": "Europe/Zagreb"},
    {"city": "Vienna", "country": "Austria", "latitude": 48.2082, "longitude": 16.3738, "timezone": "Europe/Vienna"},
    {"city": "Budapest", "country": "Hungary", "latitude": 47.4979, "longitude": 19.0402, "timezone": "Europe/Budapest"},
    {"city": "Prague", "country": "Czechia", "latitude": 50.0755, "longitude": 14.4378, "timezone": "Europe/Prague"},
    {"city": "Berlin", "country": "Germany", "latitude": 52.5200, "longitude": 13.4050, "timezone": "Europe/Berlin"},
    {"city": "Paris", "country": "France", "latitude": 48.8566, "longitude": 2.3522, "timezone": "Europe/Paris"},
    {"city": "Rome", "country": "Italy", "latitude": 41.9028, "longitude": 12.4964, "timezone": "Europe/Rome"},
]

HISTORICAL_DAYS = 180
HOURLY_HISTORY_VARIABLES = [
    "temperature_2m",
    "precipitation",
    "relative_humidity_2m",
    "pressure_msl",
    "wind_speed_10m",
]
HOURLY_FORECAST_VARIABLES = [
    "temperature_2m",
    "precipitation",
    "precipitation_probability",
    "relative_humidity_2m",
    "pressure_msl",
    "wind_speed_10m",
]

