# IIS projekt: inteligentni sistem za vremensko napoved

Projekt sledi originalni ideji iz dokumenta: inteligentni sistem za napoved
vremena v nekaj evropskih prestolnicah. Sistem uporablja Open-Meteo API za
zgodovinske urne podatke in napovedne podatke, nato nauci dva napovedna modela:

- regresijski nevronski model za napoved temperature za naslednjo uro,
- klasifikacijski nevronski model za napoved, ali bodo naslednjo uro padavine.

Uporabniski vmesnik prikaze zemljevid izbranih evropskih prestolnic. Ob kliku na
mesto prikaze 24-urno napoved temperature in padavin za naslednji dan.
Administratorski pogled prikaze validacijo podatkov, metrike modelov, produkcijski
monitoring in aktivno verzijo modela.

## Lokalni zagon

```powershell
uv sync --frozen
uv run python src/data/fetch_weather_data.py
uv run python src/data/preprocess_weather_data.py
uv run python src/data/validate_weather_data.py
uv run python src/features/build_dataset.py
uv run python src/models/train_weather_models.py
uv run python src/monitoring/evaluate_production.py
uv run python src/app/serve.py --host 127.0.0.1 --port 8000
```

Nato odpri `http://127.0.0.1:8000`.

Ce se na Windows pojavi tezava z `uv` cache mapo, uporabi lokalni cache:

```powershell
$env:UV_CACHE_DIR = ".uv-cache"
uv run python src/app/serve.py --host 127.0.0.1 --port 8000
```

## Podatkovni in modelni cevovod

Projekt ima formalen DVC cevovod v `dvc.yaml`. Celoten cevovod lahko pozenes z:

```powershell
pipx install "dvc[s3]"
dvc repro
```

Faze cevovoda:

- `fetch_weather_data`: zajem Open-Meteo podatkov,
- `preprocess_weather_data`: ciscenje in zdruzevanje podatkov,
- `validate_weather_data`: avtomatska validacija podatkov,
- `build_weather_dataset`: izdelava nadzorovane ucne mnozice,
- `train_weather_models`: ucenje obeh nevronskih modelov in zapis eksperimenta,
- `evaluate_production`: ovrednotenje aktivnih modelov na zadnjem produkcijskem oknu.

Posamezne faze lahko pozenes tudi rocno:

```powershell
uv run python src/data/fetch_weather_data.py
uv run python src/data/preprocess_weather_data.py
uv run python src/data/validate_weather_data.py
uv run python src/features/build_dataset.py
uv run python src/models/train_weather_models.py
uv run python src/monitoring/evaluate_production.py
```

Izhodi:

- `data/raw/weather/history/*.json`: surovi zgodovinski Open-Meteo podatki,
- `data/raw/weather/forecast/*.json`: surovi Open-Meteo forecast podatki,
- `data/preprocessed/weather/*.csv`: ocisceni podatki po prestolnicah,
- `data/processed/weather/weather_supervised.csv`: ucna mnozica,
- `reports/data_validation.json`: avtomatska validacija podatkov,
- `models/weather/*.pkl`: naucena modela,
- `reports/model_evaluation.json`: metrika treninga in testnega dela,
- `reports/production_monitoring.json`: metrika zadnjega produkcijskega okna.

Mape `data`, `models`, `reports` in `predictions` so namenjene DVC/DagsHub
verzioniranju, ne neposrednemu Git commitanju.

## Testi

```powershell
uv run python -m unittest discover -s tests
```

## Docker

```powershell
docker build -t iis-weather .
docker run --rm -p 8000:8000 iis-weather
```

Zdravstveni endpoint za preverjanje deploya:

```powershell
Invoke-WebRequest http://127.0.0.1:8000/health
```

GitHub Actions workflow `.github/workflows/docker.yml` samodejno zgradi Docker
sliko. Na push v `main` ali `master` jo objavi v GitHub Container Registry
(`ghcr.io`). Ce je nastavljen secret `RENDER_DEPLOY_HOOK_URL`, workflow po buildu
sprozi se produkcijski deploy preko Render deploy hooka.

Za produkcijski deploy moras v GitHub repozitoriju nastaviti:

- `DAGSHUB_ACCESS_KEY_ID`,
- `DAGSHUB_SECRET_ACCESS_KEY`,
- `RENDER_DEPLOY_HOOK_URL` za Render deploy hook.

Natancni koraki za nastavitev produkcijskega deploya so opisani v
`DEPLOYMENT.md`.

## Izpolnjene zahteve projekta

- avtomatiziran zajem in predobdelava Open-Meteo podatkov,
- formalen DVC cevovod za zajem, obdelavo, validacijo, ucenje in monitoring,
- validacija podatkov z JSON porocilom in DVC metriko,
- feature engineering za urne casovne vrste,
- dva napovedna modela z nevronskimi mrezami,
- lokalno sledenje eksperimentom in modelni register v JSON obliki,
- produkcijski monitoring nad zadnjim podatkovnim oknom,
- administratorski pogled v web UI,
- Dockerfile in GitHub Actions workflow za build/push/deploy Docker slike.

## Offline razvoj

Ce Open-Meteo trenutno ni dosegljiv ali lokalno okolje nima interneta, lahko
ustvaris deterministican demo nabor podatkov:

```powershell
uv run python src/data/fetch_weather_data.py --demo --days 120
uv run python src/data/preprocess_weather_data.py
```
