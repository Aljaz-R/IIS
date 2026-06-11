FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

COPY pyproject.toml uv.lock README.md ./
RUN pip install --no-cache-dir uv && uv sync --frozen --no-install-project

COPY . .

ARG IIS_BOOTSTRAP_ARTIFACTS=true
ARG IIS_BOOTSTRAP_DAYS=180
RUN if [ "$IIS_BOOTSTRAP_ARTIFACTS" = "true" ]; then \
    uv run --no-sync python src/app/production.py bootstrap --days "$IIS_BOOTSTRAP_DAYS"; \
    fi

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "python", "src/app/production.py", "serve"]
