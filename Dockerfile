FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

COPY pyproject.toml uv.lock README.md ./
RUN pip install --no-cache-dir uv && uv sync --frozen --no-install-project

COPY . .

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "python", "src/app/serve.py", "--host", "0.0.0.0", "--port", "8000"]
