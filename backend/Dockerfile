# Use distroless Python (smaller, fewer dependencies, fewer pulls)
# Or use ghcr.io mirror to avoid Docker Hub rate limits
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    libffi-dev \
    libssl-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

FROM python:3.11-slim AS runtime

RUN groupadd -r app && useradd --no-log-init -r -g app app

WORKDIR /app

COPY --from=builder /wheels /wheels
COPY . .

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    RELOAD=false \
    PORT=8000

RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
 && rm -rf /wheels

RUN mkdir -p /var/log/llm-apm && chown -R app:app /var/log/llm-apm

EXPOSE 8000

USER app

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT} $([ \"${RELOAD}\" = 'true' ] && echo '--reload') --log-level info"]