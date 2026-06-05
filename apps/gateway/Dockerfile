FROM python:3.12-slim

WORKDIR /app
ENV PYTHONPATH=/app

COPY apps/gateway/pyproject.toml /app/pyproject.toml
COPY apps/gateway/alembic.ini /app/alembic.ini
COPY apps/gateway/src /app/src
RUN pip install --no-cache-dir .

EXPOSE 8080
CMD ["sh", "-c", "alembic upgrade head && uvicorn src.main:app --host ${APP_HOST:-0.0.0.0} --port ${APP_PORT:-8080}"]
