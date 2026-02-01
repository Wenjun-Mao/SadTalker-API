Quick service launch guide

General
- Each non-SadTalker service uses uv and has its own pyproject.toml.
- Run commands from the service directory.
- Configure secrets via Docker secrets in production or .env for local dev.

s1-ingest-nocodb
- Install deps and lock (first time):
  uv sync
- Local env (example):
  copy ../.env.example to .env and fill values
- Start the API:
  cd services/s1-ingest-nocodb
  uv run uvicorn ingest_nocodb.app:app --host 0.0.0.0 --port 5101
- Run tests:
  uv run pytest

Other services
- Currently placeholders. Add an app entrypoint first, then run with uvicorn similarly.
