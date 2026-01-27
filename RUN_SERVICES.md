Quick service launch guide

General
- Each non-SadTalker service uses uv and has its own pyproject.toml.
- Run commands from the service directory.

s1-ingest-nocodb
- Install deps and lock (first time):
  uv sync
- Start the API:
  uv run uvicorn ingest_nocodb.app:app --host 0.0.0.0 --port 8001
- Run tests:
  uv run pytest

Other services
- Currently placeholders. Add an app entrypoint first, then run with uvicorn similarly.
