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
  uv run uvicorn ingest_nocodb.app:app --host 0.0.0.0 --port 7101
- Run tests:
  uv run pytest

Other services
- s2-download-mp4
- s3-tts-voice
- s4-sadtalker (legacy SadTalker service)
- s5-broll-selector
- s6-video-compositor
- s7-storage-uploader
- s8-nocodb-updater

These are placeholders unless noted. Add an app entrypoint first, then run with uvicorn similarly.
