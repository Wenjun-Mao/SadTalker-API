FROM python:3.12-slim

WORKDIR /app

RUN python -m pip install --no-cache-dir uv

COPY pyproject.toml /app/pyproject.toml
COPY src /app/src

RUN uv sync --no-dev

CMD ["/bin/sh", "-c", "uv run dramatiq tts_voice.worker -Q ${S3_QUEUE:-s3-tts-voice} -p 1 -t 1"]
