FROM python:3.12-slim

WORKDIR /app

RUN python -m pip install --no-cache-dir uv

COPY pyproject.toml /app/pyproject.toml
COPY src /app/src

RUN uv sync --no-dev

CMD ["/bin/sh", "-c", "uv run dramatiq download_mp4.worker -Q ${S2_QUEUE:-s2-download-mp4} -p 1 -t 1"]
