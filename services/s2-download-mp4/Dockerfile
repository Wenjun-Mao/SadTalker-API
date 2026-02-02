FROM python:3.12-slim

WORKDIR /app

RUN python -m pip install --no-cache-dir uv

COPY pyproject.toml /app/pyproject.toml
COPY src /app/src

RUN uv sync --no-dev

CMD ["sleep", "infinity"]
