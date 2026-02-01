FROM python:3.12-slim

WORKDIR /app

# Install uv
RUN python -m pip install --no-cache-dir uv

# Copy project metadata first for better layer caching
COPY pyproject.toml /app/pyproject.toml

# Copy source code
COPY src /app/src

# Install runtime dependencies
RUN uv sync --no-dev

EXPOSE 5101

CMD ["uv", "run", "uvicorn", "ingest_nocodb.app:app", "--host", "0.0.0.0", "--port", "8111"]
