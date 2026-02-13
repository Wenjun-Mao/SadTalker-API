# SadTalker-API Project Context

## Overview
**SadTalker-API** is a microservices-based pipeline designed to automate the generation of talking-head videos. It wraps the core SadTalker algorithm (for 3D audio-driven face animation) into a scalable, asynchronous workflow triggered by NocoDB records.

## Architecture
The system follows a sequential pipeline architecture where a single job (identified by a `record_id`) is passed through multiple specialized stages.

*   **Communication:** Services communicate asynchronously using **RabbitMQ** as the broker and **Dramatiq** as the task queue library.
*   **Orchestration:** Services are containerized and orchestrated via **Docker Compose**.
*   **Configuration:** Services use `pydantic-settings` to load configuration from environment variables (local dev) or Docker secrets (production).

### Pipeline Flow
1.  **s1-ingest-nocodb:** Ingests trigger from NocoDB, initializes the job.
2.  **s2-download-mp4:** Downloads the source video (e.g., from Douyin).
3.  **s3-tts-voice:** Generates audio speech from text.
4.  **s4-sadtalker:** (Legacy Core) Generates the talking head animation using the audio and source image.
5.  **s5-broll-selector:** Selects background/B-roll footage.
6.  **s6-video-compositor:** Composites the avatar, background, and audio.
7.  **s7-storage-uploader:** Uploads the final result to storage.
8.  **s8-nocodb-updater:** Updates the original NocoDB record with the result URL.

## Directory Structure

### `services/`
Each service is independently deployable and manages its own dependencies.

*   `s1-ingest-nocodb`: Webhook entry point (FastAPI).
*   `s2-download-mp4`: Video downloader worker.
*   `s3-tts-voice`: Text-to-Speech worker.
*   `s4-sadtalker`: Core SadTalker implementation (Legacy structure, uses `requirements.txt`).
*   `s5-broll-selector`: B-roll selection logic.
*   `s6-video-compositor`: FFmpeg-based composition.
*   `s7-storage-uploader`: Cloud storage interface.
*   `s8-nocodb-updater`: NocoDB API client for status updates.

### `packages/`
Shared Python packages for common functionality.

*   `core`: Shared data models, logging configurations, and job contracts.
*   `media`: FFmpeg wrappers and video processing utilities.
*   `nocodb`: Shared NocoDB API client logic.
*   `storage`: Shared storage interface (S3/MinIO compatible).

### `infra/`
Infrastructure configuration.
*   `docker-compose/`: Defines the service stack and secrets mapping.

## Development Conventions

*   **Best Practices First:** Always implement the most robust, idiomatic, and maintainable solution from the start. Avoid "quick fixes" that compromise code integrity or architectural clarity.
*   **Modern Code Style:** The project uses modern Python (3.12+) and library (SQLAlchemy 2.0, Pydantic v2) features. This includes using `list` and `|` for typing instead of `typing.List` and `typing.Optional`.
*   **Package Management:** Modern services (non-s4) use `uv` for fast dependency management and virtual environments. `s4` uses standard `pip` and `requirements.txt`.
*   **Settings:** 
    *   Use `pydantic-settings` via `BaseSettings`.
    *   Prioritize Docker secrets (`/run/secrets`) > Environment variables > Defaults.
    *   Prefix environment variables with the service name (e.g., `S2_RABBITMQ_URL`).
*   **Logging:** Use `loguru` for structured logging.
*   **Testing:** `pytest` is the standard test runner.

## Code Style Preferences
*   **Self-Documenting Code:** Prioritize code that is readable and self-documenting to aid long-term maintenance.
*   **Explanatory Comments:** For complex functions or methods, add comments that explain the logic. A preferred style is a high-level "map" in the docstring that outlines the steps, with "step" markers in the code itself.

## Key Commands

### Local Development (Service Level)
Commands should be run from within the specific service directory (e.g., `services/s1-ingest-nocodb`).

```bash
# Install dependencies
uv sync

# Run the service (FastAPI example)
uv run uvicorn ingest_nocodb.app:app --host 0.0.0.0 --port 7101 --reload

# Run the worker (Dramatiq example)
uv run dramatiq download_mp4.worker

# Run tests
uv run pytest
```

### Infrastructure (Root Level)

```bash
# Start the full stack
docker-compose -f infra/docker-compose/docker-compose.yml up -d

# Build specific services
docker-compose -f infra/docker-compose/docker-compose.yml build s1-ingest-nocodb
```

## Setup Notes
*   **Secrets:** For local development, copy `.env.example` to `.env` in the root or service directory. Docker Compose maps files from `infra/docker-compose/secrets/` to the containers.
*   **Legacy Service:** `s4-sadtalker` requires a specific environment setup (Python 3.10 + PyTorch) often managed via `launcher.py` or its specific Dockerfile.
