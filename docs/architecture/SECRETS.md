Secrets strategy (Docker + local dev)

Recommendation
- Use Docker secrets in production.
- Use a root .env for local development only.
- Keep real secrets out of git; commit only .env.example.

How it works with pydantic-settings
- Pydantic-settings reads, in order:
  1) environment variables (S1_*)
  2) .env file (local dev)
  3) Docker secrets mounted as files in /run/secrets
- With secrets_dir="/run/secrets", it loads files named after the field name.
  For s1-ingest-nocodb, expected files:
  - /run/secrets/rabbitmq_url
  - /run/secrets/nocodb_api_key
  - /run/secrets/nocodb_base_url

Local dev example
1) Copy .env.example to .env at repo root or service directory.
2) Fill values for:
   - S1_RABBITMQ_URL
   - S1_NOCODB_API_KEY
   - S1_NOCODB_BASE_URL
3) Run: uv run uvicorn ingest_nocodb.app:app --host 0.0.0.0 --port 8111

Docker Compose example

services:
  s1-ingest-nocodb:
    build: ./services/s1-ingest-nocodb
    ports:
      - "8111:8111"
    environment:
      - S1_RABBITMQ_URL=amqp://guest:guest@rabbitmq:5672/
    secrets:
      - nocodb_api_key
      - nocodb_base_url
    command: uvicorn ingest_nocodb.app:app --host 0.0.0.0 --port 8111

secrets:
  nocodb_api_key:
    file: ./secrets/nocodb_api_key
  nocodb_base_url:
    file: ./secrets/nocodb_base_url

Notes
- For RabbitMQ URL, env var is fine if it is not sensitive; otherwise put it in a secret file named rabbitmq_url.
- In Kubernetes, map secrets to /run/secrets or use env vars; pydantic-settings supports both.
