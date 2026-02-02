Local development compose files live here.

Current services
- RabbitMQ (broker)
- s1-ingest-nocodb

Quick start
1) Create secret files under infra/docker-compose/secrets:
	- nocodb_api_key
	- nocodb_base_url
2) Run from this folder:
	docker compose up --build

Optional services (profiles)
- s2-download-mp4: docker compose --profile s2 up --build
- s3-tts-voice: docker compose --profile s3 up --build
- s5-broll-selector: docker compose --profile s5 up --build
- s6-video-compositor: docker compose --profile s6 up --build
- s7-storage-uploader: docker compose --profile s7 up --build
- s8-nocodb-updater: docker compose --profile s8 up --build

How Docker secrets work (Compose)
- Secrets are files mounted at /run/secrets/<name> inside the container.
- This repo reads secrets via pydantic-settings with secrets_dir=/run/secrets.
- Each secret file should contain only the value (no quotes, no key names).
- Secret files are unprefixed (e.g., nocodb_api_key), while env vars use the S1_ prefix.

Example
- /run/secrets/nocodb_api_key contains: abcd1234
- /run/secrets/nocodb_base_url contains: https://nocodb.example.com

Notes
- Environment variables still work and can override secrets if set.
- For RabbitMQ URL, we use env var by default.
