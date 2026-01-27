Dependency layout

- Each service owns its own dependencies and should remain independently deployable.
- Non-SadTalker services use uv with per-service pyproject.toml and uv.lock.
- SadTalker keeps its original requirements files under services/s3-sadtalker.
- Shared code lives in packages/* and is imported via editable installs or built artifacts in deployment.
- Do not install dependencies locally; only update dependency files.

Notes
- Keep root-level requirements.txt absent to avoid conflating service deps.
