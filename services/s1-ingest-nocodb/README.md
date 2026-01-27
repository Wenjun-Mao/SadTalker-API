Ingest NocoDB rows, download source video, remove watermark.

Dependencies: managed with uv via pyproject.toml.

Endpoint
- POST /webhook (module path: ingest_nocodb.app:app)

Inputs:
- NocoDB record
Outputs:
- video_bg_path
