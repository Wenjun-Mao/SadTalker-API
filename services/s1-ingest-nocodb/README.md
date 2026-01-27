Ingest NocoDB rows, download source video, remove watermark.

Dependencies: managed with uv via pyproject.toml.

Endpoint
- POST /webhook

Inputs:
- NocoDB record
Outputs:
- video_bg_path
