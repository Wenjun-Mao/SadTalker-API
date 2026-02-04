Download MP4 for the pipeline.

Dependencies: managed with uv via pyproject.toml.

Worker
- dramatiq worker module: download_mp4.worker
- actor name: s2_download_mp4.process

Required settings (env or Docker secrets)
- S2_RABBITMQ_URL
- S2_VIDEO_PARSE_API_TOKEN

Optional settings
- S2_API_URL (default https://api.xiazaitool.com/api/parseVideoUrl)
- S2_OUTPUT_DIR (default /data/s2)
- S2_REQUEST_TIMEOUT_S (default 60)
- S2_QUEUE (default s2-download-mp4)
- S2_DOWNSTREAM_QUEUE (default s3-tts-voice)
- S2_DOWNSTREAM_ACTOR (default s3_tts_voice.process)

Message in
- record_id, title, source_url, content, original_text

Message out
- record_id, title, source_url, content, original_text, download_url, video_path
