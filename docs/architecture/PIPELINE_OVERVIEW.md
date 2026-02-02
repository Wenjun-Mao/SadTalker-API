Pipeline overview

1. Ingest NocoDB rows and download background video. (s1-ingest-nocodb)
2. Download source MP4. (s2-download-mp4)
3. Generate audio from script (TTS). (s3-tts-voice)
4. Generate avatar video with SadTalker. (s4-sadtalker)
5. Pick a guide video. (s5-broll-selector)
6. Composite background + avatar, align length, then append guide video. (s6-video-compositor)
7. Upload final video to storage and update NocoDB. (s7-storage-uploader, s8-nocodb-updater)

Queueing
- RabbitMQ broker with Dramatiq workers.
- Single job_id passed through all services.
