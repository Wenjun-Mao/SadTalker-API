Pipeline overview

1. Ingest NocoDB rows and download background video. (s1-ingest-nocodb)
2. Generate audio from script. (s2-tts-voice)
3. Generate avatar video with SadTalker. (s3-sadtalker)
4. Pick a guide video. (s4-broll-selector)
5. Composite background + avatar, align length, then append guide video. (s5-video-compositor)
6. Upload final video to storage and update NocoDB. (s6-storage-uploader, s7-nocodb-updater)

Queueing
- RabbitMQ broker with Dramatiq workers.
- Single job_id passed through all services.
