from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import dramatiq
import httpx
from dramatiq.brokers.rabbitmq import RabbitmqBroker
from loguru import logger

from tts_voice.settings import get_settings

# Initialize settings and broker
settings = get_settings()

_url_parts = settings.rabbitmq_url.split("@")
_masked_url = _url_parts[-1] if len(_url_parts) > 1 else settings.rabbitmq_url

logger.info(f"Initializing s3-tts-voice worker (broker={_masked_url}, queue={settings.downstream_queue})")

broker = RabbitmqBroker(url=settings.rabbitmq_url)
dramatiq.set_broker(broker)


def _request_tts_url(settings: Any, text: str) -> str:
    """
    Call the external TTS API to generate a voice URL.
    Request example: https://ou-han.cn/DouBao/textToVoice?text=...&sex=2&token=...
    """
    params = {
        "text": text,
        "sex": "2",  # Defaulting to 2 as per user example
        "token": settings.api_token,
    }
    timeout = httpx.Timeout(settings.request_timeout_s)
    
    with httpx.Client(timeout=timeout) as client:
        response = client.get(settings.api_url, params=params)
        response.raise_for_status()
        payload = response.json()

    if settings.debug_log_payload:
        logger.info(
            "TTS API response:{}",
            json.dumps(payload, ensure_ascii=False, indent=2),
        )

    # Expected: {"status":1001,"info":"success！","voiceUrl":"..."}
    if payload.get("status") != 1001:
        raise RuntimeError(
            f"TTS API returned error status: {payload.get('status')}. "
            f"Info: {payload.get('info')}"
        )
    
    voice_url = payload.get("voiceUrl")
    if not voice_url:
        raise RuntimeError(
            f"TTS API response missing voiceUrl: {json.dumps(payload, ensure_ascii=False)[:1000]}"
        )
    
    return voice_url


def _download_audio(audio_url: str, output_dir: str, record_id: int) -> str:
    """
    Download the generated audio file.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filename = f"record_{record_id}_{uuid4().hex}.mp3"
    target_path = Path(output_dir) / filename
    
    timeout = httpx.Timeout(120.0)
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        with client.stream("GET", audio_url) as response:
            response.raise_for_status()
            with target_path.open("wb") as file_handle:
                for chunk in response.iter_bytes():
                    if chunk:
                        file_handle.write(chunk)

    return str(target_path)


def _enqueue_downstream(
    settings: Any,
    record_id: int,
    title: str,
    url: str,
    content: str,
    original_text: str,
    douyin_download_url: str,
    douyin_video_path: str,
    tts_audio_path: str,
) -> None:
    """
    Pass the job along with the newly generated TTS audio path to the next service.
    """
    broker = dramatiq.get_broker()
    message = dramatiq.Message(
        queue_name=settings.downstream_queue,
        actor_name=settings.downstream_actor,
        args=[
            record_id,
            title,
            url,
            content,
            original_text,
            douyin_download_url,
            douyin_video_path,
            tts_audio_path,
        ],
        kwargs={},
        options={},
    )
    broker.enqueue(message)


@dramatiq.actor(actor_name="s3_tts_voice.ping", queue_name="s3-tts-voice")
def ping() -> None:
    logger.info("PONG! s3-tts-voice worker is alive and reachable.")


@dramatiq.actor(
    actor_name="s3_tts_voice.process",
    queue_name="s3-tts-voice",
)
def process(
    record_id: int,
    title: str,
    url: str,
    content: str,
    original_text: str,
    douyin_download_url: str,
    douyin_video_path: str,
) -> None:
    """
    Main worker process for generating TTS audio from content.
    """
    logger.info(f"Received TTS job for record_id={record_id}")
    settings = get_settings()
    
    if settings.debug_log_payload:
        args = {
            "record_id": record_id,
            "title": title,
            "content": content,
            "douyin_video_path": douyin_video_path,
        }
        logger.info("Job details:{}", json.dumps(args, indent=2, default=str))

    try:
        # Step 1: Request TTS Voice URL
        voice_url = _request_tts_url(settings, content)
        
        # Step 2: Download the audio file
        tts_audio_path = _download_audio(
            voice_url,
            settings.output_dir,
            record_id,
        )

        # Step 3: Enqueue for downstream (s4-sadtalker)
        _enqueue_downstream(
            settings,
            record_id=record_id,
            title=title,
            url=url,
            content=content,
            original_text=original_text,
            douyin_download_url=douyin_download_url,
            douyin_video_path=douyin_video_path,
            tts_audio_path=tts_audio_path,
        )

        if settings.debug_log_payload:
            logger.info(f"Completed TTS job for record_id={record_id}. Path: {tts_audio_path}")

    except Exception:
        logger.bind(
            record_id=record_id,
            content=content[:100],
        ).exception("Failed TTS job")
        raise
    finally:
        os.sync() if hasattr(os, "sync") else None


# Log registered actors for verification
logger.info(f"Worker started with actors: {[a.actor_name for a in broker.actors.values()]}")
