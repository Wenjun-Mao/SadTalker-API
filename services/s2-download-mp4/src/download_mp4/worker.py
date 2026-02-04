from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import dramatiq
import httpx
from dramatiq.brokers.rabbitmq import RabbitmqBroker
from loguru import logger

from download_mp4.settings import Settings, get_settings


_broker: Optional[RabbitmqBroker] = None


def _startup() -> None:
    try:
        settings = get_settings()
        init_broker(settings)
        if settings.debug_log_payload:
            # Mask password in URL for safe logging
            url_parts = settings.rabbitmq_url.split("@")
            masked_url = url_parts[-1] if len(url_parts) > 1 else settings.rabbitmq_url
            logger.bind(
                queue=os.getenv("S2_QUEUE", "s2-download-mp4"),
                broker_host=masked_url,
            ).info("Worker initialized and connected to broker")
    except Exception:
        logger.exception("Worker initialization failed")
        raise


def init_broker(settings: Settings) -> RabbitmqBroker:
    global _broker
    if _broker is None:
        _broker = RabbitmqBroker(url=settings.rabbitmq_url)
        dramatiq.set_broker(_broker)
    return _broker


def _extract_douyin_download_url(payload: dict[str, Any]) -> Optional[str]:
    if isinstance(payload.get("data"), dict):
        data = payload["data"]
        for key in ("url", "download_url", "video_url", "videoUrl"):
            value = data.get(key)
            if isinstance(value, str) and value:
                return value
    for key in ("url", "download_url", "video_url", "videoUrl"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _request_douyin_download_url(settings: Settings, source_url: str) -> str:
    headers = {"Content-Type": "application/json"}
    body = {"url": source_url, "token": settings.api_token}
    timeout = httpx.Timeout(settings.request_timeout_s)
    with httpx.Client(timeout=timeout) as client:
        response = client.post(settings.api_url, headers=headers, json=body)
        response.raise_for_status()
        payload = response.json()

    if settings.debug_log_payload:
        logger.info(
            "External API response:\n{}",
            json.dumps(payload, ensure_ascii=False, indent=2),
        )

    douyin_download_url = _extract_douyin_download_url(payload)
    if not douyin_download_url:
        raise RuntimeError(
            "parseVideoUrl response missing download URL: "
            f"{json.dumps(payload, ensure_ascii=False)[:1000]}"
        )
    return douyin_download_url


def _download_mp4(douyin_download_url: str, output_dir: str, record_id: int) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filename = f"record_{record_id}_{uuid4().hex}.mp4"
    target_path = Path(output_dir) / filename
    timeout = httpx.Timeout(120.0)
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        with client.stream("GET", douyin_download_url) as response:
            response.raise_for_status()
            with target_path.open("wb") as file_handle:
                for chunk in response.iter_bytes():
                    if chunk:
                        file_handle.write(chunk)

    return str(target_path)


def _enqueue_downstream(
    settings: Settings,
    record_id: int,
    title: str,
    url: str,
    content: str,
    original_text: str,
    douyin_download_url: str,
    douyin_video_path: str,
) -> None:
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
        ],
        kwargs={},
        options={},
    )
    broker.enqueue(message)


@dramatiq.actor(actor_name="s2_download_mp4.process")
def process(
    record_id: int,
    title: str,
    url: str,
    content: str,
    original_text: str,
) -> None:
    settings = get_settings()
    if settings.debug_log_payload:
        args = {
            "record_id": record_id,
            "title": title,
            "url": url,
            "content": content,
            "original_text": original_text,
        }
        logger.info("Received job:\n{}", json.dumps(args, indent=2, default=str))

    try:
        douyin_download_url = _request_douyin_download_url(settings, url)
        douyin_video_path = _download_mp4(
            douyin_download_url,
            settings.output_dir,
            record_id,
        )

        _enqueue_downstream(
            settings,
            record_id=record_id,
            title=title,
            url=url,
            content=content,
            original_text=original_text,
            douyin_download_url=douyin_download_url,
            douyin_video_path=douyin_video_path,
        )

        if settings.debug_log_payload:
            result_args = {
                "record_id": record_id,
                "title": title,
                "url": url,
                "content": content,
                "original_text": original_text,
                "douyin_download_url": douyin_download_url,
                "douyin_video_path": douyin_video_path,
                "downstream_queue": settings.downstream_queue,
                "downstream_actor": settings.downstream_actor,
            }
            logger.info("Completed job:\n{}", json.dumps(result_args, indent=2, default=str))
    except Exception:
        logger.bind(
            record_id=record_id,
            title=title,
            url=url,
        ).exception("Failed job")
        raise
    finally:
        # Ensure all filesystem buffers are flushed to disk (Linux/Unix only; no-op on Windows)
        os.sync() if hasattr(os, "sync") else None


_startup()