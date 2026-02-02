from __future__ import annotations

from typing import Optional

import dramatiq
from dramatiq.brokers.rabbitmq import RabbitmqBroker

from ingest_nocodb.settings import Settings


_broker: Optional[RabbitmqBroker] = None


def init_broker(settings: Settings) -> RabbitmqBroker:
    global _broker
    if _broker is None:
        _broker = RabbitmqBroker(url=settings.rabbitmq_url)
        dramatiq.set_broker(_broker)
    return _broker


def enqueue_downstream(
    settings: Settings,
    title: str,
    url: str,
    content: str,
    original_text: str,
) -> None:
    broker = init_broker(settings)
    message = dramatiq.Message(
        queue_name=settings.downstream_queue,
        actor_name=settings.downstream_actor,
        args=[title, url, content, original_text],
        kwargs={},
        options={},
    )
    broker.enqueue(message)
