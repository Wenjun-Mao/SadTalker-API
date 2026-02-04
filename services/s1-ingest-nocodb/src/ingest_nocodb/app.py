from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from ingest_nocodb.messaging import enqueue_downstream, init_broker
from ingest_nocodb.settings import get_settings

app = FastAPI(title="s1-ingest-nocodb")


# Sanity check: ensure settings (from env vars and secrets) and broker are initialized at startup.
# This will fail fast if config or RabbitMQ connection is invalid,
# so deployment errors are caught early (not on first request).
@app.on_event("startup")
async def startup_check() -> None:
    settings = get_settings()
    init_broker(settings)


class NocoDbRow(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    record_id: int = Field(alias="Id")
    created_at: datetime = Field(alias="CreatedAt")
    updated_at: datetime = Field(alias="UpdatedAt")
    title: Optional[str] = Field(default=None, alias="Title")
    url: Optional[str] = Field(default=None, alias="url")
    content: Optional[str] = Field(default=None, alias="content")
    original_text: Optional[str] = Field(default=None, alias="originaltext")
    image1: Optional[str] = Field(default=None, alias="image1")
    image2: Optional[str] = Field(default=None, alias="image2")
    image3: Optional[str] = Field(default=None, alias="image3")
    image4: Optional[str] = Field(default=None, alias="image4")
    image5: Optional[str] = Field(default=None, alias="image5")
    cover_image: Optional[str] = Field(default=None, alias="fengmianimage")
    final_url: Optional[str] = Field(default=None, alias="chengpinurl")


class NocoDbWebhookData(BaseModel):
    model_config = ConfigDict(extra="allow")

    table_id: str
    table_name: str
    rows: List[NocoDbRow]


class NocoDbWebhookPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    id: str
    version: str
    data: NocoDbWebhookData


class WebhookAck(BaseModel):
    ok: bool
    received_rows: int


@app.post("/webhook", response_model=WebhookAck)
async def webhook(payload: NocoDbWebhookPayload) -> WebhookAck:
    if not payload.data.rows:
        raise HTTPException(status_code=400, detail="No rows provided in payload.")
    settings = get_settings()

    for row in payload.data.rows:
        missing_fields = [
            name
            for name, value in {
                "Title": row.title,
                "url": row.url,
                "content": row.content,
                "originaltext": row.original_text,
            }.items()
            if not value
        ]
        if missing_fields:
            raise HTTPException(
                status_code=400,
                detail=f"Row {row.record_id} missing fields: {', '.join(missing_fields)}",
            )

        enqueue_downstream(
            settings,
            record_id=row.record_id,
            title=row.title,
            url=row.url,
            content=row.content,
            original_text=row.original_text,
        )

    return WebhookAck(ok=True, received_rows=len(payload.data.rows))
