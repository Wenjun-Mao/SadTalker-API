from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="S2_",
        env_file=".env",
        env_file_encoding="utf-8",
        secrets_dir="/run/secrets",
        extra="ignore",
    )

    rabbitmq_url: str = Field(
        ...,
        description="RabbitMQ connection URL",
        validation_alias=AliasChoices("S2_RABBITMQ_URL", "rabbitmq_url"),
    )
    api_token: str = Field(
        ...,
        description="xiazaitool parse-video API token",
        validation_alias=AliasChoices(
            "S2_VIDEO_PARSE_API_TOKEN",
            "video_parse_api_token",
            "api_token",
        ),
    )
    api_url: str = Field(
        "https://api.xiazaitool.com/api/parseVideoUrl",
        description="xiazaitool parse URL endpoint",
        validation_alias=AliasChoices("S2_API_URL", "api_url"),
    )
    output_dir: str = Field(
        "/data/s2",
        description="Directory to store downloaded MP4 files",
        validation_alias=AliasChoices("S2_OUTPUT_DIR", "output_dir"),
    )
    request_timeout_s: float = Field(
        60.0,
        description="HTTP request timeout in seconds",
        validation_alias=AliasChoices("S2_REQUEST_TIMEOUT_S", "request_timeout_s"),
    )
    downstream_queue: str = Field(
        "s3-tts-voice",
        description="Dramatiq queue for downstream service",
        validation_alias=AliasChoices("S2_DOWNSTREAM_QUEUE", "downstream_queue"),
    )
    downstream_actor: str = Field(
        "s3_tts_voice.process",
        description="Dramatiq actor name for downstream service",
        validation_alias=AliasChoices("S2_DOWNSTREAM_ACTOR", "downstream_actor"),
    )


def get_settings() -> Settings:
    return Settings()