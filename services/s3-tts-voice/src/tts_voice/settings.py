from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="S3_",
        env_file=".env",
        env_file_encoding="utf-8",
        secrets_dir="/run/secrets",
        extra="ignore",
    )

    rabbitmq_url: str = Field(
        ...,
        description="RabbitMQ connection URL",
        validation_alias=AliasChoices("S3_RABBITMQ_URL", "rabbitmq_url"),
    )
    api_token: str = Field(
        ...,
        description="TTS API token",
        validation_alias=AliasChoices(
            "S3_TTS_API_TOKEN",
            "tts_api_token",
            "api_token",
        ),
    )
    api_url: str = Field(
        "https://ou-han.cn/DouBao/textToVoice",
        description="TTS API endpoint",
        validation_alias=AliasChoices("S3_API_URL", "api_url"),
    )
    output_dir: str = Field(
        "/data/s3",
        description="Directory to store generated audio files",
        validation_alias=AliasChoices("S3_OUTPUT_DIR", "output_dir"),
    )
    request_timeout_s: float = Field(
        60.0,
        description="HTTP request timeout in seconds",
        validation_alias=AliasChoices("S3_REQUEST_TIMEOUT_S", "request_timeout_s"),
    )
    downstream_queue: str = Field(
        "s4-sadtalker",
        description="Dramatiq queue for downstream service",
        validation_alias=AliasChoices("S3_DOWNSTREAM_QUEUE", "downstream_queue"),
    )
    downstream_actor: str = Field(
        "s4_sadtalker.process",
        description="Dramatiq actor name for downstream service",
        validation_alias=AliasChoices("S3_DOWNSTREAM_ACTOR", "downstream_actor"),
    )
    debug_log_payload: bool = Field(
        False,
        description="Enable verbose logging of incoming args and outputs",
        validation_alias=AliasChoices(
            "DEBUG_LOG_PAYLOAD",
            "debug_log_payload",
            "S3_DEBUG_LOG_PAYLOAD",
        ),
    )


def get_settings() -> Settings:
    return Settings()
