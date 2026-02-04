from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="S1_",
        env_file=".env",
        env_file_encoding="utf-8",
        secrets_dir="/run/secrets",
        extra="ignore",
    )

    rabbitmq_url: str = Field(
        ...,
        description="RabbitMQ connection URL",
        validation_alias=AliasChoices("S1_RABBITMQ_URL", "rabbitmq_url"),
    )
    nocodb_api_key: str = Field(
        ...,
        description="NocoDB API key",
        validation_alias=AliasChoices("S1_NOCODB_API_KEY", "nocodb_api_key"),
    )
    nocodb_base_url: str = Field(
        ...,
        description="NocoDB base URL, e.g. https://nocodb.example.com",
        validation_alias=AliasChoices("S1_NOCODB_BASE_URL", "nocodb_base_url"),
    )
    downstream_queue: str = Field(
        "s2-download-mp4",
        description="Dramatiq queue for downstream service",
        validation_alias=AliasChoices("S1_DOWNSTREAM_QUEUE", "downstream_queue"),
    )
    downstream_actor: str = Field(
        "s2_download_mp4.process",
        description="Dramatiq actor name for downstream service",
        validation_alias=AliasChoices("S1_DOWNSTREAM_ACTOR", "downstream_actor"),
    )
    debug_log_payload: bool = Field(
        False,
        description="Enable verbose logging of incoming rows and enqueued messages",
        validation_alias=AliasChoices(
            "DEBUG_LOG_PAYLOAD",
            "debug_log_payload",
            "S1_DEBUG_LOG_PAYLOAD",
        ),
    )


def get_settings() -> Settings:
    return Settings()
