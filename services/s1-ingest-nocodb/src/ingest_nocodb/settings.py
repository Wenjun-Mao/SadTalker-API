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


def get_settings() -> Settings:
    return Settings()
