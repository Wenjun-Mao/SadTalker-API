from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="S1_",
        env_file=".env",
        env_file_encoding="utf-8",
        secrets_dir="/run/secrets",
        extra="ignore",
    )

    rabbitmq_url: str = Field(..., description="RabbitMQ connection URL")
    nocodb_api_key: str = Field(..., description="NocoDB API key")
    nocodb_base_url: str = Field(..., description="NocoDB base URL, e.g. https://nocodb.example.com")


def get_settings() -> Settings:
    return Settings()
