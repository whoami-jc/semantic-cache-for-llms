from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: SecretStr = SecretStr("")
    openai_model: str = Field(default="gpt-4.1-mini-2025-04-14", min_length=1)
    openai_timeout_seconds: float = Field(default=60, gt=0)
    openai_max_output_tokens: int = Field(default=1024, ge=16)
