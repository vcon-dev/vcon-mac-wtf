"""Configuration management using Pydantic Settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    # MLX Whisper
    mlx_model: str = "mlx-community/whisper-turbo"
    preload_model: bool = True

    # Limits
    max_audio_size_mb: int = 100


settings = Settings()
