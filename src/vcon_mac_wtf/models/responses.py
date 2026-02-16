"""Health and error response models."""

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    version: str = "0.1.0"


class ReadyResponse(BaseModel):
    status: str
    model: str | None = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
