"""OpenAI-compatible request/response models."""

from pydantic import BaseModel


class TranscriptionResponseJson(BaseModel):
    """Minimal JSON response (response_format=json)."""

    text: str


class TranscriptionResponseVerbose(BaseModel):
    """Verbose JSON response matching OpenAI's format (response_format=verbose_json)."""

    task: str = "transcribe"
    language: str
    duration: float
    text: str
    words: list[dict] | None = None
    segments: list[dict] | None = None


class ModelObject(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "mlx-community"
    aliases: list[str] | None = None


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelObject]
