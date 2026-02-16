"""High-level transcription orchestration."""

import logging
from typing import Any

from ..engine.mlx_engine import mlx_engine

logger = logging.getLogger(__name__)


async def transcribe_audio_bytes(
    audio_bytes: bytes,
    suffix: str = ".wav",
    model: str | None = None,
    language: str | None = None,
    word_timestamps: bool = True,
) -> dict[str, Any]:
    """Transcribe audio bytes and return the raw MLX Whisper result dict."""
    logger.info(
        "Transcribing %d bytes (model=%s, language=%s)",
        len(audio_bytes),
        model,
        language,
    )
    result = await mlx_engine.transcribe_bytes(
        audio_bytes=audio_bytes,
        suffix=suffix,
        model=model,
        language=language,
        word_timestamps=word_timestamps,
    )
    logger.info("Transcription complete: %d chars", len(result.get("text", "")))
    return result
