"""OpenAI-compatible transcription endpoint: POST /v1/audio/transcriptions."""

import logging
import time
from typing import Optional

from fastapi import APIRouter, Form, HTTPException, UploadFile

from ..config import settings
from ..engine.mlx_engine import mlx_engine
from ..services.transcription import transcribe_audio_bytes
from ..services.wtf_converter import convert_result_to_wtf

logger = logging.getLogger(__name__)

router = APIRouter(tags=["transcription"])

# Map media types to file suffixes for temp files
MEDIATYPE_SUFFIXES: dict[str, str] = {
    "audio/wav": ".wav",
    "audio/wave": ".wav",
    "audio/x-wav": ".wav",
    "audio/mp3": ".mp3",
    "audio/mpeg": ".mp3",
    "audio/mp4": ".mp4",
    "audio/x-m4a": ".m4a",
    "audio/m4a": ".m4a",
    "audio/flac": ".flac",
    "audio/ogg": ".ogg",
    "audio/webm": ".webm",
}


@router.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile,
    model: str = Form(default=""),
    response_format: str = Form(default="verbose_json"),
    language: Optional[str] = Form(default=None),
    timestamp_granularities: Optional[list[str]] = Form(default=None),
):
    """OpenAI-compatible audio transcription endpoint."""
    effective_model = model if model else settings.mlx_model

    # Determine word timestamps from granularities
    want_words = True
    if timestamp_granularities:
        want_words = "word" in timestamp_granularities

    # Read audio bytes
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    max_bytes = settings.max_audio_size_mb * 1024 * 1024
    if len(audio_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Audio file too large ({len(audio_bytes)} bytes, max {max_bytes})",
        )

    # Determine file suffix from content type
    content_type = file.content_type or "audio/wav"
    suffix = MEDIATYPE_SUFFIXES.get(content_type, ".wav")
    # Also try suffix from filename
    if file.filename:
        from pathlib import Path

        file_suffix = Path(file.filename).suffix
        if file_suffix:
            suffix = file_suffix

    start = time.monotonic()
    result = await transcribe_audio_bytes(
        audio_bytes=audio_bytes,
        suffix=suffix,
        model=effective_model,
        language=language,
        word_timestamps=want_words,
    )
    processing_time = time.monotonic() - start

    # Format response
    if response_format == "text":
        return result.get("text", "")

    if response_format == "json":
        return {"text": result.get("text", "")}

    if response_format == "wtf":
        wtf_doc = convert_result_to_wtf(result, effective_model, processing_time)
        return wtf_doc

    # Default: verbose_json
    response: dict = {
        "task": "transcribe",
        "language": result.get("language", "en"),
        "duration": result.get("duration", 0.0),
        "text": result.get("text", ""),
    }

    segments = result.get("segments", [])
    if segments:
        response["segments"] = segments

    # Flatten words from segments for top-level words array
    if want_words and segments:
        all_words = []
        for seg in segments:
            for w in seg.get("words", []):
                all_words.append(
                    {
                        "word": w.get("word", ""),
                        "start": w.get("start", 0.0),
                        "end": w.get("end", 0.0),
                    }
                )
        if all_words:
            response["words"] = all_words

    return response
