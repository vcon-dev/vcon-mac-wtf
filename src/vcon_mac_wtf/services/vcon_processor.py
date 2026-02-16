"""vCon processing: parse, extract audio, transcribe, enrich."""

import base64
import logging
import time
from typing import Any

from ..config import settings
from .transcription import transcribe_audio_bytes
from .wtf_converter import convert_result_to_wtf

logger = logging.getLogger(__name__)

# Map mediatypes to file suffixes
AUDIO_SUFFIXES: dict[str, str] = {
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


async def process_vcon(
    vcon_data: dict[str, Any],
    model: str | None = None,
    language: str | None = None,
    word_timestamps: bool = True,
) -> dict[str, Any]:
    """Process a vCon: find audio dialogs, transcribe, and enrich with WTF analysis.

    Returns the enriched vCon dict with analysis entries appended.
    """
    effective_model = model or settings.mlx_model
    dialogs = vcon_data.get("dialog", [])
    analysis = list(vcon_data.get("analysis", []))

    stats = {"processed": 0, "skipped": 0, "failed": 0, "total_time_ms": 0}

    for i, dialog in enumerate(dialogs):
        # Only process recording dialogs with audio mediatypes
        if dialog.get("type") != "recording":
            stats["skipped"] += 1
            continue

        mediatype = dialog.get("mediatype", "")
        if not mediatype.startswith("audio/"):
            stats["skipped"] += 1
            continue

        body = dialog.get("body")
        if not body:
            stats["skipped"] += 1
            continue

        try:
            # Decode base64url body to bytes
            audio_bytes = _decode_audio_body(body, dialog.get("encoding", "base64url"))
            suffix = AUDIO_SUFFIXES.get(mediatype, ".wav")

            start = time.monotonic()
            result = await transcribe_audio_bytes(
                audio_bytes=audio_bytes,
                suffix=suffix,
                model=effective_model,
                language=language,
                word_timestamps=word_timestamps,
            )
            elapsed = time.monotonic() - start

            # Convert to WTF
            wtf_doc = convert_result_to_wtf(result, effective_model, elapsed)

            # Append analysis entry
            analysis.append(
                {
                    "type": "wtf_transcription",
                    "dialog": i,
                    "mediatype": "application/json",
                    "vendor": "mlx-whisper",
                    "product": effective_model,
                    "schema": "wtf-1.0",
                    "body": wtf_doc,
                    "encoding": "json",
                }
            )

            stats["processed"] += 1
            stats["total_time_ms"] += int(elapsed * 1000)
            logger.info("Dialog %d transcribed (%.1fs)", i, elapsed)

        except Exception:
            stats["failed"] += 1
            logger.exception("Failed to transcribe dialog %d", i)

    enriched = dict(vcon_data)
    enriched["analysis"] = analysis
    return enriched, stats


def _decode_audio_body(body: str, encoding: str) -> bytes:
    """Decode the dialog body to raw audio bytes."""
    if encoding == "base64url":
        # base64url: replace URL-safe chars and add padding
        padded = body + "=" * (-len(body) % 4)
        return base64.urlsafe_b64decode(padded)
    if encoding == "base64":
        return base64.b64decode(body)
    # If no encoding, assume it's already base64
    try:
        return base64.b64decode(body)
    except Exception:
        return base64.urlsafe_b64decode(body + "=" * (-len(body) % 4))
