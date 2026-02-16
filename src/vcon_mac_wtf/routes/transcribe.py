"""vCon-native transcription endpoint: POST /transcribe."""

import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query

from ..services.vcon_processor import process_vcon

logger = logging.getLogger(__name__)

router = APIRouter(tags=["transcription"])


@router.post("/transcribe")
async def transcribe_vcon(
    body: dict[str, Any],
    model: Optional[str] = Query(default=None, description="MLX Whisper model override"),
    language: Optional[str] = Query(default=None, description="Language hint (e.g. en, es)"),
    word_timestamps: bool = Query(default=True, description="Include word-level timestamps"),
):
    """Accept a vCon, transcribe audio dialogs, return enriched vCon with WTF analysis."""
    # Basic validation
    if "dialog" not in body:
        raise HTTPException(status_code=400, detail="Missing 'dialog' field in vCon")

    dialogs = body.get("dialog", [])
    audio_dialogs = [
        d
        for d in dialogs
        if d.get("type") == "recording" and (d.get("mediatype", "")).startswith("audio/")
    ]
    if not audio_dialogs:
        raise HTTPException(status_code=422, detail="No audio recording dialogs found in vCon")

    enriched, stats = await process_vcon(
        vcon_data=body,
        model=model,
        language=language,
        word_timestamps=word_timestamps,
    )

    # Return enriched vCon with stats in headers
    from fastapi.responses import JSONResponse

    headers = {
        "X-Dialogs-Processed": str(stats["processed"]),
        "X-Dialogs-Skipped": str(stats["skipped"]),
        "X-Dialogs-Failed": str(stats["failed"]),
        "X-Processing-Time-Ms": str(stats["total_time_ms"]),
        "X-Provider": "mlx-whisper",
        "X-Model": model or "",
    }
    return JSONResponse(content=enriched, headers=headers)
