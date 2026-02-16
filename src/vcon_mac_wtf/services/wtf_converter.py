"""Bridge to the wtf-transcript-converter library."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def convert_result_to_wtf(
    whisper_result: dict[str, Any],
    model_name: str,
    processing_time_seconds: float,
) -> dict[str, Any]:
    """Convert an MLX Whisper result dict to WTF format using WhisperConverter.

    Returns the WTF document as a JSON-serializable dict.
    """
    from wtf_transcript_converter.providers.whisper import WhisperConverter

    # Augment the result with metadata the converter expects
    augmented = dict(whisper_result)
    augmented["model"] = model_name
    augmented["processing_time"] = processing_time_seconds

    converter = WhisperConverter()
    wtf_doc = converter.convert_to_wtf(augmented)
    return wtf_doc.model_dump(exclude_none=True)
