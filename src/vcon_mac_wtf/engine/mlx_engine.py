"""Core MLX Whisper engine wrapping mlx_whisper.transcribe()."""

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Any

from .model_manager import model_manager

logger = logging.getLogger(__name__)


class MLXWhisperEngine:
    """Wraps mlx_whisper for async-compatible transcription on Apple Silicon."""

    def __init__(self):
        self._loaded_model: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self._loaded_model is not None

    @property
    def loaded_model(self) -> str | None:
        return self._loaded_model

    def load_model(self, model_name: str) -> None:
        """Pre-load a model by running a tiny transcription to warm the cache."""
        import mlx_whisper

        resolved = model_manager.resolve_model_name(model_name)
        logger.info("Loading MLX Whisper model: %s", resolved)
        # Create a tiny silent WAV to trigger model download and load
        _warm_up_model(resolved)
        self._loaded_model = resolved
        logger.info("Model loaded: %s", resolved)

    async def transcribe(
        self,
        audio_path: str,
        model: str | None = None,
        language: str | None = None,
        word_timestamps: bool = True,
    ) -> dict[str, Any]:
        """Transcribe an audio file using MLX Whisper.

        Runs the blocking mlx_whisper.transcribe() in a thread pool to avoid
        blocking the FastAPI event loop.
        """
        resolved_model = model_manager.resolve_model_name(model) if model else self._loaded_model
        if not resolved_model:
            raise RuntimeError("No model loaded. Call load_model() first or pass a model name.")

        result = await asyncio.to_thread(
            _run_transcribe,
            audio_path=audio_path,
            model=resolved_model,
            language=language,
            word_timestamps=word_timestamps,
        )
        return result

    async def transcribe_bytes(
        self,
        audio_bytes: bytes,
        suffix: str = ".wav",
        model: str | None = None,
        language: str | None = None,
        word_timestamps: bool = True,
    ) -> dict[str, Any]:
        """Transcribe audio from bytes by writing to a temp file first."""
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            return await self.transcribe(
                audio_path=tmp.name,
                model=model,
                language=language,
                word_timestamps=word_timestamps,
            )


def _run_transcribe(
    audio_path: str,
    model: str,
    language: str | None,
    word_timestamps: bool,
) -> dict[str, Any]:
    """Synchronous wrapper around mlx_whisper.transcribe() for use in a thread."""
    import mlx_whisper

    kwargs: dict[str, Any] = {
        "path_or_hf_repo": model,
        "word_timestamps": word_timestamps,
    }
    if language:
        kwargs["language"] = language

    result = mlx_whisper.transcribe(audio_path, **kwargs)
    return result


def _warm_up_model(model: str) -> None:
    """Warm up the model by loading it. mlx_whisper downloads on first use."""
    import struct
    import mlx_whisper

    # Create a minimal valid WAV file (44 bytes header + 1600 bytes of silence = 0.1s at 16kHz)
    num_samples = 1600
    sample_rate = 16000
    data_size = num_samples * 2  # 16-bit = 2 bytes per sample
    wav_header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,  # chunk size
        1,  # PCM
        1,  # mono
        sample_rate,
        sample_rate * 2,  # byte rate
        2,  # block align
        16,  # bits per sample
        b"data",
        data_size,
    )
    silence = b"\x00" * data_size

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp.write(wav_header + silence)
        tmp.flush()
        mlx_whisper.transcribe(tmp.name, path_or_hf_repo=model)


mlx_engine = MLXWhisperEngine()
