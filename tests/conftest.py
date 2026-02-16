"""Shared test fixtures."""

import struct
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def sample_whisper_result():
    """A realistic mlx_whisper.transcribe() return value."""
    return {
        "text": " Hello, this is a test transcription.",
        "language": "en",
        "segments": [
            {
                "id": 0,
                "seek": 0,
                "start": 0.0,
                "end": 2.5,
                "text": " Hello, this is a test",
                "tokens": [50364, 2425, 11, 341, 307, 257, 1500],
                "temperature": 0.0,
                "avg_logprob": -0.234,
                "compression_ratio": 1.56,
                "no_speech_prob": 0.012,
                "words": [
                    {"word": " Hello,", "start": 0.0, "end": 0.4, "probability": 0.98},
                    {"word": " this", "start": 0.5, "end": 0.7, "probability": 0.95},
                    {"word": " is", "start": 0.7, "end": 0.9, "probability": 0.97},
                    {"word": " a", "start": 0.9, "end": 1.0, "probability": 0.99},
                    {"word": " test", "start": 1.0, "end": 1.4, "probability": 0.96},
                ],
            },
            {
                "id": 1,
                "seek": 250,
                "start": 2.5,
                "end": 4.0,
                "text": " transcription.",
                "tokens": [50364, 1159, 13],
                "temperature": 0.0,
                "avg_logprob": -0.189,
                "compression_ratio": 1.23,
                "no_speech_prob": 0.008,
                "words": [
                    {
                        "word": " transcription.",
                        "start": 2.5,
                        "end": 3.8,
                        "probability": 0.94,
                    },
                ],
            },
        ],
    }


@pytest.fixture
def mock_mlx_engine(sample_whisper_result):
    """Patch the mlx_engine singleton so no real MLX inference runs."""
    with patch("vcon_mac_wtf.engine.mlx_engine.mlx_engine") as mock_eng:
        mock_eng.is_loaded = True
        mock_eng.loaded_model = "mlx-community/whisper-turbo"

        async def fake_transcribe_bytes(**kwargs):
            return sample_whisper_result

        async def fake_transcribe(**kwargs):
            return sample_whisper_result

        mock_eng.transcribe_bytes = MagicMock(side_effect=fake_transcribe_bytes)
        mock_eng.transcribe = MagicMock(side_effect=fake_transcribe)
        yield mock_eng


@pytest.fixture
def client(mock_mlx_engine):
    """FastAPI TestClient with mocked MLX engine."""
    # Patch the engine at the module level before importing the app
    with patch("vcon_mac_wtf.main.mlx_engine", mock_mlx_engine), \
         patch("vcon_mac_wtf.routes.health.mlx_engine", mock_mlx_engine), \
         patch("vcon_mac_wtf.services.transcription.mlx_engine", mock_mlx_engine):
        from vcon_mac_wtf.main import app
        with TestClient(app) as c:
            yield c


@pytest.fixture
def sample_wav_bytes():
    """Minimal valid WAV file (0.1s of silence at 16kHz, mono, 16-bit)."""
    num_samples = 1600
    sample_rate = 16000
    data_size = num_samples * 2
    wav_header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM
        1,  # mono
        sample_rate,
        sample_rate * 2,
        2,
        16,
        b"data",
        data_size,
    )
    return wav_header + b"\x00" * data_size


@pytest.fixture
def sample_vcon(sample_wav_bytes):
    """A minimal vCon with one recording dialog containing inline base64 audio."""
    import base64

    audio_b64 = base64.urlsafe_b64encode(sample_wav_bytes).decode("ascii")
    return {
        "vcon": "0.0.1",
        "uuid": "test-uuid-1234",
        "parties": [{"name": "Alice"}, {"name": "Bob"}],
        "dialog": [
            {
                "type": "recording",
                "start": "2024-01-15T10:00:00Z",
                "parties": [0, 1],
                "mediatype": "audio/wav",
                "body": audio_b64,
                "encoding": "base64url",
            }
        ],
        "analysis": [],
    }
