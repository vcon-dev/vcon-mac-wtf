"""Integration tests requiring Apple Silicon and MLX.

Run with: pytest tests/test_integration.py -v
These are excluded by default: pytest tests/ -m "not integration"
"""

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def real_client():
    """TestClient that uses the real MLX engine (no mocking)."""
    from fastapi.testclient import TestClient

    from vcon_mac_wtf.main import app

    with TestClient(app) as c:
        yield c


def test_real_transcription(real_client, sample_wav_bytes):
    """Test real MLX Whisper inference on a tiny audio file."""
    import io

    resp = real_client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", io.BytesIO(sample_wav_bytes), "audio/wav")},
        data={"model": "mlx-community/whisper-tiny", "response_format": "verbose_json"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "text" in data
    assert "segments" in data
