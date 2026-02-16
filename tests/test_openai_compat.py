"""Tests for the OpenAI-compatible endpoint."""

import io


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_health_ready(client):
    resp = client.get("/health/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model"] == "mlx-community/whisper-turbo"


def test_list_models(client):
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) >= 6


def test_transcribe_verbose_json(client, sample_wav_bytes):
    resp = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", io.BytesIO(sample_wav_bytes), "audio/wav")},
        data={"model": "turbo", "response_format": "verbose_json"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "text" in data
    assert "segments" in data
    assert data["task"] == "transcribe"
    assert data["language"] == "en"


def test_transcribe_json(client, sample_wav_bytes):
    resp = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", io.BytesIO(sample_wav_bytes), "audio/wav")},
        data={"model": "turbo", "response_format": "json"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "text" in data
    assert "segments" not in data


def test_transcribe_text(client, sample_wav_bytes):
    resp = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", io.BytesIO(sample_wav_bytes), "audio/wav")},
        data={"model": "turbo", "response_format": "text"},
    )
    assert resp.status_code == 200
    # Text format returns a plain string (JSON-encoded)
    assert "Hello" in resp.text or "test" in resp.text


def test_transcribe_empty_file(client):
    resp = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", io.BytesIO(b""), "audio/wav")},
        data={"model": "turbo"},
    )
    assert resp.status_code == 400
