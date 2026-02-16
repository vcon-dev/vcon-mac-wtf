"""Tests for vCon processing pipeline."""


def test_transcribe_vcon(client, sample_vcon):
    resp = client.post("/transcribe", json=sample_vcon)
    assert resp.status_code == 200
    data = resp.json()

    # Should have the original fields
    assert data["vcon"] == "0.0.1"
    assert data["uuid"] == "test-uuid-1234"
    assert len(data["parties"]) == 2
    assert len(data["dialog"]) == 1

    # Should have enriched analysis
    assert len(data["analysis"]) == 1
    analysis = data["analysis"][0]
    assert analysis["type"] == "wtf_transcription"
    assert analysis["vendor"] == "mlx-whisper"
    assert analysis["schema"] == "wtf-1.0"
    assert analysis["encoding"] == "json"
    assert analysis["dialog"] == 0

    # WTF body should have standard fields
    wtf = analysis["body"]
    assert "transcript" in wtf
    assert "segments" in wtf
    assert "metadata" in wtf


def test_transcribe_vcon_no_dialog(client):
    resp = client.post("/transcribe", json={"vcon": "0.0.1", "parties": []})
    assert resp.status_code == 400


def test_transcribe_vcon_no_audio_dialog(client):
    vcon = {
        "vcon": "0.0.1",
        "parties": [],
        "dialog": [{"type": "text", "body": "hello"}],
        "analysis": [],
    }
    resp = client.post("/transcribe", json=vcon)
    assert resp.status_code == 422


def test_transcribe_vcon_stats_headers(client, sample_vcon):
    resp = client.post("/transcribe", json=sample_vcon)
    assert resp.status_code == 200
    assert resp.headers.get("X-Dialogs-Processed") == "1"
    assert resp.headers.get("X-Dialogs-Skipped") == "0"
    assert resp.headers.get("X-Dialogs-Failed") == "0"
    assert resp.headers.get("X-Provider") == "mlx-whisper"
