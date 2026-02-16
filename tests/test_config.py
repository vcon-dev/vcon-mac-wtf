"""Tests for configuration."""

from vcon_mac_wtf.config import Settings


def test_default_settings():
    s = Settings()
    assert s.host == "0.0.0.0"
    assert s.port == 8000
    assert s.mlx_model == "mlx-community/whisper-turbo"
    assert s.preload_model is True
    assert s.max_audio_size_mb == 100


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("PORT", "9000")
    monkeypatch.setenv("MLX_MODEL", "mlx-community/whisper-small")
    monkeypatch.setenv("PRELOAD_MODEL", "false")
    s = Settings()
    assert s.port == 9000
    assert s.mlx_model == "mlx-community/whisper-small"
    assert s.preload_model is False
