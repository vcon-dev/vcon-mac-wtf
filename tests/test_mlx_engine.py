"""Tests for MLX engine (with mocked mlx_whisper)."""

from unittest.mock import patch

from vcon_mac_wtf.engine.model_manager import model_manager


def test_model_alias_resolution():
    assert model_manager.resolve_model_name("turbo") == "mlx-community/whisper-turbo"
    assert model_manager.resolve_model_name("tiny") == "mlx-community/whisper-tiny"
    # Full name passes through
    assert (
        model_manager.resolve_model_name("mlx-community/whisper-large-v3")
        == "mlx-community/whisper-large-v3"
    )
    # Unknown name passes through
    assert model_manager.resolve_model_name("custom/model") == "custom/model"


def test_list_models():
    models = model_manager.list_models()
    assert len(models) >= 6
    ids = [m["id"] for m in models]
    assert "mlx-community/whisper-turbo" in ids
    assert "mlx-community/whisper-tiny" in ids
    for m in models:
        assert "id" in m
        assert "object" in m
        assert m["object"] == "model"
