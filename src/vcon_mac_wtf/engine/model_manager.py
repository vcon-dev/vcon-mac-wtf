"""Model management: listing, short-name aliases, cache info."""

from pathlib import Path

# Short name -> HuggingFace model ID
MODEL_ALIASES: dict[str, str] = {
    "tiny": "mlx-community/whisper-tiny",
    "base": "mlx-community/whisper-base",
    "small": "mlx-community/whisper-small",
    "medium": "mlx-community/whisper-medium",
    "large-v3": "mlx-community/whisper-large-v3",
    "turbo": "mlx-community/whisper-turbo",
}

# All known models (alias values + some additional)
ALL_MODELS: list[str] = list(MODEL_ALIASES.values())


class ModelManager:
    """Manages MLX Whisper model resolution and listing."""

    def resolve_model_name(self, name: str) -> str:
        """Resolve a short name or pass through a full model ID."""
        return MODEL_ALIASES.get(name, name)

    def list_models(self) -> list[dict]:
        """List available models in OpenAI-compatible format."""
        models = []
        for alias, model_id in MODEL_ALIASES.items():
            models.append(
                {
                    "id": model_id,
                    "object": "model",
                    "owned_by": "mlx-community",
                    "aliases": [alias],
                }
            )
        return models

    def is_cached(self, model_name: str) -> bool:
        """Check if a model is locally cached (best-effort heuristic)."""
        resolved = self.resolve_model_name(model_name)
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        # HuggingFace caches as models--org--name
        slug = resolved.replace("/", "--")
        return (cache_dir / f"models--{slug}").exists()


model_manager = ModelManager()
