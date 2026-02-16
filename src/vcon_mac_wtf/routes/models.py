"""Model listing endpoint (OpenAI-compatible)."""

from fastapi import APIRouter

from ..engine.model_manager import model_manager
from ..models.openai_compat import ModelListResponse, ModelObject

router = APIRouter(tags=["models"])


@router.get("/v1/models")
async def list_models() -> ModelListResponse:
    raw = model_manager.list_models()
    data = [ModelObject(**m) for m in raw]
    return ModelListResponse(data=data)
