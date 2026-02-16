"""Health check endpoints."""

from fastapi import APIRouter

from ..engine.mlx_engine import mlx_engine
from ..models.responses import HealthResponse, ReadyResponse

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> HealthResponse:
    return HealthResponse()


@router.get("/health/ready")
async def ready() -> ReadyResponse:
    if mlx_engine.is_loaded:
        return ReadyResponse(status="ok", model=mlx_engine.loaded_model)
    return ReadyResponse(status="not_ready", model=None)
