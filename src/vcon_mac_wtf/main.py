"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv

from fastapi import FastAPI

load_dotenv()  # Load .env so HF_TOKEN etc. are available to huggingface_hub
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .engine.mlx_engine import mlx_engine
from .routes import health, models, openai_compat, transcribe

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
    logger.info(
        "Starting vcon-mac-wtf server (model=%s, preload=%s)",
        settings.mlx_model,
        settings.preload_model,
    )
    if settings.preload_model:
        logger.info("Preloading MLX Whisper model: %s", settings.mlx_model)
        mlx_engine.load_model(settings.mlx_model)
        logger.info("Model loaded successfully")
    yield
    logger.info("Shutting down vcon-mac-wtf server")


app = FastAPI(
    title="vcon-mac-wtf",
    description="MLX Whisper transcription server for Apple Silicon with WTF/vCon support",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(models.router)
app.include_router(openai_compat.router)
app.include_router(transcribe.router)


def run() -> None:
    """CLI entry point for `vcon-mac-wtf` command."""
    import uvicorn

    uvicorn.run(
        "vcon_mac_wtf.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
    )


if __name__ == "__main__":
    run()
