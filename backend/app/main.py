"""
Stroke Detection System — FastAPI Application Factory

Creates and configures the FastAPI application with:
  • CORS middleware
  • Static file serving for result images
  • All API routers
  • Model loading on startup
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.api.routes import upload, diagnosis, health
from app.services.inference import inference_service

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    setup_logging()
    logger.info(f"Starting {settings.app_name} ({settings.app_env})")

    # Ensure upload directory exists
    os.makedirs(settings.upload_dir, exist_ok=True)

    # Model loading is now handled lazily in the inference service
    # to avoid Render startup timeouts or OOM on the 512MB free tier.
    yield  # ← application is running

    logger.info("Shutting down …")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        description=(
            "AI-powered stroke detection system that identifies hemorrhagic "
            "and ischemic strokes from CT brain scans and provides rapid "
            "diagnostic assistance to clinicians."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS ─────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Static files (result images) ─────────
    os.makedirs(settings.upload_dir, exist_ok=True)
    app.mount(
        "/static",
        StaticFiles(directory=settings.upload_dir),
        name="static",
    )

    # ── Routers ──────────────────────────────
    app.include_router(health.router, prefix="/api/v1")
    app.include_router(upload.router, prefix="/api/v1")
    app.include_router(diagnosis.router, prefix="/api/v1")

    return app


app = create_app()
