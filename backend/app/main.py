"""
Stroke Detection System — FastAPI Application Factory

Creates and configures the FastAPI application with:
  • CORS middleware (supports Vercel previews)
  • Static file serving for result images
  • All API routers
  • Model loading on startup
"""

import os
import traceback
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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

    # Models are now strictly lazy-loaded on demand to stay within Render's 512MB RAM tier.
    yield  # ← application is running

    logger.info("Shutting down …")


def _get_cors_origin(request: Request):
    """Return the request origin if it's in the allowed list or is a Vercel app."""
    origin = request.headers.get("origin")
    if not origin:
        return None
    
    # Check against allowed list
    if origin in settings.allowed_origins_list:
        return origin
    
    # Allow all vercel apps
    if origin.endswith(".vercel.app"):
        return origin
        
    return None


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
    # Using allow_origin_regex for Vercel subdomains
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins_list,
        allow_origin_regex="https://.*\\.vercel\\.app",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Root Health Check (Required for some health check monitors) ──────────
    @app.get("/", tags=["Health"])
    async def root():
        return {"status": "online", "app": settings.app_name}

    # ── Exception handler for proper CORS ──────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        # Determine status code and detail
        if isinstance(exc, HTTPException):
            status_code = exc.status_code
            detail = exc.detail
        else:
            logger.error(f"Unhandled exception: {exc}\n{traceback.format_exc()}")
            status_code = 500
            detail = f"Internal server error: {str(exc)}"
            
        origin = _get_cors_origin(request)
        headers = {}
        if origin:
            headers["Access-Control-Allow-Origin"] = origin
            headers["Access-Control-Allow-Credentials"] = "true"
            
        return JSONResponse(
            status_code=status_code,
            content={"detail": detail},
            headers=headers,
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
