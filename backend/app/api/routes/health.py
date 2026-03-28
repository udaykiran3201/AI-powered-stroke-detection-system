"""
Stroke Detection System — Health-check Route
"""

from fastapi import APIRouter
from app.core.config import get_settings
from app.schemas.schemas import HealthResponse
from app.services.inference import inference_service

router = APIRouter(tags=["Health"])
settings = get_settings()

APP_VERSION = "1.0.0"


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health_check():
    return HealthResponse(
        status="healthy",
        version=APP_VERSION,
        models_loaded=inference_service.models_loaded,
        environment=settings.app_env,
    )
