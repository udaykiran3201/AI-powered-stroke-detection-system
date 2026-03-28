"""
Stroke Detection System — Application Configuration
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List
from functools import lru_cache
import os


class Settings(BaseSettings):
    """Central configuration loaded from environment variables / .env file."""

    # ---- Application ----
    app_name: str = "StrokeDetectionAPI"
    app_env: str = "development"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000

    # ---- Security ----
    secret_key: str = "change-me-to-a-random-secret"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # ---- AWS S3 ----
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"
    s3_bucket_name: str = "stroke-detection-scans"

    # ---- Model Paths ----
    classification_model_path: str = "models/weights/stroke_classifier.pth"
    segmentation_model_path: str = "models/weights/unet_segmentation.pth"

    # ---- CORS ----
    allowed_origins: str = "http://localhost:3000,http://localhost:5173"

    # ---- Upload ----
    upload_dir: str = "uploads"
    max_upload_size_mb: int = 50

    @property
    def allowed_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",")]

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
