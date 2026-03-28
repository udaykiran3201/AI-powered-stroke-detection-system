"""
Stroke Detection System — Logging Configuration
"""

import sys
from loguru import logger
from app.core.config import get_settings


def setup_logging() -> None:
    """Configure structured logging with loguru."""
    settings = get_settings()

    # Remove default handler
    logger.remove()

    # Console handler
    log_level = "DEBUG" if settings.debug else "INFO"
    logger.add(
        sys.stdout,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # File handler (rotating)
    logger.add(
        "logs/stroke_detection_{time:YYYY-MM-DD}.log",
        level="INFO",
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
            "{name}:{function}:{line} — {message}"
        ),
    )

    logger.info(f"Logging initialised — level={log_level}, env={settings.app_env}")
