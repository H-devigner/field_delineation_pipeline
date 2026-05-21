"""
Structured JSON logging for the field delineation pipeline.

Provides a configured logger that outputs structured JSON in production
and human-readable text in development. Configurable via FIELD_LOG_LEVEL
and FIELD_ENV environment variables.

Usage:
    from src.monitoring.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Processing region", extra={"region": 5, "tiles": 128})
"""

import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Optional

try:
    from pythonjsonlogger import jsonlogger
    HAS_JSON_LOGGER = True
except ImportError:
    HAS_JSON_LOGGER = False

# ── Run context ──────────────────────────────────────────────
_RUN_ID = uuid.uuid4().hex[:12]
_ENV = os.environ.get("FIELD_ENV", "dev")


class ContextFilter(logging.Filter):
    """Add run context to every log record."""

    def __init__(self, run_id: str, env: str):
        super().__init__()
        self.run_id = run_id
        self.env = env

    def filter(self, record):
        record.run_id = self.run_id
        record.env = self.env
        record.timestamp = datetime.now(timezone.utc).isoformat()
        return True


def get_logger(
    name: str,
    level: Optional[str] = None,
    log_format: Optional[str] = None,
) -> logging.Logger:
    """
    Get a configured logger.

    Parameters
    ----------
    name : str
        Logger name (usually __name__).
    level : str or None
        Log level. Defaults to FIELD_LOG_LEVEL env var or INFO.
    log_format : str or None
        "json" or "text". Defaults to "json" in prod, "text" in dev.
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    level = level or os.environ.get("FIELD_LOG_LEVEL", "INFO")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    log_format = log_format or ("json" if _ENV == "prod" else "text")

    handler = logging.StreamHandler(sys.stdout)

    if log_format == "json" and HAS_JSON_LOGGER:
        formatter = jsonlogger.JsonFormatter(
            "%(timestamp)s %(name)s %(levelname)s %(message)s %(run_id)s %(env)s",
            rename_fields={"levelname": "level", "name": "logger"},
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s │ %(name)-30s │ %(levelname)-7s │ %(message)s",
            datefmt="%H:%M:%S",
        )

    handler.setFormatter(formatter)
    handler.addFilter(ContextFilter(_RUN_ID, _ENV))
    logger.addHandler(handler)
    logger.propagate = False

    return logger


def configure_root_logger(level: str = "INFO", log_format: str = "auto"):
    """Configure the root logger for the entire application."""
    if log_format == "auto":
        log_format = "json" if _ENV == "prod" else "text"

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear existing handlers
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)

    if log_format == "json" and HAS_JSON_LOGGER:
        formatter = jsonlogger.JsonFormatter(
            "%(timestamp)s %(name)s %(levelname)s %(message)s %(run_id)s %(env)s",
            rename_fields={"levelname": "level", "name": "logger"},
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s │ %(name)-30s │ %(levelname)-7s │ %(message)s",
            datefmt="%H:%M:%S",
        )

    handler.setFormatter(formatter)
    handler.addFilter(ContextFilter(_RUN_ID, _ENV))
    root.addHandler(handler)

    return _RUN_ID
