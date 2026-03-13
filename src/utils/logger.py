"""
src/utils/logger.py — Structured logging configuration.

Uses structlog + rich for human-readable console output and JSON lines for
machine-readable file output.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import structlog


def _get_log_level() -> int:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    return getattr(logging, level_name, logging.INFO)


def configure_logging(log_file: Path | None = None) -> None:
    """Configure structlog + stdlib logging pipeline."""
    level = _get_log_level()

    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if sys.stdout.isatty():
        # Pretty console output for local development
        renderer = structlog.dev.ConsoleRenderer()
    else:
        # JSON lines for production / CI
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Keep stdlib logging synchronised (for third-party libraries)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        logging.root.addHandler(file_handler)


def get_logger(name: str) -> structlog.BoundLogger:
    """Return a named structured logger."""
    return structlog.get_logger(name)
