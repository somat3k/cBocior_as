"""
src/utils/logger.py — Structured logging configuration.

Uses structlog routed through the stdlib logging pipeline so that both
console and file handlers receive every log event.

Architecture:
  structlog processors (shared + renderer)
      └─► structlog.stdlib.ProcessorFormatter
              ├─► StreamHandler (stdout/stderr)
              └─► FileHandler   (optional JSON-lines file)
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
    """Configure structlog + stdlib logging pipeline.

    All structlog events are forwarded to the stdlib root logger so that
    file handlers (when requested) receive the same events as the console.
    """
    level = _get_log_level()

    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    structlog.configure(
        processors=shared_processors
        + [
            # Prepare the event_dict for the stdlib ProcessorFormatter
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )

    # Choose console renderer based on whether we have an interactive terminal
    if sys.stdout.isatty():
        console_renderer = structlog.dev.ConsoleRenderer()
    else:
        console_renderer = structlog.processors.JSONRenderer()

    formatter = structlog.stdlib.ProcessorFormatter(
        # These run *after* the shared_processors above
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            console_renderer,
        ],
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    # Remove any handlers added by previous configure_logging() calls
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)

    # Optional JSON-lines file handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        json_formatter = structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(),
            ],
        )
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(json_formatter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> structlog.BoundLogger:
    """Return a named structured logger."""
    return structlog.get_logger(name)
