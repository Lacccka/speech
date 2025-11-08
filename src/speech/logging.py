"""Logging configuration helpers for the Speech project."""

from __future__ import annotations

import logging
import os
from logging import Logger
from typing import Final


_LOG_LEVEL_ENV: Final[str] = "LOG_LEVEL"
_LOG_FORMAT: Final[str] = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"


def configure_logging(default_level: int = logging.INFO) -> None:
    """Configure basic logging for the application.

    The log level can be overridden via the ``LOG_LEVEL`` environment variable.
    """

    level_name = os.getenv(_LOG_LEVEL_ENV)
    level = getattr(logging, level_name.upper(), default_level) if level_name else default_level

    logging.basicConfig(level=level, format=_LOG_FORMAT)


def get_logger(name: str) -> Logger:
    """Return a configured :class:`~logging.Logger` instance."""

    return logging.getLogger(name)

