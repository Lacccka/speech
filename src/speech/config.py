"""Utilities for loading application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass(slots=True)
class BotConfig:
    """Configuration options for the Telegram bot."""

    token: str


@dataclass(slots=True)
class AppConfig:
    """Container for application configuration."""

    bot: BotConfig
    project_root: Path


def load_config(env_file: Optional[os.PathLike[str] | str] = None) -> AppConfig:
    """Load configuration from environment variables.

    Parameters
    ----------
    env_file:
        Optional path to a ``.env`` file. When provided, the file is loaded
        before reading environment variables.

    Returns
    -------
    AppConfig
        Populated configuration dataclass.

    Raises
    ------
    ValueError
        If required configuration values are missing.
    """

    if env_file is not None:
        load_dotenv(env_file)
    else:
        load_dotenv()

    token = os.getenv("BOT_TOKEN")
    if not token:
        raise ValueError("BOT_TOKEN environment variable is not set")

    return AppConfig(bot=BotConfig(token=token), project_root=Path.cwd())

