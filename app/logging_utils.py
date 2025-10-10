"""Logging utilities."""

from __future__ import annotations

import logging


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure the root logger and return an app scoped logger."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    return logging.getLogger("transcription_app")

