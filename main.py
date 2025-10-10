"""Entry point for the Gradio transcription application."""

from __future__ import annotations

import os

from app.config import DEVICE
from app.logging_utils import setup_logging
from app.ui import build_interface


def main() -> None:
    logger = setup_logging()
    logger.info("Device available: %s (CUDA: %s)", DEVICE, hasattr(DEVICE, "type") and DEVICE.type == "cuda")
    app = build_interface()
    app.launch(server_name=os.environ.get("GRADIO_SERVER", "127.0.0.1"), server_port=int(os.environ.get("GRADIO_PORT", 7860)), inbrowser=False, show_api=False)


if __name__ == "__main__":
    main()

