"""Entrypoint: ``python -m dmi_nowcast_sidecar``."""
from __future__ import annotations

import uvicorn

from .app import create_app
from .config import load_config
from .logging_setup import configure_logging


def main() -> None:
    config = load_config()
    configure_logging(level=config.server.log_level, fmt=config.server.log_format)
    app = create_app(config)
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_config=None,  # we've already configured logging via structlog
        access_log=False,  # quiet; structured access logging can be added later
    )


if __name__ == "__main__":
    main()
