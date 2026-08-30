"""Structlog wiring — pretty in dev, JSON in prod.

Single ``configure_logging(level, fmt)`` call should happen once at
startup. Library code calls ``structlog.get_logger(__name__)`` and
that's it.
"""
from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(level: str = "INFO", fmt: str = "pretty") -> None:
    """Configure structlog + stdlib logging in one go.

    ``fmt='pretty'`` gives colorized human-readable lines for terminal
    use; ``fmt='json'`` produces one JSON object per line for production
    (so docker/journald can ingest cleanly).
    """
    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)

    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.stdlib.add_logger_name,
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if fmt == "json":
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    # ProcessorFormatter chain expects the renderer at the end; structlog's
    # own logger emits via stdlib so the foreign stdlib handler can format
    # both worlds identically.
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )

    # Also route stdlib logging (uvicorn, apscheduler, etc.) through structlog.
    stdlib_handler = logging.StreamHandler(sys.stderr)
    stdlib_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared_processors,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                renderer,
            ],
        )
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(stdlib_handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Quiet down chatty libraries.
    logging.getLogger("apscheduler").setLevel(logging.WARNING)
    logging.getLogger("aiohttp.access").setLevel(logging.WARNING)
