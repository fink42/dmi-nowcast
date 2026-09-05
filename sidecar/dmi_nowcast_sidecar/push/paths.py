"""Where the push key and the subscription DB live.

``PushConfig`` deliberately does not know about ``StorageConfig`` — a
config section that reaches into another one cannot be validated on its
own. The two defaults are resolved here instead, against the whole
``Config``, so the data volume stays the single place both files land in.
"""
from __future__ import annotations

from pathlib import Path

from ..config import Config

# Everything push owns lives in one subdirectory of the data volume, so a
# backup or a wipe is one path. No route serves this directory (see
# ``tests/test_push_backend.py`` — the path-safety probes).
PUSH_SUBDIR = "push"
DB_FILENAME = "subscriptions.sqlite"
KEY_FILENAME = "vapid_private.pem"

# The fitted threshold table is NOT under ``push/``: it is a published
# artifact — fitted on the private instance, served at
# ``/calibration/push_thresholds.json``, pulled by the public instance's
# sync task — and lands beside the other synced files at the root of the
# data volume. ``push/`` holds the two secrets, and nothing else.
THRESHOLDS_FILENAME = "push_thresholds.json"


def push_dir(config: Config) -> Path:
    """``<storage.data_dir>/push`` — the default home of both files."""
    return Path(config.storage.data_dir) / PUSH_SUBDIR


def resolved_db_path(config: Config) -> Path:
    """Configured ``push.db_path``, else ``<data_dir>/push/<DB_FILENAME>``."""
    if config.push.db_path is not None:
        return Path(config.push.db_path)
    return push_dir(config) / DB_FILENAME


def resolved_key_path(config: Config) -> Path:
    """Configured ``push.vapid_private_key_file``, else the data-dir default."""
    if config.push.vapid_private_key_file is not None:
        return Path(config.push.vapid_private_key_file)
    return push_dir(config) / KEY_FILENAME


def resolved_thresholds_path(config: Config) -> Path:
    """Configured ``push.thresholds_path``, else ``<data_dir>/<name>``.

    One function, one answer: the nightly fit writes here, the sync task
    writes here, ``/calibration/push_thresholds.json`` serves this, and
    ``push.thresholds.ThresholdTable`` reads it. A second opinion about
    this path is silent — the file appears and nothing loads it.
    """
    if config.push.thresholds_path is not None:
        return Path(config.push.thresholds_path)
    return Path(config.storage.data_dir) / THRESHOLDS_FILENAME


__all__ = [
    "DB_FILENAME",
    "KEY_FILENAME",
    "PUSH_SUBDIR",
    "push_dir",
    "resolved_db_path",
    "resolved_key_path",
    "resolved_thresholds_path",
    "THRESHOLDS_FILENAME",
]
