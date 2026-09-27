"""On-disk persistence for ``state.json``.

Atomic writes via a fsynced tempfile + ``os.replace``; keeps the previous
good ``state.json`` at ``state.json.prev`` so a future cycle that crashes
mid-write doesn't leave the consumer with no readable state, and a reader
never finds ``state.json`` missing between two writes.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

from .state_schema import State


STATE_FILENAME = "state.json"
PREV_STATE_FILENAME = "state.json.prev"


class StateStore:
    """File-backed state with atomic writes and last-good rollback."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @property
    def state_path(self) -> Path:
        return self.data_dir / STATE_FILENAME

    @property
    def prev_state_path(self) -> Path:
        return self.data_dir / PREV_STATE_FILENAME

    def load(self) -> State | None:
        """Read the current state, falling back to ``state.json.prev`` when
        ``state.json`` is missing or unreadable. None when neither exists."""
        if not self.state_path.is_file():
            return self._load_prev()
        try:
            raw = json.loads(self.state_path.read_text())
            return State.model_validate(raw)
        except Exception:  # noqa: BLE001
            # Corrupt file — fall back to the previous-good if available.
            return self._load_prev()

    def _load_prev(self) -> State | None:
        if not self.prev_state_path.is_file():
            return None
        try:
            return State.model_validate_json(self.prev_state_path.read_text())
        except Exception:  # noqa: BLE001
            return None

    def write(self, state: State) -> None:
        """Atomically replace ``state.json`` with the new payload.

        Blocking (fsync): an async caller runs it via ``asyncio.to_thread``.

        Algorithm — ``state.json`` exists at every instant once written:
          1. Write the new content to a tempfile in the same directory and
             fsync it. Nothing on disk has changed yet, so a failure here
             leaves both ``state.json`` and ``.prev`` exactly as they were.
          2. Point ``state.json.prev`` at the current ``state.json``: a
             hard link to it under a temporary name, ``os.replace``-d over
             ``.prev`` (a byte copy where links are unsupported).
             ``state.json`` itself is untouched.
          3. ``os.replace`` the tempfile to ``state.json`` — atomic on
             POSIX and NTFS — and fsync the directory.

        Until 2026-09-27 the order was promote-then-write: ``state.json``
        was renamed to ``.prev`` first, and a reader landing between the
        two renames found no ``state.json`` at all.
        """
        payload = state.model_dump_json(indent=2)
        tmp_fd, tmp_name = tempfile.mkstemp(
            prefix=".state-", suffix=".json", dir=str(self.data_dir),
        )
        try:
            with os.fdopen(tmp_fd, "w") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            if self.state_path.is_file():
                self._snapshot_current_to_prev()
            os.replace(tmp_name, self.state_path)
        except Exception:
            # Best-effort cleanup of the tempfile; the live files are either
            # untouched or already fully replaced.
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass
            raise
        _fsync_dir(self.data_dir)

    def _snapshot_current_to_prev(self) -> None:
        """``state.json.prev`` := the current ``state.json``, atomically."""
        fd, link_name = tempfile.mkstemp(
            prefix=".state-prev-", suffix=".json", dir=str(self.data_dir),
        )
        os.close(fd)
        os.unlink(link_name)
        try:
            try:
                os.link(self.state_path, link_name)
            except OSError:
                shutil.copyfile(self.state_path, link_name)
            os.replace(link_name, self.prev_state_path)
        except Exception:
            try:
                os.unlink(link_name)
            except FileNotFoundError:
                pass
            raise


def _fsync_dir(path: Path) -> None:
    """Make the renames durable. Best effort: not every platform lets a
    directory be opened for fsync."""
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _utc_now_iso() -> str:
    """Helper for tests that need a fixed-format clock; only used here so
    state_schema.py stays free of clock imports."""
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"
