"""On-disk persistence for ``state.json``.

Atomic writes via ``.tmp`` + ``os.replace``; keeps the previous good
``state.json`` at ``state.json.prev`` so a future cycle that crashes
mid-write doesn't leave the consumer with no readable state.
"""
from __future__ import annotations

import json
import os
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
        """Read the current state. Returns None if no state has ever been written."""
        if not self.state_path.is_file():
            return None
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

        Algorithm:
          1. Promote the current ``state.json`` → ``state.json.prev``
             (just before the new file is moved into place; this is the
             window where ``state.json`` may briefly not exist on disk).
          2. Write new content to a tempfile in the same directory.
          3. ``os.replace`` the tempfile to ``state.json`` — atomic on
             POSIX, atomic on NTFS.
        """
        payload = state.model_dump_json(indent=2)
        # Promote current → prev.
        if self.state_path.exists():
            self.state_path.replace(self.prev_state_path)
        # New temp inside the same dir so replace is atomic.
        tmp_fd, tmp_name = tempfile.mkstemp(
            prefix=".state-", suffix=".json", dir=str(self.data_dir),
        )
        try:
            with os.fdopen(tmp_fd, "w") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, self.state_path)
        except Exception:
            # Best-effort cleanup of the tempfile.
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass
            # Restore prev if we promoted but couldn't write.
            if self.prev_state_path.exists() and not self.state_path.exists():
                self.prev_state_path.replace(self.state_path)
            raise


def _utc_now_iso() -> str:
    """Helper for tests that need a fixed-format clock; only used here so
    state_schema.py stays free of clock imports."""
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"
