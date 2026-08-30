"""Shared test fixtures."""
from __future__ import annotations

from pathlib import Path

import pytest

from dmi_nowcast_sidecar.config import Config


@pytest.fixture
def minimal_config(tmp_path: Path) -> Config:
    """A Config object with required fields filled in, defaults elsewhere.

    Storage paths are scoped to ``tmp_path`` so tests don't try to mkdir the
    production-default ``/var/lib/dmi-nowcast-corpus``.
    """
    return Config(
        home={"lat": 55.33, "lon": 10.32},  # type: ignore[arg-type]
        # Both curve files scoped to tmp_path (missing → raw behaviour) so a
        # real /var/lib file on the VM can never leak into a test run.
        calibration={  # type: ignore[arg-type]
            "curves_path": tmp_path / "curves.json",
            "national_curves_path": tmp_path / "national_curves.json",
        },
        storage={  # type: ignore[arg-type]
            "data_dir": tmp_path / "data",
            "corpus_dir": tmp_path / "corpus",
        },
        # Scope the strike archive too, or CycleEngine mkdir's the prod default
        # /var/lib/... on any env where it's writable (CI/VM), polluting a real corpus.
        lightning={"archive_dir": tmp_path / "strikes"},  # type: ignore[arg-type]
    )


@pytest.fixture
def config_yaml(tmp_path: Path) -> Path:
    """A minimal valid YAML config file path."""
    p = tmp_path / "config.yaml"
    p.write_text(
        "home:\n"
        "  lat: 55.33\n"
        "  lon: 10.32\n"
        f"storage:\n"
        f"  data_dir: {tmp_path / 'data'}\n"
    )
    return p
