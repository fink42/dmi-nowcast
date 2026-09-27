"""Review R4b in the cycle: the per-cycle STEPS seed and the advection backend.

* ``run_ensemble`` keeps its fixed default seed: the per-cycle seed
  (``ensemble_seed(radar_ts)``) failed the served-p_post leg of the R4b
  gate and waits for a post-processor refit.
* The deterministic native-grid series is advected with
  ``forecast.advection_backend`` (``cv2`` by default, ``scipy`` the
  rollback), read from config at every cycle.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from dmi_nowcast_core.parse import parse_composite
from dmi_nowcast_core.probabilistic import ensemble_seed
from dmi_nowcast_sidecar import compute as compute_mod
from dmi_nowcast_sidecar.compute import CycleEngine
from dmi_nowcast_sidecar.config import load_config

from tests.test_compute_ensemble import (  # noqa: F401 — fixtures
    _make_fake_run_ensemble,
    engine,
    synthetic_paths,
)


def _spy_advection(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    seen: list[dict] = []
    real = compute_mod.advect_field_series

    def spy(*args, **kwargs):
        seen.append(dict(kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(compute_mod, "advect_field_series", spy)
    return seen


def test_run_ensemble_keeps_the_fixed_default_seed(
    engine: CycleEngine,
    synthetic_paths: list[Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R4b gate: the per-cycle seed is NOT enabled yet (it waits for a
    post-processor refit on per-cycle-seed replays), so the cycle passes
    no seed and run_ensemble's default applies. Enabling it is
    ``seed=ensemble_seed(composite_now.timestamp_utc)``."""
    calls: list[dict] = []
    monkeypatch.setattr(compute_mod, "run_ensemble", _make_fake_run_ensemble(calls))
    engine._compute_sync(synthetic_paths, fetch_ms=0.0)
    assert "seed" not in calls[0]
    newest = parse_composite(synthetic_paths[-1]).timestamp_utc
    assert ensemble_seed(newest) == int(newest.timestamp()) // 60


@pytest.mark.parametrize("backend", ["cv2", "scipy"])
def test_advection_uses_the_configured_backend(
    engine: CycleEngine,
    synthetic_paths: list[Path],
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
) -> None:
    monkeypatch.setattr(
        compute_mod, "run_ensemble", _make_fake_run_ensemble([]),
    )
    engine.config.forecast.advection_backend = backend
    seen = _spy_advection(monkeypatch)
    engine._compute_sync(synthetic_paths, fetch_ms=0.0)
    assert seen, "the cycle advected nothing"
    assert seen[0]["backend"] == backend


def _write_yaml(tmp_path: Path, block: str = "") -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "home:\n  lat: 55.33\n  lon: 10.32\n"
        f"storage:\n  data_dir: {tmp_path / 'data'}\n" + block
    )
    return p


def test_advection_backend_defaults_to_cv2(tmp_path: Path) -> None:
    assert load_config(_write_yaml(tmp_path)).forecast.advection_backend == "cv2"


def test_advection_backend_rolls_back_to_scipy(tmp_path: Path) -> None:
    cfg = load_config(_write_yaml(
        tmp_path, "forecast:\n  advection_backend: scipy\n",
    ))
    assert cfg.forecast.advection_backend == "scipy"


def test_advection_backend_is_constrained(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_config(_write_yaml(
            tmp_path, "forecast:\n  advection_backend: numba\n",
        ))
