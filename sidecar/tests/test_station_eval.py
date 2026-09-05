"""F3 — the live gauge scoreboard (``station_eval``).

The step that runs after the push fan-out, samples the national grids at
every rain gauge, runs the same decision engine the subscribers go
through, and appends the result to the corpus.

No STEPS anywhere: the cycle's products are a hand-built
:class:`NationalProducts` over a 4×4 grid with a fake geo, so every
assertion is about the *step* — the guards, the state file, the parquet
merge, the thread it runs on, and the promise that none of it can cost a
radar cycle.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pyarrow")

from dmi_nowcast_core.geo import GridIndex
from dmi_nowcast_core.national import NationalProducts
from dmi_nowcast_sidecar.config import Config
from dmi_nowcast_sidecar.station_eval import (
    StationEvalService,
    append_rows,
    load_points,
    partition_path,
    state_from_json,
    state_path,
    state_to_json,
)

RADAR_TS = datetime(2026, 9, 5, 6, 20, tzinfo=timezone.utc)
GENERATED_AT = RADAR_TS + timedelta(minutes=14)
POINTS = {
    "version": 2,
    "points": [
        {"id": "06180", "lat": 55.614, "lon": 12.6454, "region": "Hovedstaden"},
        {"id": "06120", "lat": 55.4735, "lon": 10.3297, "region": "Syddanmark"},
    ],
}


# ---------------------------------------------------------------------------
# Fixtures — a cycle without a cycle
# ---------------------------------------------------------------------------


class FakeGeo:
    """Maps each station's (lon, lat) to a native index; nothing else."""

    def __init__(self, mapping: dict[tuple[float, float], tuple[float, float]]):
        self._mapping = mapping

    def lonlat_to_grid(self, lon: float, lat: float) -> GridIndex:
        row, col = self._mapping[(round(lon, 4), round(lat, 4))]
        return GridIndex(row=row, col=col)


def _grid(value: float) -> np.ndarray:
    return np.full((4, 4), value, dtype=np.float32)


def _products(p_rain_30: float = 0.9) -> NationalProducts:
    return NationalProducts(
        p_rain={10: _grid(0.1), 20: _grid(0.5), 30: _grid(p_rain_30)},
        eta_min=_grid(25.0),
        intensity_mm_h=_grid(1.8),
        leads_min=(10, 20, 30),
        threshold_mm_h=0.5,
        timestep_min=10.0,
        frame_age_min=14.0,
        downsample_factor=4,
        n_members=24,
    )


def _geo() -> FakeGeo:
    # ×4 downsample: native row 4 → product row 1, native row 8 → row 2.
    return FakeGeo({
        (12.6454, 55.614): (4.0, 4.0),
        (10.3297, 55.4735): (8.0, 8.0),
    })


class Snapshot(tuple):
    """Mirrors ``compute.NationalSnapshot``: a 2-tuple carrying extras."""


def _engine(products, *, observed=0.0, forecast=0.0, radar_ts=RADAR_TS):
    snap = Snapshot((products, radar_ts))
    snap.observed_mm_h = None if observed is None else _grid(observed)
    snap.forecast_mm_h = None if forecast is None else {0: _grid(forecast)}
    snap.generated_at_utc = radar_ts + timedelta(minutes=14)
    return SimpleNamespace(national_latest=snap, geo=_geo())


def _cycle_result(radar_ts=RADAR_TS):
    return SimpleNamespace(
        state=SimpleNamespace(radar=SimpleNamespace(latest_ts=radar_ts)),
        error=None,
    )


@pytest.fixture
def points_file(tmp_path: Path) -> Path:
    path = tmp_path / "station_points.json"
    path.write_text(json.dumps(POINTS))
    return path


@pytest.fixture
def config(tmp_path: Path, points_file: Path) -> Config:
    return Config(
        home={"lat": 55.33, "lon": 10.32},  # type: ignore[arg-type]
        calibration={  # type: ignore[arg-type]
            "curves_path": tmp_path / "curves.json",
            "national_curves_path": tmp_path / "national_curves.json",
        },
        storage={  # type: ignore[arg-type]
            "data_dir": tmp_path / "data",
            "corpus_dir": tmp_path / "corpus",
        },
        lightning={"archive_dir": tmp_path / "strikes"},  # type: ignore[arg-type]
        station_eval={  # type: ignore[arg-type]
            "enabled": True,
            "points_file": str(points_file),
        },
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_defaults_are_off_and_carry_the_live_rule(tmp_path: Path) -> None:
    cfg = Config(home={"lat": 55.33, "lon": 10.32})  # type: ignore[arg-type]
    assert cfg.station_eval.enabled is False
    assert cfg.station_eval.points_file is None
    assert cfg.station_eval.rules.model_dump() == {
        "threshold_pct": 40,
        "lead_min": 30,
        "rearm_after_min": 60,
        "persistence_obs": 1,
    }


def test_enabled_requires_a_points_file() -> None:
    with pytest.raises(ValueError, match="requires station_eval.points_file"):
        Config(
            home={"lat": 55.33, "lon": 10.32},  # type: ignore[arg-type]
            station_eval={"enabled": True},  # type: ignore[arg-type]
        )


def test_the_public_stack_refuses_to_enable_it(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="server.public_mode"):
        Config(
            home={"lat": 55.33, "lon": 10.32},  # type: ignore[arg-type]
            server={"public_mode": True},  # type: ignore[arg-type]
            station_eval={  # type: ignore[arg-type]
                "enabled": True, "points_file": str(tmp_path / "p.json"),
            },
        )


def test_enabled_requires_a_corpus_dir(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="storage.corpus_dir"):
        Config(
            home={"lat": 55.33, "lon": 10.32},  # type: ignore[arg-type]
            storage={  # type: ignore[arg-type]
                "data_dir": tmp_path / "data", "corpus_dir": None,
            },
            station_eval={  # type: ignore[arg-type]
                "enabled": True, "points_file": str(tmp_path / "p.json"),
            },
        )


def test_lead_must_be_a_served_national_lead(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="national.leads_min"):
        Config(
            home={"lat": 55.33, "lon": 10.32},  # type: ignore[arg-type]
            storage={  # type: ignore[arg-type]
                "data_dir": tmp_path / "data", "corpus_dir": tmp_path / "corpus",
            },
            station_eval={  # type: ignore[arg-type]
                "enabled": True,
                "points_file": str(tmp_path / "p.json"),
                "rules": {"lead_min": 35},
            },
        )


# ---------------------------------------------------------------------------
# Points + state persistence
# ---------------------------------------------------------------------------


def test_load_points_reads_the_v2_file(points_file: Path) -> None:
    points = load_points(points_file)
    assert [p["id"] for p in points] == ["06180", "06120"]
    assert points[0]["lat"] == pytest.approx(55.614)


def test_load_points_rejects_another_version(tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"version": 1, "points": POINTS["points"]}))
    with pytest.raises(ValueError, match="version-2"):
        load_points(path)


def test_state_json_round_trips(config: Config) -> None:
    from dmi_nowcast_sidecar.push.engine import SubState

    original = SubState(
        armed=False,
        streak=3,
        below_since_utc=RADAR_TS - timedelta(minutes=30),
        last_eval_radar_ts=RADAR_TS,
    )
    assert state_from_json(state_to_json(original)) == original
    # Anything unreadable restarts armed rather than raising.
    assert state_from_json(None).armed is True
    assert state_from_json("garbage").streak == 0


async def test_state_survives_a_restart(config: Config) -> None:
    service = StationEvalService(config, _engine(_products()))
    await service.after_cycle(_cycle_result())
    saved = json.loads(state_path(config).read_text())
    assert saved["version"] == 1
    assert set(saved["stations"]) == {"06180", "06120"}
    assert saved["stations"]["06180"]["armed"] is False  # it fired

    # A fresh process reads the file instead of starting armed, so the
    # subscription is not re-notified on the next frame.
    later = RADAR_TS + timedelta(minutes=10)
    reborn = StationEvalService(config, _engine(_products(), radar_ts=later))
    await reborn.after_cycle(_cycle_result(later))
    rows = _read_partition(config, later)
    second = [r for r in rows if r["radar_ts"] == later]
    assert {r["action"] for r in second} == {"none"}
    assert all(r["armed_after"] is False for r in second)


# ---------------------------------------------------------------------------
# The step
# ---------------------------------------------------------------------------


def _read_partition(config: Config, instant=RADAR_TS) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(partition_path(config, instant)).to_pylist()


async def test_the_step_appends_one_row_per_station(config: Config) -> None:
    service = StationEvalService(config, _engine(_products(), observed=0.0))
    await service.after_cycle(_cycle_result())

    assert partition_path(config, RADAR_TS) == (
        Path(config.storage.corpus_dir) / "stations" / "eval" / "2026" / "09.parquet"
    )
    rows = _read_partition(config)
    assert len(rows) == 2
    row = [r for r in rows if r["station_id"] == "06180"][0]
    assert row["radar_ts"] == RADAR_TS
    assert row["generated_at"] == GENERATED_AT
    assert row["p_rain"] == pytest.approx(0.9)     # the rule's lead, 30 min
    assert row["eta_min"] == pytest.approx(25.0)
    assert row["intensity_mm_h"] == pytest.approx(1.8)
    assert row["observed_mm_h"] == pytest.approx(0.0)
    assert row["forecast_now_mm_h"] == pytest.approx(0.0)
    # 0.9 over the 40 % threshold, persistence 1, dry at the point → a push.
    assert row["action"] == "notify"
    assert row["armed_after"] is False
    assert row["streak_after"] == 1
    assert service.last_summary["stations"] == 2


async def test_observed_rain_at_the_point_silences_the_warning(
    config: Config,
) -> None:
    service = StationEvalService(config, _engine(_products(), observed=2.0))
    await service.after_cycle(_cycle_result())
    rows = _read_partition(config)
    assert {r["action"] for r in rows} == {"already_raining"}


async def test_a_below_threshold_probability_does_nothing(config: Config) -> None:
    service = StationEvalService(config, _engine(_products(p_rain_30=0.2)))
    await service.after_cycle(_cycle_result())
    rows = _read_partition(config)
    assert {r["action"] for r in rows} == {"none"}
    assert all(r["armed_after"] is True for r in rows)


async def test_the_append_is_idempotent_on_radar_ts_and_station(
    config: Config,
) -> None:
    service = StationEvalService(config, _engine(_products()))
    await service.after_cycle(_cycle_result())
    assert len(_read_partition(config)) == 2

    # A second service (fresh process) re-evaluating the same frame must
    # correct the rows, never double them.
    again = StationEvalService(config, _engine(_products()))
    await again.after_cycle(_cycle_result())
    rows = _read_partition(config)
    assert len(rows) == 2
    assert {(r["radar_ts"], r["station_id"]) for r in rows} == {
        (RADAR_TS, "06180"), (RADAR_TS, "06120"),
    }


async def test_rows_from_two_frames_accumulate_in_one_month(
    config: Config,
) -> None:
    service = StationEvalService(config, _engine(_products()))
    await service.after_cycle(_cycle_result())
    later = RADAR_TS + timedelta(minutes=10)
    service.engine = _engine(_products(), radar_ts=later)
    await service.after_cycle(_cycle_result(later))
    rows = _read_partition(config, later)
    assert len(rows) == 4
    assert [r["radar_ts"] for r in rows] == sorted(r["radar_ts"] for r in rows)


async def test_a_repeated_radar_frame_is_not_evaluated_twice(
    config: Config,
) -> None:
    """Half the cycles re-emit the previous frame; a streak must not move."""
    service = StationEvalService(config, _engine(_products(p_rain_30=0.5)))
    await service.after_cycle(_cycle_result())
    first = service.last_summary
    await service.after_cycle(_cycle_result())      # same radar_ts
    assert service.last_summary is first
    rows = _read_partition(config)
    assert len(rows) == 2


async def test_products_from_another_frame_are_refused(config: Config) -> None:
    stale = RADAR_TS - timedelta(minutes=10)
    service = StationEvalService(config, _engine(_products(), radar_ts=stale))
    await service.after_cycle(_cycle_result(RADAR_TS))
    assert service.last_summary is None
    assert not partition_path(config, RADAR_TS).exists()


async def test_no_national_products_is_a_skip_not_a_failure(
    config: Config,
) -> None:
    service = StationEvalService(
        config, SimpleNamespace(national_latest=None, geo=None),
    )
    await service.after_cycle(_cycle_result())
    assert service.last_summary is None


async def test_public_mode_is_refused_at_run_time_too(config: Config) -> None:
    # Config already refuses the combination; the step checks again so a
    # future wiring change cannot start writing a corpus on the public box.
    config.server.public_mode = True
    service = StationEvalService(config, _engine(_products()))
    await service.after_cycle(_cycle_result())
    assert service.last_summary is None
    assert not state_path(config).exists()


async def test_disabled_does_nothing(config: Config) -> None:
    config.station_eval.enabled = False
    service = StationEvalService(config, _engine(_products()))
    await service.after_cycle(_cycle_result())
    assert service.last_summary is None


async def test_a_failure_inside_the_step_never_reaches_the_cycle(
    config: Config, monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = StationEvalService(config, _engine(_products()))

    def boom(*args, **kwargs):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(service, "_evaluate_and_append", boom)
    await service.after_cycle(_cycle_result())      # must not raise
    assert service.last_summary is None
    # And the frame is not marked evaluated, so a later retry can still run.
    assert service._last_radar_ts is None


async def test_a_missing_points_file_is_survivable(config: Config) -> None:
    config.station_eval.points_file = Path("/nonexistent/points.json")
    service = StationEvalService(config, _engine(_products()))
    await service.after_cycle(_cycle_result())
    assert service.last_summary is None


async def test_all_the_blocking_work_runs_off_the_event_loop(
    config: Config, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sampling, the parquet rewrite and the state write are executor work."""
    service = StationEvalService(config, _engine(_products()))
    seen: list[bool] = []
    original = service._evaluate_and_append

    def spy(*args, **kwargs):
        seen.append(threading.current_thread() is threading.main_thread())
        return original(*args, **kwargs)

    monkeypatch.setattr(service, "_evaluate_and_append", spy)
    await service.after_cycle(_cycle_result())
    assert seen == [False], "the blocking work ran on the event loop thread"


# ---------------------------------------------------------------------------
# The partition merge, on its own
# ---------------------------------------------------------------------------


def _row(station: str, ts: datetime, action: str = "none") -> dict:
    return {
        "radar_ts": ts,
        "generated_at": ts + timedelta(minutes=14),
        "station_id": station,
        "p_rain": 0.5,
        "eta_min": None,
        "intensity_mm_h": None,
        "observed_mm_h": None,
        "forecast_now_mm_h": None,
        "action": action,
        "armed_after": True,
        "streak_after": 0,
    }


def test_append_rows_replaces_the_key_and_sorts(tmp_path: Path) -> None:
    import pyarrow.parquet as pq

    path = tmp_path / "09.parquet"
    later = RADAR_TS + timedelta(minutes=10)
    append_rows(path, [_row("06180", later), _row("06180", RADAR_TS)])
    append_rows(path, [_row("06180", RADAR_TS, action="notify")])
    rows = pq.read_table(path).to_pylist()
    assert len(rows) == 2
    assert [r["radar_ts"] for r in rows] == [RADAR_TS, later]
    assert rows[0]["action"] == "notify"       # rewritten, not duplicated


def test_append_rows_replaces_a_corrupt_partition(tmp_path: Path) -> None:
    path = tmp_path / "09.parquet"
    path.write_bytes(b"not a parquet file")
    assert append_rows(path, [_row("06180", RADAR_TS)]) == 1


def test_append_rows_writes_atomically(tmp_path: Path) -> None:
    path = tmp_path / "09.parquet"
    append_rows(path, [_row("06180", RADAR_TS)])
    # No temp files left behind for a reader to trip over.
    assert [p.name for p in tmp_path.iterdir()] == ["09.parquet"]


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


async def test_after_cycle_hooks_run_in_order_and_are_isolated() -> None:
    from dmi_nowcast_sidecar.app import _build_after_cycle

    calls: list[str] = []

    class Hook:
        def __init__(self, name: str, fail: bool = False) -> None:
            self.name, self.fail = name, fail

        async def after_cycle(self, result) -> None:
            calls.append(self.name)
            if self.fail:
                raise RuntimeError("boom")

    assert _build_after_cycle(None, None) is None
    chained = _build_after_cycle(Hook("push", fail=True), Hook("station_eval"))
    await chained(_cycle_result())
    # Push runs first, and its failure does not cost the scoreboard its turn.
    assert calls == ["push", "station_eval"]


def test_app_builds_the_service_when_enabled(config: Config) -> None:
    from dmi_nowcast_sidecar.app import create_app

    app = create_app(config, auto_start_scheduler=False)
    assert isinstance(app.state.station_eval_service, StationEvalService)


def test_app_leaves_it_none_when_disabled(config: Config) -> None:
    from dmi_nowcast_sidecar.app import create_app

    config.station_eval.enabled = False
    app = create_app(config, auto_start_scheduler=False)
    assert app.state.station_eval_service is None
