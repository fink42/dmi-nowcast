"""Gauge-truth poller (Phase F, F1): config, scheduling, async discipline.

Four things must hold, and they are the four ways this can hurt in
production:

1. **Config refuses the dangerous combinations.** The public instance
   cannot enable gauge polling (no corpus volume, and it would burn DMI's
   fair-use budget writing into a container filesystem nobody reads), and
   neither can an instance with ``storage.corpus_dir: null``. Both are
   rejected at config load, before a single request is made.
2. **The poll works with no network.** A fake client drives the whole
   task; what lands in the store is asserted, not mocked away.
3. **Nothing blocks the event loop.** The Parquet rewrite is the one
   genuinely blocking call in this task, and the test proves it runs on a
   different thread than the loop — not merely that ``to_thread`` was
   spelled correctly.
4. **A metObs failure is contained.** One parameter erroring must not
   sink the others, and must never propagate into the radar cycle.
"""
from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from dmi_nowcast_core.metobs import Observation, Station
from dmi_nowcast_core.station_store import StationObsStore
from dmi_nowcast_sidecar.config import Config, StationObsConfig
from dmi_nowcast_sidecar.station_obs import (
    StationObsPoller,
    build_station_obs_poller,
)

NOW = datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc)


class FakeClient:
    """An ``AsyncMetObsClient`` stand-in: records calls, returns canned rows."""

    def __init__(self, rows: dict[str, list[Observation]] | None = None,
                 fail: set[str] | None = None) -> None:
        self.rows = rows or {}
        self.fail = fail or set()
        self.calls: list[tuple[str, datetime, datetime]] = []
        self.closed = False
        self.last_stats = type("S", (), {"skipped": 0})()

    async def fetch_observations(self, parameter_id, start_utc, end_utc, **kw):
        self.calls.append((parameter_id, start_utc, end_utc))
        if parameter_id in self.fail:
            raise RuntimeError("metObs is down")
        return list(self.rows.get(parameter_id, []))

    async def aclose(self) -> None:
        self.closed = True


def _config(tmp_path: Path, **station_obs) -> Config:
    settings = {"enabled": True, **station_obs}
    return Config(
        home={"lat": 55.33, "lon": 10.32},  # type: ignore[arg-type]
        calibration={  # type: ignore[arg-type]
            "curves_path": tmp_path / "curves.json",
            "national_curves_path": tmp_path / "national_curves.json",
        },
        storage={"data_dir": tmp_path / "data", "corpus_dir": tmp_path / "corpus"},  # type: ignore[arg-type]
        lightning={"archive_dir": tmp_path / "strikes"},  # type: ignore[arg-type]
        station_obs=settings,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_disabled_by_default() -> None:
    settings = StationObsConfig()
    assert settings.enabled is False
    assert settings.interval_min == 10
    assert settings.lookback_min == 40
    assert settings.parameters == ["precip_past10min", "precip_dur_past10min"]
    assert settings.base_url.startswith("https://opendataapi.dmi.dk/v2/metObs")
    assert settings.api_key is None


def test_lookback_must_cover_the_interval() -> None:
    """Otherwise a slipped poll leaves a hole nothing ever revisits."""
    with pytest.raises(ValueError, match="lookback_min"):
        StationObsConfig(enabled=True, interval_min=30, lookback_min=10)


@pytest.mark.parametrize("parameters", [[], ["precip_past10min", "precip_past10min"], [" "]])
def test_invalid_parameter_lists_are_rejected(parameters: list[str]) -> None:
    with pytest.raises(ValueError):
        StationObsConfig(parameters=parameters)


def test_public_mode_cannot_enable_gauge_polling(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="public_mode"):
        Config(
            home={"lat": 55.33, "lon": 10.32},  # type: ignore[arg-type]
            server={"public_mode": True},  # type: ignore[arg-type]
            storage={"data_dir": tmp_path, "corpus_dir": tmp_path / "corpus"},  # type: ignore[arg-type]
            station_obs={"enabled": True},  # type: ignore[arg-type]
        )


def test_enabling_without_a_corpus_dir_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="corpus_dir"):
        Config(
            home={"lat": 55.33, "lon": 10.32},  # type: ignore[arg-type]
            storage={"data_dir": tmp_path, "corpus_dir": None},  # type: ignore[arg-type]
            station_obs={"enabled": True},  # type: ignore[arg-type]
        )


def test_public_mode_with_polling_off_is_fine(tmp_path: Path) -> None:
    config = Config(
        home={"lat": 55.33, "lon": 10.32},  # type: ignore[arg-type]
        server={"public_mode": True},  # type: ignore[arg-type]
        storage={"data_dir": tmp_path, "corpus_dir": None},  # type: ignore[arg-type]
    )
    assert config.station_obs.enabled is False
    assert build_station_obs_poller(config) is None


def test_builder_returns_none_when_disabled(tmp_path: Path) -> None:
    config = _config(tmp_path, enabled=False)
    assert build_station_obs_poller(config) is None


def test_builder_returns_a_poller_when_enabled(tmp_path: Path) -> None:
    poller = build_station_obs_poller(_config(tmp_path))
    assert isinstance(poller, StationObsPoller)
    assert poller.store.root == tmp_path / "corpus"


def test_builder_refuses_public_mode_even_off_config_path(tmp_path: Path) -> None:
    """``Config`` already rejects this; the builder is the second line of
    defence for a config assembled in code rather than loaded."""
    config = _config(tmp_path)
    object.__setattr__(config.server, "public_mode", True)
    assert build_station_obs_poller(config) is None


# ---------------------------------------------------------------------------
# Polling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_poll_writes_observations_to_the_store(tmp_path: Path) -> None:
    client = FakeClient({
        "precip_past10min": [
            Observation("06126", NOW - timedelta(minutes=10), "precip_past10min", 0.4),
            Observation("06188", NOW - timedelta(minutes=10), "precip_past10min", 0.0),
        ],
        "precip_dur_past10min": [
            Observation("06126", NOW - timedelta(minutes=10), "precip_dur_past10min", 7.0),
        ],
    })
    poller = StationObsPoller(_config(tmp_path), client=client)  # type: ignore[arg-type]
    result = await poller.poll_once(now=NOW)

    assert result.ok
    assert result.fetched == 3
    assert result.new_rows == 3
    table = poller.store.read(NOW - timedelta(hours=1), NOW)
    assert table.num_rows == 3
    assert set(table.column("parameter_id").to_pylist()) == {
        "precip_past10min", "precip_dur_past10min",
    }


@pytest.mark.asyncio
async def test_poll_window_is_the_configured_lookback(tmp_path: Path) -> None:
    client = FakeClient()
    poller = StationObsPoller(
        _config(tmp_path, lookback_min=40), client=client,  # type: ignore[arg-type]
    )
    await poller.poll_once(now=NOW)
    assert [c[0] for c in client.calls] == [
        "precip_past10min", "precip_dur_past10min",
    ]
    for _, start, end in client.calls:
        assert end == NOW
        assert start == NOW - timedelta(minutes=40)


@pytest.mark.asyncio
async def test_repeated_polls_do_not_duplicate_rows(tmp_path: Path) -> None:
    """The overlapping lookback is only free because the store dedupes."""
    rows = [Observation("06126", NOW - timedelta(minutes=10), "precip_past10min", 0.4)]
    client = FakeClient({"precip_past10min": rows})
    poller = StationObsPoller(
        _config(tmp_path, parameters=["precip_past10min"]), client=client,  # type: ignore[arg-type]
    )
    first = await poller.poll_once(now=NOW)
    second = await poller.poll_once(now=NOW + timedelta(minutes=10))
    assert first.new_rows == 1
    assert second.new_rows == 0
    assert poller.store.read(NOW - timedelta(hours=1), NOW).num_rows == 1


@pytest.mark.asyncio
async def test_traces_are_counted_and_archived_raw(tmp_path: Path) -> None:
    client = FakeClient({"precip_past10min": [
        Observation("06126", NOW - timedelta(minutes=10), "precip_past10min", -0.1),
    ]})
    poller = StationObsPoller(
        _config(tmp_path, parameters=["precip_past10min"]), client=client,  # type: ignore[arg-type]
    )
    result = await poller.poll_once(now=NOW)
    assert result.traces == 1
    stored = poller.store.read(NOW - timedelta(hours=1), NOW)
    assert stored.column("value").to_pylist() == [pytest.approx(-0.1)]


@pytest.mark.asyncio
async def test_one_failing_parameter_does_not_sink_the_others(tmp_path: Path) -> None:
    client = FakeClient(
        rows={"precip_dur_past10min": [
            Observation("06126", NOW, "precip_dur_past10min", 4.0),
        ]},
        fail={"precip_past10min"},
    )
    poller = StationObsPoller(_config(tmp_path), client=client)  # type: ignore[arg-type]
    result = await poller.poll_once(now=NOW)
    assert not result.ok
    assert "precip_past10min" in result.errors
    assert result.new_rows == 1


@pytest.mark.asyncio
async def test_job_wrapper_never_raises(tmp_path: Path) -> None:
    """apscheduler's job target must be total — a metObs outage cannot be
    allowed to surface anywhere near the radar cycle."""
    client = FakeClient(fail={"precip_past10min", "precip_dur_past10min"})
    poller = StationObsPoller(_config(tmp_path), client=client)  # type: ignore[arg-type]
    await poller._run_once()  # must not raise


# ---------------------------------------------------------------------------
# Async discipline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parquet_write_runs_off_the_event_loop(tmp_path: Path) -> None:
    """The store rewrite is the one blocking call in this task. Proven by
    thread identity, not by the presence of the word ``to_thread``."""
    loop_thread = threading.get_ident()
    seen: list[int] = []
    real_store = StationObsStore(tmp_path / "corpus")

    class RecordingStore(StationObsStore):
        def append(self, observations):  # type: ignore[override]
            seen.append(threading.get_ident())
            return real_store.append(observations)

    client = FakeClient({"precip_past10min": [
        Observation("06126", NOW, "precip_past10min", 0.2),
    ]})
    poller = StationObsPoller(
        _config(tmp_path, parameters=["precip_past10min"]),
        client=client,  # type: ignore[arg-type]
        store=RecordingStore(tmp_path / "corpus"),
    )
    await poller.poll_once(now=NOW)
    assert seen and all(ident != loop_thread for ident in seen)


@pytest.mark.asyncio
async def test_append_failure_is_contained(tmp_path: Path) -> None:
    class BrokenStore(StationObsStore):
        def append(self, observations):  # type: ignore[override]
            raise OSError("disk full")

    client = FakeClient({"precip_past10min": [
        Observation("06126", NOW, "precip_past10min", 0.2),
    ]})
    poller = StationObsPoller(
        _config(tmp_path, parameters=["precip_past10min"]),
        client=client,  # type: ignore[arg-type]
        store=BrokenStore(tmp_path / "corpus"),
    )
    result = await poller.poll_once(now=NOW)
    assert "precip_past10min" in result.errors
    assert result.new_rows == 0


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_schedules_the_job_and_runs_one_immediately(tmp_path: Path) -> None:
    client = FakeClient()
    poller = StationObsPoller(
        _config(tmp_path, interval_min=10), client=client,  # type: ignore[arg-type]
    )
    try:
        await poller.start(run_immediately=True)
        assert len(client.calls) == 2  # one per configured parameter
        job = poller._scheduler.get_job("station_obs_poll")
        assert job is not None
        assert job.trigger.interval == timedelta(minutes=10)
    finally:
        await poller.shutdown()


@pytest.mark.asyncio
async def test_start_can_skip_the_immediate_poll(tmp_path: Path) -> None:
    client = FakeClient()
    poller = StationObsPoller(_config(tmp_path), client=client)  # type: ignore[arg-type]
    try:
        await poller.start(run_immediately=False)
        assert client.calls == []
    finally:
        await poller.shutdown()


@pytest.mark.asyncio
async def test_shutdown_does_not_close_an_injected_client(tmp_path: Path) -> None:
    client = FakeClient()
    poller = StationObsPoller(_config(tmp_path), client=client)  # type: ignore[arg-type]
    await poller.start(run_immediately=False)
    await poller.shutdown()
    assert client.closed is False


@pytest.mark.asyncio
async def test_shutdown_is_safe_before_start(tmp_path: Path) -> None:
    poller = StationObsPoller(_config(tmp_path), client=FakeClient())  # type: ignore[arg-type]
    await poller.shutdown()


# ---------------------------------------------------------------------------
# App wiring
# ---------------------------------------------------------------------------


def test_app_starts_no_poller_when_disabled(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from dmi_nowcast_sidecar.app import create_app

    app = create_app(_config(tmp_path, enabled=False), auto_start_scheduler=False)
    with TestClient(app) as client:
        client.get("/healthz")
        assert app.state.station_obs_poller is None


def test_app_exposes_the_injected_poller(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from dmi_nowcast_sidecar.app import create_app

    poller = StationObsPoller(_config(tmp_path), client=FakeClient())  # type: ignore[arg-type]
    app = create_app(
        _config(tmp_path), station_obs_poller=poller, auto_start_scheduler=False,
    )
    with TestClient(app) as client:
        client.get("/healthz")
        assert app.state.station_obs_poller is poller


def test_station_catalogue_can_be_written_through_the_store(tmp_path: Path) -> None:
    """The poller writes observations; the catalogue comes from the
    backfill script. This pins the store API both of them share."""
    store = StationObsStore(tmp_path / "corpus")
    store.write_catalogue([
        Station("06126", "Odense", "Synop", 55.47, 10.33, "DNK",
                None, None, "Active", ("precip_past10min",), "6"),
    ])
    assert [s.station_id for s in store.read_catalogue()] == ["06126"]


def test_asyncio_import_is_used_for_offloading() -> None:
    """Guard against someone 'simplifying' the executor call away."""
    import inspect

    from dmi_nowcast_sidecar import station_obs

    source = inspect.getsource(station_obs.StationObsPoller.poll_once)
    assert "asyncio.to_thread" in source
    assert asyncio is not None
