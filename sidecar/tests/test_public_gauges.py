"""S5 — the public stack computes its own ``ng_*`` block.

S1 made the cycle compute the 21 neighbour-gauge columns live, and the
public instance could not: no corpus volume, so no gauge store, so
``build_gauge_history`` returned None and the model that decides every
notification there read 21 imputed means. This suite is about the four
pieces that close that, and each test is a way the fix could be wrong
without anything failing:

1. **The permission to poll.** The public instance may now poll metObs,
   on one condition — ``station_obs.store_dir`` names a BOUNDED store on
   its own volume. Without it the old refusal stands, and the bounded
   store may never be the corpus, because it is the one store retention
   is allowed to delete months out of.
2. **The store stays small.** Retention runs after every poll, deletes
   whole month partitions that ended before the cutoff, refuses to date a
   file whose name it cannot parse, and never runs at all on the private
   instance's archive.
3. **The reader follows the writer.** ``build_gauge_history`` resolves
   the store the poller just filled and the catalogue ``sync`` copied
   over, with the private instance's two answers as the defaults — so a
   config that says nothing new behaves exactly as it did.
4. **The catalogue arrives late.** It comes by hourly ``sync``, so the
   first cycles run before it exists. That must be a null block and one
   log line, and the cycle after the file lands must pick it up without a
   restart.

Then the claim all four exist for: a public-mode cycle over a synthetic
gauge store publishes a neighbour block that is filled, and says how many
points it filled it for.

Offline and synthetic throughout: no network, no radar, no VM. The S1
fixtures are imported rather than copied — a second fixture would be a
second opinion about what the block should say.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import structlog

pytest.importorskip("pyarrow")

from dmi_nowcast_core import postprocess as pp
from dmi_nowcast_core.geo import GridIndex
from dmi_nowcast_core.metobs import Observation
from dmi_nowcast_sidecar import compute as compute_mod
from dmi_nowcast_sidecar.config import Config
from dmi_nowcast_sidecar.gauge_history import (
    GaugeHistory,
    build_gauge_history,
    resolved_gauge_points_path,
    resolved_gauge_store_dir,
)
from dmi_nowcast_sidecar.push.postprocess import point_key
from dmi_nowcast_sidecar.station_obs import (
    StationObsPoller,
    build_station_obs_poller,
    month_partition_end,
)
from dmi_nowcast_sidecar.sync import STATION_POINTS_FILE, target_path

# The S1 fixtures: one gauge archive, one catalogue, one motion frame.
from tests.test_push_postprocess import (  # noqa: E402
    GENERATED_AT,
    PIXEL_KM,
    RADAR_TS,
    _flow,
    _gauge_points_file,
    _gauge_store,
    _ng_catalogue,
    _ng_grid_features,
    _NG_NATIVE,
    _products,
    _rain_field,
)
from tests.test_station_obs import FakeClient  # noqa: E402

NOW = datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc)
LAG_MIN = pp.DEFAULT_GAUGE_LAG_MIN


def _public_config(tmp_path: Path, **over) -> Config:
    """The public stack: public mode, no corpus, a bounded gauge store."""
    sections: dict[str, object] = {
        "home": {"lat": 55.33, "lon": 10.32},
        "server": {"public_mode": True},
        "calibration": {
            "curves_path": tmp_path / "curves.json",
            "national_curves_path": tmp_path / "national_curves.json",
        },
        "storage": {"data_dir": tmp_path / "data", "corpus_dir": None},
        "lightning": {"archive_dir": tmp_path / "strikes"},
        "station_obs": {"enabled": True, "store_dir": tmp_path / "gauges"},
    }
    sections.update(over)
    return Config(**sections)  # type: ignore[arg-type]


def _private_config(tmp_path: Path, **over) -> Config:
    """The LAN stack, for the tests that say S5 did not touch it."""
    sections: dict[str, object] = {
        "home": {"lat": 55.33, "lon": 10.32},
        "calibration": {
            "curves_path": tmp_path / "curves.json",
            "national_curves_path": tmp_path / "national_curves.json",
        },
        "storage": {
            "data_dir": tmp_path / "data", "corpus_dir": tmp_path / "corpus",
        },
        "lightning": {"archive_dir": tmp_path / "strikes"},
        "station_obs": {"enabled": True},
    }
    sections.update(over)
    return Config(**sections)  # type: ignore[arg-type]


def _reading(when: datetime, value: float = 0.2) -> Observation:
    return Observation("06126", when, "precip_past10min", value)


# ---------------------------------------------------------------------------
# 1. The permission to poll
# ---------------------------------------------------------------------------


class TestThePermissionToPoll:
    def test_a_bounded_store_lets_the_public_instance_poll(
        self, tmp_path: Path,
    ) -> None:
        config = _public_config(tmp_path)
        assert config.server.public_mode is True
        assert config.storage.corpus_dir is None
        assert config.station_obs.store_dir == tmp_path / "gauges"
        assert config.station_obs.retention_days == 7

    def test_without_one_the_old_refusal_stands(self, tmp_path: Path) -> None:
        """The message still names the mode, and now names the way out."""
        with pytest.raises(ValueError, match="public_mode") as excinfo:
            _public_config(tmp_path, station_obs={"enabled": True})
        assert "store_dir" in str(excinfo.value)

    def test_a_private_instance_without_a_corpus_is_still_refused(
        self, tmp_path: Path,
    ) -> None:
        with pytest.raises(ValueError, match="corpus_dir"):
            _private_config(
                tmp_path,
                storage={"data_dir": tmp_path / "data", "corpus_dir": None},
            )

    def test_the_bounded_store_may_not_be_the_corpus(
        self, tmp_path: Path,
    ) -> None:
        """Retention deletes months. It may never be pointed at the archive."""
        with pytest.raises(ValueError, match="must not be storage.corpus_dir"):
            _private_config(
                tmp_path,
                station_obs={"enabled": True, "store_dir": tmp_path / "corpus"},
            )

    @pytest.mark.parametrize("days", [0, -1, 3651])
    def test_retention_outside_the_bounds_is_refused(
        self, tmp_path: Path, days: int,
    ) -> None:
        with pytest.raises(ValueError):
            _public_config(tmp_path, station_obs={
                "enabled": True,
                "store_dir": tmp_path / "gauges",
                "retention_days": days,
            })

    @pytest.mark.parametrize("days", [1, 7, 3650])
    def test_retention_inside_them_is_accepted(
        self, tmp_path: Path, days: int,
    ) -> None:
        config = _public_config(tmp_path, station_obs={
            "enabled": True,
            "store_dir": tmp_path / "gauges",
            "retention_days": days,
        })
        assert config.station_obs.retention_days == days

    def test_the_builder_starts_a_pruning_poller_in_public_mode(
        self, tmp_path: Path,
    ) -> None:
        poller = build_station_obs_poller(_public_config(tmp_path))
        assert poller is not None
        assert poller.store.root == tmp_path / "gauges"
        assert poller.prunes is True

    def test_the_private_poller_is_the_one_it_always_was(
        self, tmp_path: Path,
    ) -> None:
        poller = build_station_obs_poller(_private_config(tmp_path))
        assert poller is not None
        assert poller.store.root == tmp_path / "corpus"
        assert poller.prunes is False


# ---------------------------------------------------------------------------
# 2. The bounded store
# ---------------------------------------------------------------------------


class TestTheBoundedStore:
    @pytest.mark.asyncio
    async def test_the_poll_writes_under_store_dir(
        self, tmp_path: Path,
    ) -> None:
        client = FakeClient({"precip_past10min": [
            _reading(NOW - timedelta(minutes=10), 0.4),
        ]})
        poller = StationObsPoller(
            _public_config(tmp_path, station_obs={
                "enabled": True,
                "store_dir": tmp_path / "gauges",
                "parameters": ["precip_past10min"],
            }),
            client=client,  # type: ignore[arg-type]
        )
        result = await poller.poll_once(now=NOW)
        assert result.new_rows == 1
        assert poller.store.partition_path(2026, 6).is_file()
        # And nowhere else — there is no corpus on this instance.
        assert not (tmp_path / "corpus").exists()

    @pytest.mark.asyncio
    async def test_retention_deletes_the_months_that_have_ended(
        self, tmp_path: Path,
    ) -> None:
        """Whole partitions, and only once every row in them is past the cutoff.

        Three months seeded: January (long gone), May (ended a fortnight
        before the cutoff) and the current June. A week of retention keeps
        June alone — the current month is never at risk, because its end is
        in the future.
        """
        poller = StationObsPoller(
            _public_config(tmp_path), client=FakeClient(),  # type: ignore[arg-type]
        )
        poller.store.append([
            _reading(datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)),
            _reading(datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)),
            _reading(NOW - timedelta(minutes=10)),
        ])
        assert len(poller.store.partitions()) == 3

        result = await poller.poll_once(now=NOW)

        assert result.pruned == 2
        assert [p.name for p in poller.store.partitions()] == ["06.parquet"]
        # The rows that survived are the rows the features would read.
        kept = poller.store.read(NOW - timedelta(days=1), NOW)
        assert kept.num_rows == 1

    @pytest.mark.asyncio
    async def test_a_longer_retention_keeps_more_months(
        self, tmp_path: Path,
    ) -> None:
        """The knob is the knob: 90 days keeps May, 7 days did not."""
        poller = StationObsPoller(
            _public_config(tmp_path, station_obs={
                "enabled": True,
                "store_dir": tmp_path / "gauges",
                "retention_days": 90,
            }),
            client=FakeClient(),  # type: ignore[arg-type]
        )
        poller.store.append([
            _reading(datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)),
            _reading(datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)),
            _reading(NOW - timedelta(minutes=10)),
        ])
        result = await poller.poll_once(now=NOW)
        assert result.pruned == 1
        assert [p.name for p in poller.store.partitions()] == [
            "05.parquet", "06.parquet",
        ]

    def test_a_file_it_cannot_date_is_left_alone(self, tmp_path: Path) -> None:
        """An unrecognised name is not evidence of age."""
        poller = StationObsPoller(
            _public_config(tmp_path), client=FakeClient(),  # type: ignore[arg-type]
        )
        poller.store.append([
            _reading(datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)),
        ])
        stranger = poller.store.obs_dir / "archive" / "old.parquet"
        stranger.parent.mkdir(parents=True, exist_ok=True)
        stranger.write_bytes(b"not a partition either")

        removed = poller.prune_once(NOW)

        assert [p.name for p in removed] == ["01.parquet"]
        assert stranger.is_file()

    def test_an_emptied_year_directory_goes_with_it(
        self, tmp_path: Path,
    ) -> None:
        poller = StationObsPoller(
            _public_config(tmp_path), client=FakeClient(),  # type: ignore[arg-type]
        )
        poller.store.append([
            _reading(datetime(2025, 11, 15, 12, 0, tzinfo=timezone.utc)),
            _reading(NOW - timedelta(minutes=10)),
        ])
        poller.prune_once(NOW)
        assert not (poller.store.obs_dir / "2025").exists()
        assert (poller.store.obs_dir / "2026").is_dir()

    @pytest.mark.asyncio
    async def test_the_private_archive_is_never_pruned(
        self, tmp_path: Path,
    ) -> None:
        """The whole point of the corpus is that last winter is still in it."""
        poller = StationObsPoller(
            _private_config(tmp_path), client=FakeClient(),  # type: ignore[arg-type]
        )
        poller.store.append([
            _reading(datetime(2025, 12, 7, 12, 0, tzinfo=timezone.utc)),
        ])
        result = await poller.poll_once(now=NOW)
        assert result.pruned == 0
        assert poller.store.partition_path(2025, 12).is_file()

    @pytest.mark.asyncio
    async def test_retention_runs_off_the_event_loop(
        self, tmp_path: Path,
    ) -> None:
        """``unlink`` is filesystem work, and this process serves HTTP."""
        seen: list[int] = []
        poller = StationObsPoller(
            _public_config(tmp_path), client=FakeClient(),  # type: ignore[arg-type]
        )
        real = poller.prune_once

        def recording(now: datetime | None = None) -> list[Path]:
            seen.append(threading.get_ident())
            return real(now)

        poller.prune_once = recording  # type: ignore[method-assign]
        await poller.poll_once(now=NOW)
        assert seen and all(ident != threading.get_ident() for ident in seen)

    @pytest.mark.asyncio
    async def test_a_failing_unlink_costs_no_readings(
        self, tmp_path: Path,
    ) -> None:
        """Retention is housekeeping. It may never sink a poll."""
        client = FakeClient({"precip_past10min": [
            _reading(NOW - timedelta(minutes=10), 0.4),
        ]})
        poller = StationObsPoller(
            _public_config(tmp_path, station_obs={
                "enabled": True,
                "store_dir": tmp_path / "gauges",
                "parameters": ["precip_past10min"],
            }),
            client=client,  # type: ignore[arg-type]
        )
        poller.prune_once = lambda *a, **k: (_ for _ in ()).throw(  # type: ignore[method-assign]
            OSError("read-only volume"),
        )
        with structlog.testing.capture_logs() as logs:
            result = await poller.poll_once(now=NOW)
        assert result.new_rows == 1
        assert result.pruned == 0
        assert "station_obs_prune_failed" in [e["event"] for e in logs]

    @pytest.mark.parametrize(
        ("year", "month", "expected"),
        [
            (2026, 1, datetime(2026, 2, 1, tzinfo=timezone.utc)),
            (2026, 12, datetime(2027, 1, 1, tzinfo=timezone.utc)),
        ],
    )
    def test_a_partition_ends_when_the_next_month_starts(
        self, year: int, month: int, expected: datetime,
    ) -> None:
        assert month_partition_end(year, month) == expected

    def test_a_month_that_is_not_one_has_no_end(self) -> None:
        with pytest.raises(ValueError):
            month_partition_end(2026, 13)


# ---------------------------------------------------------------------------
# 3. Where the cycle reads
# ---------------------------------------------------------------------------


class TestWhereTheCycleReads:
    def test_the_store_defaults_to_the_one_this_instance_polls(
        self, tmp_path: Path,
    ) -> None:
        """One directory, named once: the reader follows the writer."""
        config = _public_config(tmp_path, postprocess={
            "gauge_points_file": tmp_path / "data" / "stations" / "points.json",
        })
        assert resolved_gauge_store_dir(config) == tmp_path / "gauges"
        history = build_gauge_history(config)
        assert history is not None
        assert history.store.root == tmp_path / "gauges"
        assert history.points_file == (
            tmp_path / "data" / "stations" / "points.json"
        )
        assert history.lag_min == pytest.approx(pp.DEFAULT_GAUGE_LAG_MIN)

    def test_an_explicit_store_overrides_it(self, tmp_path: Path) -> None:
        config = _public_config(tmp_path, postprocess={
            "gauge_store_dir": tmp_path / "elsewhere",
            "gauge_points_file": tmp_path / "points.json",
        })
        assert build_gauge_history(config).store.root == tmp_path / "elsewhere"

    def test_the_private_defaults_are_the_ones_it_always_had(
        self, tmp_path: Path,
    ) -> None:
        """Neither key set: the corpus and ``station_eval.points_file``."""
        config = _private_config(tmp_path, station_eval={
            "points_file": tmp_path / "corpus" / "stations" / "points.json",
        })
        assert resolved_gauge_store_dir(config) == tmp_path / "corpus"
        assert resolved_gauge_points_path(config) == (
            tmp_path / "corpus" / "stations" / "points.json"
        )
        history = build_gauge_history(config)
        assert history is not None
        assert history.store.root == tmp_path / "corpus"

    def test_no_catalogue_anywhere_is_still_no_history(
        self, tmp_path: Path,
    ) -> None:
        """A store with nothing to map a coordinate onto answers nothing."""
        config = _public_config(tmp_path)
        assert resolved_gauge_points_path(config) is None
        assert build_gauge_history(config) is None

    def test_no_store_anywhere_is_still_no_history(
        self, tmp_path: Path,
    ) -> None:
        config = Config(
            home={"lat": 55.33, "lon": 10.32},  # type: ignore[arg-type]
            server={"public_mode": True},  # type: ignore[arg-type]
            storage={  # type: ignore[arg-type]
                "data_dir": tmp_path / "data", "corpus_dir": None,
            },
            postprocess={"gauge_points_file": tmp_path / "p.json"},  # type: ignore[arg-type]
        )
        assert resolved_gauge_store_dir(config) is None
        assert build_gauge_history(config) is None

    def test_the_feature_switch_still_wins_over_both(
        self, tmp_path: Path,
    ) -> None:
        config = _public_config(tmp_path, postprocess={
            "gauge_features": False,
            "gauge_points_file": tmp_path / "points.json",
        })
        assert build_gauge_history(config) is None


# ---------------------------------------------------------------------------
# 4. The catalogue arrives late
# ---------------------------------------------------------------------------


def _history(tmp_path: Path, points_file: Path) -> GaugeHistory:
    """A history over the S1 gauge archive, reading ``points_file``."""
    catalogue = _ng_catalogue()
    store = _gauge_store(tmp_path, catalogue)
    return GaugeHistory(store.root, points_file, lag_min=LAG_MIN)


class TestTheCatalogueArrivesLate:
    """It comes by hourly ``sync``, so the first cycles run without it."""

    def test_a_missing_catalogue_is_a_null_read_and_one_log_line(
        self, tmp_path: Path,
    ) -> None:
        history = _history(tmp_path, tmp_path / "not_yet.json")
        with structlog.testing.capture_logs() as logs:
            read = history.read(GENERATED_AT)
        assert read.by_station == {}
        assert read.coords == {}
        assert [e["event"] for e in logs] == [
            "gauge_history_points_unreadable",
        ]

    def test_the_line_is_not_repeated_on_every_cycle(
        self, tmp_path: Path,
    ) -> None:
        """A 2-minute cycle against an hourly sync is 30 chances to shout."""
        history = _history(tmp_path, tmp_path / "not_yet.json")
        with structlog.testing.capture_logs() as logs:
            for _ in range(5):
                history.read(GENERATED_AT)
        assert [e["event"] for e in logs].count(
            "gauge_history_points_unreadable",
        ) == 1

    def test_the_cycle_after_the_sync_picks_it_up(
        self, tmp_path: Path,
    ) -> None:
        """No restart: the whole point of re-reading while the cache is empty."""
        catalogue = _ng_catalogue()
        points_file = tmp_path / "station_points.json"
        history = _history(tmp_path, points_file)
        assert history.read(GENERATED_AT).coords == {}

        # ``sync`` writes the file.
        written = _gauge_points_file(tmp_path, catalogue)
        assert written == points_file

        with structlog.testing.capture_logs() as logs:
            read = history.read(GENERATED_AT)
        assert set(read.coords) == {station for station, *_ in catalogue}
        assert read.by_station  # the store read happened too
        loaded = next(
            e for e in logs if e["event"] == "gauge_history_points_loaded"
        )
        assert loaded["stations"] == len(catalogue)

    def test_a_document_naming_nobody_is_retried_too(
        self, tmp_path: Path,
    ) -> None:
        """Half a written file looks exactly like this, and is not a catalogue."""
        points_file = tmp_path / "station_points.json"
        points_file.write_text(json.dumps({"version": 2, "points": []}))
        history = _history(tmp_path, points_file)
        with structlog.testing.capture_logs() as logs:
            assert history.read(GENERATED_AT).coords == {}
        assert [e["event"] for e in logs] == [
            "gauge_history_points_unreadable",
        ]

        points_file.write_text(json.dumps({
            "version": 2,
            "points": [{"id": s, "lat": lat, "lon": lon}
                       for s, lat, lon, _w in _ng_catalogue()],
        }))
        assert history.read(GENERATED_AT).coords

    def test_a_loaded_catalogue_is_not_re_read(self, tmp_path: Path) -> None:
        """The retry ends the moment there is something to cache."""
        catalogue = _ng_catalogue()
        points_file = _gauge_points_file(tmp_path, catalogue)
        history = _history(tmp_path, points_file)
        assert history.coords()
        points_file.unlink()
        assert set(history.coords()) == {s for s, *_ in catalogue}


# ---------------------------------------------------------------------------
# 5. The private instance publishes the catalogue, sync carries it
# ---------------------------------------------------------------------------


API_KEY = "operator-key"


def _client(config: Config):
    from fastapi.testclient import TestClient

    from dmi_nowcast_sidecar.app import create_app

    return TestClient(create_app(config, auto_start_scheduler=False))


class TestThePrivateInstancePublishesTheCatalogue:
    def test_it_serves_the_file_it_reads_itself(self, tmp_path: Path) -> None:
        points_file = _gauge_points_file(tmp_path, _ng_catalogue())
        config = _private_config(
            tmp_path, station_eval={"points_file": points_file},
        )
        with _client(config) as client:
            response = client.get("/stations/station_points.json")
        assert response.status_code == 200
        assert response.json() == json.loads(points_file.read_text())
        assert response.headers["cache-control"] == "public, max-age=300"

    def test_it_is_503_before_the_file_exists(self, tmp_path: Path) -> None:
        config = _private_config(
            tmp_path, station_eval={"points_file": tmp_path / "missing.json"},
        )
        with _client(config) as client:
            assert client.get(
                "/stations/station_points.json",
            ).status_code == 503

    def test_it_is_503_when_nothing_names_a_catalogue(
        self, tmp_path: Path,
    ) -> None:
        with _client(_private_config(tmp_path)) as client:
            assert client.get(
                "/stations/station_points.json",
            ).status_code == 503

    def test_public_mode_hides_it_like_the_other_artefacts(
        self, tmp_path: Path,
    ) -> None:
        """Pulled, never republished — the same rule as the three calibration
        files. A 404 identical to a path that was never registered."""
        points_file = _gauge_points_file(tmp_path, _ng_catalogue())
        config = _public_config(tmp_path, postprocess={
            "gauge_points_file": points_file,
        })
        config.server.api_key = API_KEY
        with _client(config) as client:
            anonymous = client.get("/stations/station_points.json")
            operator = client.get(
                "/stations/station_points.json",
                headers={"Authorization": f"Bearer {API_KEY}"},
            )
        assert anonymous.status_code == 404
        assert anonymous.json() == {"detail": "Not Found"}
        assert operator.status_code == 200


class TestTheCatalogueTravelsBySync:
    def test_it_lands_where_the_cycle_reads_it(self, tmp_path: Path) -> None:
        """Getting this wrong is silent: the file appears, nothing loads it."""
        points_file = tmp_path / "data" / "stations" / "station_points.json"
        config = _public_config(
            tmp_path, postprocess={"gauge_points_file": points_file},
        )
        assert target_path(config, STATION_POINTS_FILE) == points_file
        assert resolved_gauge_points_path(config) == points_file

    def test_without_the_key_it_lands_under_the_data_dir(
        self, tmp_path: Path,
    ) -> None:
        """Nothing reads it there — and that is the honest place for a file
        this instance was told to pull and given no reader for."""
        config = _public_config(tmp_path)
        assert target_path(config, STATION_POINTS_FILE) == (
            tmp_path / "data" / "stations" / "station_points.json"
        )

    def test_the_shipped_public_example_pulls_it_and_reads_it(self) -> None:
        """The committed config is the deployment. Assert it end to end."""
        from dmi_nowcast_sidecar.config import load_config

        path = (
            Path(__file__).resolve().parents[1]
            / "deploy" / "public" / "config.public.example.yaml"
        )
        config = load_config(path)
        assert STATION_POINTS_FILE in config.sync.files
        # Pulled to exactly where the cycle resolves its catalogue...
        assert target_path(config, STATION_POINTS_FILE) == (
            resolved_gauge_points_path(config)
        )
        # ...and the readings are NOT synced: this instance polls them.
        assert config.station_obs.enabled is True
        assert config.station_obs.store_dir is not None
        assert resolved_gauge_store_dir(config) == config.station_obs.store_dir
        assert not any("obs" in name for name in config.sync.files)
        # Which together are the two halves build_gauge_history needs.
        assert build_gauge_history(config) is not None


# ---------------------------------------------------------------------------
# 6. A public cycle, end to end
# ---------------------------------------------------------------------------


class TestAPublicCycleFillsTheBlock:
    """The claim the other five sections exist for.

    A public-mode config, a gauge store its own poller could have written,
    a catalogue ``sync`` could have copied — and a cycle whose published
    rows carry a neighbour block that is a measurement rather than 21
    nulls. The engine's history comes from ``build_gauge_history(config)``
    and not from a test injection, because the resolution IS the change.
    """

    def _cycle(self, tmp_path: Path):
        catalogue = _ng_catalogue()
        store = _gauge_store(tmp_path, catalogue)
        points_file = _gauge_points_file(tmp_path, catalogue)
        config = _public_config(
            tmp_path,
            station_obs={"enabled": True, "store_dir": store.root},
            postprocess={"gauge_points_file": points_file},
        )
        engine = compute_mod.CycleEngine(config)
        assert engine._gauge_history is not None, (
            "the public cycle must resolve a gauge history from its config"
        )
        products = _products()
        native = [
            GridIndex(row=_NG_NATIVE[s][0], col=_NG_NATIVE[s][1])
            for s, *_ in catalogue
        ]
        point_products = compute_mod._read_points(
            products, native, keep_grids=True,
        )
        assert point_products is not None
        vy, vx = _flow()
        with structlog.testing.capture_logs() as logs:
            engine._publish_postprocess(
                keys=[point_key(lat, lon) for _s, lat, lon, _w in catalogue],
                grid_features=_ng_grid_features(catalogue),
                points=point_products,
                products=products,
                observed_grid=None,
                radar_ts_utc=RADAR_TS,
                generated_at_utc=GENERATED_AT,
                frame_age_min=14.0,
                rain_mm_h=_rain_field(),
                vy=vy, vx=vx,
                pixel_km=PIXEL_KM, dt_min=10.0,
                bulk_vy=2.0, bulk_vx=1.5, stalled_share=0.011,
            )
        published = engine._postprocess_latest
        assert published is not None
        event = next(e for e in logs if e["event"] == "postprocess_cycle")
        return catalogue, published, event

    def test_the_rows_carry_the_block(self, tmp_path: Path) -> None:
        catalogue, published, _event = self._cycle(tmp_path)
        row = published.rows[[s for s, *_ in catalogue].index("06180")]
        assert row["ng_frame_ok"] == 1.0
        assert row["ng_upwet_tau_min"] == pytest.approx(80.0, abs=1.0)
        assert row["ng_near_km"] == pytest.approx(5.0, abs=0.1)
        # Every column the schema names is present, not just the three.
        assert {name for name, _d in pp.SCALAR_FEATURE_COLUMNS_NG} <= set(row)

    def test_the_three_counts_are_what_the_rows_carry(
        self, tmp_path: Path,
    ) -> None:
        """All three above zero: the public instance is no longer blind."""
        _catalogue, published, event = self._cycle(tmp_path)
        rows = published.rows
        assert event["ng_frame_ok"] == sum(
            row["ng_frame_ok"] == 1.0 for row in rows
        )
        assert event["ng_near_km"] == sum(
            row["ng_near_km"] is not None for row in rows
        )
        assert event["ng_upwet_tau_min"] == sum(
            row["ng_upwet_tau_min"] is not None for row in rows
        )
        assert event["ng_frame_ok"] > 0
        assert event["ng_near_km"] > 0
        assert event["ng_upwet_tau_min"] > 0

    def test_the_gauge_block_rides_along_on_the_same_read(
        self, tmp_path: Path,
    ) -> None:
        catalogue, published, _event = self._cycle(tmp_path)
        row = published.rows[[s for s, *_ in catalogue].index("06180")]
        assert row["g_known"] == 1.0
        assert row["g_mm_30"] == pytest.approx(3.6)
