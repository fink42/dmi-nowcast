"""Publishing the quality report (Phase F, F4).

The report is BUILT on the private instance, which owns the corpus, and
COPIED to the public one, which does not. Four things have to hold for
that split to be safe, and they are what this module tests:

1. **Config refuses the wrong side.** ``quality_report.enabled`` under
   ``server.public_mode`` is a misconfiguration that would otherwise fail
   silently — the builder would find no corpus, null every section, and
   publish an empty document over a good one. ``sync.enabled`` without a
   source is the mirror mistake.
2. **The build is atomic and never destructive.** A reader hitting the
   route mid-build must see the old document or the new one, never half
   of either, and a build that raises must leave the previous report
   exactly where it was.
3. **The pull keeps the last good copy.** 304 means "yours is current",
   500 and a dead peer mean "keep what you have". Only a 200 carrying a
   parseable document replaces a file.
4. **The public gate lets exactly one new path through.**
   ``/nowcast/quality.json`` rides the existing ``/nowcast/`` prefix;
   ``/calibration/national_curves.json`` stays private, because the public
   instance reads that file, it does not republish it.

Fully synthetic: a fake builder, a stubbed HTTP transport, no corpus and
no network.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from dmi_nowcast_sidecar.app import _PUBLIC_PATHS, _is_public_path, create_app
from dmi_nowcast_sidecar.compute import CycleEngine
from dmi_nowcast_sidecar.config import Config
from dmi_nowcast_sidecar.push.paths import resolved_thresholds_path
from dmi_nowcast_sidecar.quality_report import (
    QualityReportTask,
    build_quality_report_task,
    quality_path,
)
from dmi_nowcast_sidecar.sync import (
    CURVES_FILE,
    ArtifactSync,
    build_artifact_sync,
    target_path,
)

QUALITY_FILE = "nowcast/quality.json"

#: A minimal document that satisfies the client's "something to render"
#: floor — schema version plus a parseable generated_at.
DOC = {
    "schema_version": 1,
    "generated_at_utc": "2026-09-05T03:30:00Z",
    "windows": {"radar": None, "gauge": None, "live": None},
    "headline": {
        "reliability": {"radar": None, "gauge": None},
        "warnings": None,
        "persistence_margin": None,
    },
    "reliability": {"radar": None, "gauge": None},
    "raining_now": None,
    "stations": None,
    "events": None,
    "methods": None,
    # Every top-level section is present-as-null, thresholds included:
    # the checker treats an absent key as a producer bug, not a null.
    "thresholds": None,
}

CURVES = {
    "metadata": {"fitted_at": "2026-09-01T00:00:00+00:00"},
    "curves": {
        "30": {"raw_breakpoints": [0.0, 1.0], "calibrated_values": [0.0, 0.8]},
    },
}


def _config(tmp_path: Path, **overrides) -> Config:
    """A Config with everything scoped to tmp_path."""
    base = {
        "home": {"lat": 55.33, "lon": 10.32},
        "calibration": {
            "curves_path": tmp_path / "curves.json",
            "national_curves_path": tmp_path / "national_curves.json",
        },
        "storage": {
            "data_dir": tmp_path / "data",
            "corpus_dir": tmp_path / "corpus",
        },
        "lightning": {"archive_dir": tmp_path / "strikes"},
    }
    base.update(overrides)
    return Config(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestQualityReportConfig:
    def test_off_by_default(self, minimal_config: Config) -> None:
        assert minimal_config.quality_report.enabled is False
        assert minimal_config.quality_report.at_utc == "03:30"
        assert minimal_config.quality_report.live_days == 90
        assert minimal_config.quality_report.live_days_secondary == 30

    def test_public_mode_refuses_the_builder(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="public_mode"):
            _config(
                tmp_path,
                server={"public_mode": True},
                quality_report={"enabled": True},
            )

    def test_a_builder_without_a_corpus_dir_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="corpus_dir"):
            _config(
                tmp_path,
                storage={"data_dir": tmp_path / "data", "corpus_dir": None},
                quality_report={"enabled": True},
            )

    @pytest.mark.parametrize("bad", ["3:30", "25:00", "03:60", "0330", "night"])
    def test_at_utc_must_be_a_24_hour_clock_time(
        self, tmp_path: Path, bad: str,
    ) -> None:
        with pytest.raises(ValueError, match="at_utc"):
            _config(tmp_path, quality_report={"enabled": True, "at_utc": bad})

    def test_a_valid_private_builder_loads(self, tmp_path: Path) -> None:
        config = _config(tmp_path, quality_report={
            "enabled": True,
            "at_utc": "04:15",
            "radar_corpus": tmp_path / "radar.parquet",
            "replay_dir": tmp_path / "replay",
        })
        assert config.quality_report.enabled
        assert config.quality_report.at_utc == "04:15"
        assert build_quality_report_task(config) is not None

    def test_the_factory_refuses_a_hand_assembled_public_config(
        self, tmp_path: Path,
    ) -> None:
        """The validator can be bypassed by constructing sub-models directly.

        A second check in the factory is what makes that harmless instead
        of a public instance writing into a container filesystem.
        """
        config = _config(tmp_path, quality_report={"enabled": True})
        config.server.public_mode = True
        assert build_quality_report_task(config) is None


class TestSyncConfig:
    def test_off_by_default_with_the_two_expected_files(
        self, minimal_config: Config,
    ) -> None:
        assert minimal_config.sync.enabled is False
        assert minimal_config.sync.source_url is None
        assert minimal_config.sync.interval_min == 60
        assert minimal_config.sync.files == [QUALITY_FILE, CURVES_FILE]

    def test_enabled_requires_a_source_url(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="source_url"):
            _config(tmp_path, sync={"enabled": True})

    def test_the_source_must_be_an_http_url(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="http"):
            _config(tmp_path, sync={
                "enabled": True, "source_url": "dmi-nowcast-sidecar:8081",
            })

    @pytest.mark.parametrize(
        "bad", [["../../etc/passwd"], ["nowcast/../../escape.json"], [""], ["."]],
    )
    def test_a_file_that_escapes_the_data_dir_is_refused(
        self, tmp_path: Path, bad: list[str],
    ) -> None:
        with pytest.raises(ValueError):
            _config(tmp_path, sync={
                "enabled": True,
                "source_url": "http://private:8081",
                "files": bad,
            })

    def test_duplicate_files_are_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="unique"):
            _config(tmp_path, sync={
                "enabled": True, "source_url": "http://private:8081",
                "files": [QUALITY_FILE, QUALITY_FILE],
            })

    def test_a_leading_slash_is_normalised_away(self, tmp_path: Path) -> None:
        config = _config(tmp_path, sync={
            "enabled": True, "source_url": "http://private:8081",
            "files": ["/nowcast/quality.json"],
        })
        assert config.sync.files == [QUALITY_FILE]

    def test_sync_is_allowed_in_public_mode(self, tmp_path: Path) -> None:
        """The public instance is exactly who this is for."""
        config = _config(
            tmp_path,
            server={"public_mode": True},
            storage={"data_dir": tmp_path / "data", "corpus_dir": None},
            sync={"enabled": True, "source_url": "http://private:8081"},
        )
        assert build_artifact_sync(config) is not None

    def test_the_curves_land_where_the_engine_reads_them(
        self, tmp_path: Path,
    ) -> None:
        """Everything else goes under data_dir; the curves do not.

        Getting this wrong is silent — the file appears and nothing loads
        it — so it is pinned rather than left to the reader.
        """
        config = _config(tmp_path)
        assert target_path(config, CURVES_FILE) == tmp_path / "national_curves.json"
        assert target_path(config, QUALITY_FILE) == (
            tmp_path / "data" / "nowcast" / "quality.json"
        )


# ---------------------------------------------------------------------------
# The nightly build
# ---------------------------------------------------------------------------


class TestQualityReportTask:
    def test_it_writes_the_document_where_the_route_serves_it(
        self, tmp_path: Path,
    ) -> None:
        config = _config(tmp_path, quality_report={"enabled": True})
        task = QualityReportTask(config, builder=lambda _inputs: dict(DOC))
        result = anyio_run(task.build_once())
        assert result.ok
        path = quality_path(config)
        assert path == tmp_path / "data" / "nowcast" / "quality.json"
        assert json.loads(path.read_text())["generated_at_utc"] == (
            DOC["generated_at_utc"]
        )
        assert result.schema_problems == []
        assert result.bytes_written > 0

    def test_the_write_leaves_no_temporary_file_behind(
        self, tmp_path: Path,
    ) -> None:
        config = _config(tmp_path, quality_report={"enabled": True})
        task = QualityReportTask(config, builder=lambda _inputs: dict(DOC))
        anyio_run(task.build_once())
        leftovers = list(quality_path(config).parent.glob("*.tmp"))
        assert leftovers == []

    def test_a_failed_build_keeps_the_previous_report(
        self, tmp_path: Path,
    ) -> None:
        """A page that is a day stale is fine; a truncated one is not."""
        config = _config(tmp_path, quality_report={"enabled": True})
        good = QualityReportTask(config, builder=lambda _inputs: dict(DOC))
        anyio_run(good.build_once())
        before = quality_path(config).read_text()

        def explode(_inputs):
            raise RuntimeError("corpus unreadable")

        broken = QualityReportTask(config, builder=explode)
        result = anyio_run(broken.build_once())
        assert result.ok is False
        assert "corpus unreadable" in (result.error or "")
        assert quality_path(config).read_text() == before

    def test_a_document_that_fails_the_schema_check_is_reported(
        self, tmp_path: Path,
    ) -> None:
        """Still written — the page renders what parses — but never silent."""
        config = _config(tmp_path, quality_report={"enabled": True})
        task = QualityReportTask(
            config,
            builder=lambda _inputs: {
                **DOC, "raining_now": {"pod": "excellent"},
            },
        )
        result = anyio_run(task.build_once())
        assert result.ok
        assert result.schema_problems
        assert any("raining_now" in p for p in result.schema_problems)

    def test_the_markdown_twin_is_stamped_by_the_report_date(
        self, tmp_path: Path,
    ) -> None:
        config = _config(tmp_path, quality_report={
            "enabled": True, "markdown_dir": tmp_path / "archive",
        })
        task = QualityReportTask(
            config,
            builder=lambda _inputs: dict(DOC),
            renderer=lambda report: f"# {report['generated_at_utc']}",
        )
        anyio_run(task.build_once())
        assert (tmp_path / "archive" / "2026-09-05.md").read_text().startswith("# ")

    def test_the_inputs_default_to_the_curves_the_engine_reads(
        self, tmp_path: Path,
    ) -> None:
        config = _config(tmp_path, quality_report={"enabled": True})
        inputs = QualityReportTask(config).inputs()
        assert inputs.national_curves == config.calibration.national_curves_path
        assert inputs.corpus_dir == config.storage.corpus_dir

    def test_an_explicit_curves_path_wins(self, tmp_path: Path) -> None:
        config = _config(tmp_path, quality_report={
            "enabled": True, "national_curves": tmp_path / "other.json",
        })
        assert QualityReportTask(config).inputs().national_curves == (
            tmp_path / "other.json"
        )

    def test_the_schedule_is_the_configured_utc_time(self, tmp_path: Path) -> None:
        config = _config(tmp_path, quality_report={
            "enabled": True, "at_utc": "04:15",
        })
        task = QualityReportTask(config, builder=lambda _inputs: dict(DOC))

        async def run() -> None:
            await task.start()
            try:
                jobs = task._scheduler.get_jobs()
                assert len(jobs) == 1
                trigger = jobs[0].trigger
                assert "hour='4'" in str(trigger)
                assert "minute='15'" in str(trigger)
                # The default start does NOT rebuild: a restart at noon
                # must not spend the CPU on a report already on disk.
                assert not quality_path(config).exists()
            finally:
                await task.shutdown()

        anyio_run(run())


# ---------------------------------------------------------------------------
# The pull
# ---------------------------------------------------------------------------


def _sync_config(tmp_path: Path, **sync) -> Config:
    settings = {
        "enabled": True,
        "source_url": "http://dmi-nowcast-sidecar:8081",
    }
    settings.update(sync)
    return _config(
        tmp_path,
        server={"public_mode": True},
        storage={"data_dir": tmp_path / "data", "corpus_dir": None},
        sync=settings,
    )


class _Peer:
    """A stub private instance: scripted responses, recorded requests."""

    def __init__(self, responses: dict[str, list]) -> None:
        self.responses = {k: list(v) for k, v in responses.items()}
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        queue = self.responses.get(request.url.path.lstrip("/"), [])
        if not queue:
            return httpx.Response(404)
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        return item(request) if callable(item) else item

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))


def _ok(payload: dict, etag: str | None = None) -> httpx.Response:
    headers = {"ETag": etag} if etag else {}
    return httpx.Response(200, json=payload, headers=headers)


# ---------------------------------------------------------------------------
# The nightly push-threshold fit (Phase G, G4)
# ---------------------------------------------------------------------------


def _lead_row(threshold: int | None, warnings: int = 210) -> dict:
    return {
        "threshold_pct": threshold,
        "insufficient": threshold is None,
        "f1": 0.42, "precision": 0.48, "recall": 0.37,
        "far": 0.55, "csi": 0.27,
        "warnings": warnings, "hits": 94, "false_alarms": 101,
        "misses": 150, "late": 15,
        "plateau": None if threshold is None else [threshold - 5, threshold + 5],
        "radar_plateau": None,
        "agrees_with_radar": None,
    }


def _thresholds_doc(leads: dict[str, int | None], warnings: int = 210) -> dict:
    return {
        "schema_version": 1,
        "fitted_at_utc": "2026-09-05T03:31:00+00:00",
        "objective": {
            "metric": "f1", "min_useful_lead_min": 5.0, "plateau_frac": 0.95,
            "min_warnings": 30, "rearm_after_min": 60, "persistence_obs": 1,
            "tolerance_min": 10, "dry_min": 30,
        },
        "window": {
            "from": "2026-07-01T00:00:00+00:00",
            "to": "2026-09-01T00:00:00+00:00",
            "days": 62, "stations": 97, "rows": 1841203,
        },
        "fallback_threshold_pct": 40,
        "leads": {
            key: _lead_row(value, warnings) for key, value in leads.items()
        },
    }


class _FakeTable:
    """Stands in for the running service's ``ThresholdTable``."""

    def __init__(self) -> None:
        self.nudged = 0

    def note_changed(self) -> None:
        self.nudged += 1


def _fit_config(tmp_path: Path, **fit) -> Config:
    settings = {
        "enabled": True,
        "decisions_dirs": [tmp_path / "corpus" / "stations" / "eval"],
        "thresholds": "40:60:10",
    }
    settings.update(fit)
    return _config(tmp_path, quality_report={
        "enabled": True, "fit_thresholds": settings,
    })


class TestNightlyThresholdFit:
    """The sweep that runs in front of the report build.

    The sweep itself is tested in ``test_sweep_thresholds.py``; what
    matters here is the wiring around it — the guard, where the file
    lands, the nudge, and the promise that a failed fit costs a table
    nobody notices rather than the whole nightly build.
    """

    def test_it_is_off_by_default(self, tmp_path: Path) -> None:
        config = _config(tmp_path, quality_report={"enabled": True})
        assert config.quality_report.fit_thresholds.enabled is False
        task = QualityReportTask(config, builder=lambda _inputs: dict(DOC))
        result = anyio_run(task.build_once())
        assert result.ok is True
        assert result.thresholds_path is None
        assert not resolved_thresholds_path(config).exists()

    def test_a_first_fit_lands_where_the_service_reads_it(
        self, tmp_path: Path,
    ) -> None:
        config = _fit_config(tmp_path)
        table = _FakeTable()
        seen: list = []

        def build(inputs):
            seen.append(inputs)
            return dict(DOC)

        task = QualityReportTask(
            config, builder=build, thresholds=table,
            fitter=lambda options: {"thresholds": _thresholds_doc({"30": 45})},
        )
        result = anyio_run(task.build_once())

        out = resolved_thresholds_path(config)
        assert result.thresholds_path == out
        assert result.thresholds_guard == {"30": "first_fit"}
        written = json.loads(out.read_text())
        assert written["leads"]["30"]["threshold_pct"] == 45
        # The running service is told to re-read…
        assert table.nudged == 1
        # …and the report was built with the table that was just written,
        # so quality.json describes tonight's rule, not last night's.
        assert seen[0].thresholds_path == out

    def test_the_guard_holds_a_small_move_back(self, tmp_path: Path) -> None:
        config = _fit_config(tmp_path)
        out = resolved_thresholds_path(config)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(_thresholds_doc({"30": 44})))

        task = QualityReportTask(
            config, builder=lambda _inputs: dict(DOC),
            fitter=lambda options: {"thresholds": _thresholds_doc({"30": 45})},
        )
        result = anyio_run(task.build_once())
        assert result.thresholds_guard == {"30": "kept_previous"}
        assert json.loads(out.read_text())["leads"]["30"]["threshold_pct"] == 44

    def test_a_big_well_evidenced_move_is_published(self, tmp_path: Path) -> None:
        config = _fit_config(tmp_path)
        out = resolved_thresholds_path(config)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(_thresholds_doc({"30": 40})))

        task = QualityReportTask(
            config, builder=lambda _inputs: dict(DOC),
            fitter=lambda options: {"thresholds": _thresholds_doc({"30": 60})},
        )
        result = anyio_run(task.build_once())
        assert result.thresholds_guard == {"30": "changed"}
        assert json.loads(out.read_text())["leads"]["30"]["threshold_pct"] == 60

    def test_a_failing_fit_keeps_the_table_and_still_builds(
        self, tmp_path: Path,
    ) -> None:
        config = _fit_config(tmp_path)
        out = resolved_thresholds_path(config)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(_thresholds_doc({"30": 44})))

        def explode(options):
            raise RuntimeError("parquet on fire")

        table = _FakeTable()
        task = QualityReportTask(
            config, builder=lambda _inputs: dict(DOC),
            fitter=explode, thresholds=table,
        )
        result = anyio_run(task.build_once())
        # The report is the point of the task; the fit is a step of it.
        assert result.ok is True
        assert quality_path(config).is_file()
        assert "parquet on fire" in str(result.thresholds_error)
        assert result.thresholds_path is None
        # Untouched, and nobody was told to re-read anything.
        assert json.loads(out.read_text())["leads"]["30"]["threshold_pct"] == 44
        assert table.nudged == 0

    def test_nothing_to_fit_on_is_not_a_failure(self, tmp_path: Path) -> None:
        from dmi_nowcast_sidecar.threshold_sweep import SweepError

        def nothing(options):
            raise SweepError("no decision rows found")

        config = _fit_config(tmp_path)
        task = QualityReportTask(
            config, builder=lambda _inputs: dict(DOC), fitter=nothing,
        )
        result = anyio_run(task.build_once())
        assert result.ok is True
        assert result.thresholds_error == "no decision rows found"
        assert not resolved_thresholds_path(config).exists()

    def test_the_options_are_the_live_rule(self, tmp_path: Path) -> None:
        """The replay must run under the constants the service ships."""
        config = _fit_config(tmp_path, radar_decisions_dir=tmp_path / "radar")
        config.push.lead_options = [20, 30]
        config.push.persistence_obs = 1
        config.push.rearm_after_min = 45
        options = QualityReportTask(config)._fit_options()
        assert options.leads == (20, 30)
        assert options.thresholds == (40, 50, 60)
        assert options.persistence_obs == 1
        assert options.rearm_after_min == 45
        assert options.corpus_dir == config.storage.corpus_dir
        assert options.radar_decisions_dirs == [tmp_path / "radar"]

    def test_an_explicit_out_path_wins(self, tmp_path: Path) -> None:
        config = _fit_config(tmp_path, thresholds_out=tmp_path / "elsewhere.json")
        assert QualityReportTask(config).thresholds_out() == (
            tmp_path / "elsewhere.json"
        )


# ---------------------------------------------------------------------------
# The pull, continued
# ---------------------------------------------------------------------------


class TestArtifactSync:
    def test_a_200_lands_both_files_in_their_own_places(
        self, tmp_path: Path,
    ) -> None:
        config = _sync_config(tmp_path)
        peer = _Peer({QUALITY_FILE: [_ok(DOC)], CURVES_FILE: [_ok(CURVES)]})
        sync = ArtifactSync(config, client=peer.client())
        result = anyio_run(sync.sync_once())
        assert result.ok and result.updated == 2
        assert json.loads(target_path(config, QUALITY_FILE).read_text()) == DOC
        assert json.loads(target_path(config, CURVES_FILE).read_text()) == CURVES

    def test_a_304_leaves_the_file_alone_and_sends_the_etag(
        self, tmp_path: Path,
    ) -> None:
        config = _sync_config(tmp_path, files=[QUALITY_FILE])
        peer = _Peer({QUALITY_FILE: [
            _ok(DOC, etag='"v1"'), httpx.Response(304),
        ]})
        sync = ArtifactSync(config, client=peer.client())
        assert anyio_run(sync.sync_once()).updated == 1
        written = target_path(config, QUALITY_FILE).stat().st_mtime_ns

        second = anyio_run(sync.sync_once())
        assert second.files[0].status == "unchanged"
        assert second.files[0].http_status == 304
        assert target_path(config, QUALITY_FILE).stat().st_mtime_ns == written
        assert peer.requests[-1].headers.get("If-None-Match") == '"v1"'

    def test_identical_bytes_without_an_etag_are_still_not_rewritten(
        self, tmp_path: Path,
    ) -> None:
        """A source with no ETag must not churn the file every hour."""
        config = _sync_config(tmp_path, files=[QUALITY_FILE])
        peer = _Peer({QUALITY_FILE: [_ok(DOC)]})
        sync = ArtifactSync(config, client=peer.client())
        assert anyio_run(sync.sync_once()).updated == 1
        assert anyio_run(sync.sync_once()).files[0].status == "unchanged"

    def test_a_500_keeps_the_last_good_copy(self, tmp_path: Path) -> None:
        config = _sync_config(tmp_path, files=[QUALITY_FILE])
        peer = _Peer({QUALITY_FILE: [_ok(DOC), httpx.Response(500)]})
        sync = ArtifactSync(config, client=peer.client())
        anyio_run(sync.sync_once())
        good = target_path(config, QUALITY_FILE).read_text()

        result = anyio_run(sync.sync_once())
        assert result.failed == 1
        assert result.files[0].http_status == 500
        assert target_path(config, QUALITY_FILE).read_text() == good

    def test_a_dead_peer_keeps_the_last_good_copy(self, tmp_path: Path) -> None:
        config = _sync_config(tmp_path, files=[QUALITY_FILE])

        def refuse(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        peer = _Peer({QUALITY_FILE: [_ok(DOC), refuse]})
        sync = ArtifactSync(config, client=peer.client())
        anyio_run(sync.sync_once())
        good = target_path(config, QUALITY_FILE).read_text()

        result = anyio_run(sync.sync_once())
        assert result.failed == 1
        assert "ConnectError" in (result.files[0].error or "")
        assert target_path(config, QUALITY_FILE).read_text() == good

    def test_a_body_that_is_not_json_never_replaces_a_good_file(
        self, tmp_path: Path,
    ) -> None:
        """A proxy's HTML error page arrives with a 200 and must be refused."""
        config = _sync_config(tmp_path, files=[QUALITY_FILE])
        peer = _Peer({QUALITY_FILE: [
            _ok(DOC),
            httpx.Response(200, text="<html>502 Bad Gateway</html>"),
        ]})
        sync = ArtifactSync(config, client=peer.client())
        anyio_run(sync.sync_once())
        good = target_path(config, QUALITY_FILE).read_text()

        result = anyio_run(sync.sync_once())
        assert result.failed == 1
        assert "not valid JSON" in (result.files[0].error or "")
        assert target_path(config, QUALITY_FILE).read_text() == good

    def test_an_oversize_body_is_refused(self, tmp_path: Path) -> None:
        config = _sync_config(tmp_path, files=[QUALITY_FILE], max_bytes=1024)
        peer = _Peer({QUALITY_FILE: [
            _ok({"schema_version": 1, "pad": "x" * 4096}),
        ]})
        sync = ArtifactSync(config, client=peer.client())
        result = anyio_run(sync.sync_once())
        assert result.failed == 1
        assert "max_bytes" in (result.files[0].error or "")
        assert not target_path(config, QUALITY_FILE).exists()

    def test_one_failing_file_does_not_stop_the_other(
        self, tmp_path: Path,
    ) -> None:
        config = _sync_config(tmp_path)
        peer = _Peer({
            QUALITY_FILE: [httpx.Response(503)],
            CURVES_FILE: [_ok(CURVES)],
        })
        result = anyio_run(ArtifactSync(config, client=peer.client()).sync_once())
        assert result.failed == 1 and result.updated == 1
        assert target_path(config, CURVES_FILE).is_file()

    def test_the_bearer_is_sent_when_configured(self, tmp_path: Path) -> None:
        config = _sync_config(
            tmp_path, files=[QUALITY_FILE], api_key="private-key",
        )
        seen: list[str | None] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.headers.get("Authorization"))
            return _ok(DOC)

        sync = ArtifactSync(config, client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            headers={"Authorization": f"Bearer {config.sync.api_key}"},
        ))
        anyio_run(sync.sync_once())
        assert seen == ["Bearer private-key"]

    def test_the_urls_are_the_source_plus_the_file_name(
        self, tmp_path: Path,
    ) -> None:
        sync = ArtifactSync(_sync_config(tmp_path, source_url="http://p:8081/"))
        assert sync.url_for(QUALITY_FILE) == "http://p:8081/nowcast/quality.json"
        assert sync.url_for(CURVES_FILE) == (
            "http://p:8081/calibration/national_curves.json"
        )


# ---------------------------------------------------------------------------
# Curve hot reload
# ---------------------------------------------------------------------------


class TestCurvesHotReload:
    def test_a_curve_file_replaced_on_disk_is_re_read_at_the_next_cycle(
        self, tmp_path: Path,
    ) -> None:
        """The monthly fit must reach the public instance without a restart.

        The re-read happens at the START of a cycle, which is the one
        instant at which swapping cannot leave ``national_latest``
        calibrated with one set of curves and ``national_curve_leads``
        naming another.
        """
        config = _config(tmp_path)
        engine = CycleEngine(config)
        assert engine.national_curve_leads == frozenset()

        config.calibration.national_curves_path.write_text(json.dumps(CURVES))
        # Nothing has asked for a reload yet, so nothing changed…
        assert engine.national_curve_leads == frozenset()
        # …until the next cycle looks.
        engine._reload_national_curves_if_changed()
        assert engine.national_curve_leads == frozenset({30})
        assert engine.national_calibration_fitted_at is not None

    def test_an_unchanged_file_is_not_re_parsed(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        config.calibration.national_curves_path.write_text(json.dumps(CURVES))
        engine = CycleEngine(config)
        loaded = engine._national_curves
        engine._reload_national_curves_if_changed()
        assert engine._national_curves is loaded

    def test_the_sync_task_nudges_the_engine_after_writing_curves(
        self, tmp_path: Path,
    ) -> None:
        config = _sync_config(tmp_path, files=[CURVES_FILE])
        engine = CycleEngine(config)
        sync = build_artifact_sync(config, engine)
        assert sync is not None
        sync._client = _Peer({CURVES_FILE: [_ok(CURVES)]}).client()

        anyio_run(sync.sync_once())
        assert engine._national_curves_dirty is True
        engine._reload_national_curves_if_changed()
        assert engine.national_curve_leads == frozenset({30})

    def test_a_quality_report_write_does_not_touch_the_curves(
        self, tmp_path: Path,
    ) -> None:
        config = _sync_config(tmp_path, files=[QUALITY_FILE])
        engine = CycleEngine(config)
        sync = build_artifact_sync(config, engine)
        assert sync is not None
        sync._client = _Peer({QUALITY_FILE: [_ok(DOC)]}).client()
        anyio_run(sync.sync_once())
        assert engine._national_curves_dirty is False


# ---------------------------------------------------------------------------
# Routes and the public gate
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path: Path):
    config = _config(tmp_path)
    app = create_app(config, auto_start_scheduler=False)
    with TestClient(app) as c:
        yield config, c


class TestRoutes:
    def test_quality_json_503s_before_the_first_build(self, client) -> None:
        _config_obj, c = client
        response = c.get("/nowcast/quality.json")
        assert response.status_code == 503
        assert "quality report" in response.json()["detail"]

    def test_quality_json_is_served_with_a_five_minute_cache(
        self, client,
    ) -> None:
        config, c = client
        path = quality_path(config)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(DOC))
        response = c.get("/nowcast/quality.json")
        assert response.status_code == 200
        assert response.json() == DOC
        assert response.headers["Cache-Control"] == "public, max-age=300"
        assert response.headers["content-type"].startswith("application/json")

    def test_the_literal_path_wins_over_the_stamped_artifact_matcher(
        self, client,
    ) -> None:
        """``quality.json`` is not a cycle-stamped name.

        The generic ``/nowcast/{filename}`` route would reject it as
        unsafe, so the literal route has to be registered first — and if
        it ever is not, this test sees a 400 instead of a 200.
        """
        config, c = client
        path = quality_path(config)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(DOC))
        assert c.get("/nowcast/quality.json").status_code == 200
        # …and the immutable cache header the stamped route sets is NOT
        # what this file got.
        assert "immutable" not in c.get("/nowcast/quality.json").headers[
            "Cache-Control"
        ]

    def test_national_curves_are_served_on_the_private_api(self, client) -> None:
        config, c = client
        assert c.get("/calibration/national_curves.json").status_code == 503
        config.calibration.national_curves_path.write_text(json.dumps(CURVES))
        response = c.get("/calibration/national_curves.json")
        assert response.status_code == 200
        assert response.json() == CURVES


class TestPublicGate:
    def test_quality_json_is_public_and_nothing_new_is(self) -> None:
        """Exactly one new path is reachable anonymously.

        ``/nowcast/quality.json`` rides the ``/nowcast/`` prefix that
        already existed; neither fitted file's route does, and neither
        must — the public instance READS both from the private one.
        ``/api/push/options`` (Phase G) is subscriber-facing and IS on the
        allow-list, beside the other three push routes.
        """
        assert _is_public_path("/nowcast/quality.json")
        assert not _is_public_path("/calibration/national_curves.json")
        assert not _is_public_path("/calibration/push_thresholds.json")
        assert _PUBLIC_PATHS == frozenset({
            "/healthz", "/forecast",
            "/api/push/config", "/api/push/options",
            "/api/push/subscribe", "/api/push/unsubscribe",
        })

    def test_the_gate_serves_quality_and_hides_the_curves(
        self, tmp_path: Path,
    ) -> None:
        config = _config(
            tmp_path,
            server={"public_mode": True, "api_key": "operator"},
            storage={"data_dir": tmp_path / "data", "corpus_dir": None},
        )
        path = quality_path(config)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(DOC))
        config.calibration.national_curves_path.write_text(json.dumps(CURVES))

        with TestClient(create_app(config, auto_start_scheduler=False)) as c:
            # Public: no bearer, 200.
            assert c.get("/nowcast/quality.json").status_code == 200
            # Private: no bearer, 404 — indistinguishable from no route.
            hidden = c.get("/calibration/national_curves.json")
            assert hidden.status_code == 404
            assert hidden.json() == {"detail": "Not Found"}
            # …and reachable with the operator key.
            unlocked = c.get(
                "/calibration/national_curves.json",
                headers={"Authorization": "Bearer operator"},
            )
            assert unlocked.status_code == 200


# ---------------------------------------------------------------------------
# A tiny event-loop runner, so the module needs no anyio/asyncio plugin
# ---------------------------------------------------------------------------


def anyio_run(coro):
    """Run one coroutine to completion on a fresh loop."""
    import asyncio

    return asyncio.run(_await(coro))


async def _await(coro):
    return await coro


def test_the_helper_runs_a_coroutine() -> None:
    async def answer() -> datetime:
        return datetime(2026, 9, 5, tzinfo=timezone.utc)

    assert anyio_run(answer()).year == 2026
