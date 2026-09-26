"""S11 — the onset AND push rule.

The push decision at a lead whose fitted row carries ``onset_threshold_pct``
is ``p_onset >= a AND p_post >= b`` (b = 0: onset alone decides), where
``p_onset`` is a SECOND, push-only model's probability that rain STARTS
within the lead. Everything else about the rule is unchanged, so the claims
under test are:

1. the engine's one predicate is the AND rule, and persistence, re-arm and
   the all-clear follow it; a missing ``p_onset`` falls back to the single
   threshold and is counted;
2. the two model slots refuse each other's documents;
3. the cycle scores ``p_onset`` on the same rows and writes the column;
4. every scorer (station_eval, served_rule) grades the rule the engine
   serves, and the nightly threshold refit refuses a table carrying it;
5. the served engine's pushes equal the combined-rule study's
   indicator-column replay for the same (a, b).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import structlog

pytest.importorskip("pyarrow")

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from dmi_nowcast_core import postprocess as pp  # noqa: E402
from dmi_nowcast_sidecar.config import Config  # noqa: E402
from dmi_nowcast_sidecar.push.engine import (  # noqa: E402
    INITIAL_STATE,
    Observation,
    Rules,
    evaluate,
)
from dmi_nowcast_sidecar.push.paths import (  # noqa: E402
    resolved_onset_model_path,
    resolved_thresholds_path,
)
from dmi_nowcast_sidecar.push.postprocess import (  # noqa: E402
    CyclePostprocess,
    PostprocessTable,
    build_cycle_postprocess,
    point_key,
)
from dmi_nowcast_sidecar.push.thresholds import ThresholdTable  # noqa: E402
from dmi_nowcast_sidecar.served_rule import (  # noqa: E402
    ServedRuleDecider,
    ServedRuleOptions,
)
from dmi_nowcast_sidecar.threshold_sweep import (  # noqa: E402
    build_tracks,
    replay_station,
)
from tests import test_push_postprocess as tpp  # noqa: E402
from tests import test_served_rule as tsr  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import combined_rule_study as study  # noqa: E402
import eta_revision_study as ers  # noqa: E402

UTC = timezone.utc
T0 = datetime(2026, 9, 5, 12, 0, tzinfo=UTC)
RULES = Rules(persistence_obs=1, rearm_after_min=60)


def _obs(i: int, *, p_post, p_onset, **over) -> Observation:
    kwargs = dict(
        radar_ts_utc=T0 + timedelta(minutes=10 * i),
        p_rain=0.10, eta_min=25.0, intensity_mm_h=1.0,
        observed_mm_h=0.0, forecast_now_mm_h=0.0,
        p_post=p_post, p_source="postprocess", p_onset=p_onset,
    )
    kwargs.update(over)
    return Observation(**kwargs)  # type: ignore[arg-type]


def _step(state, obs, a=26, b=35, single=45):
    return evaluate(
        state, obs, threshold_pct=b, quiet=None, tz="UTC",
        now_utc=obs.radar_ts_utc + timedelta(minutes=14), rules=RULES,
        onset_threshold_pct=a, single_threshold_pct=single,
    )


# ---------------------------------------------------------------------------
# 1. The engine
# ---------------------------------------------------------------------------


class TestEngine:
    @pytest.mark.parametrize("p_post, p_onset, action", [
        (0.50, 0.20, "none"),     # p_post over, onset under
        (0.30, 0.40, "none"),     # onset over, p_post under
        (0.35, 0.26, "notify"),   # both exactly at: >= fires
        (0.90, 0.90, "notify"),
    ])
    def test_it_fires_only_when_both_are_over(self, p_post, p_onset, action):
        decision = _step(INITIAL_STATE, _obs(0, p_post=p_post, p_onset=p_onset))
        assert decision.action == action
        assert _obs(0, p_post=p_post, p_onset=p_onset).p_decision_source == "onset_and"

    def test_b_zero_is_onset_only(self) -> None:
        obs = _obs(0, p_post=0.01, p_onset=0.30)
        assert _step(INITIAL_STATE, obs, b=0).action == "notify"
        assert _step(
            INITIAL_STATE, _obs(0, p_post=0.99, p_onset=0.20), b=0,
        ).action == "none"

    def test_a_missing_p_onset_falls_back_to_the_single_threshold(self) -> None:
        # b = 0 must NOT mean "fire on anything" when the onset half is gone.
        low = _obs(0, p_post=0.40, p_onset=None)
        assert low.p_decision_source == "postprocess"
        assert _step(INITIAL_STATE, low, b=0, single=45).action == "none"
        high = _obs(0, p_post=0.50, p_onset=None)
        assert _step(INITIAL_STATE, high, b=0, single=45).action == "notify"

    def test_without_an_onset_threshold_the_rule_is_unchanged(self) -> None:
        obs = _obs(0, p_post=0.50, p_onset=0.01)
        decision = evaluate(
            INITIAL_STATE, obs, threshold_pct=45, quiet=None, tz="UTC",
            now_utc=obs.radar_ts_utc, rules=RULES,
        )
        assert decision.action == "notify"

    def test_the_re_arm_follows_the_rule(self) -> None:
        state = _step(INITIAL_STATE, _obs(0, p_post=0.9, p_onset=0.9)).state
        # p_post stays high but the onset half is under: the rule is NOT
        # satisfied, so this is the dry clock running.
        actions = []
        for i in range(1, 7):
            decision = _step(state, _obs(i, p_post=0.9, p_onset=0.1))
            state, _ = decision.state, actions.append(decision.action)
        assert "notify" not in actions
        # 60 minutes after the first unsatisfied reading: re-armed, and an
        # observation that satisfies the rule fires at once.
        decision = _step(state, _obs(7, p_post=0.9, p_onset=0.9))
        assert decision.action == "notify"

    def test_a_satisfied_reading_before_the_mark_restarts_the_dry_clock(self):
        state = _step(INITIAL_STATE, _obs(0, p_post=0.9, p_onset=0.9)).state
        for i in range(1, 4):
            state = _step(state, _obs(i, p_post=0.9, p_onset=0.1)).state
        state = _step(state, _obs(4, p_post=0.9, p_onset=0.9)).state
        assert state.below_since_utc is None
        for i in range(5, 11):
            state = _step(state, _obs(i, p_post=0.9, p_onset=0.1)).state
        assert _step(state, _obs(11, p_post=0.9, p_onset=0.9)).action == "notify"

    def test_the_all_clear_follows_the_rule(self) -> None:
        state = _step(INITIAL_STATE, _obs(0, p_post=0.9, p_onset=0.9)).state
        first = _step(state, _obs(1, p_post=0.9, p_onset=0.1))
        assert first.action == "none"
        second = _step(first.state, _obs(2, p_post=0.9, p_onset=0.1))
        # Two readings where the rule is not satisfied, though p_post alone
        # would still read 90 %: retracted.
        assert second.action == "all_clear"


# ---------------------------------------------------------------------------
# 2. The two model slots
# ---------------------------------------------------------------------------


def _fit(target: str) -> pp.PostprocessModel:
    rows, truth = tpp._training_rows()
    return pp.fit_postprocess(
        rows, truth, tpp.LEADS, l2=1.0, design_leads=tpp.LEADS,
        fitted_at=datetime(2026, 9, 24, 9, 28, tzinfo=UTC), target=target,
    )


@pytest.fixture(scope="module")
def wet_model() -> pp.PostprocessModel:
    return _fit(pp.TARGET_WET)


@pytest.fixture(scope="module")
def onset_model() -> pp.PostprocessModel:
    return _fit(pp.TARGET_ONSET)


class TestSlots:
    def test_the_onset_slot_takes_an_onset_model(self, tmp_path, onset_model):
        table = PostprocessTable(
            tpp._write_model(tmp_path / "o.json", onset_model),
            target=pp.TARGET_ONSET,
        )
        table.load()
        assert table.active is True

    def test_the_onset_slot_refuses_a_wet_model(self, tmp_path, wet_model):
        table = PostprocessTable(
            tpp._write_model(tmp_path / "w.json", wet_model),
            target=pp.TARGET_ONSET,
        )
        with structlog.testing.capture_logs() as logs:
            table.load()
        assert table.active is False
        assert any(e["event"] == "push_postprocess_model_invalid" for e in logs)

    def test_the_display_slot_refuses_an_onset_model(self, tmp_path, onset_model):
        """A mis-installed file cannot silently change the site."""
        table = PostprocessTable(tpp._write_model(tmp_path / "o.json", onset_model))
        table.load()
        assert table.active is False

    def test_the_default_path(self, minimal_config: Config) -> None:
        assert resolved_onset_model_path(minimal_config) == (
            Path(minimal_config.storage.data_dir) / "postprocess_push.json"
        )
        minimal_config.push.onset_model_path = Path("/x/onset.json")
        assert resolved_onset_model_path(minimal_config) == Path("/x/onset.json")


# ---------------------------------------------------------------------------
# 3. The cycle
# ---------------------------------------------------------------------------


def _cycle(table, onset_table) -> CyclePostprocess:
    """``test_push_postprocess._cycle`` with the second table handed in."""
    products = tpp._products()
    geo = tpp._geo()
    from dmi_nowcast_sidecar import compute as compute_mod

    native = [geo.lonlat_to_grid(lon, lat) for _i, lat, lon, _n in tpp.POINTS]
    points = compute_mod._read_points(products, native)
    shared = [
        {"observed_mm_h": 0.2, "eta_min": 25.0, "intensity_mm_h": 1.8}
        for _ in points.pixels
    ]
    return build_cycle_postprocess(
        table,
        radar_ts_utc=tpp.RADAR_TS,
        generated_at_utc=tpp.GENERATED_AT,
        keys=[point_key(lat, lon) for _i, lat, lon, _n in tpp.POINTS],
        grid_features=tpp._grid_features(),
        raw_fractions=points.raw_fractions,
        shared=shared,
        station_radar_km=[40.0] * len(tpp.POINTS),
        leads=products.leads_min,
        season="summer",
        hour_utc=12,
        frame_age_min=14.0,
        onset_table=onset_table,
    )


class TestCycle:
    def test_p_onset_is_scored_on_the_same_rows(
        self, tmp_path, wet_model, onset_model,
    ) -> None:
        table = PostprocessTable(tpp._write_model(tmp_path / "w.json", wet_model))
        onset = PostprocessTable(
            tpp._write_model(tmp_path / "o.json", onset_model),
            target=pp.TARGET_ONSET,
        )
        table.load()
        onset.load()
        cycle = _cycle(table, onset)
        assert sorted(cycle.p_onset) == list(tpp.LEADS)
        assert all(len(v) == len(tpp.POINTS) for v in cycle.p_onset.values())
        # The same design matrix through the onset model, by hand.
        expected = onset.predict_row(
            {**cycle.rows[0], "observed_mm_h": 0.2, "eta_min": 25.0,
             "intensity_mm_h": 1.8},
        )
        got = cycle.onset_probability(tpp.HOME_LAT, tpp.HOME_LON, 30)
        assert got == pytest.approx(expected[30])
        columns = cycle.columns(tpp.HOME_LAT, tpp.HOME_LON)
        for lead in tpp.LEADS:
            name = pp.ONSET_COLUMN_TEMPLATE.format(lead=lead)
            assert columns[name] == cycle.p_onset[lead][0]
        # The display model's own numbers are untouched by the second one.
        assert cycle.p_post == _cycle(table, None).p_post

    def test_without_an_onset_model_there_is_no_column(
        self, tmp_path, wet_model,
    ) -> None:
        table = PostprocessTable(tpp._write_model(tmp_path / "w.json", wet_model))
        table.load()
        cycle = _cycle(table, PostprocessTable(None, target=pp.TARGET_ONSET))
        assert cycle.p_onset == {}
        assert cycle.onset_probability(tpp.HOME_LAT, tpp.HOME_LON, 30) is None
        assert not any(
            k.startswith(pp.ONSET_PREFIX)
            for k in cycle.columns(tpp.HOME_LAT, tpp.HOME_LON)
        )


# ---------------------------------------------------------------------------
# 4a. The thresholds table as the service reads it
# ---------------------------------------------------------------------------


def _lead_row(threshold: int, **extra) -> dict:
    return {**tsr._LEAD_ROW, "threshold_pct": threshold, **extra}


def _thresholds(path: Path, leads: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tsr._thresholds_doc(path)
    doc = json.loads(path.read_text())
    doc["leads"] = leads
    path.write_text(json.dumps(doc))
    return path


class TestThresholdTable:
    def test_onset_rule_headline_and_snapshot(self, tmp_path: Path) -> None:
        table = ThresholdTable(_thresholds(tmp_path / "t.json", {
            "20": _lead_row(0, onset_threshold_pct=26, single_threshold_pct=35),
            "30": _lead_row(45),
        }))
        table.load()
        assert table.effective(20) == (0, "table")
        assert table.onset_rule(20) == (26, 35)
        assert table.onset_rule(30) is None
        assert table.headline(20) == (26, "table", 26)
        # No onset model: every observation is on the single rule.
        assert table.headline(20, onset_active=False) == (35, "table", None)
        assert table.snapshot([20, 30]) == {
            "20": {"threshold_pct": 26, "source": "table",
                   "onset_threshold_pct": 26, "post_threshold_pct": 0},
            "30": {"threshold_pct": 45, "source": "table"},
        }


# ---------------------------------------------------------------------------
# 4b. The fan-out
# ---------------------------------------------------------------------------


class TestFanOut:
    def test_the_and_rule_decides_and_a_missing_p_onset_is_counted(
        self, tpp_config, tmp_path, monkeypatch, wet_model,
    ) -> None:
        _thresholds(resolved_thresholds_path(tpp_config), {
            "30": _lead_row(35, onset_threshold_pct=22, single_threshold_pct=45),
        })
        table = PostprocessTable(tpp._write_model(tmp_path / "w.json", wet_model))
        table.load()
        base = tpp._cycle(table)

        def cycle(p_post, p_onset, ts=tpp.RADAR_TS):
            return CyclePostprocess(
                radar_ts_utc=ts, generated_at_utc=ts + timedelta(minutes=14),
                keys=base.keys, rows=base.rows,
                p_post={30: p_post}, p_onset={30: p_onset},
                fitted_at_utc="x",
            )

        engine = tpp._CycleStub(cycle((0.9, 0.9, 0.9), (0.1, 0.9, 0.9)))
        service = tpp._service(tpp_config, engine, monkeypatch)
        tpp._subscribe(service, threshold_pct=None)
        lines, summary = tpp._run(service)
        # Home: p_post 90 % but p_onset 10 % under 22 %: no push.
        assert lines[0]["action"] == "none"
        assert lines[0]["p_source"] == "onset_and"
        assert lines[0]["onset_threshold_pct"] == 22
        assert summary["onset_rule_fallbacks"] == 0

        later = tpp.RADAR_TS + timedelta(minutes=10)
        engine.postprocess_latest = cycle((0.9, 0.9, 0.9), (0.3, 0.9, 0.9), later)
        lines, summary = tpp._run(service, later)
        assert lines[0]["action"] == "notify"

        # No p_onset for the point: judged on the single 45 %, counted.
        service.store.delete(tpp.ENDPOINT_A)
        tpp._subscribe(service, threshold_pct=None)
        latest = later + timedelta(minutes=10)
        engine.postprocess_latest = cycle((0.40, 0.9, 0.9), (None, 0.9, 0.9), latest)
        lines, summary = tpp._run(service, latest)
        assert summary["onset_rule_fallbacks"] == 1
        assert lines[0]["p_source"] == "postprocess"
        assert lines[0]["action"] == "none"   # 40 % < 45 %, though b is 35

    def test_an_override_is_a_single_threshold_rule(
        self, tpp_config, tmp_path, monkeypatch, wet_model,
    ) -> None:
        _thresholds(resolved_thresholds_path(tpp_config), {
            "30": _lead_row(35, onset_threshold_pct=90),
        })
        table = PostprocessTable(tpp._write_model(tmp_path / "w.json", wet_model))
        table.load()
        base = tpp._cycle(table)
        engine = tpp._CycleStub(CyclePostprocess(
            radar_ts_utc=tpp.RADAR_TS, generated_at_utc=tpp.GENERATED_AT,
            keys=base.keys, rows=base.rows,
            p_post={30: (0.5, 0.5, 0.5)}, p_onset={30: (0.0, 0.0, 0.0)},
        ))
        service = tpp._service(tpp_config, engine, monkeypatch)
        tpp._subscribe(service, threshold_pct=45)
        lines, summary = tpp._run(service)
        assert lines[0]["action"] == "notify"
        assert summary["onset_rule_fallbacks"] == 0


@pytest.fixture
def tpp_config(minimal_config: Config) -> Config:
    from dmi_nowcast_sidecar.config import PushConfig

    minimal_config.push = PushConfig(
        enabled=True, vapid_subject=tpp.SUBJECT, lead_options=[20, 30, 45, 60],
    )
    return minimal_config


# ---------------------------------------------------------------------------
# 4c. The live scoreboard
# ---------------------------------------------------------------------------


class TestStationEval:
    async def test_the_scoreboard_decides_on_the_and_rule(self, tmp_path) -> None:
        from tests import test_station_eval as tse

        points = tmp_path / "station_points.json"
        points.write_text(json.dumps(tse.POINTS))
        config = Config(
            home={"lat": 55.33, "lon": 10.32},  # type: ignore[arg-type]
            calibration={  # type: ignore[arg-type]
                "curves_path": tmp_path / "curves.json",
                "national_curves_path": tmp_path / "national_curves.json",
            },
            storage={  # type: ignore[arg-type]
                "data_dir": tmp_path / "data", "corpus_dir": tmp_path / "corpus",
            },
            lightning={"archive_dir": tmp_path / "strikes"},  # type: ignore[arg-type]
            station_eval={  # type: ignore[arg-type]
                "enabled": True, "points_file": str(points),
            },
        )

        class _Table:
            def maybe_reload(self):
                return False

            def effective(self, lead_min):
                return 0, "table"

            def onset_rule(self, lead_min):
                return (30, 45)

        by_station = {
            (55.614, 12.6454): {"p_post_30": 0.9, "p_onset_30": 0.5},
            (55.4735, 10.3297): {"p_post_30": 0.4, "p_onset_30": None},
        }
        post = SimpleNamespace(
            radar_ts_utc=tse.RADAR_TS,
            columns=lambda lat, lon: dict(by_station[(lat, lon)]),
        )
        from dmi_nowcast_sidecar.station_eval import StationEvalService

        service = StationEvalService(
            config, tse._engine(tse._products(0.9), postprocess=post),
            thresholds=_Table(),
        )
        await service.after_cycle(tse._cycle_result())
        summary = service.last_summary
        assert summary["onset_threshold_pct"] == 30
        assert summary["onset_rule_fallbacks"] == 1
        rows = {r["station_id"]: r for r in tse._read_partition(config)}
        # 06180: onset 50 % >= 30 and p_post >= 0: fires.
        assert rows["06180"]["action"] == "notify"
        assert rows["06180"]["p_onset_30"] == pytest.approx(0.5)
        # 06120: no p_onset, single 45 % against p_post 40 %: silent.
        assert rows["06120"]["action"] == "none"
        assert rows["06120"]["p_onset_30"] is None


# ---------------------------------------------------------------------------
# 4d. The quality page's scoreboard
# ---------------------------------------------------------------------------


ONSET = pp.ONSET_COLUMN_TEMPLATE.format(lead=tsr.LEAD)


def _served_tree(tmp_path: Path) -> Path:
    """The served-rule fixture tree, storing p_post AND p_onset."""
    rows = tsr._rows()
    tree = tsr._write_tree(tmp_path / "replay" / "decisions", rows)
    path = next(tree.glob("*.parquet"))
    table = pq.read_table(path)
    table = table.append_column(tsr.POST, pa.array(
        [0.9 if i % 3 else 0.5 for i, _ in enumerate(rows)], type=pa.float32(),
    ))
    table = table.append_column(ONSET, pa.array(
        [None if i % 7 == 0 else (0.4 if i % 2 else 0.1)
         for i, _ in enumerate(rows)],
        type=pa.float32(),
    ))
    pq.write_table(table, path)
    return tree


class TestServedRule:
    def test_the_page_grades_the_and_rule(self, tmp_path: Path) -> None:
        tree = _served_tree(tmp_path)
        thresholds = _thresholds(tmp_path / "t.json", {
            str(tsr.LEAD): _lead_row(
                60, onset_threshold_pct=30, single_threshold_pct=70,
            ),
        })
        options = ServedRuleOptions(
            decisions_dirs=[tree], thresholds_path=thresholds,
            lead_min=tsr.LEAD, probability_source="postprocess",
            postprocess_model=None, onset_model=None,
        )
        stored = tsr._stored(tree)
        decider = ServedRuleDecider(options)
        got = decider(stored)

        tracks, _ = build_tracks(
            stored, [tsr.LEAD], column_for=lambda _l: tsr.POST,
            onset_column_for=lambda _l: ONSET,
        )
        expected = {}
        for station, track in tracks.items():
            warnings = replay_station(
                track, 0, 60, persistence_obs=1, rearm_after_min=60,
                raining_now_mm_h=0.5, with_probability=True,
                with_all_clear=True, onset_threshold_pct=30,
                single_threshold_pct=70,
            )
            if warnings:
                expected[station] = warnings
        assert got == expected and got
        # Not p_post alone against its half: that is a different list.
        alone = ServedRuleDecider(dataclass_replace(
            options, thresholds_path=_thresholds(tmp_path / "s.json", {
                str(tsr.LEAD): _lead_row(60),
            }),
        ))(stored)
        assert alone != got
        assert decider.stats["threshold_pct"] == 30
        assert decider.stats["onset_threshold_pct"] == 30
        assert decider.stats["post_threshold_pct"] == 60
        assert decider.stats["rows_onset_fallback"] > 0

    def test_the_curve_rollback_grades_the_single_rule(self, tmp_path) -> None:
        tree = _served_tree(tmp_path)
        options = ServedRuleOptions(
            decisions_dirs=[tree],
            thresholds_path=_thresholds(tmp_path / "t.json", {
                str(tsr.LEAD): _lead_row(60, onset_threshold_pct=30),
            }),
            lead_min=tsr.LEAD, probability_source="curve",
        )
        decider = ServedRuleDecider(options)
        assert decider.onset is None
        assert decider.stats["onset_threshold_pct"] is None


def dataclass_replace(options, **changes):
    import dataclasses

    return dataclasses.replace(options, **changes)


# ---------------------------------------------------------------------------
# 4e. The refits
# ---------------------------------------------------------------------------


class TestRefits:
    def test_the_threshold_refit_refuses_an_onset_table(self, tmp_path) -> None:
        from dmi_nowcast_sidecar.quality_job import run_threshold_fit

        out = _thresholds(tmp_path / "push_thresholds.json", {
            "30": _lead_row(35, onset_threshold_pct=22),
        })
        before = out.read_bytes()
        called = []
        result = run_threshold_fit(
            {"thresholds_out": str(out), "options": {}},
            fitter=lambda options: called.append(options) or {},
        )
        assert called == []
        assert "onset AND rule" in result["thresholds_skipped"]
        assert out.read_bytes() == before

    def test_a_single_threshold_table_is_still_refitted(self, tmp_path) -> None:
        from dmi_nowcast_sidecar.quality_job import run_threshold_fit

        out = _thresholds(tmp_path / "push_thresholds.json", {
            "30": _lead_row(35),
        })
        result = run_threshold_fit(
            {"thresholds_out": str(out), "options": {}},
            fitter=lambda options: {},
        )
        # Past the guard: it tried to fit (and failed on the empty stub).
        assert "thresholds_skipped" not in result
        assert result.get("thresholds_error")

    def test_the_postprocess_refit_never_touches_an_onset_model(
        self, tmp_path, onset_model,
    ) -> None:
        from dmi_nowcast_sidecar.postprocess_fit import refit_skip_reason

        # Even a logistic at-gauge v1 document — the nightly fit's own
        # shape — is refused once its target is onset.
        out = tpp._write_model(tmp_path / "postprocess_push.json", onset_model)
        reason = refit_skip_reason(
            out, SimpleNamespace(model=onset_model.kind, design=onset_model.spec.version),
        )
        assert reason is not None and "target=onset" in reason


# ---------------------------------------------------------------------------
# 4f. Sync
# ---------------------------------------------------------------------------


class TestSync:
    def test_the_onset_model_lands_where_the_cycle_reads_it(self, tmp_path):
        from tests import test_quality_publication as tqp

        from dmi_nowcast_sidecar.sync import (
            ONSET_MODEL_FILE,
            build_artifact_sync,
            target_path,
        )

        config = tqp._sync_config(tmp_path, files=[ONSET_MODEL_FILE])
        assert target_path(config, ONSET_MODEL_FILE) == (
            resolved_onset_model_path(config)
        )
        nudged = []
        engine = SimpleNamespace(
            postprocess=None,
            onset_postprocess=SimpleNamespace(note_changed=lambda: nudged.append(1)),
        )
        sync = build_artifact_sync(config, engine)
        model = {"schema_version": 1, "target": "onset", "models": {}}
        peer = tqp._Peer({ONSET_MODEL_FILE: [tqp._ok(model)]})
        sync._client = peer.client()
        sync._owns_client = False
        result = tqp.anyio_run(sync.sync_once())
        assert result.ok and result.updated == 1
        assert json.loads(resolved_onset_model_path(config).read_text()) == model
        assert nudged == [1]


# ---------------------------------------------------------------------------
# 5. Replay equivalence with the study
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("a, b", [(26, 0), (22, 35), (28, 50), (30, 55)])
def test_the_served_engine_replays_like_the_studys_indicator_column(a, b):
    """For a synthetic station series, the pushes the served engine makes
    under (a, b) are exactly the study's indicator-column replay."""
    rng = np.random.default_rng(a * 100 + b)
    n = 400
    lead = 30
    radar = np.arange(n, dtype=np.int64) * 10 * ers.US_PER_MIN
    radar[250:] += 90 * ers.US_PER_MIN  # a coverage gap
    p_post = rng.uniform(0, 1, n)
    p_onset = rng.uniform(0, 0.6, n)
    # Rows with no probability at all are skipped by both.
    gone = rng.uniform(size=n) < 0.05
    p_post[gone] = np.nan
    p_onset[gone] = np.nan
    arrays = {
        "radar": radar, "gen": radar + 14 * ers.US_PER_MIN,
        "eta": np.where(rng.uniform(size=n) < 0.5, 15.0, np.nan),
        "intensity": np.full(n, np.nan),
        "observed": np.where(rng.uniform(size=n) < 0.1, 2.0, 0.0),
        "forecast": np.full(n, np.nan),
    }
    family = "onset" if b == 0 else "and"
    column = study.rule_column(p_onset, p_post, family, a, b)
    want, _ = ers.replay_pushes({**arrays, f"p{lead}": column}, lead, 50)

    def opt(v):
        return None if v != v else float(v)

    rows = [
        {
            "radar_ts": ers.to_dt(radar[i]), "generated_at": ers.to_dt(arrays["gen"][i]),
            "station_id": "X", "eta_min": opt(arrays["eta"][i]),
            "intensity_mm_h": None, "observed_mm_h": opt(arrays["observed"][i]),
            "forecast_now_mm_h": None,
            "p_post_30": opt(p_post[i]), "p_onset_30": opt(p_onset[i]),
        }
        for i in range(n)
    ]
    tracks, _ = build_tracks(
        rows, [lead], column_for=lambda _l: "p_post_30",
        onset_column_for=lambda _l: "p_onset_30",
    )
    got = replay_station(
        tracks["X"], 0, b, persistence_obs=1, rearm_after_min=60,
        onset_threshold_pct=a, single_threshold_pct=99,
    )
    sent = {ers.to_dt(arrays["gen"][i]) for i in want}
    assert [w[0] for w in got] == sorted(sent)
    assert len(want) > 3
