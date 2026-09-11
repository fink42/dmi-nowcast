"""Re-deciding the quality page's warning scoreboard under the served rule.

The two decision trees are generated with a fixed subscriber row — 40 %
at 30 min — and the push service warns at the nightly fitted threshold on
the gauge-trained ``p_post``. So the page's scoreboard measured a rule
nobody is subscribed to. :mod:`dmi_nowcast_sidecar.served_rule` re-derives
the warnings at report build time from the probabilities the rows already
carry, without rewriting a byte of the trees.

There is one claim worth testing, and it is the same shape as the gauge
curve's:

    **the page and the push service must be one rule.**

So the assertions below compare the module's output against a DIRECT
``replay_station`` run at the served threshold on a probability column
assembled here — from the model's own ``predict``, not through the
filler — rather than against a second implementation of the same loop.
A producer that read the wrong column, applied the wrong threshold, or
skipped the engine's per-row fallback would then disagree rather than
agree by construction.

The rest guards the seams: the report's key set is the population, the
fallback rows are counted, and a deployment that serves the curve decides
on the curve.

The fixture is deliberately small and hand-built — two stations, one day
of 10-minute frames, no radar, no STEPS. Station A carries the H-P
feature columns (so the model can speak for it); station B carries none
(so every one of its rows takes the engine's fallback to ``p_rain``),
which is exactly the mix the real archive is.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")
np = pytest.importorskip("numpy")

import pyarrow.parquet as pq  # noqa: E402

from dmi_nowcast_core import postprocess as pp  # noqa: E402
from dmi_nowcast_core.warning_score import (  # noqa: E402
    decision_table,
    p_rain_column,
)
from dmi_nowcast_sidecar.served_rule import (  # noqa: E402
    ServedRuleDecider,
    ServedRuleOptions,
    resolve_threshold,
)
from dmi_nowcast_sidecar.threshold_sweep import build_tracks, replay_station

UTC = timezone.utc
DAY = datetime(2026, 9, 3, tzinfo=UTC)
LEADS = (10, 20, 30, 45, 60)
DESIGN_LEADS = LEADS
LEAD = 30
#: The served table's pick for the scoreboard's horizon. Deliberately far
#: from the trees' own 40 %, so a producer reading the stored actions
#: cannot land on the same warning list by luck.
SERVED_PCT = 60
#: The featured station; ``STATION_B`` carries no features at all.
STATION_A = "06180"
STATION_B = "06181"
STATIONS = (STATION_A, STATION_B)
#: 00:00–04:00 inclusive, the fullRange cadence.
FRAMES = 25
POST = pp.post_column(LEAD)
CURVE = p_rain_column(LEAD)


# ---------------------------------------------------------------------------
# The model, and the feature rows it is fitted on
# ---------------------------------------------------------------------------


def _features(fraction: float) -> dict[str, float]:
    """One row's feature columns from a single knob.

    ``fraction`` is how much rain the ensemble sees; every correlated
    column moves with it, which is what a real feature row does. Both the
    model and the fixture rows come through here, so the model is asked to
    score the distribution it was fitted on.
    """
    out: dict[str, float] = {
        pp.raw_fraction_column(lead): fraction for lead in DESIGN_LEADS
    }
    out.update({
        "obs_max_5km_mm_h": 3.0 * fraction,
        "up_max_20km_mm_h": 4.0 * fraction,
        "up_max_40km_mm_h": 5.0 * fraction,
        "up_dist_km": 40.0 * (1.0 - fraction),
        "up_wet_frac_40km": fraction,
        "bulk_kmh": 35.0,
        "bulk_dir_deg": 250.0,
        "local_speed_kmh": 30.0,
        "stalled_share": 0.08,
        "frame_age_min": 14.0,
        "station_radar_km": 40.0,
    })
    return out


def _design(samples: list[dict[str, float]], stamps: np.ndarray) -> dict:
    """Feature dicts → the column mapping ``PostprocessModel`` reads."""
    columns: dict = {
        name: np.array(
            [float(s.get(name, np.nan)) for s in samples], dtype=np.float64,
        )
        for name in pp.feature_source_columns(DESIGN_LEADS)
    }
    columns["season"] = pp.seasons_from_epoch(stamps)
    columns["hour_utc"] = pp.hours_from_epoch(stamps).astype(np.float64)
    return columns


@pytest.fixture(scope="module")
def model() -> "pp.PostprocessModel":
    """A model whose answers span the threshold, fitted on fixture-shaped rows.

    The truth is stochastic in ``fraction`` — a row at 0.4 is wet 40 % of
    the time — so the fit lands on both sides of 60 % instead of
    collapsing to one answer, which is what makes "the same warnings"
    say something.
    """
    rng = np.random.default_rng(7)
    n = 800
    fractions = rng.random(n)
    samples = [_features(float(f)) for f in fractions]
    stamps = np.full(
        n, int(DAY.replace(hour=8).timestamp()), dtype=np.int64,
    )
    wet = (rng.random(n) < fractions).astype(float)
    truth = {lead: (wet, np.ones(n, dtype=bool)) for lead in LEADS}
    return pp.fit_postprocess(
        _design(samples, stamps), truth, LEADS, l2=1.0,
        design_leads=DESIGN_LEADS,
        fitted_at=datetime(2026, 9, 11, 3, 40, tzinfo=UTC),
    )


# ---------------------------------------------------------------------------
# The decision tree
# ---------------------------------------------------------------------------


def _fraction(index: int) -> float:
    """A deterministic ramp over the day, with a dry spell in the middle.

    The dry spell is what makes the re-armed second half of the day a
    second warning rather than a continuation of the first, so the
    engine's hysteresis actually takes part.
    """
    if 8 <= index <= 15:
        return 0.05
    return min(1.0, 0.05 + 0.09 * index)


def _curve(index: int) -> float:
    """The served curve probability: HIGH everywhere, and unlike the model's.

    High on purpose. Station B decides on this through the engine's
    fallback, so a producer that silently skipped B (or that decided B on
    a null) would send no warnings there and fail loudly.
    """
    return 0.05 if 8 <= index <= 15 else 0.95


def _rows(*, with_features: bool = True) -> list[dict]:
    """One decision row per station per frame, in the shared schema.

    ``action`` is the tree's own decision at the 40 % rule and is never
    what this module reads — it is set to a value the served rule
    disagrees with wherever it can, so a fallback to the stored column
    would show up immediately.
    """
    out: list[dict] = []
    for index in range(FRAMES):
        radar_ts = DAY + timedelta(minutes=10 * index)
        fraction = _fraction(index)
        curve = _curve(index)
        for station in STATIONS:
            row: dict = {
                "radar_ts": radar_ts,
                # Live latency, as both writers stamp it.
                "generated_at": radar_ts + timedelta(minutes=14),
                "station_id": station,
                "p_rain": curve,
                "eta_min": 22.0,
                "intensity_mm_h": 1.2,
                # Dry at the point: the "already raining" arm must never
                # silence a warning in this fixture.
                "observed_mm_h": 0.0,
                "forecast_now_mm_h": 0.0,
                # Deliberately wrong for the served rule.
                "action": "notify",
                "armed_after": True,
                "streak_after": 0,
                "threshold_pct": 40,
                p_rain_column(LEAD): curve,
            }
            for lead in LEADS:
                row.setdefault(p_rain_column(lead), curve)
            if with_features and station == STATION_A:
                row.update(_features(fraction))
            out.append(row)
    return out


def _write_tree(directory: Path, rows: list[dict]) -> Path:
    """The rows as one day's parquet, features and all, no ``p_post``."""
    import pyarrow as pa

    table = decision_table(rows, LEADS)
    for field in pp.feature_schema(LEADS):
        table = table.append_column(field, pa.array(
            [row.get(field.name) for row in rows], type=field.type,
        ))
    assert POST not in table.schema.names, "the fixture must not store p_post"
    directory.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, directory / f"{DAY.date().isoformat()}.parquet")
    return directory


def _stored(tree: Path) -> list[dict]:
    """The rows as they come BACK off disk.

    Every expectation is built from these rather than from the dicts that
    were written, because a decision parquet is float32: 0.95 reads back
    as 0.949999988079071, and the model's answer for a float32 feature is
    not its answer for the float64 one. Comparing against the round-tripped
    values makes the assertions exact instead of approximate — and an
    approximate one could not tell a threshold crossing from a rounding
    difference, which is the thing under test.
    """
    out: list[dict] = []
    for path in sorted(tree.glob("*.parquet")):
        out.extend(pq.read_table(path).to_pylist())
    return out


#: One well-formed lead row, as ``sweep_thresholds.py`` writes it.
_LEAD_ROW = {
    "threshold_pct": SERVED_PCT,
    "insufficient": False,
    "f1": 0.41, "precision": 0.47, "recall": 0.36, "far": 0.53, "csi": 0.26,
    "warnings": 212, "hits": 99, "false_alarms": 113, "misses": 160,
    "late": 17,
    "plateau": [55, 65], "radar_plateau": [50, 70],
    "agrees_with_radar": True,
}


def _thresholds_doc(path: Path, *, pct: int = SERVED_PCT) -> Path:
    """A served threshold document with a real pick at the report's lead.

    Structurally complete on purpose: ``load_thresholds`` refuses half a
    document, so a shortcut here would be read as "not fitted yet" and the
    test would silently be about the fallback.
    """
    path.write_text(json.dumps({
        "schema_version": 1,
        "fitted_at_utc": "2026-09-11T03:40:00+00:00",
        "objective": {
            "metric": "f1", "min_useful_lead_min": 5.0, "plateau_frac": 0.95,
            "min_warnings": 30, "rearm_after_min": 60, "persistence_obs": 1,
            "tolerance_min": 10, "dry_min": 60, "onset_min_mm": 0.2,
            "probability_column": "p_post_{lead}",
        },
        "window": {
            "from": "2026-07-01T00:00:00+00:00",
            "to": "2026-09-01T00:00:00+00:00",
            "days": 62, "stations": 97, "rows": 1841203,
        },
        "fallback_threshold_pct": 40,
        "leads": {str(LEAD): {**_LEAD_ROW, "threshold_pct": pct}},
    }), encoding="utf-8")
    return path


def _model_file(path: Path, model: "pp.PostprocessModel") -> Path:
    path.write_text(model.dumps(), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# The expected answer, built independently of the module under test
# ---------------------------------------------------------------------------


def _expected_column(rows: list[dict], model) -> dict[tuple, float | None]:
    """``{(radar_ts, station): probability the rule should decide on}``.

    The model's own ``predict`` on a feature dict assembled here — NOT
    through ``ProbabilityFiller`` — with the engine's per-row fallback to
    the curve for every row the model cannot speak for. This is the
    second opinion the module is checked against. ``rows`` must be the
    round-tripped ones (:func:`_stored`).
    """
    featured = [
        r for r in rows
        if r.get(pp.raw_fraction_column(LEAD)) is not None
    ]
    stamps = np.array(
        [int(r["generated_at"].timestamp()) for r in featured], dtype=np.int64,
    )
    predicted = model.predict(_design(featured, stamps))[LEAD]
    out: dict[tuple, float | None] = {}
    for row, value in zip(featured, np.asarray(predicted, dtype=np.float64)):
        out[(row["radar_ts"], row["station_id"])] = float(value)
    for row in rows:
        key = (row["radar_ts"], row["station_id"])
        if key not in out:
            out[key] = row[CURVE]
    return out


def _expected_warnings(
    rows: list[dict], probability: dict[tuple, float | None], pct: int,
) -> dict[str, list]:
    """``replay_station`` at ``pct``, on a column set here by hand."""
    staged = [
        {**row, "_p": probability[(row["radar_ts"], row["station_id"])]}
        for row in rows
    ]
    tracks, _frames = build_tracks(
        staged, [LEAD], column_for=lambda _lead: "_p",
    )
    out: dict[str, list] = {}
    for station, track in tracks.items():
        warnings = replay_station(
            track, 0, pct,
            persistence_obs=1, rearm_after_min=60, raining_now_mm_h=0.5,
            with_probability=True,
        )
        if warnings:
            out[station] = warnings
    return out


def _options(tmp_path: Path, model, **overrides) -> ServedRuleOptions:
    """Write the fixture tree and describe the rule to re-decide it under."""
    rows = overrides.pop("rows", None) or _rows()
    tree = _write_tree(tmp_path / "replay" / "decisions", rows)
    defaults = dict(
        decisions_dirs=[tree],
        thresholds_path=_thresholds_doc(tmp_path / "push_thresholds.json"),
        lead_min=LEAD,
        probability_source="postprocess",
        postprocess_model=_model_file(tmp_path / "model.json", model),
        design_leads=DESIGN_LEADS,
        persistence_obs=1,
        rearm_after_min=60,
        raining_now_mm_h=0.5,
    )
    defaults.update(overrides)
    return ServedRuleOptions(**defaults)


# ---------------------------------------------------------------------------
# The threshold
# ---------------------------------------------------------------------------


class TestThreshold:
    def test_a_fitted_pick_is_read_from_the_document(self, tmp_path: Path) -> None:
        options = ServedRuleOptions(
            thresholds_path=_thresholds_doc(tmp_path / "t.json"),
            lead_min=LEAD,
        )
        assert resolve_threshold(options) == (SERVED_PCT, "table")

    def test_a_lead_the_table_cannot_speak_for_falls_back(
        self, tmp_path: Path,
    ) -> None:
        options = ServedRuleOptions(
            thresholds_path=_thresholds_doc(tmp_path / "t.json"),
            lead_min=45,
        )
        threshold, source = resolve_threshold(options)
        assert (threshold, source) == (40, "fallback")

    def test_no_document_falls_back_to_the_configured_percent(
        self, tmp_path: Path,
    ) -> None:
        """A deployment with no fitted table is on the station_eval rule."""
        for path in (None, tmp_path / "missing.json"):
            options = ServedRuleOptions(
                thresholds_path=path, lead_min=LEAD,
                fallback_threshold_pct=35,
            )
            assert resolve_threshold(options) == (35, "config")


# ---------------------------------------------------------------------------
# The rule itself
# ---------------------------------------------------------------------------


class TestServedRuleDecider:
    """The hook, against a directly-replayed second opinion."""

    @staticmethod
    def _tree(tmp_path: Path) -> Path:
        return tmp_path / "replay" / "decisions"

    def test_the_warnings_are_the_engines_at_the_served_threshold(
        self, tmp_path: Path, model,
    ) -> None:
        """The one claim: the page and the push rule are one computation."""
        options = _options(tmp_path, model)
        stored = _stored(self._tree(tmp_path))
        decider = ServedRuleDecider(options)
        got = decider(stored)

        expected = _expected_warnings(
            stored, _expected_column(stored, model), SERVED_PCT,
        )
        assert got == expected
        assert got, "the fixture sent no warnings at all — it proves nothing"
        # Both stations warn: A on the model, B through the fallback.
        assert set(got) == set(STATIONS)
        assert decider.stats["threshold_pct"] == SERVED_PCT
        assert decider.stats["threshold_source"] == "table"
        assert decider.stats["probability"] == "postprocess"
        assert decider.stats["probability_column"] == POST
        assert decider.stats["scored"] == "re-decided"
        assert decider.stats["error"] is None

    def test_the_stored_actions_are_not_what_is_scored(
        self, tmp_path: Path, model,
    ) -> None:
        """Every row says ``notify``; the served rule sends far fewer."""
        options = _options(tmp_path, model)
        stored = _stored(self._tree(tmp_path))
        got = ServedRuleDecider(options)(stored)
        sent = sum(len(v) for v in got.values())
        assert 0 < sent < len(stored)

    def test_rows_outside_the_reports_set_are_ignored(
        self, tmp_path: Path, model,
    ) -> None:
        """The report's keys are the population — both halves, one sample.

        The tree here holds the whole day; the report hands over only the
        first half of it. A producer that scored what it read rather than
        what it was given would warn on frames the rest of the document
        knows nothing about.
        """
        options = _options(tmp_path, model)
        stored = _stored(self._tree(tmp_path))
        cutoff = DAY + timedelta(minutes=10 * 12)
        narrowed = [r for r in stored if r["radar_ts"] < cutoff]

        decider = ServedRuleDecider(options)
        got = decider(narrowed)

        assert decider.stats["rows_loaded"] == len(stored)
        assert decider.stats["rows_matched"] == len(narrowed)
        assert got == _expected_warnings(
            narrowed, _expected_column(narrowed, model), SERVED_PCT,
        )
        for warnings in got.values():
            for sent, _eta, _p in warnings:
                assert sent < cutoff + timedelta(minutes=14)

    def test_the_fallback_rows_are_counted(
        self, tmp_path: Path, model,
    ) -> None:
        """One count per row that decided on the curve instead of the model.

        Station B carries no features at all, so the model cannot speak
        for any of its rows and every one of them takes the engine's
        per-observation fallback — which is the case the fallback exists
        for, and the case that must not be silently dropped.
        """
        options = _options(tmp_path, model)
        stored = _stored(self._tree(tmp_path))
        decider = ServedRuleDecider(options)
        decider(stored)
        assert decider.stats["rows_fallback"] == FRAMES     # all of station B
        assert decider.stats["rows_matched"] == len(stored)
        # The filler's own bookkeeping travels too: B's rows are the ones
        # it could not fill, and they SURVIVED rather than being dropped.
        assert decider.stats["fill"]["dropped"] == FRAMES
        assert decider.stats["fill"]["computed"] == FRAMES
        assert decider.stats["fill"]["rows"] == len(stored)

    def test_a_station_with_no_features_still_warns(
        self, tmp_path: Path, model,
    ) -> None:
        """The fallback is the point: one model outage must not silence it."""
        options = _options(tmp_path, model)
        stored = _stored(self._tree(tmp_path))
        got = ServedRuleDecider(options)(stored)
        assert got.get(STATION_B), "the fallback sent nothing"
        # And it fired on the curve's number, which is what it decided on.
        curve_by_key = {
            (r["radar_ts"], r["station_id"]): r[CURVE] for r in stored
        }
        for sent, _eta, probability in got[STATION_B]:
            radar_ts = sent - timedelta(minutes=14)
            assert probability == curve_by_key[(radar_ts, STATION_B)]

    def test_probability_source_curve_decides_on_p_rain(
        self, tmp_path: Path, model,
    ) -> None:
        """A deployment serving the curve is measured on the curve.

        No model is read at all, so the answer is ``p_rain_<lead>`` for
        every row — and with one curve shared by both stations, the rule
        is identical at both.
        """
        options = _options(
            tmp_path, model,
            probability_source="curve", postprocess_model=None,
        )
        stored = _stored(self._tree(tmp_path))
        decider = ServedRuleDecider(options)
        got = decider(stored)

        assert decider.stats["probability"] == "curve"
        assert decider.stats["probability_column"] == CURVE
        assert decider.stats["rows_fallback"] == 0
        assert "fill" not in decider.stats
        expected = _expected_warnings(
            stored,
            {(r["radar_ts"], r["station_id"]): r[CURVE] for r in stored},
            SERVED_PCT,
        )
        assert got == expected
        assert [w[:2] for w in got[STATION_A]] == [
            w[:2] for w in got[STATION_B]
        ]

    def test_the_threshold_is_the_tables_and_not_the_fallback(
        self, tmp_path: Path, model,
    ) -> None:
        """Change the served percent, change the warnings. Nothing else moves."""
        options = _options(tmp_path, model)
        stored = _stored(self._tree(tmp_path))
        at_60 = ServedRuleDecider(options)(stored)

        _thresholds_doc(options.thresholds_path, pct=95)
        strict = ServedRuleDecider(options)
        at_95 = strict(stored)
        assert strict.stats["threshold_pct"] == 95
        assert sum(map(len, at_95.values())) < sum(map(len, at_60.values()))

    def test_a_stored_p_post_survives_without_a_model(
        self, tmp_path: Path, model,
    ) -> None:
        """A live partition that stores the column must not be pushed onto
        the fallback just because no model file was configured."""
        import pyarrow as pa

        rows = _rows()
        tree = _write_tree(tmp_path / "replay" / "decisions", rows)
        path = next(tree.glob("*.parquet"))
        table = pq.read_table(path)
        # Every row stores a probability the curve could never produce, so
        # a producer that fell back would come back with different
        # warnings and a non-zero fallback count.
        table = table.append_column(POST, pa.array(
            [0.99 if r["station_id"] == STATION_A else 0.01 for r in rows],
            type=pa.float32(),
        ))
        pq.write_table(table, path)

        options = ServedRuleOptions(
            decisions_dirs=[tree],
            thresholds_path=_thresholds_doc(tmp_path / "push_thresholds.json"),
            lead_min=LEAD,
            probability_source="postprocess",
            postprocess_model=None,
        )
        stored = _stored(tree)
        decider = ServedRuleDecider(options)
        got = decider(stored)

        assert decider.stats["rows_fallback"] == 0
        assert "fill" not in decider.stats
        assert set(got) == {STATION_A}      # B sits at 1 %, under 60 %
        assert got == _expected_warnings(
            stored,
            {(r["radar_ts"], r["station_id"]): r[POST] for r in stored},
            SERVED_PCT,
        )

    def test_an_empty_report_set_reads_nothing(
        self, tmp_path: Path, model,
    ) -> None:
        """No rows to score is not a reason to read a season of parquet."""
        decider = ServedRuleDecider(_options(tmp_path, model))
        assert decider([]) == {}
        assert decider.stats["rows_loaded"] == 0

    def test_a_broken_tree_costs_the_scoreboard_not_the_build(
        self, tmp_path: Path, model, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Total by construction: this runs inside the nightly build."""
        options = _options(tmp_path, model)
        stored = _stored(self._tree(tmp_path))
        decider = ServedRuleDecider(options)
        monkeypatch.setattr(
            "dmi_nowcast_sidecar.threshold_sweep.load_decisions",
            lambda *a, **k: (_ for _ in ()).throw(OSError("disk went away")),
        )
        assert decider(stored) == {}
        assert "OSError" in decider.stats["error"]

    def test_the_coverage_gap_splits_the_replay_the_way_the_report_does(
        self, tmp_path: Path, model,
    ) -> None:
        """A gap in the rows re-arms the subscription, as it does everywhere.

        The state machine has to reset at exactly the instants the
        report's coverage runs break, or a warning could be scored inside
        a run the engine never re-armed for.
        """
        rows = [
            r for r in _rows()
            if r["radar_ts"] < DAY + timedelta(minutes=40)
            or r["radar_ts"] >= DAY + timedelta(minutes=170)
        ]
        options = _options(tmp_path, model, rows=rows, coverage_gap_min=20)
        stored = _stored(self._tree(tmp_path))
        got = ServedRuleDecider(options)(stored)
        assert got == _expected_warnings(
            stored, _expected_column(stored, model), SERVED_PCT,
        )
