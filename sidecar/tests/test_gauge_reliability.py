"""The quality page's gauge curve, scored on the probability the site serves.

Until 2026-09-11 that curve came from the station calibration corpus:
``raw_prob`` pushed through the served isotonic curve, graded against the
corpus's own ``gauge_outcome``. It was an honest diagram of the *curve* —
and the site had stopped serving the curve. The page therefore showed a
line far below the diagonal for a probability nobody is shown, while the
model it was replaced by verifies on the diagonal at the gauges.

This suite is about the replacement, and there is really only one claim
worth testing:

    **the page and the benchmark must be one computation.**

``scripts/benchmark_report.py``'s Layer B is what decides whether a model
ships. If the /quality page reached the same question by a second route —
its own loader, its own idea of "wet within L", its own binning — then a
disagreement between the two would be unfalsifiable: neither number would
be wrong, they would just be about different samples. So the test below
runs Layer B over the fixture, runs the page's producer over the same
fixture, and asserts bin-for-bin equality.

The rest guards the edges: a row that carries no model probability is
excluded and counted rather than imputed, the curve column remains
scoreable for a deployment that serves it, and the report builder both
prefers this block over the corpus fit and carries it over on an hourly
live refresh.

The fixture is ``test_benchmark_report``'s, reused deliberately — sharing
the fixture is what makes "the same rows" true rather than approximately
true. Its probabilities are binary fractions (0.125, 0.5, 0.875) so a
float32 round-trip cannot move one into a neighbouring bin.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")
np = pytest.importorskip("numpy")

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import benchmark_report as report_module  # noqa: E402

from dmi_nowcast_core import postprocess as pp  # noqa: E402
from dmi_nowcast_core.warning_score import decision_table  # noqa: E402
from dmi_nowcast_sidecar.decision_rows import (  # noqa: E402
    build_gauge_grid,
    column_template,
    decision_window,
    load_probabilities,
    recode_stations,
)
from dmi_nowcast_sidecar.gauge_reliability import (  # noqa: E402
    GaugeReliabilityOptions,
    gauge_reliability_from_decisions,
)

from tests.test_benchmark_report import (  # noqa: E402
    DAYS,
    FIRST_FRAME_MIN,
    FRAMES_PER_DAY,
    LAST_FRAME_MIN,
    LEADS,
    STATION_A,
    STATIONS,
    _at,
    _hhmm,
    _write_gauge,
)

#: The model's probability per station and frame. Deliberately NOT the
#: curve's: a producer that read ``p_rain_<lead>`` by mistake would then
#: land in different bins and fail loudly rather than agree by accident.
def _p_post(station: str, hhmm: str) -> tuple[float, float]:
    if station == STATION_A:
        wet = hhmm in ("07:30", "07:40", "07:50")
        return (0.75 if wet else 0.25), (0.75 if wet else 0.25)
    return 0.25, 0.25


#: The design leads the model reads, i.e. the national product leads.
DESIGN_LEADS = (10, 20, 30, 45, 60)


def _features_from(fraction: float, near: float) -> dict[str, float]:
    """One row's feature columns from two knobs.

    ``fraction`` is how much rain the ensemble sees upstream (it drives
    every correlated column, which is what real feature rows do); ``near``
    separates the two stations. Both the model below and the fixture rows
    come through here, so the model is fitted on the distribution it is
    later asked to score — without that its answers saturate at 0 and
    "the filled curve equals the stored curve" would be true of a single
    bin.
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
        "local_speed_kmh": 30.0 + 10.0 * near,
        "stalled_share": 0.08,
        "frame_age_min": 14.0,
        "station_radar_km": 40.0 + 30.0 * near,
    })
    return out


def _design(samples: list[dict[str, float]], hour: float) -> dict[str, object]:
    """Feature dicts → the column mapping ``PostprocessModel`` reads."""
    columns: dict[str, object] = {
        name: np.array(
            [float(s.get(name, np.nan)) for s in samples], dtype=np.float64,
        )
        for name in pp.feature_source_columns(DESIGN_LEADS)
    }
    columns["season"] = np.array(["winter"] * len(samples), dtype="<U8")
    columns["hour_utc"] = np.full(len(samples), hour, dtype=np.float64)
    return columns


def _model() -> "pp.PostprocessModel":
    """A model fitted on rows shaped like the fixture's.

    The truth is stochastic in ``fraction`` — a row at 0.4 is wet 40 % of
    the time — so the fit lands across the bins instead of collapsing to
    one, which is what makes the equality asserted below say something.
    """
    rng = np.random.default_rng(23)
    n = 900
    fractions = rng.random(n)
    near = rng.integers(0, 2, n).astype(float)
    samples = [
        _features_from(float(f), float(k)) for f, k in zip(fractions, near)
    ]
    columns = _design(samples, 8.0)
    wet = (rng.random(n) < fractions).astype(float)
    truth = {lead: (wet, np.ones(n, dtype=bool)) for lead in LEADS}
    return pp.fit_postprocess(
        columns, truth, LEADS, l2=1.0, design_leads=DESIGN_LEADS,
        fitted_at=datetime(2026, 9, 11, 3, 40, tzinfo=timezone.utc),
    )


#: A per-month shift on the ensemble fraction. The gauge behaves the same
#: way every month in this fixture, so without this the three months are
#: identical and a model trained on two of them predicts the third
#: perfectly — which would make "held out" and "in sample" the same
#: number and the out-of-fold test unable to fail.
MONTH_SHIFT: dict[int, float] = {1: -0.2, 4: 0.0, 6: 0.2}


def _feature_values(station: str, ts: datetime) -> dict[str, float]:
    """One fixture row's feature columns, deterministic in station and frame."""
    minute = ts.hour * 60 + ts.minute
    phase = ((minute - FIRST_FRAME_MIN) / 10.0) / 12.0
    near = 1.0 if station == STATION_A else 0.0
    fraction = min(1.0, max(0.0, 0.05 + 0.9 * phase + MONTH_SHIFT[ts.month]))
    return _features_from(fraction, near)


def _stored_post(rows: list[dict]) -> dict[int, np.ndarray]:
    """What a live cycle would have written into ``p_post_<lead>``.

    Built straight from :meth:`PostprocessModel.predict` on a feature dict
    assembled here — deliberately NOT through ``ProbabilityFiller``, so
    "filled equals stored" is a statement about two paths rather than one
    path compared with itself.
    """
    seconds = np.array(
        [int(row["generated_at"].timestamp()) for row in rows], dtype=np.int64,
    )
    features: dict[str, object] = {
        name: np.array(
            [float(row.get(name, np.nan)) for row in rows], dtype=np.float64,
        )
        for name in pp.feature_source_columns(DESIGN_LEADS)
    }
    features["season"] = pp.seasons_from_epoch(seconds)
    features["hour_utc"] = pp.hours_from_epoch(seconds).astype(np.float64)
    return {
        int(lead): np.asarray(values, dtype=np.float64)
        for lead, values in _model().predict(features).items()
    }


def _rows(day, *, with_post: bool, features: bool = False) -> list[dict]:
    out: list[dict] = []
    for minute in range(FIRST_FRAME_MIN, LAST_FRAME_MIN + 1, 10):
        radar_ts = _at(day, minute)
        for station in STATIONS:
            # A curve probability unlike the model's, so the two columns
            # can never be confused for one another.
            curve = 0.5
            row = {
                "radar_ts": radar_ts,
                "generated_at": radar_ts,
                "station_id": station,
                "p_rain": 0.99,
                "action": "none",
                "eta_min": 20.0,
                "intensity_mm_h": 1.2,
                "observed_mm_h": 0.0,
                "forecast_now_mm_h": 0.0,
                "armed_after": True,
                "streak_after": 0,
            }
            for lead in LEADS:
                row[f"p_rain_{lead}"] = curve
            if features:
                row.update(_feature_values(station, radar_ts))
            if with_post:
                post = _p_post(station, _hhmm(radar_ts))
                for lead, value in zip(LEADS, post):
                    row[pp.post_column(lead)] = value
            out.append(row)
    return out


def _write_decisions(
    directory: Path,
    *,
    with_post: bool = True,
    features: bool = False,
    model_post: bool = False,
) -> Path:
    """Plant one parquet per day.

    ``features`` adds the twenty H-P feature columns, as the replay tree
    carries them. ``model_post`` stores the model's own probability in
    ``p_post_<lead>``, as a live cycle since 2026-09-11 does.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    decisions = directory / "decisions"
    decisions.mkdir(parents=True, exist_ok=True)
    for day in DAYS:
        rows = _rows(day, with_post=with_post, features=features)
        table = decision_table(rows, LEADS)
        if features:
            for field in pp.feature_schema(DESIGN_LEADS):
                table = table.append_column(field, pa.array(
                    [row.get(field.name) for row in rows], field.type,
                ))
        if with_post or model_post:
            stored = _stored_post(rows) if model_post else {}
            for field in pp.post_schema(LEADS):
                lead = int(field.name.rsplit("_", 1)[1])
                values = (
                    stored[lead].tolist() if model_post
                    else [row[field.name] for row in rows]
                )
                table = table.append_column(
                    field, pa.array(values, field.type),
                )
        pq.write_table(
            table,
            decisions / f"{day[0]:04d}-{day[1]:02d}-{day[2]:02d}.parquet",
        )
    return directory


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    _write_gauge(tmp_path / "corpus")
    _write_decisions(tmp_path / "replay")
    return tmp_path


def _write_model(root: Path) -> Path:
    path = root / "postprocess.json"
    path.write_text(_model().dumps())
    return path


def _fit_on_own_rows(root: Path) -> Path:
    """A model fitted on the very rows it will then be graded on.

    The shape of the NIGHTLY refit: all rows, no folds
    (``training.held_out: false``). Writing one here is what lets the test
    below reproduce the bug this module was changed for — the diagram that
    came back perfectly calibrated because it was a description of its own
    training data.
    """
    rows = load_probabilities(
        [root / "replay"], LEADS,
        column_for=column_template(pp.POST_COLUMN_TEMPLATE),
        extra_columns=pp.feature_source_columns(DESIGN_LEADS),
    )
    grid, _dead, scored = build_gauge_grid(
        root / "corpus", sorted(set(rows["stations"])),
        decision_window(rows["t"]),
        dry_min=60, onset_min_mm=0.2, min_known_slots=0,
    )
    recode_stations(rows, scored)
    t = rows["t"]
    features: dict[str, object] = {
        name: rows["extra"][name]
        for name in pp.feature_source_columns(DESIGN_LEADS)
    }
    features["season"] = pp.seasons_from_epoch(t)
    features["hour_utc"] = pp.hours_from_epoch(t).astype(np.float64)
    truth = {}
    for lead in LEADS:
        outcome, usable = grid.outcome(t, rows["station"], lead)
        truth[int(lead)] = (outcome, usable & ~rows["dropped"])
    model = pp.fit_postprocess(
        features, truth, LEADS, l2=1.0, design_leads=DESIGN_LEADS,
        fitted_at=datetime(2026, 9, 11, 3, 40, tzinfo=timezone.utc),
    )
    path = root / "self_fitted.json"
    path.write_text(model.dumps())
    return path


def _options(root: Path, **overrides) -> GaugeReliabilityOptions:
    kwargs = dict(
        decisions_dirs=[root / "replay"],
        corpus_dir=root / "corpus",
        leads=LEADS,
        probability_column=pp.POST_COLUMN_TEMPLATE,
        design_leads=DESIGN_LEADS,
        min_known_slots=0,
    )
    kwargs.update(overrides)
    return GaugeReliabilityOptions(**kwargs)  # type: ignore[arg-type]


def gauge_options_json(root: Path) -> dict:
    """``_options`` as the job config carries it across the process boundary."""
    from dmi_nowcast_sidecar.quality_job import gauge_reliability_options_to_json

    return gauge_reliability_options_to_json(_options(root))


def _layer_b(root: Path, template: str) -> dict:
    """Layer B over the same rows, through the benchmark's own entry points."""
    rows = load_probabilities(
        [root / "replay"], LEADS, column_for=column_template(template),
    )
    grid, _dead, scored = build_gauge_grid(
        root / "corpus", sorted(set(rows["stations"])),
        decision_window(rows["t"]),
        dry_min=60, onset_min_mm=0.2, min_known_slots=0,
    )
    recode_stations(rows, scored)
    return report_module.layer_b(
        rows, None, grid, LEADS, n_resamples=0, seed=1, ci=0.9,
    )


# ---------------------------------------------------------------------------
# 1. One computation, two callers
# ---------------------------------------------------------------------------


class TestAgreementWithLayerB:
    def test_every_bin_matches_the_benchmarks(self, corpus: Path) -> None:
        """Bin for bin, against Layer B on the same rows.

        Layer B scores whatever probability column its rows carry — its
        out-of-sample-ness comes from the file, since
        ``scripts/fit_postprocess.py`` writes its held-out predictions
        into a copy of the run. So the comparison is made on the
        IN-SAMPLE path explicitly (``out_of_fold=False``): both sides then
        score the identical stored column, and the equality is about the
        two implementations rather than about two fitting regimes.
        """
        block = gauge_reliability_from_decisions(
            _options(corpus, out_of_fold=False),
        )
        assert block is not None
        assert block["calibration"] == "in-sample"
        bench = _layer_b(corpus, pp.POST_COLUMN_TEMPLATE)
        curves = {int(c["lead_min"]): c for c in block["curves"]}
        assert set(curves) == set(LEADS)
        for lead in LEADS:
            scored = bench["leads"][str(lead)]["all"]["baseline"]
            assert scored is not None
            page = curves[lead]
            assert page["n"] == scored["n"]
            assert page["brier"] == pytest.approx(scored["brier"])
            # The benchmark omits empty bins, the page keeps them and nulls
            # the two means. Same numbers, two audiences — so compare the
            # occupied ones and assert the rest really are empty.
            occupied = {int(b["bin"]): b for b in scored["reliability_table"]}
            for index, shown in enumerate(page["bins"]):
                row = occupied.get(index)
                if row is None:
                    assert shown["n"] == 0
                    assert shown["forecast_mean"] is None
                    assert shown["observed_freq"] is None
                    continue
                assert shown["n"] == row["n"]
                assert shown["forecast_mean"] == pytest.approx(row["mean_p"])
                assert shown["observed_freq"] == pytest.approx(row["observed"])

    def test_it_scored_the_model_and_not_the_curve(self, corpus: Path) -> None:
        """The two columns differ; the block has to be of the right one."""
        model = gauge_reliability_from_decisions(
            _options(corpus, out_of_fold=False),
        )
        curve = gauge_reliability_from_decisions(
            _options(corpus, probability_column=None),
        )
        assert model is not None and curve is not None
        assert model["mode"] == "postprocess"
        assert curve["mode"] == "served"
        # The curve was never fitted on a gauge, so scoring it at the
        # gauges is already out-of-sample; there is nothing to hold out.
        assert curve["calibration"] == "served"
        assert model["probability_column"] == "p_post_{lead}"
        assert curve["probability_column"] == "p_rain_{lead}"
        # The curve fixture is a flat 0.5 everywhere, so its whole diagram
        # lives in bin 5; the model's does not.
        curve_bins = {
            index for index, b in enumerate(curve["curves"][0]["bins"])
            if b["n"] > 0
        }
        assert curve_bins == {5}
        model_bins = {
            index for index, b in enumerate(model["curves"][0]["bins"])
            if b["n"] > 0
        }
        assert model_bins == {2, 7}

    def test_the_window_describes_the_rows_it_scored(self, corpus: Path) -> None:
        block = gauge_reliability_from_decisions(
            _options(corpus, out_of_fold=False),
        )
        assert block is not None
        window = block["window"]
        assert window["from"].startswith("2026-01-15T07:00")
        assert window["to"].startswith("2026-06-15T09:00")
        assert window["points"] == len(STATIONS)
        # One "event" per decision instant, of which there are 13 a day.
        assert window["events"] == 13 * len(DAYS)


# ---------------------------------------------------------------------------
# 2. Rows the model never spoke for
# ---------------------------------------------------------------------------


class TestRowsWithoutTheModel:
    def test_they_are_excluded_and_counted(self, tmp_path: Path) -> None:
        """An archive older than the model is depth, not evidence.

        Imputing anything here — a zero, the curve's value — would make
        the diagram a statement about how far back the parquet goes.
        """
        _write_gauge(tmp_path / "corpus")
        _write_decisions(tmp_path / "replay", with_post=False)
        block = gauge_reliability_from_decisions(_options(tmp_path))
        assert block is None

    def test_a_mixed_archive_scores_only_the_rows_that_carry_it(
        self, tmp_path: Path,
    ) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        _write_gauge(tmp_path / "corpus")
        _write_decisions(tmp_path / "replay", with_post=True)
        # Rewrite the first day WITHOUT the model's columns, as a file
        # written before it shipped would be.
        day = DAYS[0]
        rows = _rows(day, with_post=False)
        pq.write_table(
            decision_table(rows, LEADS),
            tmp_path / "replay" / "decisions"
            / f"{day[0]:04d}-{day[1]:02d}-{day[2]:02d}.parquet",
        )
        block = gauge_reliability_from_decisions(_options(tmp_path))
        assert block is not None
        full = _layer_b(tmp_path, pp.POST_COLUMN_TEMPLATE)
        for curve in block["curves"]:
            lead = int(curve["lead_min"])
            scored = full["leads"][str(lead)]["all"]["baseline"]
            assert curve["n"] == scored["n"]
            # Every scoreable row of the missing day is counted, not hidden.
            assert curve["n_excluded"] > 0
            assert isinstance(pa.array([curve["n_excluded"]])[0].as_py(), int)


# ---------------------------------------------------------------------------
# 3. Nothing to score
# ---------------------------------------------------------------------------


class TestDegradesToNone:
    def test_no_decision_dirs(self, corpus: Path) -> None:
        assert gauge_reliability_from_decisions(
            _options(corpus, decisions_dirs=[]),
        ) is None

    def test_no_gauge_store(self, tmp_path: Path) -> None:
        _write_decisions(tmp_path / "replay")
        assert gauge_reliability_from_decisions(
            _options(tmp_path, corpus_dir=tmp_path / "nothing-here"),
        ) is None

    def test_an_unreadable_decision_tree(self, corpus: Path) -> None:
        assert gauge_reliability_from_decisions(
            _options(corpus, decisions_dirs=[corpus / "does-not-exist"]),
        ) is None


# ---------------------------------------------------------------------------
# 3b. Filling the column the archive does not carry
# ---------------------------------------------------------------------------


class TestFillingFromFeatures:
    """The archive is mostly features, and the diagram has to cover it.

    On the VM the decision rows are ~444 000 replay rows written before
    the serving path existed — twenty feature columns each and no
    ``p_post_<lead>`` — plus the live rows since 2026-09-11 06:33Z. A
    reader that scored the stored column alone would draw the page's
    gauge curve from a few hours and silently drop ten months, which is a
    measurement of the archive's depth rather than of the service.

    So the column is filled by the served model, per file and inside the
    read, through the same ``ProbabilityFiller`` the nightly threshold
    sweep uses. The claim under test is that filling changes nothing but
    coverage: a features-only tree scores exactly as a tree that carries
    the model's own stored probability.
    """

    def test_features_alone_score_identically_to_a_stored_column(
        self, tmp_path: Path,
    ) -> None:
        """The seam: two archives, one curve.

        ``filled`` is the replay tree as it exists — features, no
        probability. ``stored`` is the same rows with the model's answer
        already written in, as a live cycle writes it. If the filling
        used a different feature set, derived season or hour differently,
        or landed on the wrong rows, these two curves would part company.
        """
        filled_root = tmp_path / "filled"
        stored_root = tmp_path / "stored"
        for root, kwargs in (
            (filled_root, {"with_post": False, "features": True}),
            (stored_root, {"with_post": False, "features": True,
                           "model_post": True}),
        ):
            _write_gauge(root / "corpus")
            _write_decisions(root / "replay", **kwargs)
        model = _write_model(tmp_path)

        filled = gauge_reliability_from_decisions(
            _options(filled_root, postprocess_model=model),
        )
        stored = gauge_reliability_from_decisions(
            _options(stored_root, postprocess_model=model),
        )
        assert filled is not None and stored is not None
        assert [c["lead_min"] for c in filled["curves"]] == list(LEADS)
        for left, right in zip(filled["curves"], stored["curves"]):
            assert left["n"] == right["n"] > 0
            assert left["brier"] == pytest.approx(right["brier"])
            assert left["bins"] == right["bins"]
        # And the diagram really does span more than one bin, so the
        # agreement above is not the trivial kind.
        occupied = {
            index for index, b in enumerate(filled["curves"][0]["bins"])
            if b["n"] > 0
        }
        assert len(occupied) >= 2

        # The two arrived by different routes, and the counts say so.
        assert filled["fill"]["stored"] == 0
        assert filled["fill"]["computed"] > 0
        assert stored["fill"]["stored"] > 0
        assert stored["fill"]["computed"] == 0

    def test_without_a_model_the_replay_tree_scores_nothing(
        self, tmp_path: Path,
    ) -> None:
        """The regression this exists to prevent.

        Same rows, no model to fill with: every row carries features and
        no stored probability, so there is nothing to score and the block
        is null — which is exactly what the page would have shown for ten
        months of evidence.
        """
        _write_gauge(tmp_path / "corpus")
        _write_decisions(tmp_path / "replay", with_post=False, features=True)
        assert gauge_reliability_from_decisions(_options(tmp_path)) is None

    def test_a_stored_value_wins_over_the_model(self, tmp_path: Path) -> None:
        """What the engine used beats what the model would say now.

        The fixture's stored probability (0.25 / 0.75) is nothing like
        what the model produces from these features, so a filler that
        overwrote a stored value would move the whole diagram out of the
        two bins asserted here.
        """
        _write_gauge(tmp_path / "corpus")
        _write_decisions(tmp_path / "replay", with_post=True, features=True)
        # The in-sample path on purpose: this is about which value the
        # FILLER keeps, and the out-of-fold refit replaces every stored
        # value by design, which would hide the answer.
        block = gauge_reliability_from_decisions(
            _options(
                tmp_path,
                postprocess_model=_write_model(tmp_path),
                out_of_fold=False,
            ),
        )
        assert block is not None
        occupied = {
            index for index, b in enumerate(block["curves"][0]["bins"])
            if b["n"] > 0
        }
        assert occupied == {2, 7}
        assert block["fill"]["computed"] == 0
        assert block["fill"]["stored"] == block["fill"]["rows"]
        assert block["fill"]["dropped"] == 0

    def test_rows_with_neither_are_the_only_exclusions(
        self, tmp_path: Path,
    ) -> None:
        """A row with no features and no stored value cannot be scored.

        Imputing one would call every frame of the oldest partitions
        something, which is not a measurement. So it is dropped at the
        read and counted, and the count is in the block.
        """
        import pyarrow.parquet as pq

        _write_gauge(tmp_path / "corpus")
        _write_decisions(tmp_path / "replay", with_post=False, features=True)
        # Rewrite one day with neither, as a partition older than both.
        day = DAYS[0]
        bare = _rows(day, with_post=False, features=False)
        pq.write_table(
            decision_table(bare, LEADS),
            tmp_path / "replay" / "decisions"
            / f"{day[0]:04d}-{day[1]:02d}-{day[2]:02d}.parquet",
        )
        block = gauge_reliability_from_decisions(
            _options(tmp_path, postprocess_model=_write_model(tmp_path)),
        )
        assert block is not None
        fill = block["fill"]
        # ``rows``, ``stored`` and ``dropped`` count ROWS; ``computed``
        # counts the (row, lead) cells actually written, which is the
        # filler's own unit and the one the nightly sweep logs. Named
        # here because the asymmetry is otherwise easy to misread.
        rows_seen = FRAMES_PER_DAY * len(STATIONS) * len(DAYS)
        scoreable = rows_seen - len(bare)
        assert fill["rows"] == rows_seen
        assert fill["dropped"] == len(bare)
        assert fill["stored"] == 0
        assert fill["computed"] == scoreable * len(LEADS)
        # And the window shrinks to the days that could be scored.
        assert block["window"]["from"].startswith("2026-04-15")

    def test_an_unreadable_model_narrows_the_claim_but_does_not_fail(
        self, tmp_path: Path,
    ) -> None:
        """A junk model file must not take the nightly build with it."""
        _write_gauge(tmp_path / "corpus")
        _write_decisions(
            tmp_path / "replay", with_post=True, features=True,
        )
        junk = tmp_path / "junk.json"
        junk.write_text("{ not json")
        block = gauge_reliability_from_decisions(
            _options(tmp_path, postprocess_model=junk),
        )
        assert block is not None
        # The stored rows still score; nothing was filled, and with no
        # model there is nothing to refit either, so the block says so.
        assert block["fill"]["computed"] == 0
        assert block["fill"]["stored"] > 0
        assert block["calibration"] == "in-sample"


# ---------------------------------------------------------------------------
# 3c. Out-of-fold, or the diagram measures nothing
# ---------------------------------------------------------------------------


class TestOutOfFold:
    """The tautology this replaced, and the guard against it coming back.

    The nightly refit fits on ALL rows (``training.held_out: false``).
    Binning that model's own predictions against its own training rows
    draws a perfect diagonal: the first live build came back 0.149 →
    0.149, 0.754 → 0.753, 0.949 → 0.949 over hundreds of thousands of
    rows, which says only that isotonic regression can describe its own
    training data. Exactly the trap the radar curve had.

    So the shipped diagram is of predictions from a model refitted per
    ``(year, month)`` without the month it is graded on. What is asserted
    here is that the two really are different numbers, that the block says
    which one it is, and that a fold the refit cannot fit degrades to a
    labelled, counted fallback rather than a hole in the curve.
    """

    @pytest.fixture()
    def featured(self, tmp_path: Path) -> tuple[Path, Path]:
        _write_gauge(tmp_path / "corpus")
        _write_decisions(tmp_path / "replay", with_post=False, features=True)
        return tmp_path, _write_model(tmp_path)

    def test_it_differs_from_the_in_sample_curve_and_says_so(
        self, featured: tuple[Path, Path],
    ) -> None:
        root, model = featured
        folded = gauge_reliability_from_decisions(
            _options(root, postprocess_model=model),
        )
        in_sample = gauge_reliability_from_decisions(
            _options(root, postprocess_model=model, out_of_fold=False),
        )
        assert folded is not None and in_sample is not None

        assert folded["calibration"] == "out-of-fold"
        assert folded["mode"] == "postprocess_cv"
        # One fold per (year, month); the fixture spans January, April
        # and June.
        assert folded["cv_folds"] == len(DAYS)
        assert folded["fold"] == "month"

        assert in_sample["calibration"] == "in-sample"
        assert in_sample["mode"] == "postprocess"
        assert in_sample["cv_folds"] == 0
        assert in_sample["fold"] is None

        # And they are genuinely different numbers. A model that never saw
        # the month cannot describe it as well as one that did.
        assert folded["curves"][0]["bins"] != in_sample["curves"][0]["bins"]
        assert folded["curves"][0]["brier"] != in_sample["curves"][0]["brier"]

    def test_the_in_sample_curve_is_the_tautology_it_is_labelled_as(
        self, tmp_path: Path,
    ) -> None:
        """The reported failure, reproduced and then removed.

        The model here is fitted on the very rows it is then graded on —
        the nightly refit's shape exactly. In sample it lands ON the
        diagonal to three decimals, which is the 0.149 → 0.149 the first
        live build showed and is a statement about isotonic regression
        rather than about the service. Held out, it does not.
        """
        _write_gauge(tmp_path / "corpus")
        _write_decisions(tmp_path / "replay", with_post=False, features=True)
        model = _fit_on_own_rows(tmp_path)

        in_sample = gauge_reliability_from_decisions(
            _options(tmp_path, postprocess_model=model, out_of_fold=False),
        )
        assert in_sample is not None
        assert in_sample["calibration"] == "in-sample"
        populated = [
            b for b in in_sample["curves"][0]["bins"]
            if b["n"] >= 5 and b["forecast_mean"] is not None
        ]
        assert len(populated) >= 2, "the fixture must occupy real bins"
        for b in populated:
            assert b["observed_freq"] == pytest.approx(
                b["forecast_mean"], abs=1e-3,
            ), "the in-sample diagram should be the perfect diagonal"

        folded = gauge_reliability_from_decisions(
            _options(tmp_path, postprocess_model=model),
        )
        assert folded is not None
        assert folded["calibration"] == "out-of-fold"
        off = [
            b for b in folded["curves"][0]["bins"]
            if b["n"] >= 5 and b["forecast_mean"] is not None
            and abs(b["forecast_mean"] - b["observed_freq"]) > 1e-3
        ]
        assert off, "held out, the model must stop describing its own rows"

    def test_the_baseline_stays_the_served_curve_on_the_same_rows(
        self, featured: tuple[Path, Path],
    ) -> None:
        """``brier_raw`` is not refitted — it is what the site used to show."""
        root, model = featured
        folded = gauge_reliability_from_decisions(
            _options(root, postprocess_model=model),
        )
        in_sample = gauge_reliability_from_decisions(
            _options(root, postprocess_model=model, out_of_fold=False),
        )
        assert folded is not None and in_sample is not None
        for left, right in zip(folded["curves"], in_sample["curves"]):
            assert left["brier_raw"] == right["brier_raw"]
            # ``p_rain_<lead>`` is a flat 0.5 in the fixture, so the
            # paired baseline is a fixed number either way.
            assert left["brier_raw"] == pytest.approx(0.25)

    def test_one_month_is_no_fold(self, tmp_path: Path) -> None:
        """A single month cannot hold anything out, and must not pretend to."""
        import pyarrow.parquet as pq

        _write_gauge(tmp_path / "corpus")
        _write_decisions(tmp_path / "replay", with_post=False, features=True)
        for day in DAYS[1:]:
            (
                tmp_path / "replay" / "decisions"
                / f"{day[0]:04d}-{day[1]:02d}-{day[2]:02d}.parquet"
            ).unlink()
        assert len(list((tmp_path / "replay" / "decisions").glob("*.parquet"))) == 1
        block = gauge_reliability_from_decisions(
            _options(tmp_path, postprocess_model=_write_model(tmp_path)),
        )
        assert block is not None
        assert block["calibration"] == "in-sample"
        assert block["cv_folds"] == 0
        assert pq  # the import is the reason the day files could be removed

    def test_a_fold_that_cannot_be_fitted_falls_back_and_is_counted(
        self, featured: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A hole in the curve would be the quieter lie.

        The refit leaves NaN for a fold it could not fit. Those rows take
        the served model's own prediction — in-sample for them — and the
        block counts exactly how many, so a reader can see how much of the
        diagram is not out-of-sample.
        """
        from dmi_nowcast_sidecar import gauge_reliability as module

        real = module.core_postprocess.leave_one_month_out

        def holed(*args, **kwargs):
            out = real(*args, **kwargs)
            for values in out["out_of_fold"].values():
                values[: len(values) // 3] = np.nan
            return out

        monkeypatch.setattr(
            module.core_postprocess, "leave_one_month_out", holed,
        )
        root, model = featured
        block = gauge_reliability_from_decisions(
            _options(root, postprocess_model=model),
        )
        assert block is not None
        assert block["calibration"] == "out-of-fold"
        assert block["in_sample_fallbacks"] > 0
        # Nothing was dropped for it: every scoreable row still has a
        # probability, so the curve covers the same sample.
        assert all(curve["n_excluded"] == 0 for curve in block["curves"])

    def test_a_refit_failure_degrades_to_the_served_model(
        self, featured: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """One diagram is not worth the nightly build."""
        from dmi_nowcast_sidecar import gauge_reliability as module

        def explode(*_args, **_kwargs):
            raise RuntimeError("a singular design")

        monkeypatch.setattr(
            module.core_postprocess, "leave_one_month_out", explode,
        )
        root, model = featured
        block = gauge_reliability_from_decisions(
            _options(root, postprocess_model=model),
        )
        assert block is not None
        # It still publishes a curve — labelled for what it is, never
        # labelled out-of-fold.
        assert block["calibration"] == "in-sample"
        assert block["curves"]


# ---------------------------------------------------------------------------
# 4. The nightly wiring
# ---------------------------------------------------------------------------


class TestJobWiring:
    """How the block reaches the report builder.

    The producer lives in the sidecar package because the decision-row
    loader does; the builder lives in the core package and must keep
    importing nothing from the sidecar. So the block is computed by the
    job and injected — and the seams worth pinning are that the nightly
    config names it, that a live refresh does not, and that the builder
    is only handed the keyword when there is something to hand it (an
    injected test builder takes the inputs and nothing else).
    """

    @staticmethod
    def _config(tmp_path: Path, **overrides):
        from dmi_nowcast_sidecar.config import Config

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

    def test_it_is_on_by_default_and_borrows_the_fits_rows(
        self, tmp_path: Path,
    ) -> None:
        from dmi_nowcast_sidecar.quality_report import QualityReportTask

        config = self._config(tmp_path, quality_report={
            "enabled": True,
            "fit_thresholds": {
                "enabled": True,
                "decisions_dirs": [tmp_path / "replay", tmp_path / "eval"],
                "leads": [20, 30],
            },
        })
        payload = QualityReportTask(config).job_config()
        block = payload["gauge_reliability"]
        assert block["enabled"] is True
        options = block["options"]
        # The threshold fit's rows, in the fit's order: replay first, the
        # live scoreboard second, later winning a tie.
        assert options["decisions_dirs"] == [
            str(tmp_path / "replay"), str(tmp_path / "eval"),
        ]
        assert options["leads"] == [20, 30]
        # The column the service decides on, which is what makes the page
        # a diagram of the number it shows.
        assert options["probability_column"] == "p_post_{lead}"
        assert options["corpus_dir"] == str(tmp_path / "corpus")
        # And the model to FILL that column with on the ten months of
        # replay rows that predate it — the same file the sweep is
        # handed, so the page scores what the thresholds were fitted on.
        from dmi_nowcast_sidecar.push.paths import resolved_postprocess_path

        assert options["postprocess_model"] == str(
            resolved_postprocess_path(config),
        )
        assert options["design_leads"] == list(
            config.forecast.national.leads_min,
        )

    def test_the_curve_deployment_scores_the_curve(self, tmp_path: Path) -> None:
        from dmi_nowcast_sidecar.quality_report import QualityReportTask

        config = self._config(
            tmp_path,
            push={"probability_source": "curve"},
            quality_report={
                "enabled": True,
                "fit_thresholds": {
                    "enabled": True, "decisions_dirs": [tmp_path / "replay"],
                },
            },
        )
        payload = QualityReportTask(config).job_config()
        options = payload["gauge_reliability"]["options"]
        assert options["probability_column"] is None
        # Nothing to fill a curve column with, and nothing that would
        # need filling: ``p_rain_<lead>`` is on every row ever written.
        assert options["postprocess_model"] is None

    def test_without_rows_there_is_nothing_to_score(self, tmp_path: Path) -> None:
        from dmi_nowcast_sidecar.quality_report import QualityReportTask

        config = self._config(tmp_path, quality_report={"enabled": True})
        payload = QualityReportTask(config).job_config()
        assert payload["gauge_reliability"] == {"enabled": False}

    def test_a_live_refresh_never_carries_the_step(self, tmp_path: Path) -> None:
        from dmi_nowcast_sidecar.quality_report import QualityReportTask

        config = self._config(tmp_path, quality_report={
            "enabled": True,
            "fit_thresholds": {
                "enabled": True, "decisions_dirs": [tmp_path / "replay"],
            },
        })
        payload = QualityReportTask(config).job_config(live_only=True)
        assert payload["gauge_reliability"] == {"enabled": False}

    def test_the_options_survive_the_process_boundary(self, corpus: Path) -> None:
        from dmi_nowcast_sidecar.quality_job import (
            gauge_reliability_options_from_json,
            gauge_reliability_options_to_json,
        )

        options = _options(corpus)
        back = gauge_reliability_options_from_json(
            gauge_reliability_options_to_json(options),
        )
        assert back == options

    def test_the_job_hands_the_block_to_the_builder(self, corpus: Path) -> None:
        from dmi_nowcast_sidecar.quality_job import run_job

        seen: dict = {}

        def builder(_inputs, **kwargs):
            seen.update(kwargs)
            return {"schema_version": 1, "generated_at_utc": "2026-09-11T03:30:00Z"}

        config = {
            "quality": {
                "out_json": str(corpus / "out" / "quality.json"),
                "markdown_dir": None,
                "inputs": {},
            },
            "gauge_reliability": {
                "enabled": True,
                "options": gauge_options_json(corpus),
            },
        }
        summary = run_job(config, builder=builder, renderer=lambda _r: "")
        assert summary["ok"] is True
        assert [c["lead_min"] for c in seen["gauge_reliability"]["curves"]] == list(LEADS)
        assert summary["gauge_reliability"]["probability_column"] == "p_post_{lead}"

    def test_a_failing_step_falls_back_instead_of_killing_the_build(
        self, corpus: Path,
    ) -> None:
        """One diagram is not worth the scoreboard, the map and the stamps."""
        from dmi_nowcast_sidecar.quality_job import run_job

        seen: list[dict] = []

        def builder(_inputs, **kwargs):
            seen.append(kwargs)
            return {"schema_version": 1, "generated_at_utc": "2026-09-11T03:30:00Z"}

        def explode(_options):
            raise RuntimeError("a corrupt parquet")

        config = {
            "quality": {
                "out_json": str(corpus / "out" / "quality.json"),
                "markdown_dir": None,
                "inputs": {},
            },
            "gauge_reliability": {
                "enabled": True, "options": gauge_options_json(corpus),
            },
        }
        summary = run_job(
            config, builder=builder, renderer=lambda _r: "", gauge_scorer=explode,
        )
        assert summary["ok"] is True
        assert summary["gauge_reliability"] is None
        # No keyword at all: the builder falls back to the corpus fit.
        assert seen == [{}]
