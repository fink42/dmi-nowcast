"""The Layer A harness plumbing: horizons, scores, day blocks, comparison.

Everything here is pure arithmetic on small arrays and dicts — the parts
of ``scripts/persistence_vs_advection.py`` and ``scripts/compare_layer_a.py``
that do not need an HDF5 archive. Two contracts matter most and are
pinned explicitly:

1. the legacy 2x2x2 cells at 0.5 mm/h are untouched by the new threshold
   list, so ``archive/persistence_vs_advection_20260905.md`` stays
   reproducible;
2. the per-day blocks the bootstrap resamples agree with the per-case
   aggregation they were derived from.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
# The harness is a script, not a package — same pattern as
# test_calibration_points.py.
sys.path.insert(0, str(REPO / "scripts"))

import compare_layer_a as cmp_mod  # noqa: E402
import persistence_vs_advection as pva  # noqa: E402


# ---------------------------------------------------------------------------
# horizons
# ---------------------------------------------------------------------------
def test_snap_horizon_ties_go_to_the_later_frame():
    assert pva.snap_horizon(10) == 10
    assert pva.snap_horizon(30) == 30
    assert pva.snap_horizon(44) == 40
    assert pva.snap_horizon(46) == 50
    # 45 is equidistant: the later (harder) frame wins, so the reported
    # skill understates rather than overstates.
    assert pva.snap_horizon(45) == 50
    # Nothing snaps below one frame.
    assert pva.snap_horizon(1) == 10


def test_case_spec_derives_offsets_and_frame_reach():
    spec = pva.CaseSpec()
    assert spec.horizons_min == (10, 20, 30, 45)
    assert spec.offsets_min == (10, 20, 30, 50)
    assert spec.frames_ahead == 5
    assert spec.fss_scales_px == (1, 2, 4, 8, 16)


def test_case_spec_with_the_legacy_horizons_reaches_only_two_frames():
    """``--horizons 10,20`` restores the pre-Phase-H case selection."""
    spec = pva.CaseSpec(horizons_min=(10, 20))
    assert spec.offsets_min == (10, 20)
    assert spec.frames_ahead == 2


def test_case_spec_validation():
    with pytest.raises(ValueError):
        pva.CaseSpec(horizons_min=())
    with pytest.raises(ValueError):
        pva.CaseSpec(horizons_min=(20, 10))
    with pytest.raises(ValueError):
        pva.CaseSpec(horizons_min=(0,))


# ---------------------------------------------------------------------------
# horizon_metrics
# ---------------------------------------------------------------------------
NAN = float("nan")
OBS = np.array([
    [1.0, 0.0, 2.0, NAN],
    [0.0, 5.0, 0.0, 1.0],
    [1.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 6.0, 0.0],
])
ADV = np.array([
    [0.0, 1.0, 2.0, 1.0],
    [0.0, 5.0, 0.0, NAN],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
])
TRUTH = np.array([
    [1.0, 1.0, 0.0, 1.0],
    [0.0, 5.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 2.0],
    [0.0, 0.0, 6.0, 0.0],
])
SPEC = pva.CaseSpec(
    horizons_min=(10,), thresholds_mm_h=(0.5, 4.0), fss_scales_km=(2,),
)


def test_horizon_metrics_masks_pixels_missing_from_any_field():
    """Two of sixteen pixels are non-finite somewhere, so fourteen count."""
    block = pva.horizon_metrics(OBS, ADV, TRUTH, SPEC)
    assert block["n_pixels"] == 14


def test_horizon_metrics_legacy_cells_are_hand_counted():
    block = pva.horizon_metrics(OBS, ADV, TRUTH, SPEC)
    assert block["o1a1t1"] == 1 and block["o1a1t0"] == 2
    assert block["o1a0t1"] == 3 and block["o1a0t0"] == 0
    assert block["o0a1t1"] == 1 and block["o0a1t0"] == 0
    assert block["o0a0t1"] == 1 and block["o0a0t0"] == 6
    assert sum(block[c] for c in pva._CELLS) == 14


def test_horizon_metrics_legacy_cells_ignore_the_threshold_list():
    """The archived tables are at 0.5 mm/h whatever ``--thresholds`` says."""
    wide = pva.CaseSpec(
        horizons_min=(10,), thresholds_mm_h=(1.0, 4.0, 8.0), fss_scales_km=(2,),
    )
    a = pva.horizon_metrics(OBS, ADV, TRUTH, SPEC)
    b = pva.horizon_metrics(OBS, ADV, TRUTH, wide)
    for cell in pva._CELLS:
        assert a[cell] == b[cell]
    assert a["n_pixels"] == b["n_pixels"]


def test_horizon_metrics_contingency_counts_by_hand():
    block = pva.horizon_metrics(OBS, ADV, TRUTH, SPEC)
    assert block["thresholds"]["0.5"]["persistence"] == [4, 2, 2, 6]
    assert block["thresholds"]["0.5"]["advection"] == [2, 4, 2, 6]
    # At 4 mm/h only the two heavy pixels are wet; advection keeps one.
    assert block["thresholds"]["4"]["persistence"] == [2, 0, 0, 12]
    assert block["thresholds"]["4"]["advection"] == [1, 1, 0, 12]


def test_horizon_metrics_contingency_agrees_with_the_legacy_cells():
    """The 0.5 mm/h contingency must be the same event as the 2x2x2 table."""
    block = pva.horizon_metrics(OBS, ADV, TRUTH, SPEC)
    hits, misses, fa, _ = block["thresholds"]["0.5"]["advection"]
    assert hits == block["o1a1t1"] + block["o0a1t1"]
    assert fa == block["o1a1t0"] + block["o0a1t0"]
    assert misses == block["o1a0t1"] + block["o0a0t1"]


def test_horizon_metrics_fss_components_at_one_pixel_neighbourhood():
    """size-1 neighbourhood: fractions ARE the binary fields.

    Persistence disagrees with the truth on 4 of 16 grid cells and both
    have 6 wet cells, so mse = 4/16 and mse_ref = 12/16 → FSS = 2/3.
    """
    block = pva.horizon_metrics(OBS, ADV, TRUTH, SPEC)
    p_mse, p_ref, a_mse, a_ref = block["fss"]["0.5"]["2"]
    assert p_mse == pytest.approx(4 / 16)
    assert p_ref == pytest.approx(12 / 16)
    assert pva.fss_of([p_mse, p_ref, a_mse, a_ref], "persistence") == \
        pytest.approx(round(2 / 3, 4))
    # Advection: 6 disagreements, 4 + 6 wet cells.
    assert a_mse == pytest.approx(6 / 16)
    assert a_ref == pytest.approx(10 / 16)


# ---------------------------------------------------------------------------
# day blocks
# ---------------------------------------------------------------------------
def _case(month: str, speed: float, obs=OBS, adv=ADV, truth=TRUTH) -> dict:
    block = pva.horizon_metrics(obs, adv, truth, SPEC)
    block["truth_offset_min"] = 10
    return {
        "t": f"{month.replace('-', '')}010000",
        "month": month,
        "wet_fraction": 0.2,
        "bulk_speed_kmh": speed,
        "stall": {
            "bulk_kmh": speed, "stalled_share": 0.1,
            "applicable": speed >= pva.SPEED_SPLIT_KMH,
            "n_wet": 100, "profile_kmh": [1.0, 2.0],
        },
        "horizons": {"10": block},
    }


def test_strata_split_by_month_season_and_speed():
    assert pva.strata_of(_case("2026-01", 5.0)) == (
        "pooled", "month 2026-01", "winter (Dec-Mar)", "speed < 20 km/h",
    )
    assert pva.strata_of(_case("2026-07", 30.0)) == (
        "pooled", "month 2026-07", "summer (May-Sep)", "speed >= 20 km/h",
    )


def test_day_blocks_sum_cases_and_score_them():
    blocks: dict = {}
    for case in (_case("2026-01", 5.0), _case("2026-01", 30.0)):
        pva.accumulate_day_blocks(blocks, case)

    pooled = blocks["pooled"]["10"]
    assert pooled["n_cases"] == 2
    assert pooled["n_pixels"] == 28
    assert pooled["thresholds"]["0.5"]["advection"] == [4, 8, 4, 12]
    # Each speed stratum saw exactly one of the two cases.
    assert blocks["speed < 20 km/h"]["10"]["n_cases"] == 1
    assert blocks["speed >= 20 km/h"]["10"]["n_cases"] == 1

    scored = pva.block_scores(pooled)
    assert scored["thresholds"]["0.5"]["advection"]["CSI"] == pytest.approx(0.25)
    assert scored["thresholds"]["0.5"]["persistence"]["CSI"] == pytest.approx(0.5)
    assert scored["truth_offset_min"] == 10


def test_merge_blocks_pools_across_days():
    day_a: dict = {}
    pva.accumulate_day_blocks(day_a, _case("2026-01", 5.0))
    day_b: dict = {}
    pva.accumulate_day_blocks(day_b, _case("2026-02", 30.0))

    agg = pva.aggregate_blocks({"2026-01-01": day_a, "2026-02-01": day_b})
    assert agg["pooled"]["10"]["n_cases"] == 2
    assert agg["pooled"]["10"]["n_pixels"] == 28
    assert set(agg) >= {"pooled", "month 2026-01", "month 2026-02",
                        "winter (Dec-Mar)", "speed < 20 km/h",
                        "speed >= 20 km/h"}
    # Ratios come from the pooled counts, so a stratum of one case matches
    # the single case's own score.
    assert agg["month 2026-01"]["10"]["thresholds"]["0.5"]["advection"]["CSI"] \
        == pytest.approx(0.25)


def test_day_blocks_and_case_aggregation_describe_the_same_pixels():
    """The 0.5 mm/h day-block table must equal the legacy cell sums."""
    cases = [_case("2026-01", 5.0), _case("2026-07", 30.0)]
    blocks: dict = {}
    for case in cases:
        pva.accumulate_day_blocks(blocks, case)
    legacy = pva.aggregate(cases)["pooled"]["10"]
    pooled = pva.block_scores(blocks["pooled"]["10"])["thresholds"]["0.5"]

    assert pooled["advection"]["hits"] == legacy["advection"]["hits"]
    assert pooled["advection"]["misses"] == legacy["advection"]["misses"]
    assert pooled["advection"]["false_alarms"] == legacy["advection"]["false_alarms"]
    assert pooled["advection"]["CSI"] == pytest.approx(legacy["advection"]["CSI"])
    assert pooled["persistence"]["CSI"] == pytest.approx(legacy["persistence"]["CSI"])


def test_strip_detail_removes_only_the_bulky_keys():
    case = _case("2026-01", 5.0)
    pva.strip_detail(case)
    horizon = case["horizons"]["10"]
    assert "thresholds" not in horizon and "fss" not in horizon
    assert horizon["n_pixels"] == 14
    assert horizon["truth_offset_min"] == 10
    assert horizon["o1a0t1"] == 3


def test_aggregate_ignores_the_extra_horizon_keys():
    """``aggregate`` predates Phase H; the new keys must not disturb it."""
    cases = [_case("2026-01", 5.0)]
    with_detail = pva.aggregate(cases)
    without = pva.aggregate([pva.strip_detail(_case("2026-01", 5.0))])
    assert with_detail == without


# ---------------------------------------------------------------------------
# stall aggregation
# ---------------------------------------------------------------------------
def test_aggregate_stall_uses_applicable_pairs_only():
    cases = [_case("2026-01", 5.0), _case("2026-01", 30.0), _case("2026-01", 40.0)]
    cases[1]["stall"]["stalled_share"] = 0.2
    cases[2]["stall"]["stalled_share"] = 0.4
    out = pva.aggregate_stall(cases)["pooled"]

    assert out["n_pairs"] == 3
    assert out["n_applicable"] == 2
    assert out["applicable_share"] == pytest.approx(2 / 3, abs=1e-4)
    assert out["stalled_share_median"] == pytest.approx(0.3)
    assert out["mean_profile_kmh"] == [1.0, 2.0]


def test_aggregate_stall_without_applicable_pairs_reports_nulls():
    out = pva.aggregate_stall([_case("2026-01", 5.0)])["pooled"]
    assert out["n_applicable"] == 0
    assert out["stalled_share_median"] is None
    assert out["mean_profile_kmh"] == []


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------
def test_markdown_report_keeps_the_legacy_tables_and_adds_the_new_ones():
    cases = [_case("2026-01", 30.0)]
    blocks: dict = {}
    pva.accumulate_day_blocks(blocks, cases[0])
    agg = pva.aggregate(cases)
    scored = pva.aggregate_blocks({"2026-01-01": blocks})
    stall = pva.aggregate_stall(cases)

    legacy_only = pva.markdown_report(agg, {"days": 1})
    full = pva.markdown_report(agg, {"days": 1}, scored, stall)

    # Every legacy line survives verbatim.
    for line in legacy_only.splitlines():
        assert line in full
    assert "### skill by rain-rate threshold" in full
    assert "## Fractions Skill Score" in full
    assert "## Flow stall diagnostic" in full


def test_horizon_label_flags_a_snapped_truth_frame():
    assert pva._horizon_label("20", {"truth_offset_min": 20}) == "+20 min"
    assert pva._horizon_label("45", {"truth_offset_min": 50}) == \
        "+45 min (truth +50)"


# ---------------------------------------------------------------------------
# compare_layer_a
# ---------------------------------------------------------------------------
def _payload(variant: str, per_day: dict[str, list[int]]) -> dict:
    """A minimal persistence_vs_advection payload with one stratum/horizon."""
    days = {}
    for day, counts in per_day.items():
        days[day] = {"pooled": {"10": {
            "truth_offset_min": 10,
            "n_cases": 1,
            "n_pixels": sum(counts),
            "thresholds": {"0.5": {
                "persistence": [5, 5, 5, 85],
                "advection": counts,
            }},
            "fss": {"0.5": {"2": [0.2, 1.0, 0.1, 1.0]}},
        }}}
    return {"meta": {"variant": variant}, "days": days}


def test_compare_identical_runs_reports_a_tie():
    payload = _payload("production", {
        "2026-01-01": [10, 5, 5, 80], "2026-01-02": [4, 12, 6, 78],
    })
    result = cmp_mod.compare(payload, payload, n_resamples=50)
    csi_rows = [r for r in result["rows"] if r["metric"] == "CSI"]

    assert len(csi_rows) == 1
    row = csi_rows[0]
    assert row["diff"] == pytest.approx(0.0)
    assert (row["lo"], row["hi"]) == pytest.approx((0.0, 0.0))
    assert cmp_mod.verdict(row) == "tie"
    assert row["n_days"] == 2
    assert not result["skipped"]


def test_compare_detects_a_uniform_improvement():
    """Every day improves by the same amount → the paired CI clears zero."""
    baseline = _payload("production", {
        "2026-01-01": [5, 2, 3, 90], "2026-01-02": [5, 2, 3, 90],
    })
    candidate = _payload("gated", {
        "2026-01-01": [6, 2, 2, 90], "2026-01-02": [6, 2, 2, 90],
    })
    result = cmp_mod.compare(baseline, candidate, n_resamples=50)
    row = next(r for r in result["rows"] if r["metric"] == "CSI")

    assert row["baseline"] == pytest.approx(0.5)
    assert row["candidate"] == pytest.approx(0.6)
    assert row["diff"] == pytest.approx(0.1)
    assert row["lo"] > 0
    assert cmp_mod.verdict(row) == "better"


def test_compare_marks_a_regression_as_worse():
    baseline = _payload("production", {"2026-01-01": [6, 2, 2, 90]})
    candidate = _payload("bad", {"2026-01-01": [5, 2, 3, 90]})
    row = next(
        r for r in cmp_mod.compare(baseline, candidate, n_resamples=20)["rows"]
        if r["metric"] == "CSI"
    )
    assert row["diff"] == pytest.approx(-0.1)
    assert cmp_mod.verdict(row) == "worse"


def test_compare_pools_fss_components_rather_than_averaging():
    baseline = _payload("production", {"2026-01-01": [5, 2, 3, 90]})
    candidate = _payload("cand", {"2026-01-01": [5, 2, 3, 90]})
    candidate["days"]["2026-01-01"]["pooled"]["10"]["fss"]["0.5"]["2"] = \
        [0.2, 1.0, 0.05, 1.0]
    row = next(
        r for r in cmp_mod.compare(baseline, candidate, n_resamples=20)["rows"]
        if r["metric"].startswith("FSS")
    )
    assert row["baseline"] == pytest.approx(0.9)   # 1 - 0.1/1.0
    assert row["candidate"] == pytest.approx(0.95)
    assert row["diff"] == pytest.approx(0.05)


def test_compare_intersects_on_days_and_reports_nothing_shared():
    a = _payload("a", {"2026-01-01": [5, 2, 3, 90]})
    b = _payload("b", {"2026-02-01": [5, 2, 3, 90]})
    result = cmp_mod.compare(a, b, n_resamples=10)
    assert result["rows"] == []
    assert result["skipped"]


def test_compare_rejects_a_run_without_day_blocks():
    with pytest.raises(SystemExit):
        cmp_mod.compare({"aggregate": {}}, _payload("b", {"d": [1, 1, 1, 1]}))


def test_compare_markdown_renders_a_table():
    payload = _payload("production", {"2026-01-01": [10, 5, 5, 80]})
    result = cmp_mod.compare(payload, payload, n_resamples=10)
    text = cmp_mod.markdown_report(result, {"method": "advection"})
    assert "## pooled" in text
    assert "| +10 min | 0.5 mm/h | CSI |" in text
    assert "FSS 2 km" in text


def test_fss_statistic_is_nan_without_a_denominator():
    assert math.isnan(cmp_mod.fss_statistic([(0.0, 0.0)]))


# ---------------------------------------------------------------------------
# run_case, end to end on synthetic composites
# ---------------------------------------------------------------------------
from datetime import datetime, timedelta, timezone  # noqa: E402

from dmi_nowcast_core.parse import RadarComposite  # noqa: E402

T0 = datetime(2026, 9, 2, 3, 0, tzinfo=timezone.utc)
SIZE = 64


def _composite(ts: datetime, centre_col: float) -> RadarComposite:
    """A smooth echo blob at ``centre_col``, on the real dBZ floor."""
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    blob = 70.0 * np.exp(
        -(((xx - centre_col) ** 2) / (2 * 8.0 ** 2)
          + ((yy - SIZE / 2) ** 2) / (2 * 12.0 ** 2))
    )
    return RadarComposite(
        reflectivity_dbz=(blob - 32.0).astype(np.float32),
        timestamp_utc=ts,
        projection="+proj=stere",
        xscale_m=500.0, yscale_m=500.0,
        corners_lonlat={}, zr_a=200.0, zr_b=1.6,
        quantity="DBZH", source_path=Path("synthetic"),
    )


class _FakeCache:
    """A CompositeCache stand-in: a band translating east at 8 px/frame.

    8 px/frame is 24 km/h on the 500 m grid at a 10-minute cadence, i.e.
    above the plan's 20 km/h bulk threshold, so the stall diagnostic is
    applicable on this synthetic.
    """

    def get(self, ts: datetime) -> RadarComposite:
        steps = (ts - T0).total_seconds() / 600.0
        return _composite(ts, 20.0 + 8.0 * steps)


def test_run_case_end_to_end_produces_every_block():
    spec = pva.CaseSpec(
        horizons_min=(10, 20), thresholds_mm_h=(0.5, 1.0), fss_scales_km=(2, 8),
    )
    case = pva.run_case(_FakeCache(), T0, spec)

    assert case is not None
    assert case["t"] == "202609020300"
    assert case["month"] == "2026-09"
    assert set(case["horizons"]) == {"10", "20"}
    for key, block in case["horizons"].items():
        assert block["truth_offset_min"] == int(key)
        assert block["n_pixels"] > 0
        assert set(block["thresholds"]) == {"0.5", "1"}
        assert set(block["fss"]["0.5"]) == {"2", "8"}
    # The band really is moving, so the diagnostic is applicable.
    assert case["stall"]["bulk_kmh"] > pva.SPEED_SPLIT_KMH
    assert case["stall"]["applicable"] is True
    assert case["bulk_speed_kmh"] > pva.SPEED_SPLIT_KMH


def test_run_case_advection_beats_persistence_on_a_translating_band():
    spec = pva.CaseSpec(
        horizons_min=(10, 20), thresholds_mm_h=(0.5,), fss_scales_km=(2,),
    )
    case = pva.run_case(_FakeCache(), T0, spec)
    for block in case["horizons"].values():
        pers = pva.table_of(block["thresholds"]["0.5"]["persistence"])
        adv = pva.table_of(block["thresholds"]["0.5"]["advection"])
        from dmi_nowcast_core.evaluate import csi as _csi
        assert _csi(adv) > _csi(pers)


def test_run_case_with_the_persistence_variant_scores_both_columns_alike():
    """Zero flow means the 'advection' forecast IS the observation.

    The strongest available check that the variant switch reaches the
    forecast and not just the metadata.
    """
    spec = pva.CaseSpec(
        variant="persistence", horizons_min=(10, 20),
        thresholds_mm_h=(0.5, 1.0), fss_scales_km=(2,),
    )
    case = pva.run_case(_FakeCache(), T0, spec)

    assert case["stall"]["bulk_kmh"] == pytest.approx(0.0)
    assert case["stall"]["applicable"] is False
    for block in case["horizons"].values():
        for threshold in ("0.5", "1"):
            assert block["thresholds"][threshold]["advection"] == \
                block["thresholds"][threshold]["persistence"]
        # No pixel can disagree between the observation and the forecast.
        assert block["o1a0t1"] == 0 and block["o1a0t0"] == 0
        assert block["o0a1t1"] == 0 and block["o0a1t0"] == 0


def test_run_case_skips_a_dry_frame():
    class _DryCache:
        def get(self, ts: datetime) -> RadarComposite:
            comp = _composite(ts, 20.0)
            return RadarComposite(
                reflectivity_dbz=np.full((SIZE, SIZE), -32.0, dtype=np.float32),
                timestamp_utc=comp.timestamp_utc,
                projection=comp.projection,
                xscale_m=comp.xscale_m, yscale_m=comp.yscale_m,
                corners_lonlat={}, zr_a=200.0, zr_b=1.6,
                quantity="DBZH", source_path=Path("synthetic"),
            )

    assert pva.run_case(_DryCache(), T0, pva.CaseSpec(horizons_min=(10,))) is None


def test_run_case_skips_when_a_truth_frame_is_off_cadence():
    class _SkewedCache(_FakeCache):
        def get(self, ts: datetime) -> RadarComposite:
            comp = super().get(ts)
            if ts > T0:
                shifted = ts + timedelta(minutes=4)
                return _composite(shifted, 20.0)
            return comp

    assert pva.run_case(_SkewedCache(), T0, pva.CaseSpec(horizons_min=(10,))) is None


# ---------------------------------------------------------------------------
# H4: variants that need more than two frames
# ---------------------------------------------------------------------------
from dmi_nowcast_core import variants as variants_mod  # noqa: E402


class _RecordingVariant:
    """A variant that records the keywords it was called with.

    Declares its needs through the same attributes a real entry uses, so
    what is under test is the harness's reading of the declaration and
    not a special case for the two shipped candidates.
    """

    def __init__(self, needs_history: int = 0, needs_future: bool = False):
        self.needs_history = needs_history
        self.needs_future = needs_future
        self.calls: list[dict] = []

    def __call__(self, prev_dbz, curr_dbz, rain_now_mm_h, *, pixel_km, **kwargs):
        self.calls.append({"pixel_km": pixel_km, **kwargs})
        zeros = np.zeros(np.asarray(curr_dbz).shape, dtype=np.float32)
        return zeros, zeros.copy()


@pytest.fixture
def registered():
    """Register throwaway variants and always remove them again.

    The registry is process-global and ``register_variant`` refuses to
    shadow, so a test that leaked a name would break the next run of the
    suite rather than itself — the worst kind of flake.
    """
    added: list[str] = []

    def _add(name: str, variant) -> str:
        variants_mod.register_variant(name, variant)
        added.append(name)
        return name

    yield _add
    for name in added:
        variants_mod._VARIANTS.pop(name, None)


def test_a_plain_variant_is_called_without_the_extra_keywords(registered):
    """The contract that keeps every pre-H4 entry working.

    ``bulk``'s signature has no ``history_dbz``; passing one would be a
    TypeError on every case. So the harness must pass nothing at all,
    not ``history_dbz=[]``.
    """
    spy = _RecordingVariant()
    name = registered("h4-plain", spy)
    spec = pva.CaseSpec(variant=name, horizons_min=(10,), thresholds_mm_h=(0.5,),
                        fss_scales_km=(2,))

    assert pva.run_case(_FakeCache(), T0, spec) is not None
    assert spy.calls == [{"pixel_km": 0.5}]


def test_a_history_variant_gets_the_older_frames_oldest_first(registered):
    """``history_dbz=[t-30, t-20]`` for ``needs_history=2``.

    The order is load-bearing: ``median3`` pairs them as (h0, h1),
    (h1, prev), so a reversed list would build the median out of three
    backwards estimates and produce a field pointing upwind — which no
    aggregate score would identify as an ordering bug.
    """
    spy = _RecordingVariant(needs_history=2)
    name = registered("h4-history", spy)
    spec = pva.CaseSpec(variant=name, horizons_min=(10,), thresholds_mm_h=(0.5,),
                        fss_scales_km=(2,))
    cache = _FakeCache()

    assert pva.run_case(cache, T0, spec) is not None
    (call,) = spy.calls
    assert set(call) == {"pixel_km", "history_dbz"}
    history = call["history_dbz"]
    assert len(history) == 2
    # The band translates east at 8 px/frame, so an older frame's echo
    # sits further west. Compare the centre of mass, oldest first.
    centres = [
        float((np.arange(SIZE) * np.maximum(f + 32.0, 0.0).sum(axis=0)).sum()
              / np.maximum(f + 32.0, 0.0).sum())
        for f in history
    ]
    assert centres[0] < centres[1]
    prev_frame = cache.get(T0 - timedelta(minutes=10)).reflectivity_dbz
    prev_centre = float(
        (np.arange(SIZE) * np.maximum(prev_frame + 32.0, 0.0).sum(axis=0)).sum()
        / np.maximum(prev_frame + 32.0, 0.0).sum()
    )
    assert centres[1] < prev_centre


def test_a_future_variant_gets_the_next_frame(registered):
    spy = _RecordingVariant(needs_future=True)
    name = registered("h4-future", spy)
    spec = pva.CaseSpec(variant=name, horizons_min=(10,), thresholds_mm_h=(0.5,),
                        fss_scales_km=(2,))
    cache = _FakeCache()

    assert pva.run_case(cache, T0, spec) is not None
    (call,) = spy.calls
    assert set(call) == {"pixel_km", "future_dbz"}
    np.testing.assert_array_equal(
        call["future_dbz"],
        cache.get(T0 + timedelta(minutes=10)).reflectivity_dbz,
    )


def test_case_spec_widens_the_window_for_a_declaring_variant(registered):
    plain = registered("h4-reach-plain", _RecordingVariant())
    hist = registered("h4-reach-hist", _RecordingVariant(needs_history=2))
    fut = registered("h4-reach-fut", _RecordingVariant(needs_future=True))

    assert pva.CaseSpec(variant=plain).frames_before == 1
    assert pva.CaseSpec(variant=hist).frames_before == 3
    # The future frame is t+10, which every default horizon already
    # reaches — but a spec whose furthest offset is shorter must still
    # cover it, or run_case would parse a frame nobody checked.
    short = pva.CaseSpec(variant=fut, horizons_min=(10,), offsets_min=(5,))
    assert short.frames_ahead == 1
    assert pva.CaseSpec(
        variant=plain, horizons_min=(10,), offsets_min=(5,)).frames_ahead == 0


def test_cache_grows_by_one_slot_per_declared_frame(registered):
    """One extra parsed composite per declared frame, and no more."""
    plain = registered("h4-cache-plain", _RecordingVariant())
    both = registered("h4-cache-both", _RecordingVariant(2, True))

    assert pva.cache_maxsize(pva.CaseSpec(variant=plain)) == pva.BASE_CACHE_FRAMES
    assert pva.cache_maxsize(pva.CaseSpec(variant=both)) == \
        pva.BASE_CACHE_FRAMES + 3


def test_run_case_counts_a_history_frame_that_is_off_cadence(registered):
    """A file that exists but whose timestamp is wrong is a skip, not a case."""
    spy = _RecordingVariant(needs_history=2)
    name = registered("h4-skew", spy)
    spec = pva.CaseSpec(variant=name, horizons_min=(10,), thresholds_mm_h=(0.5,),
                        fss_scales_km=(2,))

    class _SkewedHistory(_FakeCache):
        def get(self, ts: datetime):
            comp = super().get(ts)
            if ts < T0 - timedelta(minutes=10):
                return _composite(ts + timedelta(minutes=3), 20.0)
            return comp

    skips = pva.empty_skips()
    assert pva.run_case(_SkewedHistory(), T0, spec, skips) is None
    assert skips["history"] == 1
    assert skips["window"] == 0 and skips["future"] == 0
    assert spy.calls == []


def test_run_case_counts_a_missing_history_frame(registered):
    """An absent file is a skip too, and it is counted the same way."""
    spy = _RecordingVariant(needs_history=2)
    name = registered("h4-missing", spy)
    spec = pva.CaseSpec(variant=name, horizons_min=(10,), thresholds_mm_h=(0.5,),
                        fss_scales_km=(2,))

    class _HoleyCache(_FakeCache):
        def get(self, ts: datetime):
            if ts <= T0 - timedelta(minutes=20):
                raise OSError("no such frame")
            return super().get(ts)

    skips = pva.empty_skips()
    assert pva.run_case(_HoleyCache(), T0, spec, skips) is None
    assert skips["history"] == 1


def test_run_case_counts_the_prev_frame_gap_as_a_window_skip():
    """The pre-H4 silent ``continue`` is now a counted ``window`` skip."""
    class _BadPrev:
        def get(self, ts: datetime):
            # prev lands 4 minutes before t instead of 10.
            if ts < T0:
                return _composite(T0 - timedelta(minutes=4), 20.0)
            return _composite(ts, 20.0)

    skips = pva.empty_skips()
    assert pva.run_case(_BadPrev(), T0, pva.CaseSpec(horizons_min=(10,)),
                        skips) is None
    assert skips["window"] == 1


def test_run_case_without_a_skip_dict_still_just_returns_none():
    """The counter is optional; the old two-argument call still works."""
    assert pva.run_case(_FakeCache(), T0, pva.CaseSpec(horizons_min=(10,))) \
        is not None


def test_contiguous_accepts_the_cadence_and_rejects_a_hole():
    step = timedelta(minutes=pva.FRAME_INTERVAL_MIN)
    run = [T0 + i * step for i in range(4)]
    assert pva.contiguous(run)
    assert pva.contiguous([])
    holed = run[:2] + [t + timedelta(minutes=10) for t in run[2:]]
    assert not pva.contiguous(holed)


def test_variant_flow_refuses_to_run_a_needy_variant_without_frames(registered):
    """A caller that forgot the frames gets told, not a wrong field."""
    name = registered("h4-strict", _RecordingVariant(needs_history=2))
    comp = _composite(T0, 20.0)
    rain = np.ones((SIZE, SIZE), dtype=np.float32)
    with pytest.raises(ValueError, match="needs 2 history frames"):
        pva.variant_flow(comp, comp, rain, name)

    fut = registered("h4-strict-future", _RecordingVariant(needs_future=True))
    with pytest.raises(ValueError, match="needs the frame after now"):
        pva.variant_flow(comp, comp, rain, fut)


def test_variant_flow_passes_only_the_newest_history_frames(registered):
    """Given more history than declared, the newest ``needs_history`` win."""
    spy = _RecordingVariant(needs_history=1)
    name = registered("h4-trim", spy)
    old = _composite(T0 - timedelta(minutes=30), 4.0)
    mid = _composite(T0 - timedelta(minutes=20), 12.0)
    comp = _composite(T0, 20.0)
    rain = np.ones((SIZE, SIZE), dtype=np.float32)

    pva.variant_flow(comp, comp, rain, name, [old, mid])
    (call,) = spy.calls
    assert len(call["history_dbz"]) == 1
    np.testing.assert_array_equal(call["history_dbz"][0], mid.reflectivity_dbz)


# --- the skip tally reaches the report --------------------------------
def test_run_day_reports_its_skips(tmp_path: Path):
    """No archive on disk, so every candidate is a window skip of zero cases."""
    day, cases, blocks, err, skips = pva.run_day(
        str(tmp_path), "2026-09-02", 1, pva.CaseSpec(horizons_min=(10,)),
    )
    assert day == "2026-09-02" and err is None
    assert cases == [] and blocks == {}
    assert set(skips) == set(pva.SKIP_REASONS)
    assert sum(skips.values()) == 0  # no frames at all means no candidates


def test_empty_skips_has_every_reason_at_zero():
    assert pva.empty_skips() == {"window": 0, "history": 0, "future": 0}
    assert pva.SKIP_REASONS == ("window", "history", "future")


def test_markdown_header_carries_the_skip_tally():
    """The count has to be readable next to the case count, not only in JSON.

    A candidate that scored 1,700 of the baseline's 1,925 cases is not a
    paired comparison, and the person reading the markdown is the one who
    has to notice.
    """
    meta = {
        "variant": "median3",
        "cases (frames)": 1700,
        "skipped cases": 225,
        "skipped by reason": "window=15, history=210, future=0",
        "extra frames needed": "history=2, future=no",
    }
    report = pva.markdown_report({"pooled": {}}, meta)
    assert "**skipped cases**: 225" in report
    assert "window=15, history=210, future=0" in report
    assert "**extra frames needed**: history=2, future=no" in report


# --- the CLI switch ---------------------------------------------------
def test_variant_list_prints_every_name_and_exits():
    """``--variant-list`` must work WITHOUT --archive-dir.

    The caller is ``sidecar/deploy/layer_a.sh`` checking a name before it
    starts a four-hour container; it has no archive path to offer.
    """
    with pytest.raises(SystemExit) as excinfo:
        pva.main(["--variant-list"])
    assert excinfo.value.code == 0


def test_variant_list_output_is_the_registry(capsys):
    from dmi_nowcast_core.variants import list_variants as _names

    with pytest.raises(SystemExit):
        pva.main(["--variant-list"])
    printed = capsys.readouterr().out.split()
    assert tuple(printed) == _names()
    assert "oracle" in printed and "lucaskanade" in printed


def test_variant_choices_do_not_swamp_the_usage_line(capsys):
    """31 names inline would make ``--help`` unreadable; metavar keeps it short."""
    with pytest.raises(SystemExit):
        pva.main(["--help"])
    usage = capsys.readouterr().out
    assert "[--variant NAME]" in usage
    assert "farneback_w21_l3_p5" not in usage.split("options:")[0]


def test_align_with_unions_the_frame_requirements():
    """A baseline aligned with a hungry candidate skips the same cases."""
    from persistence_vs_advection import CaseSpec

    plain = CaseSpec(variant="confidence", horizons_min=(10, 20))
    aligned = CaseSpec(variant="confidence", horizons_min=(10, 20), align_with="median3")
    assert plain.requirements.history == 0 and aligned.requirements.history == 2
    assert aligned.frames_before == plain.frames_before + 2
    future = CaseSpec(variant="confidence", horizons_min=(10,), align_with="oracle")
    assert future.requirements.future is True and future.frames_ahead >= 1
