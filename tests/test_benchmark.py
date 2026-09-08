"""Hand-computed cases for every metric in ``dmi_nowcast_core.benchmark``.

The Phase H acceptance gate is a number produced by these functions, so a
silent arithmetic slip here would not fail anywhere else — it would just
ship the wrong candidate. Every test therefore pins a value worked out by
hand or from a published definition (sklearn's average precision, the
Mann-Whitney identity for ROC, the Hersbach CRPS estimator), never a
value read back from an earlier run of this same code.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from dmi_nowcast_core.benchmark import (
    DECORRELATION_LEVEL,
    block_bootstrap,
    brier_decomposition,
    contingency_sum,
    crps_ensemble,
    decorrelation_time,
    field_correlation,
    paired_block_bootstrap,
    pr_auc,
    rank_histogram,
    roc_auc,
    stall_diagnostic,
)
from dmi_nowcast_core.evaluate import ContingencyTable, csi


# ---------------------------------------------------------------------------
# brier_decomposition
# ---------------------------------------------------------------------------
def test_brier_decomposition_on_bin_centres_is_exact():
    """p on the bin representatives: BS = REL - RES + UNC to the last bit.

    Two bins, forecasts at their centres. By hand:
    BS = (0.0625 + 0.5625 + 0.0625 + 0.0625)/4 = 0.1875;
    bin 0 has p̄=0.25, ō=0.5; bin 1 has p̄=0.75, ō=1.0; ō=0.75.
    REL = 0.0625, RES = 0.0625, UNC = 0.1875.
    """
    p = np.array([0.25, 0.25, 0.75, 0.75])
    y = np.array([0.0, 1.0, 1.0, 1.0])
    d = brier_decomposition(p, y, n_bins=2)

    assert d["brier"] == pytest.approx(0.1875)
    assert d["reliability"] == pytest.approx(0.0625)
    assert d["resolution"] == pytest.approx(0.0625)
    assert d["uncertainty"] == pytest.approx(0.1875)
    assert d["base_rate"] == pytest.approx(0.75)
    assert d["n"] == 4
    # Identity, exactly, because every p IS its bin's mean.
    assert d["brier_binned"] == pytest.approx(d["brier"], abs=1e-15)
    assert d["bss"] == pytest.approx(0.0)


def test_brier_decomposition_perfect_forecast():
    p = np.array([0.0, 0.0, 1.0, 1.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    d = brier_decomposition(p, y)

    assert d["brier"] == pytest.approx(0.0)
    assert d["reliability"] == pytest.approx(0.0)
    assert d["resolution"] == pytest.approx(0.25)
    assert d["uncertainty"] == pytest.approx(0.25)
    assert d["bss"] == pytest.approx(1.0)


def test_brier_decomposition_identity_on_random_data():
    """On raw p the identity holds only for the WITHIN-BIN MEAN forecast.

    Computed here independently: replace every p by its bin's mean and
    score that. The raw Brier differs by the within-bin spread, which is
    small with ten bins but not zero — the test pins both facts.
    """
    rng = np.random.default_rng(3)
    p = rng.random(4000)
    y = (rng.random(4000) < p).astype(float)
    d = brier_decomposition(p, y, n_bins=10)

    index = np.clip(np.floor(p * 10).astype(int), 0, 9)
    means = np.array([
        p[index == k].mean() if np.any(index == k) else 0.0 for k in range(10)
    ])
    binned_brier = float(np.mean((means[index] - y) ** 2))

    identity = d["reliability"] - d["resolution"] + d["uncertainty"]
    assert identity == pytest.approx(binned_brier, abs=1e-12)
    assert d["brier_binned"] == pytest.approx(binned_brier, abs=1e-12)
    # ...and the raw score is close, but not identical.
    assert d["brier"] == pytest.approx(binned_brier, abs=0.01)


def test_brier_decomposition_climatological_forecast_has_zero_skill():
    y = np.array([0.0, 0.0, 0.0, 1.0])
    p = np.full(4, 0.25)
    d = brier_decomposition(p, y)
    assert d["brier"] == pytest.approx(d["uncertainty"])
    assert d["bss"] == pytest.approx(0.0)
    assert d["resolution"] == pytest.approx(0.0)


def test_brier_decomposition_drops_non_finite_pairs():
    p = np.array([0.25, np.nan, 0.75, 0.75])
    y = np.array([0.0, 1.0, 1.0, np.nan])
    d = brier_decomposition(p, y, n_bins=2)
    assert d["n"] == 2


def test_brier_decomposition_edge_cases():
    empty = brier_decomposition(np.array([]), np.array([]))
    assert empty["n"] == 0
    assert math.isnan(empty["brier"])

    # Every outcome identical: climatology is perfect, BSS is 0/0.
    constant = brier_decomposition(np.array([0.3, 0.4]), np.array([1.0, 1.0]))
    assert constant["uncertainty"] == pytest.approx(0.0)
    assert math.isnan(constant["bss"])

    with pytest.raises(ValueError):
        brier_decomposition(np.array([0.5]), np.array([0.5]))
    with pytest.raises(ValueError):
        brier_decomposition(np.array([1.5]), np.array([1.0]))
    with pytest.raises(ValueError):
        brier_decomposition(np.array([0.5, 0.5]), np.array([1.0]))


# ---------------------------------------------------------------------------
# roc_auc / pr_auc
# ---------------------------------------------------------------------------
def test_roc_auc_four_point_case():
    """Three of the four positive/negative pairs are ordered correctly."""
    p = np.array([0.1, 0.4, 0.35, 0.8])
    y = np.array([0, 0, 1, 1])
    assert roc_auc(p, y) == pytest.approx(0.75)


def test_roc_auc_ties_count_as_half():
    assert roc_auc(np.array([0.5, 0.5]), np.array([0, 1])) == pytest.approx(0.5)
    # One tied pair out of four, the rest correctly ordered → 3.5/4.
    p = np.array([0.1, 0.4, 0.4, 0.8])
    y = np.array([0, 0, 1, 1])
    assert roc_auc(p, y) == pytest.approx(0.875)


def test_roc_auc_extremes_and_empty_class():
    assert roc_auc(np.array([1.0, 0.0]), np.array([1, 0])) == pytest.approx(1.0)
    assert roc_auc(np.array([0.0, 1.0]), np.array([1, 0])) == pytest.approx(0.0)
    assert math.isnan(roc_auc(np.array([0.2, 0.8]), np.array([0, 0])))
    assert math.isnan(roc_auc(np.array([0.2, 0.8]), np.array([1, 1])))


def test_pr_auc_with_ties():
    """AP over the distinct scores: 0.5·0.5 + 0.5·(2/3) = 0.58333.

    The tied top pair is one threshold, so precision there is 1/2 rather
    than the 1.0 a tie-splitting implementation would report.
    """
    p = np.array([0.9, 0.9, 0.5, 0.1])
    y = np.array([1, 0, 1, 0])
    assert pr_auc(p, y) == pytest.approx(0.5 * 0.5 + 0.5 * (2.0 / 3.0))


def test_pr_auc_perfect_and_inverted():
    p = np.array([0.9, 0.8, 0.2, 0.1])
    assert pr_auc(p, np.array([1, 1, 0, 0])) == pytest.approx(1.0)
    # Inverted: precision 1/3 at recall 0.5, 1/2 at recall 1.0.
    assert pr_auc(p, np.array([0, 0, 1, 1])) == pytest.approx(
        0.5 * (1.0 / 3.0) + 0.5 * 0.5
    )


def test_pr_auc_without_positives_is_nan():
    assert math.isnan(pr_auc(np.array([0.1, 0.9]), np.array([0, 0])))


def test_pr_auc_is_independent_of_input_order():
    rng = np.random.default_rng(11)
    p = rng.random(200)
    y = (rng.random(200) < 0.3).astype(float)
    shuffle = rng.permutation(200)
    assert pr_auc(p, y) == pytest.approx(pr_auc(p[shuffle], y[shuffle]))


# ---------------------------------------------------------------------------
# crps_ensemble / rank_histogram
# ---------------------------------------------------------------------------
def test_crps_zero_when_every_member_equals_the_observation():
    members = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    obs = np.array([1.0, 2.0])
    mean, per_point = crps_ensemble(members, obs)
    assert mean == pytest.approx(0.0)
    assert per_point == pytest.approx([0.0, 0.0])


def test_crps_single_member_is_absolute_error():
    mean, per_point = crps_ensemble(np.array([[3.0, -1.0]]), np.array([1.0, 1.0]))
    assert per_point == pytest.approx([2.0, 2.0])
    assert mean == pytest.approx(2.0)


def test_crps_two_member_case_by_hand():
    """members {0, 2}, obs 1: mean|x−y| = 1, mean|x−x'| = 1, CRPS = 0.5."""
    mean, _ = crps_ensemble(np.array([[0.0], [2.0]]), np.array([1.0]))
    assert mean == pytest.approx(0.5)


def test_crps_matches_the_naive_double_sum():
    rng = np.random.default_rng(7)
    members = rng.normal(size=(9, 40))
    obs = rng.normal(size=40)
    _, per_point = crps_ensemble(members, obs)
    naive = (
        np.abs(members - obs).mean(axis=0)
        - 0.5 * np.abs(members[:, None, :] - members[None, :, :]).mean(axis=(0, 1))
    )
    assert per_point == pytest.approx(naive)


def test_crps_skips_non_finite_points():
    members = np.array([[1.0, 1.0, np.nan], [1.0, 1.0, 1.0]])
    obs = np.array([1.0, np.nan, 1.0])
    mean, per_point = crps_ensemble(members, obs)
    assert mean == pytest.approx(0.0)
    assert math.isnan(per_point[1])
    assert math.isnan(per_point[2])


def test_rank_histogram_degenerate_ensemble():
    """Identical members, observation above them all → the top rank only."""
    members = np.zeros((3, 5))
    counts = rank_histogram(members, np.ones(5))
    assert counts.tolist() == [0, 0, 0, 5]

    counts_below = rank_histogram(members, np.full(5, -1.0))
    assert counts_below.tolist() == [5, 0, 0, 0]


def test_rank_histogram_ties_are_spread_and_reproducible():
    """An observation equal to every member lands anywhere, but the same
    anywhere for the same seed — otherwise a dry field would pile into
    rank 0 and fake an under-dispersive ensemble."""
    members = np.zeros((3, 400))
    obs = np.zeros(400)
    counts = rank_histogram(members, obs, seed=0)
    assert counts.sum() == 400
    assert counts.size == 4
    assert np.all(counts > 0)  # all four ranks reachable
    assert counts.tolist() == rank_histogram(members, obs, seed=0).tolist()
    assert counts.tolist() != rank_histogram(members, obs, seed=1).tolist()


def test_rank_histogram_places_the_observation_between_members():
    members = np.array([[0.0], [2.0], [4.0]])
    assert rank_histogram(members, np.array([1.0])).tolist() == [0, 1, 0, 0]
    assert rank_histogram(members, np.array([3.0])).tolist() == [0, 0, 1, 0]


def test_rank_histogram_skips_non_finite():
    members = np.array([[0.0, 0.0], [0.0, np.nan]])
    counts = rank_histogram(members, np.array([1.0, 1.0]))
    assert counts.sum() == 1


# ---------------------------------------------------------------------------
# field_correlation / decorrelation_time
# ---------------------------------------------------------------------------
def test_field_correlation_perfect_and_inverted():
    obs = np.arange(12, dtype=float).reshape(3, 4)
    assert field_correlation(obs, obs) == pytest.approx(1.0)
    assert field_correlation(-obs, obs) == pytest.approx(-1.0)


def test_field_correlation_honours_mask_and_non_finite():
    obs = np.array([[1.0, 2.0], [3.0, 100.0]])
    pred = np.array([[1.0, 2.0], [3.0, -100.0]])
    mask = np.array([[True, True], [True, False]])
    assert field_correlation(pred, obs, mask) == pytest.approx(1.0)

    pred_nan = np.array([[1.0, 2.0], [3.0, np.nan]])
    obs_nan = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert field_correlation(pred_nan, obs_nan) == pytest.approx(1.0)


def test_field_correlation_undefined_cases_are_nan():
    assert math.isnan(field_correlation(np.ones((2, 2)), np.ones((2, 2))))
    assert math.isnan(field_correlation(np.array([1.0]), np.array([2.0])))


def test_decorrelation_time_interpolates_between_leads():
    """0.5 → 0.2 across 20→30 min crosses 1/e at 24.404 min."""
    leads = [10.0, 20.0, 30.0]
    corr = [0.9, 0.5, 0.2]
    expected = 20.0 + 10.0 * (0.5 - DECORRELATION_LEVEL) / (0.5 - 0.2)
    assert decorrelation_time(leads, corr) == pytest.approx(expected)
    assert decorrelation_time(leads, corr) == pytest.approx(24.4040, abs=1e-3)


def test_decorrelation_time_never_crossing_is_nan():
    assert math.isnan(decorrelation_time([10, 20, 30], [0.99, 0.95, 0.9]))


def test_decorrelation_time_already_below_at_the_first_lead():
    """Cannot be bracketed, so the first lead is returned as an upper bound."""
    assert decorrelation_time([10, 20], [0.2, 0.1]) == pytest.approx(10.0)


def test_decorrelation_time_validation():
    assert math.isnan(decorrelation_time([10, 20], [np.nan, np.nan]))
    with pytest.raises(ValueError):
        decorrelation_time([20, 10], [0.9, 0.2])
    with pytest.raises(ValueError):
        decorrelation_time([10, 20], [0.9])


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------
def _table(hits: int, misses: int, false_alarms: int, cn: int = 100):
    return ContingencyTable(
        hits=hits, misses=misses, false_alarms=false_alarms,
        correct_negatives=cn,
    )


def test_contingency_sum_pools_counts():
    total = contingency_sum([_table(1, 2, 3, 4), _table(10, 20, 30, 40)])
    assert (total.hits, total.misses, total.false_alarms, total.correct_negatives) \
        == (11, 22, 33, 44)
    empty = contingency_sum([])
    assert empty.total == 0


def test_block_bootstrap_on_a_deterministic_statistic_has_zero_width():
    blocks = [_table(10, 5, 5)] * 5
    point, lo, hi = block_bootstrap(
        blocks, lambda bs: csi(contingency_sum(bs)), n_resamples=50,
    )
    assert point == pytest.approx(0.5)
    assert lo == pytest.approx(0.5)
    assert hi == pytest.approx(0.5)


def test_block_bootstrap_widens_when_days_differ():
    blocks = [_table(10, 5, 5), _table(1, 19, 19), _table(15, 1, 1)]
    point, lo, hi = block_bootstrap(
        blocks, lambda bs: csi(contingency_sum(bs)), n_resamples=400, seed=1,
    )
    assert lo < point < hi


def test_block_bootstrap_pools_rather_than_averages():
    """A big day and a tiny one: the pooled CSI is not their mean."""
    blocks = [_table(1000, 0, 0), _table(0, 1, 0)]
    point, _, _ = block_bootstrap(
        blocks, lambda bs: csi(contingency_sum(bs)), n_resamples=10,
    )
    assert point == pytest.approx(1000 / 1001)


def test_block_bootstrap_is_reproducible_and_validated():
    blocks = [_table(h, 20 - h, 20 - h) for h in (1, 4, 7, 10, 14, 19)]
    stat = lambda bs: csi(contingency_sum(bs))  # noqa: E731
    first = block_bootstrap(blocks, stat, n_resamples=100, seed=4)
    assert first == block_bootstrap(blocks, stat, n_resamples=100, seed=4)
    # A different seed draws different days, so the interval moves; the
    # point estimate, which never resamples, does not.
    other = block_bootstrap(blocks, stat, n_resamples=100, seed=5)
    assert other[0] == pytest.approx(first[0])
    assert (other[1], other[2]) != (first[1], first[2])

    assert all(math.isnan(v) for v in block_bootstrap([], stat))
    with pytest.raises(ValueError):
        block_bootstrap(blocks, stat, ci=1.5)
    with pytest.raises(ValueError):
        block_bootstrap(blocks, stat, n_resamples=0)


def test_paired_block_bootstrap_zero_difference_when_runs_are_identical():
    blocks = [_table(10, 5, 5), _table(3, 7, 2)]
    point, lo, hi = paired_block_bootstrap(
        blocks, list(blocks), lambda bs: csi(contingency_sum(bs)),
        n_resamples=100,
    )
    assert (point, lo, hi) == pytest.approx((0.0, 0.0, 0.0))


def test_paired_block_bootstrap_constant_offset_has_zero_width():
    """Every day improves by exactly the same amount → the CI collapses.

    This is the pairing doing its work: the individual CSIs vary across
    days, so an unpaired interval on either run would be wide.
    """
    candidate = [_table(6, 2, 2)] * 4      # CSI 0.6
    baseline = [_table(5, 2, 3)] * 4       # CSI 0.5
    point, lo, hi = paired_block_bootstrap(
        candidate, baseline, lambda bs: csi(contingency_sum(bs)),
        n_resamples=100,
    )
    assert point == pytest.approx(0.1)
    assert lo == pytest.approx(0.1)
    assert hi == pytest.approx(0.1)


def test_paired_block_bootstrap_uses_the_same_days_for_both():
    """A perfectly-correlated pair must stay perfectly correlated.

    If the two runs were resampled with independent indices the interval
    on the difference would be wide even here, where the difference is
    zero on every single day.
    """
    candidate = [_table(h, 20 - h, 5) for h in (2, 8, 14, 19)]
    baseline = list(candidate)
    _, lo, hi = paired_block_bootstrap(
        candidate, baseline, lambda bs: csi(contingency_sum(bs)),
        n_resamples=200, seed=2,
    )
    assert (lo, hi) == pytest.approx((0.0, 0.0))


def test_paired_block_bootstrap_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        paired_block_bootstrap([1, 2], [1], lambda bs: float(sum(bs)))


# ---------------------------------------------------------------------------
# stall_diagnostic
# ---------------------------------------------------------------------------
def _stalled_band(fast_px_per_frame: float = 20.0):
    """A 20-column band whose rear half is frozen and front half moves east."""
    grid = np.zeros((40, 40), dtype=np.float32)
    grid[10:30, 0:20] = 1.0  # rain, 1 mm/h
    vy = np.zeros((40, 40), dtype=np.float32)
    vx = np.zeros((40, 40), dtype=np.float32)
    vx[:, 10:20] = fast_px_per_frame
    return vy, vx, grid


def test_stall_diagnostic_half_stalled_band():
    """Rear half frozen: stalled share 0.5 and a two-level profile.

    500 m pixels at a 10-min cadence make 1 px/frame = 3 km/h, so the
    moving half runs at 60 km/h, the bulk at 30 km/h (rain-weighted over
    equal areas), and the pair is applicable.
    """
    vy, vx, rain = _stalled_band()
    out = stall_diagnostic(
        vy, vx, rain, pixel_km=0.5, dt_min=10.0, n_bins=2,
    )

    assert out["bulk_kmh"] == pytest.approx(30.0)
    assert out["bulk_dir_unit"] == pytest.approx((0.0, 1.0))
    assert out["stalled_share"] == pytest.approx(0.5)
    assert out["profile_kmh"] == pytest.approx([0.0, 60.0])
    assert out["applicable"] is True
    assert out["n_wet"] == 20 * 20


def test_stall_diagnostic_profile_runs_rear_to_front():
    vy, vx, rain = _stalled_band()
    out = stall_diagnostic(vy, vx, rain, pixel_km=0.5, dt_min=10.0, n_bins=4)
    assert out["profile_kmh"] == pytest.approx([0.0, 0.0, 60.0, 60.0])


def test_stall_diagnostic_uniform_band_is_flat_and_unstalled():
    grid = np.zeros((40, 40), dtype=np.float32)
    grid[10:30, 0:20] = 1.0
    vy = np.zeros((40, 40), dtype=np.float32)
    vx = np.full((40, 40), 10.0, dtype=np.float32)
    out = stall_diagnostic(vy, vx, grid, pixel_km=0.5, dt_min=10.0, n_bins=5)

    assert out["stalled_share"] == pytest.approx(0.0)
    assert out["profile_kmh"] == pytest.approx([30.0] * 5)
    assert out["applicable"] is True


def test_stall_diagnostic_not_applicable_below_the_bulk_threshold():
    vy, vx, rain = _stalled_band(fast_px_per_frame=2.0)  # bulk 3 km/h
    out = stall_diagnostic(vy, vx, rain, pixel_km=0.5, dt_min=10.0, n_bins=2)
    assert out["bulk_kmh"] == pytest.approx(3.0)
    assert out["applicable"] is False
    # The share is still reported so a report can say how it was computed.
    assert out["stalled_share"] == pytest.approx(0.5)


def test_stall_diagnostic_direction_follows_the_bulk_vector():
    """A southward band: the rear is the northern (low-row) end."""
    grid = np.zeros((40, 40), dtype=np.float32)
    grid[0:20, 10:30] = 1.0
    vy = np.zeros((40, 40), dtype=np.float32)
    vy[10:20, :] = 20.0  # the southern half moves
    vx = np.zeros((40, 40), dtype=np.float32)
    out = stall_diagnostic(vy, vx, grid, pixel_km=0.5, dt_min=10.0, n_bins=2)

    assert out["bulk_dir_unit"] == pytest.approx((1.0, 0.0))
    assert out["profile_kmh"] == pytest.approx([0.0, 60.0])


def test_stall_diagnostic_speed_conversion_follows_pixel_and_cadence():
    grid = np.ones((10, 10), dtype=np.float32)
    vy = np.zeros((10, 10), dtype=np.float32)
    vx = np.full((10, 10), 4.0, dtype=np.float32)
    # 4 px/frame × 2 km / (5/60 h) = 96 km/h.
    out = stall_diagnostic(vy, vx, grid, pixel_km=2.0, dt_min=5.0, n_bins=1)
    assert out["bulk_kmh"] == pytest.approx(96.0)


def test_stall_diagnostic_dry_and_degenerate_fields():
    dry = np.zeros((8, 8), dtype=np.float32)
    zero = np.zeros((8, 8), dtype=np.float32)
    out = stall_diagnostic(zero, zero, dry, pixel_km=0.5, dt_min=10.0)
    assert out["n_wet"] == 0
    assert math.isnan(out["bulk_kmh"])
    assert out["applicable"] is False
    assert all(math.isnan(v) for v in out["profile_kmh"])

    # Wet everywhere but motionless: a bulk direction does not exist.
    wet = np.ones((8, 8), dtype=np.float32)
    still = stall_diagnostic(zero, zero, wet, pixel_km=0.5, dt_min=10.0)
    assert still["bulk_kmh"] == pytest.approx(0.0)
    assert still["stalled_share"] == pytest.approx(1.0)
    assert all(math.isnan(v) for v in still["profile_kmh"])


def test_stall_diagnostic_validation():
    field = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        stall_diagnostic(field, field, field, pixel_km=0.0, dt_min=10.0)
    with pytest.raises(ValueError):
        stall_diagnostic(field, field, field, pixel_km=0.5, dt_min=0.0)
    with pytest.raises(ValueError):
        stall_diagnostic(field, field, np.zeros((3, 3)), pixel_km=0.5, dt_min=10.0)
