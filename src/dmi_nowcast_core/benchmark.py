"""Verification metrics for the Phase H benchmark suite.

Why this module exists separately from :mod:`dmi_nowcast_core.evaluate`:
``evaluate`` holds the categorical scores the live pipeline and the older
backtests already depend on, and several callers pin its API. Phase H
needs a second, larger family — a *decomposed* probability score, ranked
and ensemble scores, a lifetime estimate, and the resampling machinery
that turns any score into a confidence interval on a **difference**.
Keeping them here leaves ``evaluate`` stable and gives the benchmark one
import.

The three layers of the Phase H suite consume different parts:

* **Layer A** (field skill, radar truth): :func:`field_correlation` and
  :func:`decorrelation_time` for the Imhoff-style lifetime, plus
  :func:`crps_ensemble` and :func:`rank_histogram` on the STEPS members,
  plus :func:`stall_diagnostic` on the completed flow.
* **Layer B** (probability skill, gauge truth): :func:`brier_decomposition`,
  :func:`reliability_bins`, :func:`roc_auc`, :func:`pr_auc`.
* **Both**: :func:`block_bootstrap` / :func:`paired_block_bootstrap` for the
  day-block confidence intervals the acceptance gate is written in terms
  of, and :func:`contingency_sum` so a pooled CSI over resampled days is
  computed from summed tables rather than averaged per-day scores.

Conventions, uniform across the module:

* numpy arrays in, floats / dicts / small tuples out. No pandas.
* Non-finite inputs are **dropped in pairs**, never imputed. Every
  function reports how many points survived where the count matters.
* An undefined score is ``NaN``, not an exception and not a sentinel
  like ``0``: an empty ROC class and a genuinely zero AUC must not read
  the same in a report.
* Probability bins follow ``quality_report._bin_index`` /
  ``sql/reliability_pooled.sql``: ``bin k`` covers ``[k/K, (k+1)/K)``
  with ``p == 1.0`` folded into the last bin. One binning convention in
  the project, so a number here and a number from DuckDB agree.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import numpy as np

from .evaluate import ContingencyTable

__all__ = [
    "brier_decomposition",
    "reliability_bins",
    "roc_auc",
    "pr_auc",
    "crps_ensemble",
    "rank_histogram",
    "field_correlation",
    "decorrelation_time",
    "block_bootstrap",
    "paired_block_bootstrap",
    "contingency_sum",
    "stall_diagnostic",
    "DECORRELATION_LEVEL",
]

#: Germann & Zawadzki's "lifetime" level: the lead at which the Eulerian /
#: Lagrangian correlation has fallen to 1/e. Imhoff et al. 2020 quote
#: 25 min for 1-h events on Dutch lowland catchments with this convention,
#: which is the external anchor the plan calibrates Denmark against.
DECORRELATION_LEVEL = 1.0 / math.e

#: Probabilities are allowed to arrive this far outside [0, 1] from a float
#: round-trip before it is treated as a caller bug rather than noise.
_PROB_TOLERANCE = 1e-9


# ---------------------------------------------------------------------------
# Probability scores
# ---------------------------------------------------------------------------
def brier_decomposition(
    p: np.ndarray | Sequence[float],
    y: np.ndarray | Sequence[float],
    n_bins: int = 10,
) -> dict[str, float | int]:
    """Murphy's decomposition of the Brier score into REL − RES + UNC.

    ``p`` is a forecast probability in [0, 1]; ``y`` is a binary outcome.
    Pairs where either is non-finite are dropped.

    Returned keys:

    ``brier``
        ``mean((p − y)²)`` on the raw probabilities — the score itself.
    ``reliability``
        ``Σ n_k (p̄_k − ō_k)² / N``. Lower is better; it is the calibration
        error, the distance from the diagonal of the reliability diagram.
    ``resolution``
        ``Σ n_k (ō_k − ō)² / N``. Higher is better; it is how far the
        forecast moves the outcome frequency away from climatology.
    ``uncertainty``
        ``ō(1 − ō)`` — the Brier score of a constant climatological
        forecast, and therefore the BSS denominator. A property of the
        sample, not of the forecast.
    ``brier_binned``
        ``reliability − resolution + uncertainty``, which is exactly the
        Brier score of the *binned* forecast (every ``p`` replaced by its
        bin's mean). The three-term identity holds for this number
        exactly and for ``brier`` only up to the within-bin spread, so
        both are reported rather than pretending the difference away.
    ``bss``
        ``1 − brier / uncertainty``: the Brier skill score against the
        sample climatology. NaN when every outcome is identical, because
        a constant forecast is then perfect and the ratio is 0/0.
    ``base_rate``
        ``ō``, the observed event frequency.
    ``n``
        Pairs used.

    Raises ``ValueError`` on a shape mismatch, a non-binary ``y``, or a
    ``p`` outside [0, 1] by more than float noise — all three are caller
    bugs that would otherwise produce a plausible-looking wrong number.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    prob, outcome = _paired_finite(p, y, "p", "y")
    n = int(prob.size)
    if n == 0:
        return {
            "brier": float("nan"), "reliability": float("nan"),
            "resolution": float("nan"), "uncertainty": float("nan"),
            "brier_binned": float("nan"), "bss": float("nan"),
            "base_rate": float("nan"), "n": 0,
        }
    _check_binary(outcome)
    prob = _check_probability(prob)

    index = np.clip(np.floor(prob * n_bins).astype(np.int64), 0, n_bins - 1)
    counts = np.bincount(index, minlength=n_bins).astype(np.float64)
    sum_p = np.bincount(index, weights=prob, minlength=n_bins)
    sum_y = np.bincount(index, weights=outcome, minlength=n_bins)
    occupied = counts > 0
    p_bar = np.divide(sum_p, counts, out=np.zeros(n_bins), where=occupied)
    o_bar = np.divide(sum_y, counts, out=np.zeros(n_bins), where=occupied)

    base = float(outcome.mean())
    brier = float(np.mean((prob - outcome) ** 2))
    reliability = float(
        np.sum(counts[occupied] * (p_bar[occupied] - o_bar[occupied]) ** 2) / n
    )
    resolution = float(
        np.sum(counts[occupied] * (o_bar[occupied] - base) ** 2) / n
    )
    uncertainty = base * (1.0 - base)
    bss = float("nan") if uncertainty <= 0.0 else 1.0 - brier / uncertainty
    return {
        "brier": brier,
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "brier_binned": reliability - resolution + uncertainty,
        "bss": bss,
        "base_rate": base,
        "n": n,
    }


def reliability_bins(
    p: np.ndarray | Sequence[float],
    y: np.ndarray | Sequence[float],
    n_bins: int = 10,
) -> list[dict[str, float | int]]:
    """Reliability curve with counts, one entry per **occupied** bin.

    Bin *k* covers ``[k/K, (k+1)/K)`` with ``p == 1`` folded into the last
    one — :func:`brier_decomposition`'s binning, ``quality_report``'s and
    ``sql/reliability_pooled.sql``'s, so a number here, a number from the
    nightly report and a number from DuckDB agree. Empty bins are omitted
    rather than reported as zeroes: an unoccupied bin has no observed
    frequency, and 0.0 is a value.

    Pairs where either input is non-finite are dropped, as everywhere else
    in this module. Returns ``[]`` for an empty sample.
    """
    prob, outcome = _paired_finite(p, y, "p", "y")
    if prob.size == 0:
        return []
    _check_binary(outcome)
    prob = _check_probability(prob)
    index = np.clip(np.floor(prob * n_bins).astype(np.int64), 0, n_bins - 1)
    counts = np.bincount(index, minlength=n_bins)
    sum_p = np.bincount(index, weights=prob, minlength=n_bins)
    sum_y = np.bincount(index, weights=outcome, minlength=n_bins)
    out: list[dict[str, float | int]] = []
    for k in range(n_bins):
        total = int(counts[k])
        if total == 0:
            continue
        out.append({
            "bin": k,
            "p_lo": k / n_bins,
            "p_hi": (k + 1) / n_bins,
            "n": total,
            "mean_p": float(sum_p[k] / total),
            "observed": float(sum_y[k] / total),
        })
    return out


def roc_auc(
    p: np.ndarray | Sequence[float],
    y: np.ndarray | Sequence[float],
) -> float:
    """Area under the ROC curve, by the Mann–Whitney rank identity.

    ``AUC = P(score of a random positive > score of a random negative)``,
    with ties counted as half. Computing it from ranks rather than by
    sweeping thresholds is exact, O(n log n), and handles ties without a
    tolerance parameter.

    NaN when either class is empty — an AUC needs both.
    """
    from scipy.stats import rankdata

    prob, outcome = _paired_finite(p, y, "p", "y")
    if prob.size == 0:
        return float("nan")
    _check_binary(outcome)
    positive = outcome > 0.5
    n_pos = int(np.count_nonzero(positive))
    n_neg = int(positive.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(prob)  # average ranks for ties → the half-credit rule
    rank_sum = float(ranks[positive].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def pr_auc(
    p: np.ndarray | Sequence[float],
    y: np.ndarray | Sequence[float],
) -> float:
    """Average precision — the step-wise area under precision–recall.

    ``AP = Σ (R_n − R_{n−1}) · P_n`` over the distinct forecast values in
    decreasing order, which is scikit-learn's ``average_precision_score``.
    Deliberately *not* the trapezoidal or interpolated area: those flatter
    a forecast at the low-recall end, and the plan uses PR-AUC precisely
    because the ~7 % event rate makes optimistic summaries dangerous.

    Tied forecast values are resolved as one threshold, so a forecast that
    cannot separate a block of points gets no credit for the order they
    happen to arrive in. NaN when there are no positives.
    """
    prob, outcome = _paired_finite(p, y, "p", "y")
    if prob.size == 0:
        return float("nan")
    _check_binary(outcome)
    n_pos = float(np.count_nonzero(outcome > 0.5))
    if n_pos == 0:
        return float("nan")

    # Descending score order; mergesort keeps it stable, so the result does
    # not depend on the input order of tied points.
    order = np.argsort(-prob, kind="mergesort")
    scores = prob[order]
    labels = (outcome[order] > 0.5).astype(np.float64)
    tps = np.cumsum(labels)
    fps = np.arange(1, scores.size + 1, dtype=np.float64) - tps

    # One point per distinct score: the last index of each run of equals.
    last_of_run = np.r_[np.nonzero(np.diff(scores))[0], scores.size - 1]
    tp = tps[last_of_run]
    fp = fps[last_of_run]
    precision = tp / (tp + fp)
    recall = tp / n_pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


# ---------------------------------------------------------------------------
# Ensemble scores
# ---------------------------------------------------------------------------
def crps_ensemble(
    members: np.ndarray,
    obs: np.ndarray | Sequence[float],
) -> tuple[float, np.ndarray]:
    """Continuous Ranked Probability Score of an ensemble, per point.

    ``members`` has shape ``(n_ens, N)``, ``obs`` shape ``(N,)``. The
    estimator is the standard empirical (NRG / Hersbach) one::

        CRPS = mean_i |x_i − y| − ½ · mean_{i,j} |x_i − x_j|

    which is the exact CRPS of the ensemble's empirical CDF — the same
    quantity ``pysteps.verification.probscores.CRPS`` accumulates through
    its α/β decomposition. The second term is evaluated from the sorted
    members via ``Σ_{i,j}|x_i − x_j| = 2 Σ_i (2i − m + 1) x_(i)``, so the
    cost is ``O(m log m · N)`` and no ``m × m × N`` array is ever
    allocated — on a Layer A grid the naive form would be gigabytes.

    Returns ``(mean_over_points, per_point)``. A point is scored only when
    the observation and every member are finite; the rest are NaN in
    ``per_point`` and excluded from the mean. Lower is better; CRPS is in
    the units of the field and reduces to the absolute error for a
    single-member ensemble.
    """
    ens = np.asarray(members, dtype=np.float64)
    truth = np.asarray(obs, dtype=np.float64).ravel()
    if ens.ndim != 2:
        raise ValueError(f"members must be 2-D (n_ens, N), got {ens.shape}")
    if ens.shape[1] != truth.size:
        raise ValueError(
            f"members has {ens.shape[1]} points, obs has {truth.size}"
        )
    m = ens.shape[0]
    if m == 0:
        raise ValueError("members must hold at least one ensemble member")

    per_point = np.full(truth.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(truth) & np.all(np.isfinite(ens), axis=0)
    if not np.any(valid):
        return float("nan"), per_point

    x = ens[:, valid]
    y = truth[valid]
    spread_term = np.abs(x - y).mean(axis=0)
    x_sorted = np.sort(x, axis=0)
    coefficients = (2 * np.arange(m, dtype=np.float64) - m + 1)[:, None]
    pairwise_mean = 2.0 * (coefficients * x_sorted).sum(axis=0) / (m * m)
    per_point[valid] = spread_term - 0.5 * pairwise_mean
    return float(np.mean(per_point[valid])), per_point


def rank_histogram(
    members: np.ndarray,
    obs: np.ndarray | Sequence[float],
    *,
    seed: int = 0,
) -> np.ndarray:
    """Talagrand rank histogram: counts over ``n_ens + 1`` ranks.

    Rank ``k`` counts the points where the observation fell between the
    k-th and (k+1)-th sorted member (rank 0 = below every member,
    rank ``n_ens`` = above every one). A flat histogram means the
    ensemble spread matches the error; a U shape means it is
    under-dispersive, a dome that it is over-dispersive.

    **Ties matter here.** Radar rain fields are full of exact zeros, so a
    "first member strictly above" rule would pile every dry point into
    rank 0 and manufacture a U shape out of nothing. The standard fix is
    to place a tied observation uniformly at random among the ranks the
    tie spans, which is what ``seed`` makes reproducible.

    Points where the observation or any member is non-finite are skipped,
    so ``counts.sum()`` is the number of scored points, not ``N``.
    """
    ens = np.asarray(members, dtype=np.float64)
    truth = np.asarray(obs, dtype=np.float64).ravel()
    if ens.ndim != 2:
        raise ValueError(f"members must be 2-D (n_ens, N), got {ens.shape}")
    if ens.shape[1] != truth.size:
        raise ValueError(
            f"members has {ens.shape[1]} points, obs has {truth.size}"
        )
    m = ens.shape[0]
    valid = np.isfinite(truth) & np.all(np.isfinite(ens), axis=0)
    counts = np.zeros(m + 1, dtype=np.int64)
    if not np.any(valid):
        return counts

    x = ens[:, valid]
    y = truth[valid]
    below = np.count_nonzero(x < y, axis=0)
    tied = np.count_nonzero(x == y, axis=0)
    rng = np.random.default_rng(seed)
    ranks = below + rng.integers(0, tied + 1)
    counts += np.bincount(ranks, minlength=m + 1).astype(np.int64)
    return counts


# ---------------------------------------------------------------------------
# Field correlation and lifetime
# ---------------------------------------------------------------------------
def field_correlation(
    pred: np.ndarray,
    obs: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Pearson correlation between two fields over their valid pixels.

    Valid = finite in both fields, and inside ``mask`` when one is given
    (the harness uses the per-case mask of pixels whose backward
    trajectory stayed inside the composite). NaN when fewer than two
    pixels survive or when either field is constant, because a
    correlation is undefined there rather than zero.
    """
    a = np.asarray(pred, dtype=np.float64)
    b = np.asarray(obs, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: pred {a.shape} vs obs {b.shape}")
    valid = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape != a.shape:
            raise ValueError(f"mask shape {m.shape} != field shape {a.shape}")
        valid &= m
    if np.count_nonzero(valid) < 2:
        return float("nan")
    av = a[valid]
    bv = b[valid]
    a_dev = av - av.mean()
    b_dev = bv - bv.mean()
    denom = math.sqrt(float((a_dev ** 2).sum()) * float((b_dev ** 2).sum()))
    if denom <= 0.0:
        return float("nan")
    return float((a_dev * b_dev).sum() / denom)


def decorrelation_time(
    leads_min: Sequence[float] | np.ndarray,
    correlations: Sequence[float] | np.ndarray,
) -> float:
    """Lead, in minutes, at which the correlation first falls below 1/e.

    The Germann & Zawadzki "lifetime" convention that Imhoff et al. 2020
    report their 25 / 40 / 56 / 116 min numbers in, so a number produced
    here is directly comparable with the plan's external anchor. The
    crossing is linearly interpolated between the bracketing leads, since
    a 10-minute lead grid would otherwise quantise the answer far too
    coarsely to compare against 25 min.

    Non-finite pairs are dropped; leads must then be strictly increasing.
    Returns NaN when the correlation never drops below the level (the
    honest answer is "longer than the longest lead scored", not the
    longest lead). When the *first* remaining lead is already below the
    level the crossing cannot be bracketed, so that lead is returned — an
    upper bound on the true lifetime, and flagged as such here rather
    than silently extrapolated to a negative time.
    """
    leads = np.asarray(leads_min, dtype=np.float64).ravel()
    corr = np.asarray(correlations, dtype=np.float64).ravel()
    if leads.shape != corr.shape:
        raise ValueError(
            f"shape mismatch: leads {leads.shape} vs correlations {corr.shape}"
        )
    keep = np.isfinite(leads) & np.isfinite(corr)
    leads = leads[keep]
    corr = corr[keep]
    if leads.size == 0:
        return float("nan")
    if np.any(np.diff(leads) <= 0):
        raise ValueError("leads_min must be strictly increasing")

    below = np.nonzero(corr < DECORRELATION_LEVEL)[0]
    if below.size == 0:
        return float("nan")
    k = int(below[0])
    if k == 0:
        return float(leads[0])
    c0, c1 = corr[k - 1], corr[k]
    t0, t1 = leads[k - 1], leads[k]
    span = c0 - c1
    if span <= 0:
        return float(t1)
    return float(t0 + (t1 - t0) * (c0 - DECORRELATION_LEVEL) / span)


# ---------------------------------------------------------------------------
# Day-block resampling
# ---------------------------------------------------------------------------
def contingency_sum(blocks: Sequence[ContingencyTable]) -> ContingencyTable:
    """Pool contingency tables by summing their four counts.

    The reason this helper exists rather than callers averaging per-day
    CSIs: CSI, POD and FAR are ratios, and the mean of per-day ratios is
    not the ratio over the pooled days. A quiet day with 200 wet pixels
    would carry the same weight as a frontal day with two million. Every
    resample in :func:`block_bootstrap` must therefore re-pool the counts
    and only then form the score.
    """
    hits = misses = false_alarms = correct_negatives = 0
    for table in blocks:
        hits += table.hits
        misses += table.misses
        false_alarms += table.false_alarms
        correct_negatives += table.correct_negatives
    return ContingencyTable(
        hits=hits,
        misses=misses,
        false_alarms=false_alarms,
        correct_negatives=correct_negatives,
    )


def block_bootstrap(
    blocks: Sequence[Any],
    statistic: Callable[[list[Any]], float],
    n_resamples: int = 500,
    seed: int = 0,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Percentile bootstrap over whole days.

    ``blocks`` is one opaque object per day — a contingency table, a pair
    of FSS sums, a list of rows, whatever ``statistic`` understands.
    ``statistic(list_of_blocks)`` must **re-pool** and return the score,
    not average the per-block scores (see :func:`contingency_sum`).

    Resampling by day rather than by pixel or by frame is the whole
    point: frames five minutes apart share the same rain, so a
    pixel-level bootstrap would report intervals an order of magnitude
    too narrow. The plan fixes 500 resamples and 95 %.

    Returns ``(point, lo, hi)`` — the statistic on the observed blocks and
    the percentile interval. Non-finite resample values are dropped
    before the percentiles; if none survive the interval is NaN.
    """
    return _bootstrap(
        [list(blocks)],
        lambda samples: float(statistic(samples[0])),
        n_resamples=n_resamples, seed=seed, ci=ci,
    )


def paired_block_bootstrap(
    blocks_a: Sequence[Any],
    blocks_b: Sequence[Any],
    statistic: Callable[[list[Any]], float],
    n_resamples: int = 500,
    seed: int = 0,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Bootstrap CI on ``statistic(a) − statistic(b)``, paired by day.

    ``blocks_a[i]`` and ``blocks_b[i]`` must be the SAME day scored two
    ways (candidate and baseline), and each resample draws one set of day
    indices used for both. That pairing is what the acceptance gate needs:
    the two pipelines see identical weather, so the day-to-day variance
    that dominates either score individually cancels in the difference and
    the interval on the difference is far tighter than the difference of
    the two intervals.

    Returns ``(point_difference, lo, hi)``. An interval that excludes zero
    is the plan's evidence that a candidate really moved the score.
    """
    a = list(blocks_a)
    b = list(blocks_b)
    if len(a) != len(b):
        raise ValueError(
            f"paired bootstrap needs equal block counts, got {len(a)} and {len(b)}"
        )
    return _bootstrap(
        [a, b],
        lambda samples: float(statistic(samples[0])) - float(statistic(samples[1])),
        n_resamples=n_resamples, seed=seed, ci=ci,
    )


def _bootstrap(
    block_sets: list[list[Any]],
    combine: Callable[[list[list[Any]]], float],
    *,
    n_resamples: int,
    seed: int,
    ci: float,
) -> tuple[float, float, float]:
    """Shared resampling loop: one index draw applied to every block set."""
    if not 0.0 < ci < 1.0:
        raise ValueError(f"ci must be in (0, 1), got {ci}")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")
    n = len(block_sets[0])
    if n == 0:
        return float("nan"), float("nan"), float("nan")

    point = combine(block_sets)
    rng = np.random.default_rng(seed)
    values = np.empty(n_resamples, dtype=np.float64)
    for r in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        values[r] = combine([[bs[i] for i in idx] for bs in block_sets])
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float(point), float("nan"), float("nan")
    alpha = (1.0 - ci) / 2.0
    lo = float(np.percentile(finite, 100.0 * alpha))
    hi = float(np.percentile(finite, 100.0 * (1.0 - alpha)))
    return float(point), lo, hi


# ---------------------------------------------------------------------------
# Flow stall diagnostic (H-F)
# ---------------------------------------------------------------------------
def stall_diagnostic(
    vy: np.ndarray,
    vx: np.ndarray,
    rain_mm_h: np.ndarray,
    *,
    pixel_km: float,
    dt_min: float,
    wet_threshold_mm_h: float = 0.5,
    slow_kmh: float = 5.0,
    bulk_min_kmh: float = 20.0,
    n_bins: int = 5,
) -> dict[str, Any]:
    """Measure the Farnebäck stall inside broad echo (plan §3 H-F).

    The failure this quantifies, reproduced in
    ``archive/flow_stall_20260908/README.md``: inside a broad, flat or
    speckled rain band the local polynomial expansion has nothing to
    match, the least-squares displacement collapses toward zero, and
    ``complete_flow`` cannot rescue it because that relaxation is applied
    only OFF the echo. The advected field then stretches — the leading
    edge advances while the interior and rear stay put — and STEPS
    inherits the same flow.

    Method, exactly as in the archived diagnostic:

    1. Wet pixels are ``rain_mm_h >= wet_threshold_mm_h`` with a finite
       velocity.
    2. The bulk vector is the **rain-weighted** mean velocity over wet
       pixels — weighted, so a scatter of drizzle at the domain edge
       cannot out-vote the main band.
    3. Every wet pixel is projected onto the bulk direction, giving an
       along-motion position; the wet pixels are split into ``n_bins``
       equal-count bins of that position, rear (upwind) first.
    4. ``profile_kmh`` is the median along-motion speed **component** per
       bin. A healthy band is flat; a stalled one rises from rear to
       front.
    5. ``stalled_share`` is the fraction of wet pixels whose full speed is
       below ``slow_kmh``.

    Speeds convert as ``px/frame × pixel_km / (dt_min / 60)``.

    ``applicable`` is ``bulk_kmh >= bulk_min_kmh``. Below that the
    diagnostic is meaningless — slow-moving rain is *supposed* to have
    slow pixels — so aggregates should be taken over applicable pairs
    only. The dict is still returned so the share of applicable pairs can
    be reported.

    Returns keys ``bulk_kmh``, ``bulk_dir_unit`` (``(uy, ux)``, NaN when
    the bulk vector is degenerate), ``stalled_share``, ``profile_kmh``
    (list of ``n_bins`` medians, rear → front), ``applicable`` and
    ``n_wet``.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    if pixel_km <= 0:
        raise ValueError(f"pixel_km must be > 0, got {pixel_km}")
    if dt_min <= 0:
        raise ValueError(f"dt_min must be > 0, got {dt_min}")

    vy_arr = np.asarray(vy, dtype=np.float64)
    vx_arr = np.asarray(vx, dtype=np.float64)
    rain = np.asarray(rain_mm_h, dtype=np.float64)
    if vy_arr.shape != vx_arr.shape or vy_arr.shape != rain.shape:
        raise ValueError("vy, vx and rain_mm_h must all have the same shape")
    if vy_arr.ndim != 2:
        raise ValueError(f"expected 2-D fields, got {vy_arr.shape}")

    empty: dict[str, Any] = {
        "bulk_kmh": float("nan"),
        "bulk_dir_unit": (float("nan"), float("nan")),
        "stalled_share": float("nan"),
        "profile_kmh": [float("nan")] * n_bins,
        "applicable": False,
        "n_wet": 0,
    }

    wet = (
        np.isfinite(rain) & (rain >= wet_threshold_mm_h)
        & np.isfinite(vy_arr) & np.isfinite(vx_arr)
    )
    n_wet = int(np.count_nonzero(wet))
    if n_wet == 0:
        return empty

    kmh_per_px_frame = float(pixel_km) * 60.0 / float(dt_min)
    weights = rain[wet]
    weight_sum = float(weights.sum())
    if not math.isfinite(weight_sum) or weight_sum <= 0.0:
        return {**empty, "n_wet": n_wet}

    wet_vy = vy_arr[wet]
    wet_vx = vx_arr[wet]
    bulk_vy = float((wet_vy * weights).sum() / weight_sum)
    bulk_vx = float((wet_vx * weights).sum() / weight_sum)
    bulk_px = math.hypot(bulk_vy, bulk_vx)
    bulk_kmh = bulk_px * kmh_per_px_frame

    speeds_kmh = np.hypot(wet_vy, wet_vx) * kmh_per_px_frame
    stalled_share = float(np.count_nonzero(speeds_kmh < slow_kmh) / n_wet)

    if bulk_px <= 0.0:
        return {
            "bulk_kmh": bulk_kmh,
            "bulk_dir_unit": (float("nan"), float("nan")),
            "stalled_share": stalled_share,
            "profile_kmh": [float("nan")] * n_bins,
            "applicable": False,
            "n_wet": n_wet,
        }

    unit_y = bulk_vy / bulk_px
    unit_x = bulk_vx / bulk_px
    rows, cols = np.nonzero(wet)
    position = rows * unit_y + cols * unit_x
    along_kmh = (wet_vy * unit_y + wet_vx * unit_x) * kmh_per_px_frame

    # Equal-count (quantile) bins of the along-motion position. The inner
    # edges alone define the bins; searchsorted with side="right" then puts
    # a pixel exactly on an edge into the later (more forward) bin, which
    # keeps the rear bin from absorbing an edge-heavy tie.
    edges = np.quantile(position, np.linspace(0.0, 1.0, n_bins + 1))
    bin_index = np.searchsorted(edges[1:-1], position, side="right")
    np.clip(bin_index, 0, n_bins - 1, out=bin_index)
    profile: list[float] = []
    for k in range(n_bins):
        in_bin = bin_index == k
        profile.append(
            float(np.median(along_kmh[in_bin])) if np.any(in_bin) else float("nan")
        )

    return {
        "bulk_kmh": bulk_kmh,
        "bulk_dir_unit": (unit_y, unit_x),
        "stalled_share": stalled_share,
        "profile_kmh": profile,
        "applicable": bool(bulk_kmh >= bulk_min_kmh),
        "n_wet": n_wet,
    }


# ---------------------------------------------------------------------------
# shared validation
# ---------------------------------------------------------------------------
def _paired_finite(
    a: np.ndarray | Sequence[float],
    b: np.ndarray | Sequence[float],
    name_a: str,
    name_b: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten both inputs to float64 and drop pairs with a non-finite half."""
    arr_a = np.asarray(a, dtype=np.float64).ravel()
    arr_b = np.asarray(b, dtype=np.float64).ravel()
    if arr_a.shape != arr_b.shape:
        raise ValueError(
            f"shape mismatch: {name_a} {arr_a.shape} vs {name_b} {arr_b.shape}"
        )
    keep = np.isfinite(arr_a) & np.isfinite(arr_b)
    return arr_a[keep], arr_b[keep]


def _check_binary(y: np.ndarray) -> None:
    if not np.all((y == 0.0) | (y == 1.0)):
        raise ValueError("outcomes must be binary (0 or 1)")


def _check_probability(p: np.ndarray) -> np.ndarray:
    """Clip float noise into [0, 1]; anything larger is a caller bug."""
    if np.any(p < -_PROB_TOLERANCE) or np.any(p > 1.0 + _PROB_TOLERANCE):
        raise ValueError("probabilities must lie in [0, 1]")
    return np.clip(p, 0.0, 1.0)
