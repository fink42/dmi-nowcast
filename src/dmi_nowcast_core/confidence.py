"""Confidence score heuristic (plan §7.2).

Distinct from probability: probability is a calibrated forecast value, confidence
is a 0–1 quality-of-source score. The plan calls out **intensity volatility** as
"the most important single factor" — rapidly intensifying or weakening fields
break the advection assumption (growth and decay cannot be advected). We
implement that, plus horizon, frame age, motion divergence, and frame count.

The score is the product of per-factor weights so any single bad factor drags
the score down. Factors are exposed individually for diagnostic attributes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConfidenceResult:
    score: float
    factors: dict[str, float]


def compute_confidence(
    *,
    horizon_minutes: float,
    frame_age_seconds: float,
    intensity_volatility: float,
    motion_divergence: float = 0.0,
    n_frames: int = 3,
    horizon_full_decay_min: float = 90.0,
    frame_age_full_decay_s: float = 1800.0,
) -> ConfidenceResult:
    """Combine factors into a 0–1 confidence score.

    ``intensity_volatility`` is a relative change between frames (0 = stable,
    1 = total change). ``motion_divergence`` is the mean absolute divergence of
    the flow field in 1/pixel — a smooth, uniform translation is ~0; a chaotic
    flow is ≥ 0.1. The decay constants are deliberately conservative so a 60-min
    horizon still reports ≥ 0.3 confidence.
    """
    factors = {
        "horizon": max(0.1, 1.0 - horizon_minutes / horizon_full_decay_min),
        "frame_age": max(0.0, 1.0 - frame_age_seconds / frame_age_full_decay_s),
        # Linear (not 2×): volatility 0.5 halves the factor, volatility 1.0
        # zeros it. A 2× multiplier zeroed the factor on any dry-to-wet
        # transition, which made confidence essentially always 0 right when
        # the user most needs a meaningful signal.
        "intensity_volatility": max(0.0, 1.0 - 1.0 * intensity_volatility),
        "motion_divergence": max(0.0, 1.0 - 5.0 * motion_divergence),
        "frame_count": min(1.0, max(0.0, (n_frames - 1) / 4.0)),
    }
    score = float(np.prod(list(factors.values())))
    return ConfidenceResult(score=score, factors=factors)


def intensity_volatility_from_disc(prev_max: float, curr_max: float) -> float:
    """Change in disc-max rain rate between consecutive frames, normalised so
    that small absolute differences register as small volatility regardless
    of the relative change.

    0 = identical, 1 = max volatility (≥ ~5 mm/h change in absolute terms or
    ≥ 100 % change relative to a wet baseline). A pure dry→light-rain step
    (e.g. 0 → 0.5 mm/h) returns ~0.1 rather than 1.0, so the confidence
    heuristic doesn't zero itself the moment rain starts.
    """
    if not (np.isfinite(prev_max) and np.isfinite(curr_max)):
        return 1.0  # missing data → maximum volatility (worst case)
    diff = abs(curr_max - prev_max)
    # Floor the denominator at 5 mm/h so dry→light transitions stay mild.
    denom = max(prev_max, curr_max, 5.0)
    return float(min(1.0, diff / denom))


def motion_divergence(vy: np.ndarray, vx: np.ndarray) -> float:
    """Mean |∂vx/∂x + ∂vy/∂y| over the grid, in 1/pixel.

    A clean uniform translation has divergence ≈ 0. Mesoscale convergence
    or divergence (and noisy flow) raises this number.
    """
    dvy_dy = np.gradient(vy, axis=0)
    dvx_dx = np.gradient(vx, axis=1)
    div = dvy_dy + dvx_dx
    finite = div[np.isfinite(div)]
    if finite.size == 0:
        return 0.0
    return float(np.mean(np.abs(finite)))
