"""H-F: confidence-gated flow completion (hotfix, 2026-09-08).

Evidence for every number asserted here:
``archive/flow_stall_20260908/README.md`` in the private repo — two cases
(a stratiform shield at 17:00-17:50Z, a convective band at 07:10-08:00Z),
both reproduced with this code on public DMI frames and the evening one
verified against the live sidecar's own served motion grid.

The defect: OpenCV Farnebäck returns near-zero displacement inside broad,
flat or speckled echo, and ``complete_flow`` used to relax only OFF-echo
pixels — so the stall survived into the advection and into STEPS, and the
rain-weighted bulk the far field relaxed toward was itself dragged down by
the stalled majority.

What is pinned here:

* ``complete_flow(confidence=None)`` is bit-for-bit the pre-hotfix
  function (``_legacy_complete_flow`` below is a verbatim copy of the body
  as of commit 9232e18);
* on a synthetic stratiform block Farnebäck really does stall, and the
  gate really does un-stall the advected field;
* on textured cells the gate keeps the measured field (it is not a
  smoother);
* ``robust_bulk`` survives a stalled majority;
* ``estimate_motion(completion="bulk")`` is the legacy inline sequence.
"""
from __future__ import annotations

import numpy as np
import pytest

from dmi_nowcast_core.dense_flow import (
    DEFAULT_EFOLD_KM,
    DEFAULT_SUPPORT_DILATION_PX,
    MotionEstimate,
    complete_flow,
    dense_flow,
    distance_to_support,
    estimate_motion,
    flow_confidence,
    rain_weighted_bulk,
    robust_bulk,
    stalled_share,
)

PIXEL_KM = 0.5
DT_MIN = 10.0
THRESHOLD = 0.5
# px/frame ↔ km/h on the DMI grid: 0.5 km × 60 / 10 min = 3.0.
PX_TO_KMH = PIXEL_KM * 60.0 / DT_MIN


# ---------------------------------------------------------------------------
# Reference: complete_flow exactly as it was before the hotfix
# ---------------------------------------------------------------------------


def _legacy_complete_flow(
    vy: np.ndarray,
    vx: np.ndarray,
    rain_mm_h: np.ndarray,
    *,
    pixel_km: float,
    support_threshold_mm_h: float = 0.5,
    efold_km: float = DEFAULT_EFOLD_KM,
    dilation_px: float = DEFAULT_SUPPORT_DILATION_PX,
) -> tuple[np.ndarray, np.ndarray]:
    """Verbatim copy of ``complete_flow``'s body before 2026-09-08.

    Kept as a copy rather than a stored snapshot so the comparison covers
    every input the test cares to throw at it, not one frozen array pair.
    """
    vy_arr = np.asarray(vy, dtype=np.float32)
    vx_arr = np.asarray(vx, dtype=np.float32)
    rain = np.asarray(rain_mm_h, dtype=np.float32)
    if vy_arr.shape != vx_arr.shape or vy_arr.shape != rain.shape:
        raise ValueError("vy, vx, rain_mm_h must all have the same shape")
    if pixel_km <= 0:
        raise ValueError(f"pixel_km must be > 0, got {pixel_km}")

    finite_v = np.isfinite(vy_arr) & np.isfinite(vx_arr)
    support = np.isfinite(rain) & (rain >= support_threshold_mm_h)

    weights = np.where(support & finite_v, rain, np.float32(0.0))
    w_sum = float(weights.sum())
    if not np.isfinite(w_sum) or w_sum <= 0.0:
        return vy_arr.copy(), vx_arr.copy()
    vy_clean = np.where(finite_v, vy_arr, np.float32(0.0))
    vx_clean = np.where(finite_v, vx_arr, np.float32(0.0))
    bulk_vy = float((vy_clean * weights).sum() / w_sum)
    bulk_vx = float((vx_clean * weights).sum() / w_sum)

    tau_px = float(efold_km) / float(pixel_km)
    if tau_px <= 0:
        weight = support.astype(np.float32)
    else:
        distance_px = distance_to_support(support)
        np.subtract(distance_px, np.float32(max(0.0, dilation_px)), out=distance_px)
        np.maximum(distance_px, np.float32(0.0), out=distance_px)
        weight = np.exp(-distance_px / np.float32(tau_px))

    weight = np.where(finite_v, weight, np.float32(0.0))
    out_vy = weight * vy_clean + (np.float32(1.0) - weight) * np.float32(bulk_vy)
    out_vx = weight * vx_clean + (np.float32(1.0) - weight) * np.float32(bulk_vx)
    return out_vy.astype(np.float32), out_vx.astype(np.float32)


# ---------------------------------------------------------------------------
# Synthetic fields
# ---------------------------------------------------------------------------


def _stratiform_pair(
    shape: tuple[int, int] = (300, 300),
    shift_px: int = 10,
    seed: int = 5,
    blob_sigma: float = 24.0,
    edge_sigma: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A broad shield with a soft, irregular edge and a FLAT interior.

    The shape of a stratiform shield rather than a rectangle: the boundary
    is curved everywhere (so the boundary really does carry a recoverable
    2-D displacement — a straight edge would not, by the aperture problem)
    and the interior is a uniform plateau, which is what leaves Farnebäck
    with nothing to match. Wet fraction ~0.58, close to the 160 km box in
    the evidence file.

    Translated ``shift_px`` east between the frames, so the truth is
    ``(vy, vx) = (0, shift_px)`` everywhere it rains.

    Returns ``(prev_dbz, curr_dbz, rain_now, interior_mask)``, where
    ``interior_mask`` is the part of the echo more than 25 px from any
    edge — deeper than Farnebäck's 31-px window can reach.
    """
    from scipy.ndimage import distance_transform_edt, gaussian_filter

    h, w = shape
    rng = np.random.default_rng(seed)
    base = gaussian_filter(
        rng.normal(0.0, 1.0, size=(h, w + shift_px)).astype(np.float32), blob_sigma,
    )
    base -= base.mean()
    base /= max(float(base.std()), 1e-6)
    shield = gaussian_filter((base > -0.3).astype(np.float32), edge_sigma)
    dbz_full = (-32.0 + 62.0 * shield).astype(np.float32)

    prev = np.ascontiguousarray(dbz_full[:, shift_px:])
    curr = np.ascontiguousarray(dbz_full[:, :w])
    rain = np.where(curr > 0.0, np.float32(2.0), np.float32(0.0)).astype(np.float32)
    interior = distance_transform_edt(rain >= THRESHOLD) > 25.0
    return prev, curr, rain, interior


def _textured_pair(
    shape: tuple[int, int] = (240, 240),
    shift_px: int = 6,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random blobs translated uniformly — an echo with plenty to track."""
    rng = np.random.default_rng(seed)
    h, w = shape
    field = rng.normal(0.0, 1.0, size=(h, w + shift_px)).astype(np.float32)
    # Smooth to blob scale (~8 px) so the polynomial expansion has real
    # structure rather than pixel noise.
    from scipy.ndimage import gaussian_filter

    field = gaussian_filter(field, sigma=4.0)
    field -= field.min()
    field /= max(field.max(), 1e-6)
    dbz_full = (-32.0 + 70.0 * field).astype(np.float32)
    prev = dbz_full[:, shift_px:]
    curr = dbz_full[:, : w]
    # ``curr`` is ``prev`` translated ``shift_px`` EAST (columns increase).
    rain = np.where(curr > 15.0, np.float32(2.0), np.float32(0.0)).astype(np.float32)
    return np.ascontiguousarray(prev), np.ascontiguousarray(curr), rain


# ---------------------------------------------------------------------------
# 1. Regression: confidence=None is the old function, bit for bit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_complete_flow_without_confidence_is_bit_identical(seed: int) -> None:
    rng = np.random.default_rng(seed)
    shape = (96, 112)
    vy = rng.normal(0.0, 6.0, size=shape).astype(np.float32)
    vx = rng.normal(0.0, 6.0, size=shape).astype(np.float32)
    rain = np.where(
        rng.random(shape) < 0.3, rng.gamma(2.0, 1.5, size=shape), 0.0,
    ).astype(np.float32)
    # A few nodata pixels and a few unusable velocities, as a real frame has.
    rain[rng.random(shape) < 0.05] = np.nan
    bad = rng.random(shape) < 0.02
    vy[bad] = np.nan
    vx[bad] = np.inf

    ref_vy, ref_vx = _legacy_complete_flow(
        vy, vx, rain, pixel_km=PIXEL_KM, support_threshold_mm_h=THRESHOLD,
    )
    out_vy, out_vx = complete_flow(
        vy, vx, rain, pixel_km=PIXEL_KM, support_threshold_mm_h=THRESHOLD,
    )
    # Bit-for-bit: no tolerance, no allclose.
    assert np.array_equal(out_vy, ref_vy, equal_nan=True)
    assert np.array_equal(out_vx, ref_vx, equal_nan=True)


def test_no_echo_still_returns_the_input_unchanged() -> None:
    """The degenerate gate is unchanged on both paths."""
    shape = (32, 40)
    vy = np.full(shape, 3.0, dtype=np.float32)
    vx = np.full(shape, -2.0, dtype=np.float32)
    rain = np.zeros(shape, dtype=np.float32)
    energy = np.ones(shape, dtype=np.float32)

    for kwargs in ({}, {"confidence": energy}):
        out_vy, out_vx = complete_flow(
            vy, vx, rain, pixel_km=PIXEL_KM,
            support_threshold_mm_h=THRESHOLD, **kwargs,
        )
        assert np.array_equal(out_vy, vy)
        assert np.array_equal(out_vx, vx)


# ---------------------------------------------------------------------------
# 2. flow_confidence
# ---------------------------------------------------------------------------


def test_flow_confidence_is_zero_on_a_plateau_and_large_on_edges() -> None:
    dbz = np.full((120, 120), -32.0, dtype=np.float32)
    dbz[30:90, 30:90] = 30.0
    energy = flow_confidence(dbz, window_px=15)

    assert energy.dtype == np.float32
    assert energy.shape == dbz.shape
    assert float(energy.min()) >= 0.0
    # Deep interior of the block: nothing to track.
    assert float(energy[55:65, 55:65].max()) == pytest.approx(0.0, abs=1e-5)
    # Far field: nothing there either.
    assert float(energy[5:15, 5:15].max()) == pytest.approx(0.0, abs=1e-5)
    # The edge is where the gradient lives.
    assert float(energy[25:35, 55:65].max()) > 1e3


def test_flow_confidence_sees_the_uint8_image_not_the_float_field() -> None:
    """Structure below the uint8 quantisation step is invisible — as it is
    to Farnebäck, which is the point of measuring on ``_to_uint8``."""
    base = np.full((64, 64), 10.0, dtype=np.float32)
    sub_step = base.copy()
    # DBZ_RANGE spans 92 dBZ over 256 levels → ~0.36 dBZ per level.
    sub_step[:, ::2] += 0.05
    assert np.array_equal(
        flow_confidence(base, window_px=9), flow_confidence(sub_step, window_px=9),
    )
    # A change larger than the step is visible.
    over_step = base.copy()
    over_step[:, ::2] += 5.0
    assert float(flow_confidence(over_step, window_px=9).mean()) > 0.0


def test_flow_confidence_rejects_a_degenerate_window() -> None:
    with pytest.raises(ValueError, match="window_px"):
        flow_confidence(np.zeros((8, 8), dtype=np.float32), window_px=0)


# ---------------------------------------------------------------------------
# 3. robust_bulk
# ---------------------------------------------------------------------------


def test_robust_bulk_ignores_a_stalled_majority() -> None:
    """70 % stalled low-texture pixels, 30 % textured at (2, 3) px.

    This is the evening case in miniature: the rain-weighted mean lands
    near 0.3 × the true motion, the robust median lands on it.
    """
    shape = (100, 100)
    rng = np.random.default_rng(11)
    rain = np.full(shape, 2.0, dtype=np.float32)
    vy = np.zeros(shape, dtype=np.float32)
    vx = np.zeros(shape, dtype=np.float32)
    energy = np.zeros(shape, dtype=np.float32)

    textured = rng.random(shape) < 0.30
    vy[textured] = 2.0
    vx[textured] = 3.0
    energy[textured] = 5000.0

    bulk_vy, bulk_vx = robust_bulk(
        vy, vx, rain, energy,
        support_threshold_mm_h=THRESHOLD, texture_percentile=60.0,
    )
    assert bulk_vy == pytest.approx(2.0, abs=1e-6)
    assert bulk_vx == pytest.approx(3.0, abs=1e-6)

    # The statistic it replaces is dragged toward the stalled majority.
    weighted = rain_weighted_bulk(vy, vx, rain, support_threshold_mm_h=THRESHOLD)
    assert weighted is not None
    assert weighted[1] < 0.5 * 3.0


def test_robust_bulk_falls_back_to_the_rain_weighted_mean() -> None:
    """Too few textured pixels → the old statistic, not a noisy median."""
    shape = (60, 60)
    rain = np.full(shape, 2.0, dtype=np.float32)
    vy = np.full(shape, 1.0, dtype=np.float32)
    vx = np.full(shape, 4.0, dtype=np.float32)
    energy = np.zeros(shape, dtype=np.float32)
    energy[0, :10] = 1000.0  # 10 pixels, far below min_pixels
    vy[0, :10] = -50.0
    vx[0, :10] = -50.0

    out = robust_bulk(
        vy, vx, rain, energy,
        support_threshold_mm_h=THRESHOLD, min_pixels=200,
    )
    weighted = rain_weighted_bulk(vy, vx, rain, support_threshold_mm_h=THRESHOLD)
    assert weighted is not None
    assert out == pytest.approx(weighted, abs=1e-6)


def test_robust_bulk_on_a_dry_composite_is_zero() -> None:
    shape = (20, 20)
    zeros = np.zeros(shape, dtype=np.float32)
    assert robust_bulk(
        zeros, zeros, zeros, zeros, support_threshold_mm_h=THRESHOLD,
    ) == (0.0, 0.0)


def test_robust_bulk_is_nan_safe() -> None:
    shape = (40, 40)
    rng = np.random.default_rng(3)
    rain = np.full(shape, 2.0, dtype=np.float32)
    rain[rng.random(shape) < 0.1] = np.nan
    vy = np.full(shape, 2.0, dtype=np.float32)
    vx = np.full(shape, 3.0, dtype=np.float32)
    vy[rng.random(shape) < 0.05] = np.nan
    energy = np.full(shape, 1000.0, dtype=np.float32)
    energy[rng.random(shape) < 0.05] = np.nan

    bulk_vy, bulk_vx = robust_bulk(
        vy, vx, rain, energy, support_threshold_mm_h=THRESHOLD, min_pixels=10,
    )
    assert np.isfinite(bulk_vy) and np.isfinite(bulk_vx)
    assert bulk_vy == pytest.approx(2.0, abs=1e-6)
    assert bulk_vx == pytest.approx(3.0, abs=1e-6)


def test_robust_bulk_rejects_a_mismatched_energy_grid() -> None:
    a = np.zeros((8, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="energy shape"):
        robust_bulk(a, a, a, np.zeros((8, 9), dtype=np.float32))


# ---------------------------------------------------------------------------
# 4. The stall, and the fix, on a synthetic stratiform shield
# ---------------------------------------------------------------------------


def test_farneback_stalls_inside_a_flat_block() -> None:
    """The defect itself, reproduced without a radar file."""
    prev, curr, rain, interior = _stratiform_pair()
    vy, vx = dense_flow(prev, curr)
    interior_speed = np.hypot(vy[interior], vx[interior])
    assert float(np.median(interior_speed)) < 1.0  # px/frame, against a true 10

    share = stalled_share(
        vy, vx, rain, pixel_km=PIXEL_KM, dt_min=DT_MIN,
        support_threshold_mm_h=THRESHOLD,
    )
    assert share > 0.3


def test_confidence_gate_unstalls_the_interior() -> None:
    prev, curr, rain, interior = _stratiform_pair()

    legacy = estimate_motion(
        prev, curr, rain,
        pixel_km=PIXEL_KM, dt_min=DT_MIN, support_threshold_mm_h=THRESHOLD,
        completion="bulk",
    )
    gated = estimate_motion(
        prev, curr, rain,
        pixel_km=PIXEL_KM, dt_min=DT_MIN, support_threshold_mm_h=THRESHOLD,
        completion="confidence",
    )

    # The raw estimate is the same field in both runs — the diagnostic on
    # it is a property of the estimator, not of the completion.
    assert gated.stalled_share == pytest.approx(legacy.stalled_share)
    assert gated.stalled_share > 0.3

    # What changes is the field the forecast is advected with.
    assert legacy.stalled_share_completed > 0.3
    assert gated.stalled_share_completed < 0.05

    # The interior now moves at the bulk speed, within 10 %.
    bulk_speed = float(np.hypot(gated.bulk_vy, gated.bulk_vx))
    interior_speed = float(np.median(np.hypot(gated.vy[interior], gated.vx[interior])))
    assert interior_speed == pytest.approx(bulk_speed, rel=0.10)

    # ...and the bulk itself is the shield's real translation (10 px east),
    # which the stalled-majority mean was never going to recover: on this
    # fixture it reads ~4 px, the same 40 %-of-truth underestimate the
    # evidence file measured on 2026-09-08 (15-17 vs 25-28 km/h).
    assert bulk_speed == pytest.approx(10.0, rel=0.10)
    assert gated.bulk_vx > 0.0
    legacy_bulk_speed = float(np.hypot(legacy.bulk_vy, legacy.bulk_vx))
    assert legacy_bulk_speed < 0.6 * bulk_speed


def test_confidence_gate_keeps_a_textured_field() -> None:
    """It is a gate, not a smoother: where there IS texture, nothing moves."""
    prev, curr, rain = _textured_pair()
    wet = rain >= THRESHOLD
    assert wet.sum() > 1000

    raw_vy, raw_vx = dense_flow(prev, curr)
    energy = flow_confidence(curr)
    gated_vy, gated_vx = complete_flow(
        raw_vy, raw_vx, rain,
        pixel_km=PIXEL_KM, support_threshold_mm_h=THRESHOLD,
        confidence=energy,
    )
    raw_speed = float(np.median(np.hypot(raw_vy[wet], raw_vx[wet])))
    gated_speed = float(np.median(np.hypot(gated_vy[wet], gated_vx[wet])))
    assert raw_speed > 1.0
    assert gated_speed == pytest.approx(raw_speed, rel=0.05)


def test_off_echo_relaxation_is_untouched_by_the_gate() -> None:
    """Only the ON-echo weight changes; the far field keeps exp(-d/tau)."""
    shape = (120, 120)
    rain = np.zeros(shape, dtype=np.float32)
    rain[50:70, 50:70] = 2.0
    vy = np.zeros(shape, dtype=np.float32)
    vx = np.full(shape, 4.0, dtype=np.float32)
    energy = np.zeros(shape, dtype=np.float32)
    energy[50:70, 50:70] = 1000.0  # fully-confident echo → weight 1 on it

    gated_vy, gated_vx = complete_flow(
        vy, vx, rain, pixel_km=PIXEL_KM,
        support_threshold_mm_h=THRESHOLD, confidence=energy,
    )
    legacy_vy, legacy_vx = complete_flow(
        vy, vx, rain, pixel_km=PIXEL_KM, support_threshold_mm_h=THRESHOLD,
    )
    off = rain < THRESHOLD
    # Same bulk (a uniform field), same weight off the echo → same answer.
    assert np.allclose(gated_vy[off], legacy_vy[off], atol=1e-6)
    assert np.allclose(gated_vx[off], legacy_vx[off], atol=1e-6)


def test_all_flat_energy_hands_the_echo_the_bulk_vector() -> None:
    """``thr <= 0``: nothing was measurable, so nothing is trusted."""
    shape = (60, 60)
    rain = np.full(shape, 2.0, dtype=np.float32)
    vy = np.zeros(shape, dtype=np.float32)
    vx = np.zeros(shape, dtype=np.float32)
    vx[0, :] = 9.0  # one row of "motion" that the gate must not preserve
    energy = np.zeros(shape, dtype=np.float32)

    out_vy, out_vx = complete_flow(
        vy, vx, rain, pixel_km=PIXEL_KM,
        support_threshold_mm_h=THRESHOLD, confidence=energy,
    )
    bulk = robust_bulk(vy, vx, rain, energy, support_threshold_mm_h=THRESHOLD)
    assert np.allclose(out_vx, bulk[1], atol=1e-5)
    assert np.allclose(out_vy, bulk[0], atol=1e-5)


def test_complete_flow_rejects_a_mismatched_confidence_grid() -> None:
    a = np.zeros((8, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="confidence shape"):
        complete_flow(
            a, a, a, pixel_km=PIXEL_KM,
            confidence=np.zeros((8, 9), dtype=np.float32),
        )


# ---------------------------------------------------------------------------
# 5. estimate_motion: the shared entry point
# ---------------------------------------------------------------------------


def _legacy_inline_sequence(
    prev: np.ndarray, curr: np.ndarray, rain: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """compute.py's motion block, verbatim, as of commit 9232e18."""
    max_px = 30.0
    vy, vx = dense_flow(prev, curr)

    vy_raw = np.nan_to_num(vy, nan=0.0).astype(np.float32)
    vx_raw = np.nan_to_num(vx, nan=0.0).astype(np.float32)
    np.clip(vy_raw, -max_px, max_px, out=vy_raw)
    np.clip(vx_raw, -max_px, max_px, out=vx_raw)

    vy, vx = _legacy_complete_flow(
        vy, vx, rain, pixel_km=PIXEL_KM, support_threshold_mm_h=THRESHOLD,
    )
    vy = np.nan_to_num(vy, nan=0.0).astype(np.float32)
    vx = np.nan_to_num(vx, nan=0.0).astype(np.float32)
    np.clip(vy, -max_px, max_px, out=vy)
    np.clip(vx, -max_px, max_px, out=vx)
    return vy, vx, vy_raw, vx_raw


def test_estimate_motion_bulk_is_the_legacy_sequence() -> None:
    prev, curr, rain, _ = _stratiform_pair()
    ref_vy, ref_vx, ref_vy_raw, ref_vx_raw = _legacy_inline_sequence(prev, curr, rain)

    got = estimate_motion(
        prev, curr, rain,
        pixel_km=PIXEL_KM, dt_min=DT_MIN, support_threshold_mm_h=THRESHOLD,
        completion="bulk",
    )
    assert np.array_equal(got.vy, ref_vy)
    assert np.array_equal(got.vx, ref_vx)
    assert np.array_equal(got.vy_raw, ref_vy_raw)
    assert np.array_equal(got.vx_raw, ref_vx_raw)
    assert got.completion == "bulk"

    weighted = rain_weighted_bulk(
        *dense_flow(prev, curr), rain, support_threshold_mm_h=THRESHOLD,
    )
    assert weighted is not None
    assert (got.bulk_vy, got.bulk_vx) == pytest.approx(weighted, abs=1e-6)


def test_estimate_motion_accepts_a_precomputed_fallback_flow() -> None:
    """The mean-motion path: a uniform shift, completed and clipped here."""
    prev, curr, rain, _ = _stratiform_pair()
    shape = rain.shape
    uniform = (
        np.full(shape, 0.0, dtype=np.float32),
        np.full(shape, 5.0, dtype=np.float32),
    )
    got = estimate_motion(
        prev, curr, rain,
        pixel_km=PIXEL_KM, dt_min=DT_MIN, support_threshold_mm_h=THRESHOLD,
        completion="confidence", flow=uniform,
    )
    # A uniform field is uniform after completion, whatever the texture.
    assert np.allclose(got.vx, 5.0, atol=1e-5)
    assert np.allclose(got.vy, 0.0, atol=1e-5)
    assert got.stalled_share == 0.0


def test_estimate_motion_clips_at_max_px_per_frame() -> None:
    shape = (40, 40)
    rain = np.full(shape, 2.0, dtype=np.float32)
    wild = (
        np.full(shape, 900.0, dtype=np.float32),
        np.full(shape, -900.0, dtype=np.float32),
    )
    got = estimate_motion(
        np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32), rain,
        pixel_km=PIXEL_KM, dt_min=DT_MIN, support_threshold_mm_h=THRESHOLD,
        completion="bulk", flow=wild, max_px_per_frame=30.0,
    )
    assert float(got.vy.max()) == pytest.approx(30.0)
    assert float(got.vx.min()) == pytest.approx(-30.0)
    assert float(got.vy_raw.max()) == pytest.approx(30.0)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"completion": "nonsense"}, "completion"),
        ({"pixel_km": 0.0}, "pixel_km"),
        ({"dt_min": 0.0}, "dt_min"),
    ],
)
def test_estimate_motion_validates_its_arguments(kwargs: dict, match: str) -> None:
    shape = (16, 16)
    a = np.zeros(shape, dtype=np.float32)
    base = dict(
        pixel_km=PIXEL_KM, dt_min=DT_MIN, support_threshold_mm_h=THRESHOLD,
        completion="bulk", flow=(a.copy(), a.copy()),
    )
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        estimate_motion(a, a, a, **base)


def test_motion_estimate_is_frozen() -> None:
    prev, curr, rain, _ = _stratiform_pair(shape=(64, 64), shift_px=4)
    got = estimate_motion(
        prev, curr, rain,
        pixel_km=PIXEL_KM, dt_min=DT_MIN, support_threshold_mm_h=THRESHOLD,
        completion="bulk",
    )
    assert isinstance(got, MotionEstimate)
    with pytest.raises(Exception):
        got.completion = "confidence"  # type: ignore[misc]


def test_stalled_share_is_zero_on_a_dry_composite() -> None:
    shape = (20, 20)
    zeros = np.zeros(shape, dtype=np.float32)
    assert stalled_share(
        zeros, zeros, zeros, pixel_km=PIXEL_KM, dt_min=DT_MIN,
        support_threshold_mm_h=THRESHOLD,
    ) == 0.0


def test_stalled_share_uses_the_5_kmh_cut() -> None:
    """The cut is physical, so it moves with pixel size and frame spacing."""
    shape = (10, 10)
    rain = np.full(shape, 2.0, dtype=np.float32)
    vy = np.zeros(shape, dtype=np.float32)
    # 1.0 px/frame = 3 km/h at 0.5 km / 10 min → stalled;
    # 2.0 px/frame = 6 km/h → moving.
    vx = np.full(shape, 1.0, dtype=np.float32)
    vx[:5, :] = 2.0
    assert stalled_share(
        vy, vx, rain, pixel_km=PIXEL_KM, dt_min=DT_MIN,
        support_threshold_mm_h=THRESHOLD,
    ) == pytest.approx(0.5)
    assert PX_TO_KMH == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# 6. Real radar: the committed 2026-09-02 Odense pair
# ---------------------------------------------------------------------------


FIXTURE_DIR = __import__("pathlib").Path(__file__).parent / "fixtures" / "odense_20260902"


def _odense_pair():
    from dmi_nowcast_core.parse import parse_composite

    try:
        prev = parse_composite(FIXTURE_DIR / "dk.com.202609020350.500_max.h5")
        now = parse_composite(FIXTURE_DIR / "dk.com.202609020400.500_max.h5")
    except FileNotFoundError:  # pragma: no cover - fixtures are committed
        pytest.skip("Odense composites missing")
    return prev, now


def test_on_real_radar_the_gate_moves_the_field_without_breaking_it() -> None:
    """End-to-end on a real DMI composite pair, native 1728×1984.

    This pair is the healthy CONTROL, not the defect: scattered cells over
    0.8 % of the grid, plenty of texture, ``stalled_share`` 0.00. So what
    it pins is that the hotfix runs on real data at the real grid size and
    that where the estimator was fine, the gate leaves it alone. The
    stall itself is the synthetic above and, at full severity, the frames
    in ``archive/flow_stall_20260908/``; whether the gate wins on skill is
    Layer A's question, not a unit test's.
    """
    from dmi_nowcast_core.transform import dbz_to_rain_rate

    prev, now = _odense_pair()
    dt_min = (now.timestamp_utc - prev.timestamp_utc).total_seconds() / 60.0
    assert dt_min == pytest.approx(10.0, abs=0.2)
    rain = dbz_to_rain_rate(now.reflectivity_dbz, zr_a=now.zr_a, zr_b=now.zr_b)
    pixel_km = float(now.xscale_m) / 1000.0

    common = dict(
        pixel_km=pixel_km, dt_min=dt_min, support_threshold_mm_h=THRESHOLD,
    )
    legacy = estimate_motion(
        prev.reflectivity_dbz, now.reflectivity_dbz, rain,
        completion="bulk", **common,
    )
    gated = estimate_motion(
        prev.reflectivity_dbz, now.reflectivity_dbz, rain,
        completion="confidence", **common,
    )

    for out in (legacy, gated):
        assert np.isfinite(out.vy).all() and np.isfinite(out.vx).all()
        assert float(np.abs(out.vy).max()) <= 30.0
        assert float(np.abs(out.vx).max()) <= 30.0

    # The estimator's own stall is a property of the estimate, so the two
    # runs must report the same number for it — and on this pair there is
    # nothing to fix.
    assert gated.stalled_share == pytest.approx(legacy.stalled_share)
    assert gated.stalled_share < 0.01
    assert gated.stalled_share_completed <= legacy.stalled_share_completed

    # Nothing to fix means: don't change the answer. The two bulk vectors
    # agree in direction to within 10°, and the on-echo field the forecast
    # advects with keeps the same median speed to within 10 %.
    import math

    bearing_gap = abs(
        math.degrees(
            math.atan2(gated.bulk_vy, gated.bulk_vx)
            - math.atan2(legacy.bulk_vy, legacy.bulk_vx)
        )
    )
    assert min(bearing_gap, 360.0 - bearing_gap) < 10.0
    wet = np.isfinite(rain) & (rain >= THRESHOLD)
    assert wet.sum() > 10_000
    speed = {
        name: float(np.median(np.hypot(out.vy[wet], out.vx[wet])))
        for name, out in (("legacy", legacy), ("gated", gated))
    }
    assert speed["gated"] == pytest.approx(speed["legacy"], rel=0.10)


def test_flow_confidence_is_cheap_on_the_native_grid() -> None:
    """One Sobel pair and one box filter — the sidecar runs under a cap."""
    import time
    import tracemalloc

    _, now = _odense_pair()
    dbz = now.reflectivity_dbz
    assert dbz.size > 3_000_000, "the fixture must be a native-resolution grid"

    flow_confidence(dbz)  # warm any lazy import
    tracemalloc.start()
    t0 = time.perf_counter()
    energy = flow_confidence(dbz)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert energy.shape == dbz.shape
    assert energy.dtype == np.float32
    # Generous bounds: this is a smoke alarm for an accidental O(n·window²)
    # or a float64 copy, not a benchmark.
    assert elapsed_ms < 2000.0, f"flow_confidence took {elapsed_ms:.0f} ms"
    assert peak < 12 * dbz.nbytes, f"peak {peak / 1e6:.0f} MB for a {dbz.nbytes / 1e6:.0f} MB grid"
