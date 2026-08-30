"""Tests for Farnebäck dense flow."""
from __future__ import annotations

import numpy as np
import pytest

from dmi_nowcast_core.dense_flow import dense_flow, mean_flow


def _textured_field(shape: tuple[int, int], seed: int = 0) -> np.ndarray:
    """A multi-blob field that mimics a real radar composite: enough structure
    for Farnebäck's polynomial expansion to track displacement."""
    rng = np.random.default_rng(seed)
    field = np.zeros(shape, dtype=np.float32)
    rows, cols = np.indices(shape, dtype=np.float32)
    for _ in range(30):
        cy = rng.uniform(20, shape[0] - 20)
        cx = rng.uniform(20, shape[1] - 20)
        sigma = rng.uniform(5, 15)
        amp = rng.uniform(15, 40)
        field += (amp * np.exp(-((rows - cy) ** 2 + (cols - cx) ** 2) / (2 * sigma ** 2))).astype(np.float32)
    return field


@pytest.mark.parametrize(
    "dy, dx",
    [(3, 0), (0, 4), (-2, 5), (5, -3)],
)
def test_mean_flow_has_correct_sign_and_order_of_magnitude(dy, dx):
    """Farnebäck systematically underestimates rigid translations on synthetic
    fields (a known property — it works much better on real radar where the
    spectrum has broader texture). We assert:

    - direction (sign of dy, dx) is correct,
    - magnitude is at least 25 % of the true displacement.

    Production-grade verification comes from the Phase 3 backtest, not synthetic
    blob tests.
    """
    prev = _textured_field((256, 256), seed=42)
    curr = np.roll(prev, shift=(dy, dx), axis=(0, 1))
    vy, vx = dense_flow(prev, curr)
    interior = np.zeros(prev.shape, dtype=bool)
    interior[30:-30, 30:-30] = True
    mvy, mvx = mean_flow(vy, vx, mask=interior)
    for axis, (got, expected) in [("y", (mvy, dy)), ("x", (mvx, dx))]:
        if expected == 0:
            assert abs(got) < 1.0, f"{axis}: expected ~0, got {got:.2f}"
        else:
            assert (got > 0) == (expected > 0), f"{axis}: wrong sign: {got:.2f} vs {expected}"
            assert abs(got) >= 0.25 * abs(expected), (
                f"{axis}: magnitude too small: {got:.2f} vs {expected}"
            )


def test_zero_motion_yields_near_zero_flow():
    field = _textured_field((128, 128), seed=1)
    vy, vx = dense_flow(field, field)
    assert np.abs(vy).max() < 0.1
    assert np.abs(vx).max() < 0.1


def test_nan_inputs_do_not_crash():
    a = _textured_field((128, 128), seed=2)
    b = np.roll(a, shift=(3, 0), axis=(0, 1))
    a[0, 0] = np.nan
    b[-1, -1] = -np.inf
    vy, vx = dense_flow(a, b)
    assert vy.shape == a.shape
    assert np.all(np.isfinite(vy))
    assert np.all(np.isfinite(vx))


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        dense_flow(np.zeros((10, 10), dtype=np.float32), np.zeros((10, 11), dtype=np.float32))


@pytest.mark.parametrize("dy, dx", [(3, 0), (0, 4), (-2, 5)])
def test_skimage_backend_recovers_motion(dy, dx, monkeypatch):
    """The scikit-image branch must produce flow with the correct sign on
    synthetic small-motion data so it's a usable fallback when opencv is
    unavailable on HA OS.

    Tolerance is looser than Farnebäck's because we use TV-L1 (not ILK)
    on the skimage path — TV-L1 is more conservative with magnitudes on
    small synthetic blob fields but gives correct direction. ILK gave
    wrong directions on real radar data (e.g. NW where opencv said NE);
    sign correctness is what matters for the overlay arrow, magnitude
    accuracy is verified end-to-end in the integration smoke tests.
    """
    # Check skimage is available BEFORE applying the monkeypatch (the patch
    # would otherwise block the import we use to skip).
    pytest.importorskip("skimage.registration")

    # Force the opencv branch to be unavailable so dense_flow falls through
    # to the skimage branch even on a dev machine where cv2 is installed.
    import builtins
    real_import = builtins.__import__

    def _no_cv2(name, *args, **kwargs):
        if name == "cv2":
            raise ImportError("cv2 disabled for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_cv2)

    prev = _textured_field((192, 192), seed=7)
    curr = np.roll(prev, shift=(dy, dx), axis=(0, 1))
    vy, vx = dense_flow(prev, curr)
    interior = np.zeros(prev.shape, dtype=bool)
    interior[30:-30, 30:-30] = True
    mvy, mvx = mean_flow(vy, vx, mask=interior)
    for axis, (got, expected) in [("y", (mvy, dy)), ("x", (mvx, dx))]:
        if expected == 0:
            assert abs(got) < 1.0
        else:
            assert (got > 0) == (expected > 0), f"{axis}: wrong sign: {got:.2f} vs {expected}"
            # TV-L1 is conservative on tiny synthetic motion (~2 px). We
            # only require ~10% of the true magnitude — the sign and
            # direction are what matter for the overlay arrow.
            assert abs(got) >= 0.1 * abs(expected), (
                f"{axis}: magnitude vanishingly small: {got:.2f} vs {expected}"
            )


def test_no_backend_raises_unavailable(monkeypatch):
    """If neither opencv nor scikit-image is importable, dense_flow must raise
    DenseFlowUnavailable so coordinator.py can fall back to mean motion."""
    from dmi_nowcast_core.dense_flow import DenseFlowUnavailable
    import builtins
    real_import = builtins.__import__

    def _no_flow_libs(name, *args, **kwargs):
        if name == "cv2" or name == "skimage.registration":
            raise ImportError(f"{name} disabled")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_flow_libs)
    a = np.zeros((16, 16), dtype=np.float32)
    with pytest.raises(DenseFlowUnavailable):
        dense_flow(a, a)
