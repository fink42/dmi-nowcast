"""Equivalence tests for the vendored-STEPS performance rewrites.

``src/dmi_nowcast_core/_vendor/pysteps_steps/`` is a redistribution of
pysteps 1.21.1 and its NOTICE enumerates every deviation from upstream.
Modifications 4 and 6 replace two upstream helpers with faster / leaner
implementations that are supposed to be **exactly** equal, not merely
close — a calibration corpus and the isotonic curves fitted from it are
only comparable across rebuilds if the ensemble is unchanged.

These tests are the proof of that claim: each rewritten helper is run
against a verbatim copy of the upstream code it replaced, over a grid of
inputs that includes the degenerate cases (empty mask, no rim, ragged
lists). No STEPS run, no network — pure functions only.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import (
    binary_dilation,
    generate_binary_structure,
    iterate_structure,
)

from dmi_nowcast_core._vendor.pysteps_steps.nowcasts.utils import (
    compute_dilated_mask,
    stack_forecast_output,
)


# ---------------------------------------------------------------------------
# Upstream reference implementations (pysteps 1.21.1, verbatim)
# ---------------------------------------------------------------------------


def upstream_compute_dilated_mask(input_mask, kr, r):
    input_mask = np.ndarray.astype(input_mask.copy(), "uint8")
    mask_dilated = binary_dilation(input_mask, kr)
    kr1 = generate_binary_structure(2, 1)
    mask = mask_dilated.astype(float)
    for _ in range(r):
        mask_dilated = binary_dilation(mask_dilated, kr1)
        mask += mask_dilated
    return mask / mask.max()


def upstream_stack(members):
    return np.stack(members)


# ---------------------------------------------------------------------------
# Modification 4 — compute_dilated_mask via one taxicab distance transform
# ---------------------------------------------------------------------------


CROSS = generate_binary_structure(2, 1)


def _masks(shape=(48, 56)):
    rng = np.random.default_rng(20260907)
    speckle = rng.random(shape) > 0.97
    blobs = binary_dilation(rng.random(shape) > 0.995, iterate_structure(CROSS, 4))
    edge = np.zeros(shape, dtype=bool)
    edge[0, :] = True
    edge[:, -1] = True
    one = np.zeros(shape, dtype=bool)
    one[shape[0] // 2, shape[1] // 2] = True
    return {
        "speckle": speckle,
        "blobs": blobs,
        "dense": rng.random(shape) > 0.3,
        "edge": edge,
        "single-pixel": one,
        "full": np.ones(shape, dtype=bool),
    }


@pytest.mark.parametrize("name", sorted(_masks()))
@pytest.mark.parametrize("iterations", [0, 1, 2, 5])
@pytest.mark.parametrize("r", [0, 1, 3, 10])
def test_compute_dilated_mask_matches_upstream_exactly(name, iterations, r):
    """Element-for-element equality, not a tolerance.

    ``i`` iterations of 3x3-cross dilation is exactly "cityblock distance
    <= i", so the accumulated rim is ``clip(r + 1 - d1, 0, r + 1)`` — the
    identity the rewrite rests on. Swept over mask shapes, structuring
    elements and rim widths, including ``r = 0`` (no rim at all).
    """
    mask = _masks()[name]
    kr = iterate_structure(CROSS, iterations) if iterations else CROSS
    expected = upstream_compute_dilated_mask(mask, kr, r)
    got = compute_dilated_mask(mask, kr, r)
    assert got.dtype == expected.dtype
    assert np.array_equal(got, expected)


@pytest.mark.parametrize("r", [0, 1, 10])
def test_compute_dilated_mask_empty_input_is_all_nan_like_upstream(r):
    """A wholly dry frame: upstream divides a zero array by its own max and
    yields NaN everywhere. The rewrite must not turn that into 1.0 — a
    dry event in the corpus's dry stratum would otherwise get a completely
    different precipitation mask."""
    empty = np.zeros((32, 40), dtype=bool)
    with np.errstate(invalid="ignore"):
        expected = upstream_compute_dilated_mask(empty, CROSS, r)
        got = compute_dilated_mask(empty, CROSS, r)
    assert np.isnan(expected).all()
    assert np.isnan(got).all()


def test_compute_dilated_mask_is_normalised_and_monotone():
    """Sanity on the semantics themselves: 1 on the buffered echo, falling
    to 0 at the rim's outer edge, never outside [0, 1]."""
    mask = _masks()["blobs"]
    out = compute_dilated_mask(mask, CROSS, 10)
    assert out.min() >= 0.0 and out.max() == pytest.approx(1.0)
    assert np.all(out[binary_dilation(mask, CROSS)] == 1.0)


# ---------------------------------------------------------------------------
# Modification 6 — stack_forecast_output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n_members,n_steps", [(1, 1), (3, 4), (16, 8)])
def test_stack_forecast_output_matches_np_stack(dtype, n_members, n_steps):
    rng = np.random.default_rng(7)
    shape = (5, 6)
    members = [
        [rng.random(shape).astype(dtype) for _ in range(n_steps)]
        for _ in range(n_members)
    ]
    expected = upstream_stack([list(m) for m in members])
    got = stack_forecast_output(members)
    assert got.shape == expected.shape
    assert got.dtype == expected.dtype
    assert np.array_equal(got, expected)


def test_stack_forecast_output_preserves_nan():
    members = [[np.array([[np.nan, 1.0]]), np.array([[2.0, np.nan]])]]
    got = stack_forecast_output([list(m) for m in members])
    assert np.array_equal(got, np.stack(members), equal_nan=True)


def test_stack_forecast_output_consumes_the_input_lists():
    """The whole point is releasing each slab as it is copied — if the
    caller's list still held them the peak would be unchanged."""
    members = [[np.zeros((4, 4)) for _ in range(3)] for _ in range(2)]
    stack_forecast_output(members)
    assert members == [None, None]


@pytest.mark.parametrize("members", [
    [],                                            # no members
    [[]],                                          # a member with no steps
    [[np.zeros((2, 2))], [np.zeros((2, 2))] * 2],  # ragged
])
def test_stack_forecast_output_degenerate_input_falls_back_to_np_stack(members):
    """Degenerate shapes keep upstream's exact behaviour, including which
    exception np.stack raises."""
    try:
        expected = upstream_stack([list(m) for m in members])
    except Exception as exc:  # noqa: BLE001
        with pytest.raises(type(exc)):
            stack_forecast_output(members)
    else:
        assert np.array_equal(stack_forecast_output(members), expected)
