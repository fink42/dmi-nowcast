"""The Phase H4 flow candidates: the H-c grid, LK, median3, oracle.

What is worth pinning here, and why each one:

1. **Every entry produces a usable field.** Finite, ±30 px clipped,
   float32, right shape. A Layer A run that scored NaN would not fail —
   it would produce a table of zeros and a plausible-looking report.
2. **The vendored Lucas–Kanade recovers a known displacement**, within
   15 %. It is 400 lines of someone else's code reached through a
   rewritten dispatcher; the only thing that says the rewrite kept the
   call path intact is that the answer is still right. The sign and axis
   convention is checked in the same test, because a transposed
   ``(vy, vx)`` is the one bug no aggregate CSI would explain.
3. **``median3`` degenerates to ``confidence``** when its three pairs
   agree. The median is only worth having if it is a no-op on agreement,
   so a difference in a Layer A report is attributable to disagreement.
4. **``oracle`` refuses to run without its frame.** It is the ceiling; an
   oracle that quietly scored as production would silently invalidate
   every "how much headroom is left" reading taken from it.
5. **The grid is the grid**: 24 named cells, the documented parameters,
   and ``winsize=31`` NOT among them (that is ``confidence``).
"""
from __future__ import annotations

import numpy as np
import pytest

from dmi_nowcast_core import flow_variants as fv
from dmi_nowcast_core.dense_flow import DEFAULT_FILL_DBZ, _to_uint8, dense_flow
from dmi_nowcast_core.variants import (
    MAX_PX_PER_FRAME,
    SUPPORT_THRESHOLD_MM_H,
    confidence_flow,
    get_variant,
    list_variants,
    variant_requirements,
)

PIXEL_KM = 0.5
SIZE = 128


def _frame(cx: float, cy: float) -> np.ndarray:
    """Three Gaussian cells on the DMI grid floor, at a given centre.

    Three rather than one so the sparse-feature estimators have more than
    a single corner to track and the interpolation has something to
    interpolate between — one blob would let LK pass on a degenerate
    "every vector identical" field.
    """
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    out = np.full((SIZE, SIZE), DEFAULT_FILL_DBZ, dtype=np.float32)
    for dx, dy, sigma in ((0.0, 0.0, 11.0), (28.0, -18.0, 7.0), (-22.0, 22.0, 9.0)):
        cell = 58.0 * np.exp(
            -(((xx - cx - dx) ** 2 + (yy - cy - dy) ** 2) / (2 * sigma ** 2))
        ) - 32.0
        out = np.maximum(out, cell)
    return out.astype(np.float32)


#: Displacement of the synthetic between frames, in px: east and north.
SHIFT_X, SHIFT_Y = 5, -2


def _sequence(n: int = 5) -> list[np.ndarray]:
    """``n`` frames translating by ``(SHIFT_Y, SHIFT_X)`` each step."""
    return [_frame(50.0 + SHIFT_X * i, 64.0 + SHIFT_Y * i) for i in range(n)]


@pytest.fixture(scope="module")
def moving_blobs():
    """``(history, prev, curr, future, rain, echo)`` for the whole module.

    Module-scoped because building it is cheap but calling 24 Farnebäck
    variants on it is not, and every test wants the same arrays.
    """
    h0, h1, prev, curr, future = _sequence(5)
    rain = np.where(curr > 0.0, curr / 10.0, 0.0).astype(np.float32)
    echo = rain >= SUPPORT_THRESHOLD_MM_H
    return [h0, h1], prev, curr, future, rain, echo


def _call(name: str, moving_blobs):
    """Invoke a registered variant, supplying whatever it declared it needs."""
    history, prev, curr, future, rain, _echo = moving_blobs
    make_flow = get_variant(name)
    req = variant_requirements(make_flow)
    extra = {}
    if req.history:
        extra["history_dbz"] = history[-req.history:]
    if req.future:
        extra["future_dbz"] = future
    return make_flow(prev, curr, rain, pixel_km=PIXEL_KM, **extra)


NEW_VARIANTS = sorted(
    set(list_variants()) - {"bulk", "confidence", "production", "persistence"}
)


# ---------------------------------------------------------------------------
# 1. every new entry produces a usable field
# ---------------------------------------------------------------------------
def test_the_registry_gained_exactly_the_h4_candidates():
    assert set(NEW_VARIANTS) == set(fv.farneback_grid()) | {
        "lucaskanade", "median3", "oracle",
    }
    assert len(NEW_VARIANTS) == 27


@pytest.mark.parametrize("name", NEW_VARIANTS)
def test_new_variant_returns_a_finite_clipped_field(name, moving_blobs):
    _history, _prev, curr, _future, _rain, _echo = moving_blobs
    vy, vx = _call(name, moving_blobs)

    assert vy.shape == curr.shape and vx.shape == curr.shape
    assert vy.dtype == np.float32 and vx.dtype == np.float32
    assert np.isfinite(vy).all() and np.isfinite(vx).all()
    assert np.abs(vy).max() <= MAX_PX_PER_FRAME
    assert np.abs(vx).max() <= MAX_PX_PER_FRAME


@pytest.mark.parametrize("name", NEW_VARIANTS)
def test_new_variant_recovers_the_synthetic_motion(name, moving_blobs):
    """Not a skill test — a "did it track anything at all" test.

    ±1.5 px is loose on purpose: a 61-px window on a 128-px grid smooths
    the estimate toward the domain mean, which is exactly the trade the
    H-c sweep exists to measure. A variant outside this band has not
    traded resolution for robustness, it has failed.
    """
    *_, echo = moving_blobs
    vy, vx = _call(name, moving_blobs)
    assert float(np.mean(vx[echo])) == pytest.approx(SHIFT_X, abs=1.5)
    assert float(np.mean(vy[echo])) == pytest.approx(SHIFT_Y, abs=1.5)


@pytest.mark.parametrize("name", NEW_VARIANTS)
def test_new_variant_is_picklable(name):
    """The harness runs one process per day under ``ProcessPoolExecutor``.

    The grid entries are class instances rather than closures precisely
    so this holds; a closure would raise here, three hours into a run.
    """
    import pickle

    assert pickle.loads(pickle.dumps(get_variant(name))) is not None


# ---------------------------------------------------------------------------
# 2. the H-c grid
# ---------------------------------------------------------------------------
def test_farneback_grid_names_and_count():
    grid = fv.farneback_grid()
    assert len(grid) == 24
    assert len(grid) == (
        len(fv.FARNEBACK_WINSIZES) * len(fv.FARNEBACK_LEVELS)
        * len(fv.FARNEBACK_POLY_N)
    )
    assert set(grid) == {
        f"farneback_w{w}_l{l}_p{p}"
        for w in (21, 41, 51, 61) for l in (3, 4, 5) for p in (5, 7)
    }
    assert grid["farneback_w41_l4_p7"].winsize == 41
    assert grid["farneback_w41_l4_p7"].levels == 4
    assert grid["farneback_w41_l4_p7"].poly_n == 7


def test_the_grid_does_not_shadow_production():
    """``winsize=31, levels=3, poly_n=7`` IS ``confidence``.

    Registering it again would put the baseline in the candidate list
    under a second name and invite a report comparing it with itself.
    """
    assert 31 not in fv.FARNEBACK_WINSIZES
    assert not any("_w31_" in name for name in list_variants())


def test_poly_sigma_follows_opencv_s_recommendation():
    """1.1 for poly_n=5, 1.5 for poly_n=7 — not swept independently."""
    assert fv.POLY_SIGMA_FOR_POLY_N == {5: 1.1, 7: 1.5}
    grid = fv.farneback_grid()
    for cell in grid.values():
        assert cell.poly_sigma == fv.POLY_SIGMA_FOR_POLY_N[cell.poly_n]


def test_farneback_variant_rejects_an_unknown_poly_n():
    with pytest.raises(ValueError, match="poly_sigma"):
        fv.FarnebackVariant(31, 3, 6)


def test_a_grid_cell_is_its_parameters_and_nothing_else(moving_blobs):
    """The cell must be ``dense_flow(params)`` through the shared completion.

    Pinned against the sequence spelled out by hand, so a refactor that
    moved a parameter or dropped the completion would fail here rather
    than shift a CSI by 0.003 in a report nobody could explain.
    """
    from dmi_nowcast_core.dense_flow import estimate_motion

    _history, prev, curr, _future, rain, _echo = moving_blobs
    cell = fv.farneback_grid()["farneback_w51_l4_p5"]
    raw = dense_flow(prev, curr, winsize=51, levels=4, poly_n=5, poly_sigma=1.1)
    ref = estimate_motion(
        prev, curr, rain, pixel_km=PIXEL_KM, dt_min=10.0,
        support_threshold_mm_h=SUPPORT_THRESHOLD_MM_H,
        completion="confidence", max_px_per_frame=MAX_PX_PER_FRAME, flow=raw,
    )
    vy, vx = cell(prev, curr, rain, pixel_km=PIXEL_KM)
    np.testing.assert_array_equal(vy, ref.vy)
    np.testing.assert_array_equal(vx, ref.vx)


def test_grid_cells_actually_differ_from_each_other(moving_blobs):
    """A sweep whose cells all produce the same field measures nothing."""
    _history, prev, curr, _future, rain, _echo = moving_blobs
    grid = fv.farneback_grid()
    small = grid["farneback_w21_l3_p5"](prev, curr, rain, pixel_km=PIXEL_KM)
    large = grid["farneback_w61_l5_p7"](prev, curr, rain, pixel_km=PIXEL_KM)
    assert not np.array_equal(small[0], large[0])


# ---------------------------------------------------------------------------
# 3. Lucas-Kanade
# ---------------------------------------------------------------------------
def test_lucaskanade_recovers_a_known_displacement(moving_blobs):
    """Within 15 %, on the raw vendored output — completion excluded.

    Taken on the RAW ``dense_lucaskanade`` field rather than on the
    registered variant's, so the number is a statement about the
    vendored code and not about ``complete_flow``. The completion is
    checked separately by the finite/clipped test above.
    """
    from dmi_nowcast_core._vendor.pysteps_motion import dense_lucaskanade

    _history, prev, curr, _future, _rain, echo = moving_blobs
    stack = np.stack([
        _to_uint8(prev, DEFAULT_FILL_DBZ), _to_uint8(curr, DEFAULT_FILL_DBZ),
    ]).astype(np.float32)
    field = dense_lucaskanade(stack)

    assert field.shape == (2, SIZE, SIZE)
    # [0] is the x- (eastward) component, [1] the y- (southward) one.
    vx, vy = field[0], field[1]
    assert abs(float(np.mean(vx[echo])) - SHIFT_X) <= 0.15 * abs(SHIFT_X)
    assert abs(float(np.mean(vy[echo])) - SHIFT_Y) <= 0.15 * abs(SHIFT_Y)


def test_lucaskanade_variant_keeps_the_dense_flow_axis_convention(moving_blobs):
    """``(vy, vx)``, positive vy south, positive vx east — as ``dense_flow``.

    The synthetic moves east and north, so the two components have
    OPPOSITE signs: a transposition or a sign flip cannot pass this by
    accident, which a purely eastward test would allow.
    """
    _history, prev, curr, _future, rain, echo = moving_blobs
    vy, vx = fv.lucaskanade_flow(prev, curr, rain, pixel_km=PIXEL_KM)
    ref_vy, ref_vx = dense_flow(prev, curr)

    assert float(np.mean(vx[echo])) > 0.0     # east
    assert float(np.mean(vy[echo])) < 0.0     # north
    # The same sense as Farnebäck's own estimate on the same pair.
    assert np.sign(np.mean(vx[echo])) == np.sign(np.mean(ref_vx[echo]))
    assert np.sign(np.mean(vy[echo])) == np.sign(np.mean(ref_vy[echo]))


def test_lk_chunking_is_memory_only_and_changes_no_number():
    """``nchunks`` partitions the OUTPUT grid, so the field is identical.

    ``lucaskanade_flow`` raises it from pysteps' default of 4 to 16
    because the default costs a 1.0 GB transient on the native
    1728x1984 composite and two batch workers share a 5 GB cap. That is
    only defensible if it is bit-for-bit the same answer, so: bit for
    bit, on a grid big enough that the chunk boundaries actually fall
    inside the domain.
    """
    from dmi_nowcast_core._vendor.pysteps_motion import dense_lucaskanade

    big = 224
    yy, xx = np.mgrid[0:big, 0:big]
    def frame(cx, cy):
        out = np.full((big, big), DEFAULT_FILL_DBZ, dtype=np.float32)
        for dx, dy, sigma in ((0, 0, 18), (60, -40, 12), (-50, 45, 15)):
            out = np.maximum(out, 58.0 * np.exp(
                -(((xx - cx - dx) ** 2 + (yy - cy - dy) ** 2) / (2 * sigma ** 2))
            ) - 32.0)
        return out.astype(np.float32)

    stack = np.stack([
        _to_uint8(frame(90, 112), DEFAULT_FILL_DBZ),
        _to_uint8(frame(97, 109), DEFAULT_FILL_DBZ),
    ]).astype(np.float32)

    default = dense_lucaskanade(stack)
    chunked = dense_lucaskanade(stack, interp_kwargs={"nchunks": fv.LK_NCHUNKS})
    np.testing.assert_array_equal(default, chunked)
    assert fv.LK_NCHUNKS > 4


def test_lucaskanade_on_a_featureless_pair_is_a_zero_field():
    """No corner to track → pysteps returns zeros → persistence.

    Documented behaviour rather than an error: "nothing was measurable"
    is a real answer on a dry composite, and the stall diagnostic is
    where it shows up.
    """
    flat = np.full((64, 64), DEFAULT_FILL_DBZ, dtype=np.float32)
    rain = np.zeros((64, 64), dtype=np.float32)
    vy, vx = fv.lucaskanade_flow(flat, flat, rain, pixel_km=PIXEL_KM)
    assert not np.any(vy) and not np.any(vx)


def test_vendored_lucaskanade_rejects_an_unvendored_method():
    """The dispatcher was rewritten; an unknown name must say so clearly."""
    from dmi_nowcast_core._vendor.pysteps_motion import dense_lucaskanade

    stack = np.zeros((2, 32, 32), dtype=np.float32)
    with pytest.raises(ValueError, match="vendored"):
        dense_lucaskanade(stack, fd_method="blob")
    with pytest.raises(ValueError, match="vendored"):
        dense_lucaskanade(stack, interp_method="rbfinterp2d")


# ---------------------------------------------------------------------------
# 4. median3
# ---------------------------------------------------------------------------
def test_median3_equals_confidence_when_the_three_pairs_agree(monkeypatch):
    """Three agreeing pair estimates ⇒ ``median3`` IS ``confidence``.

    The median of three identical fields is that field, so what is left
    is ``dense_flow(prev, curr)`` through the shared completion — which
    is ``confidence`` exactly, bit for bit. Stubbed rather than staged
    on a synthetic because "agree" has to mean *identical* for the claim
    to be sharp: a translating picture gives the three pairs the same
    displacement but not the same pixels, so Farnebäck's boundary
    handling differs by a fraction of a pixel and the equality would
    have to be softened into a tolerance that proves much less.
    """
    h0, h1, prev, curr = _sequence(4)
    rain = np.where(curr > 0.0, curr / 10.0, 0.0).astype(np.float32)
    agreed = dense_flow(prev, curr)

    monkeypatch.setattr(
        fv, "dense_flow",
        lambda older, newer, **kw: (agreed[0].copy(), agreed[1].copy()),
    )
    got_vy, got_vx = fv.median3_flow(
        prev, curr, rain, pixel_km=PIXEL_KM, history_dbz=[h0, h1],
    )
    monkeypatch.undo()
    want_vy, want_vx = confidence_flow(prev, curr, rain, pixel_km=PIXEL_KM)

    np.testing.assert_array_equal(got_vy, want_vy)
    np.testing.assert_array_equal(got_vx, want_vx)


def test_median3_stays_close_to_confidence_on_a_translating_field():
    """The real synthetic: three pairs that nearly agree, so should the fields.

    The complement of the test above — with the actual estimator the
    three pairs are the same motion of a shifted picture, so they differ
    slightly and the median is a real median. It must still land on
    ``confidence``: a candidate that moved the whole field on a case with
    no disagreement to resolve would be changing more than the plan says
    it changes.
    """
    h0, h1, prev, curr = _sequence(4)
    rain = np.where(curr > 0.0, curr / 10.0, 0.0).astype(np.float32)
    echo = rain >= SUPPORT_THRESHOLD_MM_H

    got_vy, got_vx = fv.median3_flow(
        prev, curr, rain, pixel_km=PIXEL_KM, history_dbz=[h0, h1],
    )
    want_vy, want_vx = confidence_flow(prev, curr, rain, pixel_km=PIXEL_KM)

    assert float(np.mean(got_vx[echo])) == pytest.approx(
        float(np.mean(want_vx[echo])), abs=0.05)
    assert float(np.mean(got_vy[echo])) == pytest.approx(
        float(np.mean(want_vy[echo])), abs=0.05)
    # ON THE ECHO no pixel moves by a fifth of a pixel — that is where
    # the estimate is real and where a genuine change of candidate would
    # show. Off it the two fields relax toward bulk vectors that differ
    # in the fourth decimal, and the relaxation multiplies that by the
    # distance, so the far corners of the domain part by ~1.8 px on this
    # 128-px synthetic. That is the completion doing its job, not median3
    # disagreeing, so the check is deliberately not global.
    assert float(np.abs(got_vy - want_vy)[echo].max()) < 0.2
    assert float(np.abs(got_vx - want_vx)[echo].max()) < 0.2


def test_median3_takes_the_middle_estimate_per_pixel(monkeypatch):
    """Three constant pair estimates → the middle one, everywhere.

    ``dense_flow`` is stubbed so the three pairs return known, different
    fields; anything but the median (a mean, the last pair, the first)
    lands on a different number.
    """
    size = 24
    rain = np.zeros((size, size), dtype=np.float32)
    rain[8:16, 8:16] = 1.0
    values = iter([(1.0, -7.0), (9.0, -1.0), (4.0, -3.0)])

    def fake_dense_flow(prev, curr, **kwargs):
        vy, vx = next(values)
        return (np.full((size, size), vy, dtype=np.float32),
                np.full((size, size), vx, dtype=np.float32))

    monkeypatch.setattr(fv, "dense_flow", fake_dense_flow)
    zeros = np.zeros((size, size), dtype=np.float32)
    vy, vx = fv.median3_flow(
        zeros, zeros, rain, pixel_km=PIXEL_KM, history_dbz=[zeros, zeros],
    )
    # median(1, 9, 4) = 4; median(-7, -1, -3) = -3.
    assert np.allclose(vy, 4.0)
    assert np.allclose(vx, -3.0)


def test_median3_declares_two_history_frames():
    req = variant_requirements(get_variant("median3"))
    assert req.history == 2 and req.future is False


def test_median3_uses_the_newest_history_when_given_more():
    """``history_dbz`` is chronological; extra OLD frames are ignored."""
    h_extra, h0, h1, prev, curr = _sequence(5)
    rain = np.where(curr > 0.0, curr / 10.0, 0.0).astype(np.float32)
    a = fv.median3_flow(prev, curr, rain, pixel_km=PIXEL_KM,
                        history_dbz=[h0, h1])
    b = fv.median3_flow(prev, curr, rain, pixel_km=PIXEL_KM,
                        history_dbz=[h_extra, h0, h1])
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


@pytest.mark.parametrize("history", [None, [], "one"])
def test_median3_without_enough_history_says_so(history, moving_blobs):
    _h, prev, curr, _future, rain, _echo = moving_blobs
    kwargs = {} if history == "one" else {"history_dbz": history}
    if history == "one":
        kwargs["history_dbz"] = [_h[0]]
    with pytest.raises(ValueError, match="needs 2 frames older"):
        fv.median3_flow(prev, curr, rain, pixel_km=PIXEL_KM, **kwargs)


def test_median3_rejects_a_mismatched_history_shape(moving_blobs):
    history, prev, curr, _future, rain, _echo = moving_blobs
    wrong = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="history_dbz"):
        fv.median3_flow(
            prev, curr, rain, pixel_km=PIXEL_KM, history_dbz=[history[0], wrong],
        )


# ---------------------------------------------------------------------------
# 5. oracle
# ---------------------------------------------------------------------------
def test_oracle_without_the_future_frame_raises_a_clear_error(moving_blobs):
    _history, prev, curr, _future, rain, _echo = moving_blobs
    with pytest.raises(ValueError) as excinfo:
        fv.oracle_flow(prev, curr, rain, pixel_km=PIXEL_KM)
    message = str(excinfo.value)
    assert "future_dbz" in message
    assert "needs_future" in message
    # It says what it is, so nobody wires it into the service by mistake.
    assert "ceiling" in message


def test_oracle_declares_that_it_reads_the_future():
    req = variant_requirements(get_variant("oracle"))
    assert req.future is True and req.history == 0


def test_oracle_estimates_from_curr_and_future_not_prev_and_curr(moving_blobs):
    """The pair straddling the target time, through the shared completion."""
    from dmi_nowcast_core.dense_flow import estimate_motion

    _history, prev, curr, future, rain, _echo = moving_blobs
    ref = estimate_motion(
        prev, curr, rain, pixel_km=PIXEL_KM, dt_min=10.0,
        support_threshold_mm_h=SUPPORT_THRESHOLD_MM_H,
        completion="confidence", max_px_per_frame=MAX_PX_PER_FRAME,
        flow=dense_flow(curr, future),
    )
    vy, vx = fv.oracle_flow(
        prev, curr, rain, pixel_km=PIXEL_KM, future_dbz=future,
    )
    np.testing.assert_array_equal(vy, ref.vy)
    np.testing.assert_array_equal(vx, ref.vx)


def test_oracle_differs_from_production_when_the_motion_changes():
    """A field that turns: the past pair and the future pair disagree.

    Without this the oracle could be reading ``(prev, curr)`` and every
    other test would still pass on a uniformly translating synthetic.
    """
    prev = _frame(50.0, 64.0)
    curr = _frame(56.0, 64.0)          # moved east
    future = _frame(56.0, 76.0)        # then south instead
    rain = np.where(curr > 0.0, curr / 10.0, 0.0).astype(np.float32)
    echo = rain >= SUPPORT_THRESHOLD_MM_H

    o_vy, o_vx = fv.oracle_flow(
        prev, curr, rain, pixel_km=PIXEL_KM, future_dbz=future,
    )
    p_vy, p_vx = confidence_flow(prev, curr, rain, pixel_km=PIXEL_KM)
    # The oracle sees the southward turn; production still sees eastward.
    assert float(np.mean(o_vy[echo])) > 3.0
    assert float(np.mean(p_vy[echo])) == pytest.approx(0.0, abs=1.0)
    assert float(np.mean(p_vx[echo])) > 3.0


def test_oracle_rejects_a_mismatched_future_shape(moving_blobs):
    _history, prev, curr, _future, rain, _echo = moving_blobs
    with pytest.raises(ValueError, match="future_dbz"):
        fv.oracle_flow(
            prev, curr, rain, pixel_km=PIXEL_KM,
            future_dbz=np.zeros((4, 4), dtype=np.float32),
        )
