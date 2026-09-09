"""The flow-variant registry, pinned against the live pipeline.

``variants.bulk_flow`` (the pre-H-F field) is the Layer A baseline every Phase H candidate is
measured against, so it has to be ``compute.py::_compute_sync`` and not an
approximation of it. These tests re-run the inline sequence from that
function on a synthetic pair and demand bit-for-bit equality, and then
pin the two things about the ORDER that a plausible-looking refactor
would get wrong: completion runs before the NaN fill, and the clip runs
last.
"""
from __future__ import annotations

import numpy as np
import pytest

from dmi_nowcast_core import variants
from dmi_nowcast_core.dense_flow import complete_flow, dense_flow
from dmi_nowcast_core.variants import (
    MAX_PX_PER_FRAME,
    NEEDS_FUTURE_ATTR,
    NEEDS_HISTORY_ATTR,
    SUPPORT_THRESHOLD_MM_H,
    VariantRequirements,
    get_variant,
    list_variants,
    persistence_flow,
    PRODUCTION_VARIANT,
    bulk_flow,
    confidence_flow,
    production_flow,
    register_variant,
    variant_requirements,
)

PIXEL_KM = 0.5


def _moving_blob_pair(shift: int = 4, size: int = 96):
    """Two dBZ frames with a Gaussian echo displaced east, plus its rain.

    Real reflectivity: −32 dBZ background (the DMI grid floor) with a
    smooth cell on top, so Farnebäck has something to track and
    ``complete_flow`` has an echo to relax away from.
    """
    yy, xx = np.mgrid[0:size, 0:size]

    def frame(cx: float) -> np.ndarray:
        blob = 55.0 * np.exp(-(((xx - cx) ** 2 + (yy - size / 2) ** 2) / (2 * 10.0 ** 2)))
        return (blob - 32.0).astype(np.float32)

    prev = frame(size / 2 - shift)
    curr = frame(size / 2)
    # Rain rate for the current frame, thresholded the way production does.
    rain = np.where(curr > 0.0, curr / 10.0, 0.0).astype(np.float32)
    return prev, curr, rain


def test_bulk_variant_matches_the_inline_compute_sequence():
    prev, curr, rain = _moving_blob_pair()

    # compute.py::_compute_sync, lines ~660-705, verbatim.
    vy, vx = dense_flow(prev, curr)
    vy, vx = complete_flow(
        vy, vx, rain,
        pixel_km=PIXEL_KM,
        support_threshold_mm_h=SUPPORT_THRESHOLD_MM_H,
    )
    vy = np.nan_to_num(vy, nan=0.0).astype(np.float32)
    vx = np.nan_to_num(vx, nan=0.0).astype(np.float32)
    np.clip(vy, -MAX_PX_PER_FRAME, MAX_PX_PER_FRAME, out=vy)
    np.clip(vx, -MAX_PX_PER_FRAME, MAX_PX_PER_FRAME, out=vx)

    got_vy, got_vx = bulk_flow(prev, curr, rain, pixel_km=PIXEL_KM)
    assert np.array_equal(got_vy, vy)
    assert np.array_equal(got_vx, vx)
    assert got_vy.dtype == np.float32 and got_vx.dtype == np.float32


def test_bulk_variant_recovers_the_synthetic_displacement():
    """A sanity check that the pair is trackable at all, not just equal."""
    prev, curr, rain = _moving_blob_pair(shift=4)
    vy, vx = bulk_flow(prev, curr, rain, pixel_km=PIXEL_KM)
    echo = rain >= SUPPORT_THRESHOLD_MM_H
    assert float(np.mean(vx[echo])) == pytest.approx(4.0, abs=1.5)
    assert abs(float(np.mean(vy[echo]))) < 1.0


def test_bulk_completes_before_filling_nan():
    """A NaN next to the echo must take the BULK vector, not zero.

    ``complete_flow`` gives a non-finite pixel weight 0, so it inherits
    bulk motion. If ``nan_to_num`` ran first the same pixel would be 0
    px/frame with a relaxation weight near 1 — i.e. frozen right beside
    moving rain, which is the stall this whole phase is about.
    """
    size = 20
    rain = np.zeros((size, size), dtype=np.float32)
    rain[8:12, 8:12] = 1.0
    flow_vy = np.zeros((size, size), dtype=np.float32)
    flow_vx = np.full((size, size), 5.0, dtype=np.float32)
    flow_vx[10, 13] = np.nan  # two pixels off the echo edge

    def fake_dense_flow(prev, curr, **kwargs):
        return flow_vy.copy(), flow_vx.copy()

    original = variants.dense_flow
    variants.dense_flow = fake_dense_flow
    try:
        vy, vx = bulk_flow(
            np.zeros((size, size), dtype=np.float32),
            np.zeros((size, size), dtype=np.float32),
            rain, pixel_km=PIXEL_KM,
        )
    finally:
        variants.dense_flow = original

    assert not np.any(np.isnan(vx))
    assert vx[10, 13] == pytest.approx(5.0)  # bulk, not 0
    assert vy[10, 13] == pytest.approx(0.0)


def test_bulk_clips_last_and_leaves_no_nan():
    """With no echo at all, completion is a no-op — the guards still apply."""
    size = 12
    flow = np.full((size, size), 1.0e6, dtype=np.float32)
    flow[0, 0] = np.nan

    def fake_dense_flow(prev, curr, **kwargs):
        return flow.copy(), -flow.copy()

    original = variants.dense_flow
    variants.dense_flow = fake_dense_flow
    try:
        vy, vx = bulk_flow(
            np.zeros((size, size), dtype=np.float32),
            np.zeros((size, size), dtype=np.float32),
            np.zeros((size, size), dtype=np.float32),  # nothing above 0.5 mm/h
            pixel_km=PIXEL_KM,
        )
    finally:
        variants.dense_flow = original

    assert not np.any(np.isnan(vy)) and not np.any(np.isnan(vx))
    assert vy[0, 0] == pytest.approx(0.0)
    assert vy[1, 1] == pytest.approx(MAX_PX_PER_FRAME)
    assert vx[1, 1] == pytest.approx(-MAX_PX_PER_FRAME)


def test_persistence_variant_is_exactly_zero():
    prev, curr, rain = _moving_blob_pair()
    vy, vx = persistence_flow(prev, curr, rain, pixel_km=PIXEL_KM)

    assert vy.shape == curr.shape and vx.shape == curr.shape
    assert vy.dtype == np.float32 and vx.dtype == np.float32
    assert not np.any(vy) and not np.any(vx)
    # Separate arrays: a caller writing into one must not move the other.
    vy[0, 0] = 1.0
    assert vx[0, 0] == 0.0


#: The four entries that describe the SERVICE. Everything else in the
#: registry is a Phase H candidate registered by ``flow_variants``.
SERVICE_VARIANTS = ("bulk", "confidence", "persistence", "production")


def test_registry_lookup_and_listing():
    """Sorted, complete, and the service entries are still all there.

    An equality against the whole list would now be a list of 31 names
    that has to be edited every time a candidate is added — which is the
    kind of test that gets updated without being read. The invariants
    worth holding are that the service four are present and resolve to
    the right callables, and that the listing is sorted and unique.
    """
    names = list_variants()
    assert set(names) >= set(SERVICE_VARIANTS)
    assert list(names) == sorted(names)
    assert len(set(names)) == len(names)
    assert get_variant("production") is production_flow
    assert get_variant("persistence") is persistence_flow
    with pytest.raises(KeyError) as excinfo:
        get_variant("nope")
    assert "production" in str(excinfo.value)


def test_the_phase_h_candidates_are_registered_by_importing_variants():
    """``import variants`` alone must be enough to see the whole registry.

    ``flow_variants`` registers on import and ``variants`` imports it at
    the bottom, so a caller that only ever touches ``variants`` — which
    is every caller — gets the complete list. If that import were ever
    dropped, ``--variant lucaskanade`` would fail with "unknown flow
    variant" on the VM and nowhere else.
    """
    names = list_variants()
    assert {"lucaskanade", "median3", "oracle"} <= set(names)
    assert sum(n.startswith("farneback_w") for n in names) == 24


def test_register_variant_refuses_to_shadow():
    with pytest.raises(ValueError):
        register_variant("production", persistence_flow)


def test_register_variant_adds_and_is_callable():
    name = "test-only-zero"

    def zero_flow(prev_dbz, curr_dbz, rain_now_mm_h, *, pixel_km):
        return persistence_flow(prev_dbz, curr_dbz, rain_now_mm_h, pixel_km=pixel_km)

    register_variant(name, zero_flow)
    try:
        assert name in list_variants()
        vy, _ = get_variant(name)(
            np.zeros((4, 4), dtype=np.float32),
            np.zeros((4, 4), dtype=np.float32),
            np.zeros((4, 4), dtype=np.float32),
            pixel_km=PIXEL_KM,
        )
        assert not np.any(vy)
    finally:
        variants._VARIANTS.pop(name, None)


def test_production_is_the_confidence_field_the_sidecar_serves():
    assert PRODUCTION_VARIANT == "confidence"
    assert get_variant("production") is confidence_flow
    assert production_flow is confidence_flow
    assert get_variant("bulk") is bulk_flow
    assert set(list_variants()) >= {"bulk", "confidence", "production", "persistence"}


def test_confidence_variant_routes_through_estimate_motion():
    from dmi_nowcast_core.dense_flow import estimate_motion

    prev, curr, rain = _moving_blob_pair()
    vy, vx = confidence_flow(prev, curr, rain, pixel_km=PIXEL_KM)
    ref = estimate_motion(
        prev, curr, rain, pixel_km=PIXEL_KM, dt_min=10.0,
        support_threshold_mm_h=0.5, completion="confidence", max_px_per_frame=30.0,
    )
    np.testing.assert_array_equal(vy, ref.vy)
    np.testing.assert_array_equal(vx, ref.vx)
    assert np.isfinite(vy).all() and np.abs(vy).max() <= 30.0


# ---------------------------------------------------------------------------
# the extra-frame declaration (H4)
# ---------------------------------------------------------------------------
def test_a_variant_that_declares_nothing_needs_nothing():
    """The default, and the reason every pre-H4 entry still works.

    ``variant_requirements`` reads two attributes that almost no entry
    sets. If the default were anything but "needs nothing", the harness
    would start fetching frames for ``production`` and skipping cases at
    the edges of every day — silently changing the baseline the whole
    phase is measured against.
    """
    assert variant_requirements(production_flow) == VariantRequirements(0, False)
    assert variant_requirements(persistence_flow) == VariantRequirements(0, False)
    for name in SERVICE_VARIANTS:
        assert variant_requirements(get_variant(name)) == (0, False)


def test_requirements_read_the_declared_attributes():
    def needy(prev_dbz, curr_dbz, rain_now_mm_h, *, pixel_km, **kwargs):
        return persistence_flow(prev_dbz, curr_dbz, rain_now_mm_h, pixel_km=pixel_km)

    setattr(needy, NEEDS_HISTORY_ATTR, 3)
    setattr(needy, NEEDS_FUTURE_ATTR, True)
    req = variant_requirements(needy)
    assert req == VariantRequirements(history=3, future=True)
    # A NamedTuple, so both spellings work for a caller.
    assert req.history == 3 and req[0] == 3


def test_requirements_reject_a_nonsense_declaration():
    """A bad declaration is a bug in the entry, caught before the run.

    Not a shrug-and-default: the harness sizes its frame window from
    this number, so a negative or non-integer one would either skip
    every case or index a list backwards, three hours into a worker.
    """
    def entry(prev_dbz, curr_dbz, rain_now_mm_h, *, pixel_km):
        return persistence_flow(prev_dbz, curr_dbz, rain_now_mm_h, pixel_km=pixel_km)

    setattr(entry, NEEDS_HISTORY_ATTR, -1)
    with pytest.raises(ValueError, match="must be >= 0"):
        variant_requirements(entry)

    setattr(entry, NEEDS_HISTORY_ATTR, "two")
    with pytest.raises(TypeError, match="must be an int"):
        variant_requirements(entry)


def test_the_two_h4_entries_that_need_frames_declare_it():
    assert variant_requirements(get_variant("median3")) == (2, False)
    assert variant_requirements(get_variant("oracle")) == (0, True)


def test_every_registered_variant_declares_something_readable():
    """No entry may carry a declaration the harness cannot act on.

    Cheap, and it is the one place a new candidate with a typo'd
    attribute (``needs_histroy = 2``) would be noticed — such an entry
    would read as "needs nothing", be handed no frames, and raise inside
    a worker on the first case.
    """
    for name in list_variants():
        req = variant_requirements(get_variant(name))
        assert isinstance(req.history, int) and req.history >= 0
        assert isinstance(req.future, bool)
