"""WP3 — the review bundle's frame plan and its two PNGs per stamp.

Fully synthetic: ``review_frames.write_fixture_composite`` writes real
DMI-dialect ODIM HDF5 files into a corpus-shaped ``YYYY/MM`` tree, so every
test runs the production path (``parse_composite`` → Z–R → block p90 →
quantise) rather than a stub, and the builder's ``--fixture`` mode uses the
same helper.

Covers:
- the frame window's past edge — ``window_min + pad_min + cadence_min``,
  hand-worked for an on-grid and an off-grid anchor — and dedup across
  overlapping events,
- fullRange-only by default; ``include_doppler`` interleaving tagged :x5,
- the grayscale PNG dequantising back within half a quantisation step
  (mirrors ``test_national_artifacts``'s round-trip),
- the ``grid`` block equalling ``national_artifacts._build_manifest``'s for
  the same geo and downsample factor,
- a missing composite landing in ``missing`` instead of raising,
- skip-if-exists on a second run (and a half-written pair re-rendered),
- nodata coming back as level 255 and NOT as 0 mm/h.
"""
from __future__ import annotations

import io
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dmi_nowcast_core.corpus import (
    SCAN_TYPE_DOPPLER,
    SCAN_TYPE_FULL_RANGE,
    archive_path_for,
)
from dmi_nowcast_core.geo import CompositeGeo
from dmi_nowcast_core.national import NationalProducts, observed_rain_grid
from dmi_nowcast_core.parse import parse_composite
from dmi_nowcast_core.review_schema import (
    DEFAULT_FRAME_PAD_MIN,
    DEFAULT_WINDOW_MIN,
    GRID_BLOCK_FIELDS,
)
from dmi_nowcast_core.transform import dbz_to_rain_rate
from dmi_nowcast_sidecar import national_artifacts as na
from dmi_nowcast_sidecar.national_artifacts import NODATA_LEVEL, QUANT_SPECS, dequantise
from dmi_nowcast_sidecar.review_frames import (
    DEFAULT_CADENCE_MIN,
    OBSERVED_QUANT_KEY,
    composite_filename,
    frame_plan,
    frames_manifest_block,
    grid_block,
    stamp_of,
    write_fixture_archive,
    write_fixture_composite,
    write_frames,
)

GRID_PX = 64
DOWNSAMPLE = 4
PRODUCT_PX = GRID_PX // DOWNSAMPLE   # 16×16 product grid

#: A hand-worked anchor ON the 10-min grid: 13:40 − (90 + 20 + 10) = 11:40.
ANCHOR_ON_GRID = datetime(2026, 6, 12, 13, 40, tzinfo=timezone.utc)
#: …and one OFF it, as a real ``sent_utc`` is: 13:43 − 120 = 11:43, whose
#: first grid stamp is 11:50 — still earlier than the decision edge minus
#: the pad (11:53), which is the guarantee the cadence term buys.
ANCHOR_OFF_GRID = datetime(2026, 6, 12, 13, 43, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Synthetic fields
# ---------------------------------------------------------------------------

def _dbz_field() -> np.ndarray:
    """A dBZ ramp across the grid: every quantisation level gets exercised,
    and the top of the ramp sits above the 53 dBZ hail cap."""
    ramp = np.linspace(5.0, 60.0, GRID_PX * GRID_PX, dtype=np.float64)
    return ramp.reshape(GRID_PX, GRID_PX)


def _nodata_mask() -> np.ndarray:
    """Native pixels the radar did not measure — two whole product blocks
    (rows 0:8, cols 0:8 at f = 4), so the block p90 has nothing to average
    with and the product pixel is genuinely unknown."""
    mask = np.zeros((GRID_PX, GRID_PX), dtype=bool)
    mask[0:8, 0:8] = True
    return mask


def _undetect_mask() -> np.ndarray:
    """Native pixels measured as below detection — the 'we looked, nothing
    fell' case, two product blocks immediately below the nodata ones."""
    mask = np.zeros((GRID_PX, GRID_PX), dtype=bool)
    mask[8:16, 0:8] = True
    return mask


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A corpus archive holding the frames of ``ANCHOR_ON_GRID``'s plan."""
    root = tmp_path / "corpus"
    plan = frame_plan([("ev", ANCHOR_ON_GRID)])
    write_fixture_archive(
        root, plan.stamp_strings(),
        dbz=_dbz_field(),
        nodata_mask=_nodata_mask(),
        undetect_mask=_undetect_mask(),
    )
    return root


def _expected_field(corpus_dir: Path, stamp: str) -> np.ndarray:
    """The mm/h product grid the bundle should have encoded for ``stamp``."""
    composite = parse_composite(
        archive_path_for(corpus_dir, f"dk.com.{stamp}.500_max.h5")
    )
    rain = dbz_to_rain_rate(
        composite.reflectivity_dbz, zr_a=composite.zr_a, zr_b=composite.zr_b,
    )
    return observed_rain_grid(rain, downsample_factor=DOWNSAMPLE)


def _levels(path: Path) -> np.ndarray:
    with Image.open(io.BytesIO(path.read_bytes())) as img:
        assert img.mode == "L"
        return np.array(img)


def _rgba(path: Path) -> np.ndarray:
    with Image.open(io.BytesIO(path.read_bytes())) as img:
        assert img.mode == "RGBA"
        return np.array(img)


# ---------------------------------------------------------------------------
# 1. The frame window and dedup
# ---------------------------------------------------------------------------

def test_past_edge_is_padded_by_window_pad_and_cadence() -> None:
    """The frame window is NOT ±90: a composite is 13–24 min old when a
    cycle stands on it, so the left edge reaches back one pad plus one
    cadence further than the decision edge."""
    plan = frame_plan([("ev", ANCHOR_ON_GRID)])
    assert plan.window_min == DEFAULT_WINDOW_MIN == 90
    assert plan.pad_min == DEFAULT_FRAME_PAD_MIN == 20
    assert plan.cadence_min == DEFAULT_CADENCE_MIN == 10

    event = plan.events[0]
    # 13:40 − (90 + 20 + 10) = 11:40; 13:40 + 90 = 15:10.
    assert event.frames_from_utc == datetime(2026, 6, 12, 11, 40, tzinfo=timezone.utc)
    assert event.frames_to_utc == datetime(2026, 6, 12, 15, 10, tzinfo=timezone.utc)
    assert event.stamps[0] == "202606121140"
    assert event.stamps[-1] == "202606121510"
    assert len(event.stamps) == 210 // 10 + 1

    # The DECISION edges stay separate values and are strictly narrower.
    assert event.window_from_utc == datetime(2026, 6, 12, 12, 10, tzinfo=timezone.utc)
    assert event.window_to_utc == event.frames_to_utc
    assert event.frames_from_utc < event.window_from_utc


def test_off_grid_anchor_still_covers_the_decision_edge() -> None:
    """The cadence term is what makes the guarantee hold for an anchor that
    is not on the 10-min grid — which every real ``sent_utc`` is not."""
    plan = frame_plan([("ev", ANCHOR_OFF_GRID)])
    event = plan.events[0]
    assert event.frames_from_utc == datetime(2026, 6, 12, 11, 43, tzinfo=timezone.utc)
    # First stamp on the grid at or after 11:43.
    assert event.stamps[0] == "202606121150"
    # …and still earlier than "decision edge minus the composite's age".
    earliest = datetime.strptime(event.stamps[0], "%Y%m%d%H%M").replace(
        tzinfo=timezone.utc,
    )
    assert earliest <= event.window_from_utc - timedelta(minutes=plan.pad_min)


def test_dedupes_across_overlapping_events() -> None:
    """Dedup is the bundle's whole economy: two anchors 30 min apart share
    all but three stamps, and the frames directory holds the union once."""
    anchors = [
        ("ev-a", ANCHOR_ON_GRID),
        ("ev-b", ANCHOR_ON_GRID + timedelta(minutes=30)),
    ]
    plan = frame_plan(anchors)
    a, b = plan.events
    assert plan.stamps_total == a.n_frames + b.n_frames == 44
    assert plan.stamps_unique == len(set(a.stamps) | set(b.stamps)) == 25
    assert plan.dedup_saving_pct == pytest.approx(100 * (1 - 25 / 44))
    # Chronological and unique, so the directory sorts into animation order.
    stamps = plan.stamp_strings()
    assert list(stamps) == sorted(stamps)
    assert len(set(stamps)) == len(stamps)
    # Each event keeps its own ordered scrubber list.
    assert a.stamps[0] == "202606121140" and b.stamps[0] == "202606121210"


def test_anchor_validation() -> None:
    naive = ANCHOR_ON_GRID.replace(tzinfo=None)
    with pytest.raises(ValueError, match="timezone-aware"):
        frame_plan([("ev", naive)])
    with pytest.raises(ValueError, match="duplicate event_id"):
        frame_plan([("ev", ANCHOR_ON_GRID), ("ev", ANCHOR_OFF_GRID)])
    # fullRange frames exist only on :x0 minutes, so a 7-min cadence would
    # plan stamps that cannot exist.
    with pytest.raises(ValueError, match="multiple of 10"):
        frame_plan([("ev", ANCHOR_ON_GRID)], cadence_min=7)


def test_empty_plan_is_empty_not_an_error() -> None:
    plan = frame_plan([])
    assert plan.stamps == () and plan.events == ()
    assert plan.stamps_total == 0 and plan.stamps_unique == 0
    assert plan.frames_from_utc is None and plan.frames_to_utc is None


# ---------------------------------------------------------------------------
# 2. fullRange by default, doppler only on request
# ---------------------------------------------------------------------------

def test_default_plan_is_fullrange_only() -> None:
    """The decision path ran on fullRange, so 10 min IS the decision
    cadence; a raw :x5 frame would put a coverage edge and an intensity
    step into the movie that reads as weather."""
    plan = frame_plan([("ev", ANCHOR_ON_GRID)])
    assert plan.include_doppler is False
    assert plan.products == (SCAN_TYPE_FULL_RANGE,)
    assert all(s.product == SCAN_TYPE_FULL_RANGE for s in plan.stamps)
    assert all(s.ts_utc.minute % 10 == 0 for s in plan.stamps)


def test_include_doppler_interleaves_tagged_x5_stamps() -> None:
    plan = frame_plan([("ev", ANCHOR_ON_GRID)], include_doppler=True)
    assert plan.products == (SCAN_TYPE_FULL_RANGE, SCAN_TYPE_DOPPLER)
    by_product = {s.stamp: s.product for s in plan.stamps}
    for stamp, product in by_product.items():
        minute = int(stamp[-2:]) % 10
        assert product == (SCAN_TYPE_FULL_RANGE if minute == 0 else SCAN_TYPE_DOPPLER)
    # Interleaved, not appended: :x0 and :x5 alternate through the window.
    assert [s.product for s in plan.stamps[:4]] == [
        SCAN_TYPE_FULL_RANGE, SCAN_TYPE_DOPPLER,
        SCAN_TYPE_FULL_RANGE, SCAN_TYPE_DOPPLER,
    ]
    # The fullRange skeleton is untouched by the opt-in.
    fullrange = [s.stamp for s in plan.stamps if s.product == SCAN_TYPE_FULL_RANGE]
    assert fullrange == list(frame_plan([("ev", ANCHOR_ON_GRID)]).stamp_strings())


def test_planned_filenames_are_archive_filenames() -> None:
    plan = frame_plan([("ev", ANCHOR_ON_GRID)])
    first = plan.stamps[0]
    assert first.filename == "dk.com.202606121140.500_max.h5"
    assert first.filename == composite_filename(first.ts_utc)
    assert first.overlay_name == "202606121140.overlay.png"
    assert first.observed_name == "202606121140.observed.png"


# ---------------------------------------------------------------------------
# 3. The grayscale PNG round-trips
# ---------------------------------------------------------------------------

def test_observed_png_dequantises_within_half_a_step(
    corpus: Path, tmp_path: Path,
) -> None:
    """Mirrors ``test_national_artifacts``'s round-trip: the browser reads
    mm/h under the cursor out of this PNG, so half a quantisation step is
    the error budget it is allowed to inherit."""
    plan = frame_plan([("ev", ANCHOR_ON_GRID)])
    result = write_frames(plan, corpus_dir=corpus, bundle_dir=tmp_path / "bundle")
    stamp = result.written[0]

    spec = QUANT_SPECS[OBSERVED_QUANT_KEY]
    entry = result.encodings["observed"]
    assert (entry["scale"], entry["offset"], entry["nodata"]) == (
        spec.scale, spec.offset, NODATA_LEVEL,
    )
    assert entry["units"] == "mm/h"

    levels = _levels(result.frames_dir / f"{stamp}.observed.png")
    decoded = dequantise(levels, scale=entry["scale"], offset=entry["offset"])
    expected = _expected_field(corpus, stamp)

    assert decoded.shape == expected.shape == (PRODUCT_PX, PRODUCT_PX)
    assert np.array_equal(np.isnan(decoded), np.isnan(expected))
    finite = ~np.isnan(expected)
    err = np.abs(decoded[finite] - expected[finite])
    assert float(err.max()) <= spec.scale / 2 + 1e-4


def test_overlay_is_rgba_on_the_same_product_grid(
    corpus: Path, tmp_path: Path,
) -> None:
    plan = frame_plan([("ev", ANCHOR_ON_GRID)])
    result = write_frames(plan, corpus_dir=corpus, bundle_dir=tmp_path / "bundle")
    rgba = _rgba(result.frames_dir / f"{result.written[0]}.overlay.png")
    assert rgba.shape == (PRODUCT_PX, PRODUCT_PX, 4)
    # The colormap's identity and alpha ramp travel with the bundle, so the
    # page never has to hardcode render.py's constants.
    overlay_meta = result.encodings["overlay"]
    assert overlay_meta["encoding"] == "rgba8"
    assert overlay_meta["floor_mm_h"] == pytest.approx(0.3)
    assert overlay_meta["solid_mm_h"] == pytest.approx(2.0)
    assert overlay_meta["colormap_stops"][0]["mm_h"] == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# 4. The grid block is A2's, field for field
# ---------------------------------------------------------------------------

def _national_grid_block(geo: CompositeGeo) -> dict:
    """What ``national_artifacts`` writes for the same geo and factor."""
    zeros = np.zeros((PRODUCT_PX, PRODUCT_PX), dtype=np.float32)
    products = NationalProducts(
        p_rain={10: zeros}, eta_min=zeros, intensity_mm_h=zeros,
        leads_min=(10,), threshold_mm_h=0.1, timestep_min=10.0,
        frame_age_min=4.0, downsample_factor=DOWNSAMPLE, n_members=8,
    )
    manifest = na._build_manifest(
        products=products,
        geo=geo,
        radar_utc=ANCHOR_ON_GRID,
        generated_utc=ANCHOR_ON_GRID,
        stamp=stamp_of(ANCHOR_ON_GRID),
        artifacts=[],
        overlay_shape=None,
    )
    return manifest["grid"]


def test_grid_block_matches_national_artifacts(corpus: Path, tmp_path: Path) -> None:
    """Load-bearing: the review page reuses ``map/warp.ts`` and
    ``nowcast/sampler.ts`` unchanged, and they parse this exact block."""
    plan = frame_plan([("ev", ANCHOR_ON_GRID)])
    result = write_frames(plan, corpus_dir=corpus, bundle_dir=tmp_path / "bundle")
    geo = CompositeGeo(parse_composite(
        archive_path_for(corpus, f"dk.com.{result.written[0]}.500_max.h5")
    ))

    assert tuple(result.grid) == GRID_BLOCK_FIELDS
    assert result.grid == _national_grid_block(geo)
    # Stride slicing keeps the native UL corner and scales the pixel size.
    x_ul, y_ul = geo.projection_origin_m
    assert result.grid["x_ul_m"] == x_ul and result.grid["y_ul_m"] == y_ul
    assert result.grid["pixel_scale_x_m"] == geo.composite.xscale_m * DOWNSAMPLE
    assert result.grid["shape"] == [PRODUCT_PX, PRODUCT_PX]

    # And the standalone helper agrees with what the run published.
    assert grid_block(
        geo, downsample_factor=DOWNSAMPLE, shape=(PRODUCT_PX, PRODUCT_PX),
    ) == result.grid


def test_grid_block_survives_an_all_skipped_rerun(
    corpus: Path, tmp_path: Path,
) -> None:
    """A resumed run renders nothing but still has to publish the geometry."""
    plan = frame_plan([("ev", ANCHOR_ON_GRID)])
    bundle = tmp_path / "bundle"
    first = write_frames(plan, corpus_dir=corpus, bundle_dir=bundle)
    second = write_frames(plan, corpus_dir=corpus, bundle_dir=bundle)
    assert second.written == ()
    assert second.grid == first.grid


# ---------------------------------------------------------------------------
# 5. Gaps in the archive are data, not a crash
# ---------------------------------------------------------------------------

def test_missing_composite_is_recorded_and_does_not_raise(
    corpus: Path, tmp_path: Path,
) -> None:
    """A 7 GB archive with a few gaps must not lose an eight-hour run."""
    plan = frame_plan([("ev", ANCHOR_ON_GRID)])
    gap = plan.stamps[3]
    archive_path_for(corpus, gap.filename).unlink()
    broken = plan.stamps[4]
    archive_path_for(corpus, broken.filename).write_bytes(b"not an HDF5 file")

    result = write_frames(plan, corpus_dir=corpus, bundle_dir=tmp_path / "bundle")

    assert set(result.missing) == {gap.stamp, broken.stamp}
    assert result.missing_reasons[gap.stamp] == "absent from the corpus archive"
    assert result.missing_reasons[broken.stamp].startswith("unreadable")
    assert len(result.written) == len(plan.stamps) - 2
    assert not (result.frames_dir / f"{gap.stamp}.overlay.png").exists()
    # The manifest block carries the gaps so the UI can badge them.
    block = frames_manifest_block(plan, result)
    assert set(block["missing"]) == {gap.stamp, broken.stamp}
    assert block["count"] == len(plan.stamps) - 2


def test_frames_manifest_block_keeps_the_two_edges_apart(
    corpus: Path, tmp_path: Path,
) -> None:
    plan = frame_plan([("ev", ANCHOR_ON_GRID)])
    result = write_frames(plan, corpus_dir=corpus, bundle_dir=tmp_path / "bundle")
    block = frames_manifest_block(plan, result)
    assert block["from_utc"] == "2026-06-12T11:40:00+00:00"
    assert block["to_utc"] == "2026-06-12T15:10:00+00:00"
    assert block["window_min"] == 90 and block["frame_pad_min"] == 20
    assert block["products"] == [SCAN_TYPE_FULL_RANGE]
    assert block["bytes"] == result.bytes_total > 0
    assert "grid" not in block          # top-level manifest key, not ours


# ---------------------------------------------------------------------------
# 6. Resumability
# ---------------------------------------------------------------------------

def test_second_run_writes_no_bytes_and_reports_skips(
    corpus: Path, tmp_path: Path,
) -> None:
    plan = frame_plan([("ev", ANCHOR_ON_GRID)])
    bundle = tmp_path / "bundle"
    first = write_frames(plan, corpus_dir=corpus, bundle_dir=bundle)
    assert first.skipped == () and len(first.written) == len(plan.stamps)
    before = {
        p.name: (p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(first.frames_dir.iterdir())
    }

    second = write_frames(plan, corpus_dir=corpus, bundle_dir=bundle)

    assert second.written == ()
    assert set(second.skipped) == set(plan.stamp_strings())
    assert second.bytes_written == 0
    # The budget number still reports the whole bundle, not this run.
    assert second.bytes_total == first.bytes_total > 0
    after = {
        p.name: (p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(second.frames_dir.iterdir())
    }
    assert after == before             # nothing was rewritten
    assert not list(first.frames_dir.glob("*.tmp"))


def test_half_written_pair_is_rendered_again(corpus: Path, tmp_path: Path) -> None:
    """Killed between the two writes, a stamp is incomplete — not done."""
    plan = frame_plan([("ev", ANCHOR_ON_GRID)])
    bundle = tmp_path / "bundle"
    first = write_frames(plan, corpus_dir=corpus, bundle_dir=bundle)
    orphan = first.written[0]
    (first.frames_dir / f"{orphan}.observed.png").unlink()

    second = write_frames(plan, corpus_dir=corpus, bundle_dir=bundle)

    assert second.written == (orphan,)
    assert (first.frames_dir / f"{orphan}.observed.png").is_file()
    assert second.bytes_written > 0


# ---------------------------------------------------------------------------
# 7. nodata is not zero rain
# ---------------------------------------------------------------------------

def test_nodata_block_is_level_255_not_zero_mm_h(
    corpus: Path, tmp_path: Path,
) -> None:
    """A nodata pixel read as 0 mm/h says "we measured no rain here" when
    the truth is "we did not measure here" — exactly the confusion this
    whole tool exists to expose. It must survive to the browser as null.
    """
    plan = frame_plan([("ev", ANCHOR_ON_GRID)])
    result = write_frames(plan, corpus_dir=corpus, bundle_dir=tmp_path / "bundle")
    stamp = result.written[0]
    levels = _levels(result.frames_dir / f"{stamp}.observed.png")
    entry = result.encodings["observed"]
    decoded = dequantise(levels, scale=entry["scale"], offset=entry["offset"])

    # Native rows/cols 0:8 are nodata → product blocks [0:2, 0:2].
    assert np.all(levels[0:2, 0:2] == NODATA_LEVEL)
    assert np.all(np.isnan(decoded[0:2, 0:2]))
    assert not np.any(decoded[0:2, 0:2] == 0.0)

    # Native rows 8:16, cols 0:8 are undetect — measured, nothing fell →
    # a real 0 mm/h at level 0, which is a DIFFERENT statement.
    assert np.all(levels[2:4, 0:2] == 0)
    assert np.all(decoded[2:4, 0:2] == 0.0)

    # Both are transparent in the overlay, which is why the grayscale grid
    # (not the colours) is what the read-out samples.
    rgba = _rgba(result.frames_dir / f"{stamp}.overlay.png")
    assert np.all(rgba[0:2, 0:2, 3] == 0)
    assert np.all(rgba[2:4, 0:2, 3] == 0)

    # A partly-covered block keeps the pixels it has: the whole grid is not
    # nodata just because two blocks are.
    assert np.count_nonzero(levels == NODATA_LEVEL) == 4


def test_scaling_is_read_from_the_file_not_hardcoded(tmp_path: Path) -> None:
    """The same dBZ under different HDF5 scaling must render the same mm/h.

    ``write_fixture_composite`` encodes with gain 0.5 / offset −32; this
    test re-encodes the identical dBZ values through a composite whose
    Z–R coefficients differ, and asserts the rendered field follows the
    FILE's ``zr-a``/``zr-b`` rather than the plan's 200/1.6 defaults.
    """
    ts = ANCHOR_ON_GRID
    path = write_fixture_composite(tmp_path / "c.h5", ts, dbz=_dbz_field())
    import h5py

    with h5py.File(path, "r+") as h5:
        h5["/how"].attrs["zr-a"] = 300.0

    composite = parse_composite(path)
    assert composite.zr_a == 300.0
    expected = observed_rain_grid(
        dbz_to_rain_rate(
            composite.reflectivity_dbz, zr_a=300.0, zr_b=composite.zr_b,
        ),
        downsample_factor=DOWNSAMPLE,
    )
    default_zr = observed_rain_grid(
        dbz_to_rain_rate(composite.reflectivity_dbz),
        downsample_factor=DOWNSAMPLE,
    )
    assert not np.allclose(expected, default_zr)   # the file's value matters

    corpus_dir = tmp_path / "corpus"
    target = archive_path_for(corpus_dir, composite_filename(ts))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(path.read_bytes())
    plan = frame_plan([("ev", ts)], window_min=0, pad_min=0)
    result = write_frames(
        plan, corpus_dir=corpus_dir, bundle_dir=tmp_path / "bundle",
    )
    entry = result.encodings["observed"]
    levels = _levels(result.frames_dir / f"{stamp_of(ts)}.observed.png")
    decoded = dequantise(levels, scale=entry["scale"], offset=entry["offset"])
    finite = ~np.isnan(expected)
    assert np.all(
        np.abs(decoded[finite] - expected[finite]) <= entry["scale"] / 2 + 1e-4
    )
