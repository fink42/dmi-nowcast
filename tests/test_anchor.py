"""The anchor policy: which frame a cycle stands on (Phase H, H-L / L3).

The arithmetic that matters is small and easy to get subtly wrong, so it
is pinned here rather than discovered on a seven-hour replay:

* a frame is invisible until it is **published** — ``:x0 + 13.1`` and
  ``:x5 + 8.1``, which are the same wall instant, so the freshest policy
  buys five minutes of age and *not* a faster decision cadence;
* the doppler frame is never used raw, and never outside its coverage;
* the flow and the cascade see one product at 10-minute spacing, always,
  falling all the way back to the fullRange policy when they cannot.

The synthetic-composite writer is imported from
``test_gauge_agreement_study`` rather than copied — one description of
DMI's HDF5 layout, in one place.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from dmi_nowcast_core import anchor as an
from dmi_nowcast_core import product_pairs as pp
from dmi_nowcast_core.corpus import (
    SCAN_TYPE_DOPPLER,
    SCAN_TYPE_FULL_RANGE,
    ArchiveIndex,
)
from dmi_nowcast_core.parse import parse_composite

from .test_gauge_agreement_study import archive_frame, write_composite

#: June — the "summer" stratum, and the season the fixture table is keyed
#: under, so the pooled fallback is not what is being exercised.
DAY = datetime(2026, 6, 1, tzinfo=timezone.utc)
SHAPE = (32, 32)
#: doppler's 120 km edge, in miniature: everything outside this disc is
#: nodata in the :x5 product and has to come from the :x0 one.
DISC_RADIUS_PX = 10.0


# ---------------------------------------------------------------------------
# Fixtures: fields, a harmonisation map, an archive of both products
# ---------------------------------------------------------------------------


def coverage_disc(shape: tuple[int, int] = SHAPE) -> np.ndarray:
    rows, cols = np.indices(shape)
    centre = ((shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0)
    return (
        (rows - centre[0]) ** 2 + (cols - centre[1]) ** 2
    ) <= DISC_RADIUS_PX ** 2


def fullrange_field(value: float = 25.0) -> np.ndarray:
    return np.full(SHAPE, value, dtype=np.float32)


def doppler_field(value: float = 20.0) -> np.ndarray:
    """``value`` inside the disc, ``nodata`` outside — doppler's geometry."""
    out = np.full(SHAPE, np.nan, dtype=np.float32)
    out[coverage_disc()] = value
    return out


def harmonisation_payload(shift_db: float = 3.0) -> dict:
    """A table that adds ``shift_db`` to every echo bin, in every band.

    The real one is fitted per band and season; a constant shift is what
    makes the assertions readable, and ``test_doppler_harmonisation.py``
    already pins the fit that produces one.
    """
    edges = pp.mapped_edges()
    entry = {
        "mapped_dbz": [float(e + shift_db) for e in edges],
        "n_doppler_px": 1_000_000,
        "n_fullrange_px": 1_000_000,
    }
    return {
        "schema_version": pp.HARMONISATION_SCHEMA_VERSION,
        "meta": {
            "generated": "2026-09-08T18:44:53+00:00",
            "n_fit_triples": 1417,
        },
        "bin_edges_dbz": [float(e) for e in edges],
        "tables": {
            pp.table_key(season, band): dict(entry)
            for season in pp.SEASON_ORDER
            for band in pp.DOPPLER_BANDS
        },
    }


@pytest.fixture
def harmonisation(tmp_path: Path) -> Path:
    path = tmp_path / "doppler_harmonisation.json"
    path.write_text(json.dumps(harmonisation_payload()))
    return path


def build_archive(
    root: Path,
    *,
    start: datetime = DAY + timedelta(hours=5),
    slots: int = 25,
    doppler: bool = True,
    skip: tuple[datetime, ...] = (),
) -> Path:
    """Both products on the 5-minute grid: ``:x0`` fullRange, ``:x5`` doppler."""
    for i in range(slots):
        ts = start + timedelta(minutes=5 * i)
        if ts in skip:
            continue
        is_full = ts.minute % 10 == 0
        if not is_full and not doppler:
            continue
        archive_frame(
            root, fullrange_field() if is_full else doppler_field(), ts,
        )
    return root


def frame_map(root: Path) -> dict:
    index = ArchiveIndex(root)
    return pp.frame_map(
        index.list_in_window(DAY, DAY + timedelta(days=1)),
    )


@pytest.fixture
def frames(tmp_path: Path) -> dict:
    return frame_map(build_archive(tmp_path / "corpus"))


# ---------------------------------------------------------------------------
# The archive labels the two products by minute
# ---------------------------------------------------------------------------


def test_the_archive_holds_both_products_on_the_five_minute_grid(
    frames: dict,
) -> None:
    assert set(frames) == {SCAN_TYPE_FULL_RANGE, SCAN_TYPE_DOPPLER}
    assert all(ts.minute % 10 == 0 for ts in frames[SCAN_TYPE_FULL_RANGE])
    assert all(ts.minute % 10 == 5 for ts in frames[SCAN_TYPE_DOPPLER])
    assert len(frames[SCAN_TYPE_FULL_RANGE]) == 13
    assert len(frames[SCAN_TYPE_DOPPLER]) == 12


# ---------------------------------------------------------------------------
# select_anchor: publication, freshness, staleness
# ---------------------------------------------------------------------------


def test_a_frame_is_invisible_until_its_lag_has_passed(frames: dict) -> None:
    """13.1 and 8.1 minutes are the same wall instant: :x0 + 13.1 == :x5 + 8.1."""
    just_before = DAY + timedelta(hours=6, minutes=13)
    just_after = DAY + timedelta(hours=6, minutes=14)

    early = an.select_anchor(just_before, frames, policy=an.POLICY_FRESHEST)
    assert early.scan_type == SCAN_TYPE_DOPPLER
    assert early.timestamp == DAY + timedelta(hours=5, minutes=55)
    assert early.fullrange_ts == DAY + timedelta(hours=5, minutes=50)

    late = an.select_anchor(just_after, frames, policy=an.POLICY_FRESHEST)
    assert late.timestamp == DAY + timedelta(hours=6, minutes=5)
    assert late.fullrange_ts == DAY + timedelta(hours=6, minutes=0)
    # The publication event carries both frames, so both move together.
    assert late.frame_age_min == pytest.approx(9.0)


def test_the_two_policies_differ_by_exactly_five_minutes(frames: dict) -> None:
    now = DAY + timedelta(hours=6, minutes=15)
    base = an.select_anchor(now, frames, policy=an.POLICY_FULLRANGE)
    fresh = an.select_anchor(now, frames, policy=an.POLICY_FRESHEST)
    assert base.scan_type == SCAN_TYPE_FULL_RANGE
    assert base.frame_age_min == pytest.approx(15.0)
    assert fresh.scan_type == SCAN_TYPE_DOPPLER
    assert fresh.frame_age_min == pytest.approx(10.0)
    assert base.frame_age_min - fresh.frame_age_min == pytest.approx(5.0)
    # The fullRange policy never even looks at the doppler product.
    assert base.doppler_ts is None
    assert fresh.doppler_ts == fresh.timestamp


def test_the_fullrange_frame_wins_when_doppler_lags_further_behind(
    frames: dict,
) -> None:
    """"Once its lag has passed" is a real condition, not a formality."""
    slow = an.ProductLag(fullrange_min=13.1, doppler_min=20.0)
    got = an.select_anchor(
        DAY + timedelta(hours=6, minutes=14), frames,
        policy=an.POLICY_FRESHEST, lag=slow,
    )
    assert got.scan_type == SCAN_TYPE_FULL_RANGE
    assert got.timestamp == DAY + timedelta(hours=6)
    assert got.degraded is True
    assert "fresher of the two" in got.reason


def test_a_missing_doppler_frame_falls_back_to_fullRange(tmp_path: Path) -> None:
    gap = DAY + timedelta(hours=6, minutes=5)
    frames = frame_map(build_archive(tmp_path / "corpus", skip=(gap,)))
    got = an.select_anchor(
        DAY + timedelta(hours=6, minutes=14), frames, policy=an.POLICY_FRESHEST,
    )
    assert got.scan_type == SCAN_TYPE_FULL_RANGE
    assert got.timestamp == DAY + timedelta(hours=6)
    assert got.degraded is True
    assert got.frame_age_min == pytest.approx(14.0)


def test_september_has_no_doppler_at_all_and_degrades_cleanly(
    tmp_path: Path,
) -> None:
    """The 2026-09 archive: fullRange only. The candidate becomes the baseline."""
    frames = frame_map(build_archive(tmp_path / "corpus", doppler=False))
    assert frames.get(SCAN_TYPE_DOPPLER) in (None, {})
    now = DAY + timedelta(hours=6, minutes=15)
    fresh = an.select_anchor(now, frames, policy=an.POLICY_FRESHEST)
    base = an.select_anchor(now, frames, policy=an.POLICY_FULLRANGE)
    assert fresh.timestamp == base.timestamp
    assert fresh.frame_age_min == base.frame_age_min == pytest.approx(15.0)
    assert fresh.degraded is True
    assert fresh.reason == "no doppler frame published"
    assert base.degraded is False

    counts = an.counts_template()
    an.count_selection(counts, an.anchor_history(fresh, frames))
    assert counts["degraded"] == 1
    assert counts["fullrange_anchored"] == 1
    assert counts["doppler_anchored"] == 0
    assert counts["history_fallback"] == 0


def test_a_stale_frame_is_not_an_anchor(frames: dict) -> None:
    """A gap in the archive must stop the run, not anchor it on yesterday."""
    long_after = DAY + timedelta(hours=9)
    assert an.select_anchor(long_after, frames, policy=an.POLICY_FRESHEST) is None
    assert an.select_anchor(
        long_after, frames, policy=an.POLICY_FRESHEST, max_age_min=300.0,
    ) is not None


def test_nothing_published_yet_is_not_an_anchor(frames: dict) -> None:
    assert an.select_anchor(DAY, frames, policy=an.POLICY_FULLRANGE) is None


def test_an_unknown_policy_is_refused(frames: dict) -> None:
    with pytest.raises(ValueError, match="unknown anchor policy"):
        an.select_anchor(DAY, frames, policy="doppler-only")


# ---------------------------------------------------------------------------
# decision_instants: one cycle per new frame, not one per poll
# ---------------------------------------------------------------------------


def test_both_policies_decide_at_the_same_instants_ten_minutes_apart(
    frames: dict,
) -> None:
    """The freshness is free; the decision rate does not change.

    A poll that finds the anchor it already had is not a cycle (the
    runtime's no-new-frame fast path), which is also what keeps
    ``radar_ts`` unique — every consumer deduplicates decision rows on it.
    """
    window = (DAY + timedelta(hours=6), DAY + timedelta(hours=7))
    base = an.decision_instants(*window, frames, policy=an.POLICY_FULLRANGE)
    fresh = an.decision_instants(*window, frames, policy=an.POLICY_FRESHEST)

    assert [s.now for s in base] == [s.now for s in fresh]
    # The first poll of a window always yields a cycle: there is no
    # previous anchor to compare it against, so it stands on whatever was
    # already published. In a day plan that instant anchors on the
    # previous day's last frame and is filtered out by the day it belongs
    # to; here it is the warm-up, and the steady state follows it.
    assert [s.now.minute for s in base] == [0, 5, 15, 25, 35, 45, 55]
    assert base[0].timestamp == DAY + timedelta(hours=5, minutes=40)
    assert fresh[0].timestamp == DAY + timedelta(hours=5, minutes=45)

    assert all(s.frame_age_min == pytest.approx(15.0) for s in base[1:])
    assert all(s.frame_age_min == pytest.approx(10.0) for s in fresh[1:])
    assert all(s.scan_type == SCAN_TYPE_FULL_RANGE for s in base)
    assert all(s.scan_type == SCAN_TYPE_DOPPLER for s in fresh)
    # Every cycle stands on a frame no earlier cycle stood on, and the
    # anchors advance ten minutes at a time under both policies.
    assert len({s.key for s in fresh}) == len(fresh)
    for series in (base, fresh):
        gaps = {
            (b.timestamp - a.timestamp).total_seconds() / 60.0
            for a, b in zip(series[1:], series[2:])
        }
        assert gaps == {10.0}


def test_the_flat_override_reproduces_a_pre_l3_run_to_the_minute(
    frames: dict,
) -> None:
    flat = an.ProductLag(flat=14.0)
    got = an.decision_instants(
        DAY + timedelta(hours=6), DAY + timedelta(hours=7), frames,
        policy=an.POLICY_FULLRANGE, lag=flat, poll_interval_min=0.0,
    )
    assert [s.timestamp.minute for s in got] == [50, 0, 10, 20, 30, 40]
    assert all(s.frame_age_min == pytest.approx(14.0) for s in got)
    assert all(
        s.now == s.timestamp + timedelta(minutes=14) for s in got
    )


# ---------------------------------------------------------------------------
# anchor_field: harmonise inside coverage, fill outside
# ---------------------------------------------------------------------------


def test_a_fullrange_anchor_is_returned_untouched() -> None:
    field = fullrange_field()
    got = an.anchor_field(
        field, scan_type=SCAN_TYPE_FULL_RANGE, anchor_ts=DAY,
    )
    assert np.array_equal(got.dbz, field)
    assert got.dbz is not field                 # a copy, never the caller's
    assert not got.doppler_mask.any()
    assert got.fill_ts is None


def test_a_doppler_anchor_is_harmonised_inside_and_filled_outside() -> None:
    doppler = doppler_field(20.0)
    fill = fullrange_field(25.0)
    got = an.anchor_field(
        doppler, scan_type=SCAN_TYPE_DOPPLER, anchor_ts=DAY,
        fill_dbz=fill, fill_ts=DAY - timedelta(minutes=5),
        harmonisation=harmonisation_payload(3.0),
        distance_km=np.full(SHAPE, 30.0, dtype=np.float32),
    )
    disc = coverage_disc()
    assert np.array_equal(got.doppler_mask, disc)
    # Inside: the L2 map, applied. Raw doppler would read 20 dBZ here and
    # carry 20-30 % less echo area than fullRange at that level.
    assert np.allclose(got.dbz[disc], 23.0, atol=0.01)
    # Outside: the older, wider frame, unchanged.
    assert np.allclose(got.dbz[~disc], 25.0)
    assert got.fill_ts == DAY - timedelta(minutes=5)
    assert np.isnan(doppler[~disc]).all()       # the input is not modified


def test_an_observed_dry_doppler_pixel_is_coverage_not_a_hole() -> None:
    """``undetect`` is data. Filling it would swap fresh dry for stale dry."""
    doppler = doppler_field(20.0)
    disc = coverage_disc()
    rows, cols = np.nonzero(disc)
    doppler[rows[0], cols[0]] = -np.inf
    got = an.anchor_field(
        doppler, scan_type=SCAN_TYPE_DOPPLER, anchor_ts=DAY,
        fill_dbz=fullrange_field(25.0),
        harmonisation=harmonisation_payload(),
        distance_km=np.full(SHAPE, 30.0, dtype=np.float32),
    )
    assert got.doppler_mask[rows[0], cols[0]]
    assert got.dbz[rows[0], cols[0]] == -np.inf


def test_without_a_fill_frame_the_uncovered_pixels_stay_nodata() -> None:
    got = an.anchor_field(
        doppler_field(20.0), scan_type=SCAN_TYPE_DOPPLER, anchor_ts=DAY,
        harmonisation=harmonisation_payload(),
        distance_km=np.full(SHAPE, 30.0, dtype=np.float32),
    )
    assert np.isnan(got.dbz[~coverage_disc()]).all()
    assert np.allclose(got.dbz[coverage_disc()], 23.0, atol=0.01)


def test_a_doppler_anchor_without_the_map_is_refused() -> None:
    with pytest.raises(ValueError, match="harmonisation"):
        an.anchor_field(
            doppler_field(), scan_type=SCAN_TYPE_DOPPLER, anchor_ts=DAY,
        )


def test_a_mismatched_fill_frame_is_refused() -> None:
    with pytest.raises(ValueError, match="does not match"):
        an.anchor_field(
            doppler_field(), scan_type=SCAN_TYPE_DOPPLER, anchor_ts=DAY,
            fill_dbz=np.zeros((4, 4), dtype=np.float32),
            harmonisation=harmonisation_payload(),
            distance_km=np.full(SHAPE, 30.0, dtype=np.float32),
        )


def test_fill_uncovered_leaves_a_fullrange_field_alone() -> None:
    """Its mask is all-False; composing against a fill would erase the frame."""
    field = an.anchor_field(
        fullrange_field(), scan_type=SCAN_TYPE_FULL_RANGE, anchor_ts=DAY,
    )
    got = an.fill_uncovered(field, np.zeros(SHAPE, dtype=np.float32), DAY)
    assert got is field


def test_the_season_comes_from_the_frame_not_the_wall_clock() -> None:
    """A December frame replayed in June must use December's table."""
    winter = harmonisation_payload(0.0)
    winter["tables"][pp.table_key("winter", "0-60km")]["mapped_dbz"] = [
        float(e + 7.0) for e in pp.mapped_edges()
    ]
    got = an.anchor_field(
        doppler_field(20.0), scan_type=SCAN_TYPE_DOPPLER,
        anchor_ts=datetime(2026, 1, 12, tzinfo=timezone.utc),
        harmonisation=winter,
        distance_km=np.full(SHAPE, 30.0, dtype=np.float32),
    )
    assert np.allclose(got.dbz[coverage_disc()], 27.0, atol=0.01)


# ---------------------------------------------------------------------------
# anchor_history: same product, 10 minutes apart, or fall all the way back
# ---------------------------------------------------------------------------


def test_a_doppler_anchor_is_backed_by_a_doppler_triple(frames: dict) -> None:
    selection = an.select_anchor(
        DAY + timedelta(hours=6, minutes=15), frames,
        policy=an.POLICY_FRESHEST,
    )
    history = an.anchor_history(selection, frames)
    assert history.scan_type == SCAN_TYPE_DOPPLER
    assert [ts.strftime("%H:%M") for ts in history.timestamps] == [
        "05:45", "05:55", "06:05",
    ]
    assert history.step_min == 10.0
    assert history.fallback is False
    assert history.selection is selection
    # Each doppler frame's fill partner is the :x0 frame 5 min before it.
    assert [ts.strftime("%H:%M") for ts in history.fill_timestamps] == [
        "05:40", "05:50", "06:00",
    ]


def test_a_fullrange_anchor_is_backed_by_a_fullrange_triple(frames: dict) -> None:
    selection = an.select_anchor(
        DAY + timedelta(hours=6, minutes=15), frames,
        policy=an.POLICY_FULLRANGE,
    )
    history = an.anchor_history(selection, frames)
    assert history.scan_type == SCAN_TYPE_FULL_RANGE
    assert [ts.strftime("%H:%M") for ts in history.timestamps] == [
        "05:40", "05:50", "06:00",
    ]
    assert history.fill_timestamps == (None, None, None)


def test_a_broken_doppler_triple_falls_back_to_the_fullrange_policy(
    tmp_path: Path,
) -> None:
    """Not to a mixed triple: the products never interleave along time."""
    hole = DAY + timedelta(hours=5, minutes=45)
    frames = frame_map(build_archive(tmp_path / "corpus", skip=(hole,)))
    now = DAY + timedelta(hours=6, minutes=15)
    selection = an.select_anchor(now, frames, policy=an.POLICY_FRESHEST)
    assert selection.scan_type == SCAN_TYPE_DOPPLER   # the anchor is still fresh

    history = an.anchor_history(selection, frames)
    assert history.fallback is True
    assert history.scan_type == SCAN_TYPE_FULL_RANGE
    assert history.selection.scan_type == SCAN_TYPE_FULL_RANGE
    assert history.selection.timestamp == DAY + timedelta(hours=6)
    # Five minutes older than the doppler anchor would have been, and the
    # age is recomputed rather than carried over.
    assert history.selection.frame_age_min == pytest.approx(15.0)
    assert "fell back" in history.reason

    counts = an.counts_template()
    an.count_selection(counts, history)
    assert counts["history_fallback"] == 1
    assert counts["fullrange_anchored"] == 1
    assert counts["degraded"] == 0        # counted as a fallback, not twice


def test_no_triple_at_all_is_no_history(tmp_path: Path) -> None:
    root = build_archive(
        tmp_path / "corpus", start=DAY + timedelta(hours=6), slots=3,
    )
    frames = frame_map(root)
    selection = an.select_anchor(
        DAY + timedelta(hours=6, minutes=15), frames,
        policy=an.POLICY_FRESHEST,
    )
    assert an.anchor_history(selection, frames) is None


# ---------------------------------------------------------------------------
# Loading the map, and the run's provenance
# ---------------------------------------------------------------------------


def test_the_map_is_read_once_per_process(harmonisation: Path) -> None:
    first = an.load_harmonisation(harmonisation)
    assert an.load_harmonisation(harmonisation) is first


def test_a_table_of_another_schema_version_is_refused(tmp_path: Path) -> None:
    payload = harmonisation_payload()
    payload["schema_version"] = pp.HARMONISATION_SCHEMA_VERSION + 1
    path = tmp_path / "future.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="schema version"):
        an.load_harmonisation(path)


def test_an_empty_table_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({
        "schema_version": pp.HARMONISATION_SCHEMA_VERSION, "tables": {},
    }))
    with pytest.raises(ValueError, match="empty"):
        an.load_harmonisation(path)


def test_the_stamp_identifies_the_map_a_run_applied(harmonisation: Path) -> None:
    stamp = an.harmonisation_stamp(harmonisation)
    assert stamp["schema_version"] == pp.HARMONISATION_SCHEMA_VERSION
    assert stamp["fitted_at"] == "2026-09-08T18:44:53+00:00"
    assert stamp["n_fit_triples"] == 1417
    assert len(stamp["sha256"]) == 16
    assert "summer|0-60km" in stamp["tables"]
    assert an.harmonisation_stamp(None) == {"path": None}
    missing = an.harmonisation_stamp(harmonisation.parent / "nope.json")
    assert "error" in missing


def test_frame_age_stats_describe_the_freshness_a_run_got() -> None:
    assert an.frame_age_stats([])["n"] == 0
    got = an.frame_age_stats([10.0, 10.0, 15.0])
    assert got["p50"] == 10.0
    assert got["mean"] == pytest.approx(11.67, abs=0.01)
    assert (got["min"], got["max"]) == (10.0, 15.0)


def test_counts_sum_across_day_workers() -> None:
    total = an.counts_template()
    an.sum_counts(total, {"instants": 3, "doppler_anchored": 3})
    an.sum_counts(total, {"instants": 2, "fullrange_anchored": 2})
    an.sum_counts(total, None)
    assert total["instants"] == 5
    assert total["doppler_anchored"] == 3
    assert total["fullrange_anchored"] == 2


# ---------------------------------------------------------------------------
# The distance grid: built once, not per frame
# ---------------------------------------------------------------------------


def test_the_distance_grid_is_memoised_per_geometry(tmp_path: Path) -> None:
    """~3.4 M inverse projections on the national grid — once per worker."""
    path = write_composite(tmp_path / "f.h5", fullrange_field(), DAY)
    composite = parse_composite(path)
    grid = an.distance_km_grid(composite)
    assert grid.shape == SHAPE
    assert an.distance_km_grid(parse_composite(path)) is grid
    assert np.allclose(grid, pp.radar_distance_grid(composite), atol=1e-3)
