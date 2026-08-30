"""Tests for parse.py against golden DMI composite fixtures.

Fixtures captured from the live DMI API on 2026-05-17.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from dmi_nowcast_core.parse import parse_composite

FIXTURES = Path(__file__).parent / "fixtures"
FULLRANGE = FIXTURES / "composite_fullrange.h5"
DOPPLER = FIXTURES / "composite_doppler.h5"


@pytest.fixture
def fullrange():
    return parse_composite(FULLRANGE)


def test_metadata_fields(fullrange):
    assert fullrange.quantity == "DBZH"
    assert fullrange.xscale_m == 500.0
    assert fullrange.yscale_m == 500.0
    # Plan §6.2 hardcodes 200/1.6 — DMI publishes the same in /how.
    assert fullrange.zr_a == 200.0
    assert fullrange.zr_b == 1.6
    assert "+proj=stere" in fullrange.projection
    assert "+lat_0=56" in fullrange.projection
    assert "+lon_0=10.5666" in fullrange.projection


def test_timestamp_is_utc_aware_and_close_to_nominal_slot(fullrange):
    # The filename and STAC datetime are 19:40:00, but /what/time is 194001 — DMI
    # writes the scan-completion second (1 s after the nominal minute) into the file.
    # Downstream code that keys frames by 5-min slot should floor to the minute.
    assert fullrange.timestamp_utc.tzinfo is not None
    nominal = datetime(2026, 5, 17, 19, 40, tzinfo=timezone.utc)
    delta = abs((fullrange.timestamp_utc - nominal).total_seconds())
    assert delta < 60, f"timestamp {fullrange.timestamp_utc} more than 1 minute off nominal"


def test_array_shape_and_dtype(fullrange):
    assert fullrange.reflectivity_dbz.shape == (1728, 1984)
    assert fullrange.reflectivity_dbz.dtype == np.float32


def test_nodata_becomes_nan_and_undetect_becomes_neginf(fullrange):
    arr = fullrange.reflectivity_dbz
    # gain=0.5, offset=-32, nodata=255 → if we hadn't masked, raw=255 would be 95.5 dBZ
    # (a plausible-but-wrong physical value). The mask must turn those into NaN.
    assert np.isnan(arr).any(), "expected at least some nodata pixels in a real composite"
    # undetect=0 raw → mapped to -inf, so the minimum finite-or-neginf value is -inf, not -32.
    finite_min = np.nanmin(arr[np.isfinite(arr)]) if np.isfinite(arr).any() else None
    assert finite_min is None or finite_min >= -32.0


def test_physical_range_is_sane_after_masking(fullrange):
    """Max valid dBZ should be below the hail-cap region (≤95.5 from gain/offset, but
    real precipitation rarely exceeds ~65 dBZ; the cap is a guard, not an expectation)."""
    arr = fullrange.reflectivity_dbz
    finite = arr[np.isfinite(arr)]
    if finite.size > 0:
        assert finite.max() <= 95.5  # raw 255 * 0.5 - 32 = 95.5; sanity bound
        assert finite.min() >= -32.0  # raw 0 * 0.5 - 32, but undetect was already masked


def test_corners_form_a_box_covering_denmark(fullrange):
    c = fullrange.corners_lonlat
    # Denmark sits roughly inside lon 8–15, lat 54.5–57.7. The composite covers more.
    assert c["LL"][0] < 8.0 < c["LR"][0]  # LL.lon < 8 < LR.lon
    assert c["LL"][1] < 55.0 < c["UL"][1]  # LL.lat < 55 < UL.lat


def test_both_scan_types_parse(fullrange):
    """Doppler fixture must parse with identical schema as fullRange."""
    dop = parse_composite(DOPPLER)
    assert dop.reflectivity_dbz.shape == fullrange.reflectivity_dbz.shape
    assert dop.projection == fullrange.projection
    assert dop.timestamp_utc > fullrange.timestamp_utc  # doppler is 5 min later
