"""Tests for the bundled lightning-probability calibrator (prob_calibration)."""
from __future__ import annotations

from dmi_nowcast_sidecar import prob_calibration as pc


def test_region_of_boxes():
    assert pc.region_of(55.35, 10.35) == "Denmark"   # Odense
    assert pc.region_of(45.83, 6.87) == "Alps"        # Mont Blanc
    assert pc.region_of(0.0, 0.0) == "Other"


def test_calibrate_in_range_and_region_specific():
    for raw in (0.0, 0.3, 0.6, 0.9, 1.0):
        v = pc.calibrate("Denmark", 10.0, raw)
        assert 0.0 <= v <= 1.0
    # Denmark and Alps curves differ at the same raw input.
    assert pc.calibrate("Denmark", 10.0, 0.6) != pc.calibrate("Alps", 10.0, 0.6)


def test_ring_leads_from_bundle():
    # Single source of truth for the endpoint's (ring, lead) pairs.
    assert pc.ring_leads() == [(3.0, 15.0), (10.0, 30.0)]


def test_calibrate_falls_back_to_pooled_then_raw():
    # An unknown region uses the pooled curve (≠ raw passthrough here).
    assert pc.calibrate("Other", 10.0, 0.6) != 0.6
    # A ring with no bundled curve passes the raw value straight through.
    assert pc.calibrate("Denmark", 99.0, 0.42) == 0.42


def test_corrupt_bundle_degrades_to_raw(tmp_path, monkeypatch):
    bad = tmp_path / "broken.json"
    bad.write_text("{ this is not valid json")
    monkeypatch.setattr(pc, "_BUNDLE", bad)
    # Both loaders are lru_cached — clearing only _curves would let a
    # _bundle() result warmed by an earlier test bypass the monkeypatch.
    pc._bundle.cache_clear()
    pc._curves.cache_clear()
    try:
        assert pc._curves() == {}
        assert pc.calibrate("Denmark", 10.0, 0.42) == 0.42  # raw passthrough
    finally:
        # Drop the {} so other tests reload the real bundle.
        pc._bundle.cache_clear()
        pc._curves.cache_clear()
