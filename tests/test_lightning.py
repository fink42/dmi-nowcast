"""Tests for lightning.py — geometry, clustering, velocity, and ETA.

Pure-Python module; no fixtures needed. ETA tests build synthetic strike
clouds with known motion so the expected ETA is analytic.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from dmi_nowcast_core.lightning import (
    EtaParams,
    LightningStrike,
    STATE_APPROACHING,
    STATE_INSIDE,
    STATE_INSUFFICIENT,
    STATE_RECEDING,
    STATE_STALLED,
    bearing_deg,
    cluster_strikes,
    compute_eta,
    estimate_velocity,
    haversine_km,
    smooth_eta,
    strike_probability,
    summarize_clusters,
)

NOW = datetime(2026, 6, 6, 12, 0, 0, tzinfo=timezone.utc)
TARGET_LAT, TARGET_LON = 55.0, 10.0


def _km_to_dlon(km: float, lat0: float = TARGET_LAT) -> float:
    return km / (111.320 * math.cos(math.radians(lat0)))


def _strike_west(dist_km: float, minutes_ago: float, lat: float = TARGET_LAT) -> LightningStrike:
    """A strike due west of the target at ``dist_km`` km, ``minutes_ago`` old."""
    return LightningStrike(
        lat=lat, lon=TARGET_LON - _km_to_dlon(dist_km, lat),
        t=NOW - timedelta(minutes=minutes_ago),
    )


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def test_haversine_one_degree_lat():
    # One degree of latitude ≈ 111.2 km anywhere.
    d = haversine_km(55.0, 10.0, 56.0, 10.0)
    assert 110.0 < d < 112.0


def test_haversine_zero():
    assert haversine_km(55.0, 10.0, 55.0, 10.0) == pytest.approx(0.0, abs=1e-9)


def test_bearing_cardinals():
    assert bearing_deg(55.0, 10.0, 56.0, 10.0) == pytest.approx(0.0, abs=1.0)   # N
    assert bearing_deg(55.0, 10.0, 55.0, 11.0) == pytest.approx(90.0, abs=1.0)  # E
    assert bearing_deg(55.0, 10.0, 54.0, 10.0) == pytest.approx(180.0, abs=1.0)  # S


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #
def test_cluster_separates_distant_groups():
    near = [LightningStrike(55.0, 10.0, NOW), LightningStrike(55.01, 10.01, NOW)]
    far = [LightningStrike(56.0, 11.0, NOW), LightningStrike(56.01, 11.01, NOW)]
    clusters = cluster_strikes(near + far, eps_km=15.0)
    assert len(clusters) == 2
    assert {len(c) for c in clusters} == {2}


def test_cluster_merges_close_strikes():
    strikes = [LightningStrike(55.0 + 0.01 * i, 10.0, NOW) for i in range(5)]  # ~1.1 km steps
    clusters = cluster_strikes(strikes, eps_km=15.0)
    assert len(clusters) == 1


def test_cluster_empty():
    assert cluster_strikes([], eps_km=15.0) == []


# --------------------------------------------------------------------------- #
# Velocity
# --------------------------------------------------------------------------- #
def test_estimate_velocity_recovers_eastward_motion():
    # Cell drifting east at 1 km/min, due west of target: distance shrinks as
    # the strike gets more recent (minutes_ago → 0).
    cell = [_strike_west(20.0 + m, minutes_ago=m) for m in range(11)]
    vel = estimate_velocity(cell, TARGET_LAT, TARGET_LON)
    assert vel is not None
    assert vel.east_kmmin == pytest.approx(1.0, abs=0.1)   # moving east
    assert vel.north_kmmin == pytest.approx(0.0, abs=0.1)
    assert vel.span_min == pytest.approx(10.0, abs=0.01)


def test_estimate_velocity_too_few():
    assert estimate_velocity([LightningStrike(55, 10, NOW)], TARGET_LAT, TARGET_LON) is None


# --------------------------------------------------------------------------- #
# ETA
# --------------------------------------------------------------------------- #
def _approaching_cell() -> list[LightningStrike]:
    # 11 strikes, one per minute over the last 10 min, marching east at
    # 1 km/min (= 60 km/h). Nearest (now) strike is 20 km west; 10 min ago
    # it was 30 km west.
    return [_strike_west(20.0 + m, minutes_ago=m) for m in range(11)]


def test_compute_eta_approaching():
    res = compute_eta(_approaching_cell(), TARGET_LAT, TARGET_LON, [3.0, 10.0], now=NOW)
    assert res.state == STATE_APPROACHING
    assert res.n_cells == 1
    assert res.leading_edge_km == pytest.approx(20.0, abs=1.0)
    assert res.closing_kmh == pytest.approx(60.0, abs=8.0)
    assert res.cell_bearing_deg == pytest.approx(90.0, abs=10.0)  # moving east
    by_ring = {r.ring_km: r for r in res.rings}
    # (20-10)/1 = 10 min ; (20-3)/1 = 17 min
    assert by_ring[10.0].eta_min == pytest.approx(10.0, abs=2.0)
    assert by_ring[3.0].eta_min == pytest.approx(17.0, abs=2.5)
    # Closer ring should always be reached later than the wider ring.
    assert by_ring[3.0].eta_min > by_ring[10.0].eta_min
    assert 0.0 < res.confidence <= 1.0


def test_compute_eta_receding():
    # Same cloud but marching west (away): nearest strike grows from 20 km
    # (10 min ago) to 30 km (now).
    cell = [_strike_west(30.0 - m, minutes_ago=m) for m in range(11)]
    res = compute_eta(cell, TARGET_LAT, TARGET_LON, [3.0, 10.0], now=NOW)
    assert res.state == STATE_RECEDING
    assert all(r.eta_min is None for r in res.rings)


def test_compute_eta_inside_ring():
    # A dense cell sitting right on top of the target (within 3 km).
    cell = [_strike_west(1.0, minutes_ago=m) for m in range(0, 11)]
    res = compute_eta(cell, TARGET_LAT, TARGET_LON, [3.0, 10.0], now=NOW)
    assert res.state == STATE_INSIDE
    assert all(r.inside and r.eta_min == 0.0 for r in res.rings)


def test_compute_eta_insufficient_data():
    cell = [_strike_west(20.0, minutes_ago=m) for m in range(3)]  # < min_strikes (6)
    res = compute_eta(cell, TARGET_LAT, TARGET_LON, [3.0, 10.0], now=NOW)
    assert res.state == STATE_INSUFFICIENT
    assert all(r.eta_min is None for r in res.rings)
    assert res.confidence == 0.0


def test_compute_eta_drops_stale_strikes():
    # All strikes older than the 30-min window → treated as no data.
    cell = [_strike_west(20.0, minutes_ago=40 + m) for m in range(10)]
    res = compute_eta(cell, TARGET_LAT, TARGET_LON, [3.0, 10.0], now=NOW)
    assert res.state == STATE_INSUFFICIENT


def test_compute_eta_ignores_far_strikes():
    # Strikes beyond relevance_radius_km (60) shouldn't count.
    cell = [_strike_west(200.0, minutes_ago=m) for m in range(10)]
    res = compute_eta(cell, TARGET_LAT, TARGET_LON, [3.0, 10.0], now=NOW)
    assert res.state == STATE_INSUFFICIENT


def _strike_north(dist_km: float, minutes_ago: float) -> LightningStrike:
    """A strike due north of the target at ``dist_km`` km, ``minutes_ago`` old."""
    return LightningStrike(lat=TARGET_LAT + dist_km / 110.574, lon=TARGET_LON,
                           t=NOW - timedelta(minutes=minutes_ago))


def test_compute_eta_multicell_picks_soonest_across_cells():
    # Two well-separated approaching cells:
    #  P — due west, close (11 km) but slow (0.5 km/min) → soonest to 10 km, late to 3 km.
    #  Q — due north, far (25 km) but fast (4 km/min) → soonest to 3 km, so the
    #      single-cell tracker picks Q and reports Q's (later) 10 km ETA.
    P = [_strike_west(11.0 + 0.5 * m, minutes_ago=m) for m in range(11)]
    Q = [_strike_north(25.0 + 4.0 * m, minutes_ago=m) for m in range(9)]
    strikes = P + Q

    single = compute_eta(strikes, TARGET_LAT, TARGET_LON, [3.0, 10.0], now=NOW)
    multi = compute_eta(strikes, TARGET_LAT, TARGET_LON, [3.0, 10.0], now=NOW, multicell=True)

    assert multi.n_cells == 2 and multi.state == STATE_APPROACHING
    eta10_single = next(r.eta_min for r in single.rings if r.ring_km == 10.0)
    eta10_multi = next(r.eta_min for r in multi.rings if r.ring_km == 10.0)
    assert eta10_single is not None and eta10_multi is not None
    # Multi-cell takes the soonest 10 km arrival across BOTH cells, so it can only be
    # ≤ the single-cell pick — and strictly sooner here (cell P beats Q to 10 km).
    assert eta10_multi <= eta10_single
    assert eta10_multi < eta10_single


def test_compute_eta_stalled_near_cell_does_not_mask_approaching():
    # Cell A: just outside 3 km, crawling at ~0.6 km/h (< min_closing) → STALLED.
    #   Old code ranked it by a finite raw ETA and let it win, masking cell B.
    # Cell B: 20 km north, closing ~30 km/h → genuinely approaching.
    # Single-cell compute_eta must now pick B and report its arrival (finding M1).
    A = [_strike_west(3.1 + 0.01 * m, minutes_ago=m) for m in range(11)]
    B = [_strike_north(20.0 + 0.5 * m, minutes_ago=m) for m in range(11)]
    res = compute_eta(A + B, TARGET_LAT, TARGET_LON, [3.0, 10.0], now=NOW)
    assert res.n_cells == 2
    assert res.state == STATE_APPROACHING
    eta10 = next(r.eta_min for r in res.rings if r.ring_km == 10.0)
    assert eta10 is not None  # B's 10 km arrival is reported, not masked by A


# --------------------------------------------------------------------------- #
# strike_probability (Phase 2 — probabilistic / areal forecast)
# --------------------------------------------------------------------------- #
RING_LEADS = [(3.0, 15.0), (10.0, 30.0)]


def _prob(strikes):
    out = strike_probability(strikes, TARGET_LAT, TARGET_LON, RING_LEADS, now=NOW)
    return {rp.ring_km: rp.prob for rp in out}


def test_strike_probability_approaching_high():
    # A cell closing from 20 km at ~1 km/min reaches 10 km in ~10 min (< 30 lead)
    # → high P at 10 km; 3 km is ~17 min out (> 15 lead) → lower.
    p = _prob(_approaching_cell())
    assert p[10.0] > 0.7
    assert p[3.0] < p[10.0]
    assert all(0.0 <= v <= 1.0 for v in p.values())


def test_strike_probability_receding_low():
    receding = [_strike_west(30.0 - m, minutes_ago=m) for m in range(11)]  # moving away
    p = _prob(receding)
    assert p[10.0] < 0.2


def test_strike_probability_insufficient_zero():
    p = _prob([_strike_west(20.0, minutes_ago=m) for m in range(3)])  # < min_strikes
    assert p[3.0] == 0.0 and p[10.0] == 0.0


def test_strike_probability_reproducible():
    cell = _approaching_cell()
    a = strike_probability(cell, TARGET_LAT, TARGET_LON, RING_LEADS, now=NOW, seed=7)
    b = strike_probability(cell, TARGET_LAT, TARGET_LON, RING_LEADS, now=NOW, seed=7)
    assert [r.prob for r in a] == [r.prob for r in b]


# --------------------------------------------------------------------------- #
# summarize_clusters (map / debug)
# --------------------------------------------------------------------------- #
def test_summarize_clusters_two_cells_marks_threatening():
    approaching = _approaching_cell()  # west, marching east toward target
    # A second, stationary cell far to the NE, well separated (> cluster eps).
    ne = [LightningStrike(55.6, 10.9, NOW - timedelta(minutes=m)) for m in range(6)]
    clusters = summarize_clusters(approaching + ne, TARGET_LAT, TARGET_LON, now=NOW)
    assert len(clusters) == 2
    threat = [c for c in clusters if c.threatening]
    assert len(threat) == 1
    t = threat[0]
    assert t.bearing_deg == pytest.approx(90.0, abs=15.0)  # moving east
    assert t.eta_min is not None and t.eta_min > 0
    assert t.leading_edge_km == pytest.approx(20.0, abs=2.0)
    # Every cluster reports geometry.
    assert all(c.n_strikes >= 1 and c.centroid_lat and c.spread_km >= 0 for c in clusters)


def test_summarize_clusters_empty():
    assert summarize_clusters([], TARGET_LAT, TARGET_LON, now=NOW) == []


# --------------------------------------------------------------------------- #
# smooth_eta (cross-cycle EMA)
# --------------------------------------------------------------------------- #
def test_smooth_eta_no_prior_returns_raw():
    res = compute_eta(_approaching_cell(), TARGET_LAT, TARGET_LON, [3.0, 10.0], now=NOW)
    sm, sc, se = smooth_eta(res, None, None, alpha=0.3, min_closing_kmh=5.0)
    assert sm.state == res.state
    assert sc == res.closing_kmh and se == res.leading_edge_km


def test_smooth_eta_damps_transient_stall():
    # This cycle's raw estimate is a (transient) stall: strikes not moving.
    stalled = [_strike_west(20.0, minutes_ago=m) for m in range(10)]
    raw = compute_eta(stalled, TARGET_LAT, TARGET_LON, [3.0, 10.0], now=NOW)
    assert raw.state in (STATE_STALLED, STATE_RECEDING)
    # But the prior cycles had it approaching at 60 km/h, edge 20 km.
    sm, sc, se = smooth_eta(raw, prior_closing_kmh=60.0, prior_edge_km=20.0,
                            alpha=0.3, min_closing_kmh=5.0)
    # EMA keeps it approaching instead of flipping to stalled.
    assert sm.state == STATE_APPROACHING
    assert sc == pytest.approx(0.3 * raw.closing_kmh + 0.7 * 60.0, abs=0.2)
    by = {r.ring_km: r for r in sm.rings}
    assert by[10.0].eta_min is not None


def test_smooth_eta_insufficient_passthrough():
    res = compute_eta([_strike_west(20.0, minutes_ago=m) for m in range(3)],
                      TARGET_LAT, TARGET_LON, [3.0, 10.0], now=NOW)
    assert res.state == STATE_INSUFFICIENT
    sm, sc, se = smooth_eta(res, 60.0, 20.0, 0.3, 5.0)
    assert sm is res and sc is None and se is None
