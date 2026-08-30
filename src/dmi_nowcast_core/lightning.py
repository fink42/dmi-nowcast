"""Lightning strike geometry + cell-tracking ETA (no HA, no FastAPI).

A sparse-point companion to the radar nowcast. Given recent Blitzortung
strikes (lat/lon/time) and a target point, estimate when an electrically
active storm cell's *leading edge* will reach rings (e.g. 3 km, 10 km)
around the target.

Approach (advection on sparse points, mirroring the radar nowcast idea):

  1. Cluster strikes into cells (spatial, haversine single-linkage).
  2. Estimate each cell's velocity by a least-squares fit of strike
     position vs time (OLS slope == centroid velocity).
  3. Leading edge = closest of the *most-recent* strikes to the target.
  4. Closing speed = velocity component along the cell→target direction.
  5. ``ETA_to_ring = (leading_edge - ring) / closing_speed``.

Distances use the haversine formula; the velocity fit uses a local
equirectangular ENU approximation around the target (valid at the ≤60 km
scales we care about). Pure Python + numpy so it unit-tests in isolation
and carries no ``homeassistant`` / FastAPI imports — same contract as the
rest of ``dmi_nowcast_core``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np

EARTH_RADIUS_KM = 6371.0088
# Equirectangular scale factors (km per degree) — good enough for the
# local ENU velocity fit over tens of km.
_KM_PER_DEG_LAT = 110.574
_KM_PER_DEG_LON = 111.320  # times cos(lat)

# ETA result states.
STATE_APPROACHING = "approaching"
STATE_RECEDING = "receding"
STATE_STALLED = "stalled"
STATE_INSIDE = "inside_ring"
STATE_INSUFFICIENT = "insufficient_data"


@dataclass(frozen=True)
class LightningStrike:
    """One detected strike. ``t`` must be timezone-aware UTC."""
    lat: float
    lon: float
    t: datetime


@dataclass(frozen=True)
class EtaParams:
    """Tunables for :func:`compute_eta`. Mirrors the sidecar's LightningConfig."""
    buffer_window_min: float = 30.0
    min_strikes: int = 6
    cluster_eps_km: float = 15.0
    relevance_radius_km: float = 60.0
    leading_edge_recent_min: float = 5.0
    # Closing speeds below this are treated as "stalled" rather than a usable ETA.
    min_closing_kmh: float = 5.0
    # Minimum time spread among a cell's strikes for its velocity fit to be trusted.
    min_fit_span_min: float = 3.0


@dataclass(frozen=True)
class RingEta:
    ring_km: float
    eta_min: float | None
    inside: bool


@dataclass(frozen=True)
class EtaResult:
    state: str
    rings: list[RingEta]
    leading_edge_km: float | None
    closing_kmh: float | None
    cell_speed_kmh: float | None
    # Compass bearing the cell is moving TOWARD (0°=N, 90°=E); None if unknown.
    cell_bearing_deg: float | None
    n_strikes: int
    n_cells: int
    confidence: float


# --------------------------------------------------------------------------- #
# Geometry helpers (missing from geo.py, which only does radar-grid projection)
# --------------------------------------------------------------------------- #
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two WGS84 points."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing in degrees (0°=N, clockwise) from point 1 toward point 2."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlmb = math.radians(lon2 - lon1)
    x = math.sin(dlmb) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlmb)
    return math.degrees(math.atan2(x, y)) % 360.0


def _to_enu_km(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
    """Local east/north offset in km of (lat, lon) relative to (lat0, lon0)."""
    east = (lon - lon0) * _KM_PER_DEG_LON * math.cos(math.radians(lat0))
    north = (lat - lat0) * _KM_PER_DEG_LAT
    return east, north


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #
def cluster_strikes(
    strikes: list[LightningStrike], eps_km: float
) -> list[list[LightningStrike]]:
    """Single-linkage spatial clustering: strikes within ``eps_km`` of each
    other join the same cell. O(n²) — fine for the small buffers (≤ a few
    hundred strikes) this runs on, and avoids a scikit-learn dependency.
    """
    n = len(strikes)
    if n == 0:
        return []
    # Union-find over the within-eps adjacency.
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if haversine_km(
                strikes[i].lat, strikes[i].lon, strikes[j].lat, strikes[j].lon
            ) <= eps_km:
                union(i, j)

    groups: dict[int, list[LightningStrike]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(strikes[i])
    return list(groups.values())


# --------------------------------------------------------------------------- #
# Velocity
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Velocity:
    east_kmmin: float
    north_kmmin: float
    residual_km: float
    span_min: float


def estimate_velocity(
    cell: list[LightningStrike], lat0: float, lon0: float
) -> _Velocity | None:
    """Least-squares velocity of a cell's strike cloud in local ENU km/min.

    OLS of position vs time gives the centroid's linear motion. Returns
    ``None`` when there are fewer than two strikes or zero time spread.
    """
    if len(cell) < 2:
        return None
    t0 = min(s.t for s in cell)
    t = np.array([(s.t - t0).total_seconds() / 60.0 for s in cell], dtype=float)
    span = float(t.max() - t.min())
    if span <= 0.0:
        return None
    east = np.array([_to_enu_km(s.lat, s.lon, lat0, lon0)[0] for s in cell])
    north = np.array([_to_enu_km(s.lat, s.lon, lat0, lon0)[1] for s in cell])
    A = np.column_stack([t, np.ones_like(t)])
    (ve, _ie), res_e, *_ = np.linalg.lstsq(A, east, rcond=None)
    (vn, _in), res_n, *_ = np.linalg.lstsq(A, north, rcond=None)
    # Residual: RMS of the combined fit (km).
    pred_e = A @ np.array([ve, _ie])
    pred_n = A @ np.array([vn, _in])
    resid = float(np.sqrt(np.mean((east - pred_e) ** 2 + (north - pred_n) ** 2)))
    return _Velocity(east_kmmin=float(ve), north_kmmin=float(vn),
                     residual_km=resid, span_min=span)


# --------------------------------------------------------------------------- #
# ETA
# --------------------------------------------------------------------------- #
def _confidence(n_strikes: int, span_min: float, residual_km: float,
                params: EtaParams) -> float:
    """Heuristic 0–1 trust score. Not a probability — a "how shaky are the
    inputs" signal, mirroring the rain nowcast's separate confidence."""
    n_factor = min(1.0, n_strikes / (2.0 * max(1, params.min_strikes)))
    span_factor = min(1.0, span_min / 10.0)
    resid_factor = 1.0 / (1.0 + residual_km / 3.0)  # 3 km scale
    return round(max(0.0, min(1.0, n_factor * 0.4 + span_factor * 0.3
                              + resid_factor * 0.3)), 3)


def _empty(state: str, rings: list[float], n_strikes: int, n_cells: int) -> EtaResult:
    return EtaResult(
        state=state,
        rings=[RingEta(ring_km=r, eta_min=None, inside=False) for r in sorted(rings)],
        leading_edge_km=None, closing_kmh=None, cell_speed_kmh=None,
        cell_bearing_deg=None, n_strikes=n_strikes, n_cells=n_cells, confidence=0.0,
    )


@dataclass(frozen=True)
class _CellEval:
    """Per-cluster metrics relative to a target. Shared by compute_eta (picks
    the threatening one) and summarize_clusters (reports all of them)."""
    cell: list[LightningStrike]
    n: int
    centroid_lat: float
    centroid_lon: float
    spread_km: float          # mean strike distance from centroid
    latest_age_min: float
    leading_edge_km: float    # closest recent strike to target
    closing_kmmin: float      # velocity component toward target (km/min)
    speed_kmh: float | None   # cell ground speed
    bearing_deg: float | None # direction of motion (0=N, toward)
    vel_east_kmmin: float | None
    vel_north_kmmin: float | None
    span_min: float
    residual_km: float


def _evaluate_cell(
    cell: list[LightningStrike], target_lat: float, target_lon: float,
    params: EtaParams, now: datetime,
) -> _CellEval:
    n = len(cell)
    clat = sum(s.lat for s in cell) / n
    clon = sum(s.lon for s in cell) / n
    spread = sum(haversine_km(clat, clon, s.lat, s.lon) for s in cell) / n
    latest = max(s.t for s in cell)
    latest_age = (now - latest).total_seconds() / 60.0
    recent = [s for s in cell
              if (latest - s.t).total_seconds() / 60.0 <= params.leading_edge_recent_min]
    edge_set = recent or cell
    leading_edge = min(haversine_km(target_lat, target_lon, s.lat, s.lon) for s in edge_set)

    vel = estimate_velocity(cell, target_lat, target_lon)
    closing_kmmin = 0.0
    bearing = speed_kmh = ve = vn = None
    span = resid = 0.0
    if vel is not None:
        span, resid = vel.span_min, vel.residual_km
        if vel.span_min >= params.min_fit_span_min:
            ve, vn = vel.east_kmmin, vel.north_kmmin
            cx = float(np.mean([_to_enu_km(s.lat, s.lon, target_lat, target_lon)[0] for s in cell]))
            cy = float(np.mean([_to_enu_km(s.lat, s.lon, target_lat, target_lon)[1] for s in cell]))
            norm = math.hypot(cx, cy)
            if norm > 1e-6:
                ux, uy = -cx / norm, -cy / norm
                closing_kmmin = vel.east_kmmin * ux + vel.north_kmmin * uy
            speed = math.hypot(vel.east_kmmin, vel.north_kmmin)
            speed_kmh = speed * 60.0
            if speed > 1e-6:
                bearing = math.degrees(math.atan2(vel.east_kmmin, vel.north_kmmin)) % 360.0
    return _CellEval(
        cell=cell, n=n, centroid_lat=clat, centroid_lon=clon, spread_km=spread,
        latest_age_min=latest_age, leading_edge_km=leading_edge,
        closing_kmmin=closing_kmmin, speed_kmh=speed_kmh, bearing_deg=bearing,
        vel_east_kmmin=ve, vel_north_kmmin=vn, span_min=span, residual_km=resid,
    )


def _eta_to_ring(ev: _CellEval, ring_km: float) -> float:
    """Minutes for ev's leading edge to reach ``ring_km``; inf if not closing."""
    if ev.closing_kmmin <= 0:
        return math.inf
    return max(0.0, (ev.leading_edge_km - ring_km) / ev.closing_kmmin)


def _threat_eta(ev: _CellEval, ring_km: float, min_closing_kmh: float) -> float:
    """ETA used to RANK cells for the 'most threatening' pick: 0 if already inside
    the ring, the real ETA if approaching above ``min_closing_kmh``, else +inf.

    Without the closing gate a near-stationary cell with a close edge (which the
    classifier will call STALLED and emit no ETA for) can outrank — and mask — a
    genuinely approaching cell farther out (review finding M1)."""
    if ev.leading_edge_km <= ring_km:
        return 0.0
    if ev.closing_kmmin * 60.0 < min_closing_kmh:
        return math.inf
    return (ev.leading_edge_km - ring_km) / ev.closing_kmmin


def _classify(
    leading_edge_km: float, closing_kmh: float,
    rings_sorted: list[float], min_closing_kmh: float,
) -> tuple[str, list[RingEta]]:
    """Map a leading edge + closing speed to a state + per-ring ETAs. Shared by
    compute_eta and smooth_eta so the threshold logic lives in one place."""
    closing_kmmin = closing_kmh / 60.0
    smallest = rings_sorted[0]
    if leading_edge_km <= smallest:
        state = STATE_INSIDE
    elif closing_kmh <= 0:
        state = STATE_RECEDING if closing_kmh < 0 else STATE_STALLED
    elif closing_kmh < min_closing_kmh:
        state = STATE_STALLED
    else:
        state = STATE_APPROACHING

    usable = state == STATE_APPROACHING
    rings: list[RingEta] = []
    for r in rings_sorted:
        if leading_edge_km <= r:
            rings.append(RingEta(ring_km=r, eta_min=0.0, inside=True))
        elif usable:
            rings.append(RingEta(ring_km=r, eta_min=round((leading_edge_km - r) / closing_kmmin, 1), inside=False))
        else:
            rings.append(RingEta(ring_km=r, eta_min=None, inside=False))
    return state, rings


def compute_eta(
    strikes: list[LightningStrike],
    target_lat: float,
    target_lon: float,
    rings_km: list[float],
    params: EtaParams = EtaParams(),
    now: datetime | None = None,
    *,
    multicell: bool = False,
) -> EtaResult:
    """Estimate ETA of the threatening cell's leading edge to each ring.

    See module docstring for the algorithm. ``rings_km`` is reported sorted
    ascending. ``now`` defaults to the current UTC time (override for tests).

    ``multicell=False`` (default, the deployed behaviour) tracks the single cell
    that threatens the smallest ring soonest. ``multicell=True`` evaluates *every*
    cell and reports the soonest ETA across all approaching cells per ring — a
    storm reaching the target from a different cell than the nearest one is no
    longer missed (the POD fix from the algorithm redesign).
    """
    if now is None:
        now = datetime.now(timezone.utc)
    rings_sorted = sorted(rings_km)

    # 1. Window + relevance filter.
    cutoff = now - timedelta(minutes=params.buffer_window_min)
    relevant = [
        s for s in strikes
        if s.t >= cutoff
        and haversine_km(target_lat, target_lon, s.lat, s.lon)
        <= params.relevance_radius_km
    ]
    if len(relevant) < params.min_strikes:
        return _empty(STATE_INSUFFICIENT, rings_sorted, len(relevant), 0)

    # 2. Cluster into cells.
    cells = cluster_strikes(relevant, params.cluster_eps_km)
    n_cells = len(cells)
    smallest_ring = rings_sorted[0]
    evals = [_evaluate_cell(cell, target_lat, target_lon, params, now) for cell in cells]

    if multicell:
        return _aggregate_multicell(evals, rings_sorted, params, len(relevant), n_cells)

    # 3–4. Single-cell: keep the one that threatens the target soonest (smallest
    # positive ETA to the smallest ring; tie-break on nearer edge).
    best = min(evals, key=lambda e: (_threat_eta(e, smallest_ring, params.min_closing_kmh),
                                     e.leading_edge_km))

    leading_edge = best.leading_edge_km
    closing_kmmin = best.closing_kmmin
    confidence = _confidence(best.n, best.span_min, best.residual_km, params)

    # 5. Classify + per-ring ETA (shared with smooth_eta).
    closing_kmh = closing_kmmin * 60.0
    state, ring_etas = _classify(leading_edge, closing_kmh, rings_sorted, params.min_closing_kmh)

    return EtaResult(
        state=state,
        rings=ring_etas,
        leading_edge_km=round(leading_edge, 2),
        closing_kmh=round(closing_kmh, 1),
        cell_speed_kmh=round(best.speed_kmh, 1) if best.speed_kmh is not None else None,
        cell_bearing_deg=round(best.bearing_deg, 0) if best.bearing_deg is not None else None,
        n_strikes=len(relevant),
        n_cells=n_cells,
        confidence=confidence,
    )


def _aggregate_multicell(
    evals: list[_CellEval], rings_sorted: list[float], params: EtaParams,
    n_strikes: int, n_cells: int,
) -> EtaResult:
    """Combine per-cell classifications into one result: each ring takes the
    soonest ETA across *all* approaching cells; ``state`` is approaching if any
    cell is. Scalar fields come from the primary threat (soonest-approaching cell,
    else the nearest leading edge)."""
    # Per-cell (state, ring-etas) using the shared classifier.
    per_cell = [
        _classify(e.leading_edge_km, e.closing_kmmin * 60.0, rings_sorted, params.min_closing_kmh)
        for e in evals
    ]
    # Aggregate each ring: inside if any cell inside, else soonest cell ETA.
    agg_rings: list[RingEta] = []
    for i, r in enumerate(rings_sorted):
        if any(pc[1][i].inside for pc in per_cell):
            agg_rings.append(RingEta(ring_km=r, eta_min=0.0, inside=True))
            continue
        etas = [pc[1][i].eta_min for pc in per_cell if pc[1][i].eta_min is not None]
        agg_rings.append(RingEta(ring_km=r, eta_min=round(min(etas), 1) if etas else None,
                                 inside=False))
    # Overall state: approaching dominates, then stalled, then receding/inside.
    states = [pc[0] for pc in per_cell]
    if any(s == STATE_INSIDE for s in states):
        state = STATE_INSIDE
    elif any(s == STATE_APPROACHING for s in states):
        state = STATE_APPROACHING
    elif any(s == STATE_STALLED for s in states):
        state = STATE_STALLED
    else:
        state = STATE_RECEDING
    # Primary threat = soonest-approaching to the smallest ring, else nearest edge.
    approaching = [(e, _eta_to_ring(e, rings_sorted[0]))
                   for e, pc in zip(evals, per_cell) if pc[0] == STATE_APPROACHING]
    primary = (min(approaching, key=lambda x: x[1])[0] if approaching
               else min(evals, key=lambda e: e.leading_edge_km))
    return EtaResult(
        state=state,
        rings=agg_rings,
        leading_edge_km=round(primary.leading_edge_km, 2),
        closing_kmh=round(primary.closing_kmmin * 60.0, 1),
        cell_speed_kmh=round(primary.speed_kmh, 1) if primary.speed_kmh is not None else None,
        cell_bearing_deg=round(primary.bearing_deg, 0) if primary.bearing_deg is not None else None,
        n_strikes=n_strikes,
        n_cells=n_cells,
        confidence=_confidence(primary.n, primary.span_min, primary.residual_km, params),
    )


@dataclass(frozen=True)
class ClusterSummary:
    """One cluster's geometry + motion, for the map render and /lightning/clusters
    debug endpoint. Target-relative fields (leading_edge/closing/eta) are filled
    when a target is supplied; ``threatening`` marks the cell compute_eta picks."""
    n_strikes: int
    centroid_lat: float
    centroid_lon: float
    spread_km: float
    latest_age_min: float
    speed_kmh: float | None
    bearing_deg: float | None
    vel_east_kmmin: float | None
    vel_north_kmmin: float | None
    leading_edge_km: float | None
    closing_kmh: float | None
    eta_min: float | None
    threatening: bool


def summarize_clusters(
    strikes: list[LightningStrike],
    target_lat: float,
    target_lon: float,
    params: EtaParams = EtaParams(),
    now: datetime | None = None,
    rings_km: tuple[float, ...] = (3.0, 10.0),
    max_radius_km: float = 150.0,
) -> list[ClusterSummary]:
    """All clusters in the window within ``max_radius_km`` of the target, each
    with motion + (target-relative) leading edge / closing / ETA. The cell that
    ``compute_eta`` would pick and that is actually approaching is flagged
    ``threatening``. Ordered nearest-leading-edge first."""
    if now is None:
        now = datetime.now(timezone.utc)
    cutoff = now - timedelta(minutes=params.buffer_window_min)
    relevant = [
        s for s in strikes
        if s.t >= cutoff
        and haversine_km(target_lat, target_lon, s.lat, s.lon) <= max_radius_km
    ]
    if not relevant:
        return []
    cells = cluster_strikes(relevant, params.cluster_eps_km)
    evals = [_evaluate_cell(cell, target_lat, target_lon, params, now) for cell in cells]
    smallest = min(rings_km)
    # Which cell would compute_eta pick, and is it genuinely approaching?
    best = min(evals, key=lambda e: (_threat_eta(e, smallest, params.min_closing_kmh),
                                     e.leading_edge_km))
    best_approaching = (
        best.closing_kmmin * 60.0 >= params.min_closing_kmh
        and best.leading_edge_km > smallest
    )

    out: list[ClusterSummary] = []
    for e in evals:
        eta = _eta_to_ring(e, smallest)
        approaching = e.closing_kmmin * 60.0 >= params.min_closing_kmh and e.leading_edge_km > smallest
        out.append(ClusterSummary(
            n_strikes=e.n,
            centroid_lat=round(e.centroid_lat, 5),
            centroid_lon=round(e.centroid_lon, 5),
            spread_km=round(e.spread_km, 2),
            latest_age_min=round(e.latest_age_min, 1),
            speed_kmh=round(e.speed_kmh, 1) if e.speed_kmh is not None else None,
            bearing_deg=round(e.bearing_deg, 0) if e.bearing_deg is not None else None,
            vel_east_kmmin=e.vel_east_kmmin,
            vel_north_kmmin=e.vel_north_kmmin,
            leading_edge_km=round(e.leading_edge_km, 2),
            closing_kmh=round(e.closing_kmmin * 60.0, 1),
            eta_min=round(eta, 1) if (approaching and math.isfinite(eta)) else None,
            threatening=(e is best and best_approaching),
        ))
    out.sort(key=lambda c: c.leading_edge_km if c.leading_edge_km is not None else math.inf)
    return out


def smooth_eta(
    result: EtaResult,
    prior_closing_kmh: float | None,
    prior_edge_km: float | None,
    alpha: float,
    min_closing_kmh: float,
) -> tuple[EtaResult, float | None, float | None]:
    """EMA-smooth a fresh EtaResult against the prior cycle's values.

    Blends the noisy ``closing_kmh`` and ``leading_edge_km`` with the prior
    (``alpha`` = weight on the new sample), then re-derives ``state`` + ring
    ETAs from the smoothed scalars via :func:`_classify`. Returns
    ``(smoothed_result, used_closing_kmh, used_edge_km)`` — the caller persists
    the used values as the next prior.

    Pure: all state (the prior) is passed in. No-op for ``insufficient_data``
    (nothing meaningful to smooth) and when there is no prior yet.
    """
    if result.leading_edge_km is None or result.closing_kmh is None:
        return result, None, None
    if prior_closing_kmh is None or prior_edge_km is None:
        sc, se = result.closing_kmh, result.leading_edge_km
    else:
        a = max(0.0, min(1.0, alpha))
        sc = a * result.closing_kmh + (1.0 - a) * prior_closing_kmh
        se = a * result.leading_edge_km + (1.0 - a) * prior_edge_km
    rings_sorted = sorted(r.ring_km for r in result.rings)
    state, rings = _classify(se, sc, rings_sorted, min_closing_kmh)
    smoothed = EtaResult(
        state=state,
        rings=rings,
        leading_edge_km=round(se, 2),
        closing_kmh=round(sc, 1),
        cell_speed_kmh=result.cell_speed_kmh,
        cell_bearing_deg=result.cell_bearing_deg,
        n_strikes=result.n_strikes,
        n_cells=result.n_cells,
        confidence=result.confidence,
    )
    return smoothed, sc, se


# --------------------------------------------------------------------------- #
# Probabilistic / areal forecast (Phase 2 redesign)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ProbParams:
    """Tunables for :func:`strike_probability`. A 2026-06-27 sweep tried tuning
    these (pos_sigma 2→4 etc.) — it looked good on the thin anchors but did NOT
    hold on the robust grid (anchor overfit; the deployable 10 km ring was better
    with these original values), so they stand. ProbParams has little leverage;
    the architecture is the win. See algorithm-redesign.md Phase 2."""
    n_members: int = 50
    # Per-cell velocity uncertainty (km/min, one std per axis): a fraction of the
    # cell speed + a floor + a residual-driven term. Over a lead of Δt this becomes
    # a position cone of ≈ sigma·Δt — the motion-error cone (Bowler et al. 2006).
    vel_sigma_frac: float = 0.35
    vel_sigma_floor_kmmin: float = 0.10
    vel_sigma_resid_frac: float = 0.10
    # Per-strike position jitter (km): storm-front spread + Blitzortung location error.
    pos_sigma_km: float = 2.0
    # Velocity std for cells with no reliable fit (erratic/pulse) — they spread
    # diffusely instead of advecting along a bogus vector (the regime switch).
    erratic_sigma_kmmin: float = 0.5


@dataclass(frozen=True)
class RingProb:
    ring_km: float
    lead_min: float
    prob: float  # P(≥1 strike within ring_km of target within lead_min)


def _ensemble_ring_hit(p0: np.ndarray, vp: np.ndarray, ring_km: float,
                       lead_min: float) -> np.ndarray:
    """Boolean (n_members,): does any of a cell's perturbed strikes (start ``p0``
    ENU km, shape (M, k, 2)) advected by perturbed velocity ``vp`` (M, 2) km/min
    enter ``ring_km`` within ``lead_min``? Solves |p0 + vp·t|² = r² per strike."""
    a = (vp * vp).sum(axis=1)                                    # (M,)
    b = 2.0 * (p0[:, :, 0] * vp[:, None, 0] + p0[:, :, 1] * vp[:, None, 1])  # (M,k)
    c = (p0 * p0).sum(axis=2) - ring_km * ring_km                # (M,k)
    a_k = a[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        disc = b * b - 4.0 * a_k * c
        sq = np.sqrt(np.where(disc >= 0, disc, 0.0))
        t1 = (-b - sq) / (2.0 * a_k)
        t2 = (-b + sq) / (2.0 * a_k)
    earliest = np.full(c.shape, np.inf)
    for t in (t1, t2):
        valid = (disc >= 0) & np.isfinite(t) & (t >= 0)
        earliest = np.where(valid & (t < earliest), t, earliest)
    earliest = np.where(c <= 0, 0.0, earliest)                   # already inside ⇒ t=0
    return (earliest <= lead_min).any(axis=1)                    # (M,)


def strike_probability(
    strikes: list[LightningStrike],
    target_lat: float,
    target_lon: float,
    ring_leads: list[tuple[float, float]],
    params: EtaParams = EtaParams(),
    prob: ProbParams = ProbParams(),
    now: datetime | None = None,
    seed: int = 12345,
) -> list[RingProb]:
    """Areal probabilistic nowcast: ``P(≥1 strike within ring within lead)`` for
    each ``(ring_km, lead_min)`` pair, via a Lagrangian motion-perturbation
    ensemble over the recent strikes (Phase 2 redesign — see
    the algorithm redesign).

    Each cell's recent strikes are advected by its velocity, perturbed per member
    by velocity noise (a growing motion-error cone) and position jitter; a member
    "hits" a ring if any advected strike enters it within the lead. The probability
    is the hit fraction. Cells with no reliable velocity fit spread diffusely
    (``erratic_sigma_kmmin``) rather than advecting along a bogus vector. Pure +
    seeded → reproducible."""
    if now is None:
        now = datetime.now(timezone.utc)
    base = [RingProb(ring_km=r, lead_min=l, prob=0.0) for r, l in ring_leads]

    cutoff = now - timedelta(minutes=params.buffer_window_min)
    relevant = [
        s for s in strikes
        if s.t >= cutoff
        and haversine_km(target_lat, target_lon, s.lat, s.lon) <= params.relevance_radius_km
    ]
    if len(relevant) < params.min_strikes:
        return base

    cells = cluster_strikes(relevant, params.cluster_eps_km)
    rng = np.random.default_rng(seed)
    M = prob.n_members
    member_hit = {(r, l): np.zeros(M, dtype=bool) for r, l in ring_leads}

    for cell in cells:
        latest = max(s.t for s in cell)
        recent = [s for s in cell
                  if (latest - s.t).total_seconds() / 60.0 <= params.leading_edge_recent_min] or cell
        offs = np.array([_to_enu_km(s.lat, s.lon, target_lat, target_lon) for s in recent])
        k = len(offs)
        vel = estimate_velocity(cell, target_lat, target_lon)
        if vel is not None and vel.span_min >= params.min_fit_span_min:
            v = np.array([vel.east_kmmin, vel.north_kmmin])
            speed = math.hypot(v[0], v[1])
            sigma_v = (prob.vel_sigma_frac * speed + prob.vel_sigma_floor_kmmin
                       + prob.vel_sigma_resid_frac * vel.residual_km)
        else:
            v = np.array([0.0, 0.0])
            sigma_v = prob.erratic_sigma_kmmin
        vp = v[None, :] + rng.normal(0.0, sigma_v, (M, 2))
        p0 = offs[None, :, :] + rng.normal(0.0, prob.pos_sigma_km, (M, k, 2))
        for r, l in ring_leads:
            member_hit[(r, l)] |= _ensemble_ring_hit(p0, vp, r, l)

    return [RingProb(ring_km=r, lead_min=l, prob=float(member_hit[(r, l)].mean()))
            for r, l in ring_leads]


__all__ = [
    "LightningStrike", "EtaParams", "RingEta", "EtaResult",
    "ClusterSummary", "ProbParams", "RingProb",
    "haversine_km", "bearing_deg", "cluster_strikes", "estimate_velocity",
    "compute_eta", "summarize_clusters", "smooth_eta", "strike_probability",
    "STATE_APPROACHING", "STATE_RECEDING", "STATE_STALLED",
    "STATE_INSIDE", "STATE_INSUFFICIENT",
]
