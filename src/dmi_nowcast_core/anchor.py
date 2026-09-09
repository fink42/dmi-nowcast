"""Which radar frame a nowcast cycle stands on — Phase H, H-L / L3.

DMI publishes two composites into one collection, ten minutes apart in
nominal time and, as it turns out, *simultaneously* in wall time:

===========  ========  ========  =====================================
product      minute    range     published after its nominal time
===========  ========  ========  =====================================
fullRange    ``:x0``   240 km    13.1 min (median = p90, 2026-09-06/07)
doppler      ``:x5``   120 km    8.1 min (median = p90, max 13.1)
===========  ========  ========  =====================================

``x0 + 13.1 == x5 + 8.1``: both frames land in the same publication
event, every ten minutes. The cycle that reads only fullRange therefore
stands on a frame that is **five minutes older than the freshest one it
could have had**, at every horizon, before any algorithm change (plan
§0.2: mean anchor age 18.4 min vs 13.5 min). That is what this module
buys, and it is the cheapest five minutes in the phase.

Why the doppler frame cannot simply be dropped into the chain
-------------------------------------------------------------
The Phase B addendum (2026-08-29) stopped an earlier attempt at *raw*
mixing, and it was right to. The two products are not the same view:
doppler covers ~40 % of fullRange's area, and where both see echo it
reads 0–2.5 dB low, so its echo area at 20 dBZ (~0.6 mm/h — where the
wet threshold lives) is only **0.69–0.83 of fullRange's**, i.e. 20–30 %
less echo, before any correction (plan §0.3, measured 2026-09-08). Two
views that differ by as much as ten minutes of evolution cannot alternate
inside one advection chain: every second frame would appear to lose a
fifth of its rain and grow it back.

So the doppler frame is used only after :func:`product_pairs.apply_harmonisation`
maps its dBZ onto fullRange's distribution, per distance band and season
(L2, 2026-09-08: shifts −2.0 to +2.8 dB; held-out wet-area ratio at
0.5 mm/h moves from 0.79–1.67 before mapping to 0.97–1.11 after). And
even then the two products never alternate *in time* inside one chain:

* the **anchor field** (:func:`anchor_field`) is the per-pixel freshest
  covering frame — harmonised doppler inside its 120 km range, the newest
  fullRange frame outside it. DECIDE-3 option (a).
* the **history** (:func:`anchor_history`) — the frames the flow and the
  STEPS cascade are estimated from — is always **same-type**, three
  frames at 10-minute spacing. DECIDE-3's "flow always from same-type
  pairs" and DECIDE-4's "keep the 10-minute STEPS timestep".

Nothing here fetches, parses or forecasts. It answers three questions —
*which frame, what field, which history* — from a listing of what is on
disk and a clock, so the replay (L3) and the runtime (L4) can answer them
the same way.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .corpus import SCAN_TYPE_DOPPLER, SCAN_TYPE_FULL_RANGE
from .product_pairs import (
    HARMONISATION_SCHEMA_VERSION,
    apply_harmonisation,
    radar_distance_grid,
    season_of,
)

__all__ = [
    "POLICY_FULLRANGE",
    "POLICY_FRESHEST",
    "POLICIES",
    "HISTORY_SAME_TYPE",
    "HISTORY_FILLED",
    "HISTORY_MODES",
    "DEFAULT_LAG_FULLRANGE_MIN",
    "DEFAULT_LAG_DOPPLER_MIN",
    "DEFAULT_POLL_INTERVAL_MIN",
    "DEFAULT_MAX_ANCHOR_AGE_MIN",
    "DEFAULT_STEP_MIN",
    "DEFAULT_TOLERANCE_S",
    "ProductLag",
    "AnchorSelection",
    "AnchorField",
    "HistorySelection",
    "select_anchor",
    "decision_instants",
    "anchor_field",
    "fill_uncovered",
    "anchor_history",
    "distance_km_grid",
    "load_harmonisation",
    "harmonisation_stamp",
]


#: Today's policy: the cycle sees fullRange frames and nothing else.
POLICY_FULLRANGE = SCAN_TYPE_FULL_RANGE
#: The candidate: per-pixel freshest covering frame (DECIDE-3 option a).
POLICY_FRESHEST = "freshest"
POLICIES: tuple[str, ...] = (POLICY_FULLRANGE, POLICY_FRESHEST)

#: The history is three frames of ONE product — the brief's reading of
#: DECIDE-3. Under a doppler anchor this means the flow, the cascade and
#: therefore ``p_rain`` exist only inside doppler's 120 km range: the six
#: NW-Jutland gauges (and ~a third of the fullRange domain) go dark.
HISTORY_SAME_TYPE = "same-type"
#: Every history frame is itself a per-pixel freshest composite (doppler
#: harmonised inside coverage, the fullRange frame 5 min earlier outside).
#: Each PIXEL's series is then still same-product and still 10-minute
#: spaced — the addendum's objection is to products alternating in time,
#: which this does not do — and the national grid keeps its coverage.
#: Not the default: it is a wider claim than DECIDE-3 makes, and it is
#: here so L3 can measure it rather than guess.
HISTORY_FILLED = "filled"
HISTORY_MODES: tuple[str, ...] = (HISTORY_SAME_TYPE, HISTORY_FILLED)

#: Publication lag per product, minutes after the nominal frame time.
#: Measured on 576 composite listings, 2026-09-06 00:00 → 2026-09-07
#: 23:55 UTC, from the API's ``created`` property (plan §0.2).
DEFAULT_LAG_FULLRANGE_MIN = 13.1
DEFAULT_LAG_DOPPLER_MIN = 8.1

#: The cycle's poll cadence. The scheduler wakes every 5 min ± 30 s
#: jitter; a replay cannot reproduce the jitter, so instants sit on the
#: 5-minute grid and a frame is first seen at the first poll at or after
#: its publication. This adds the same ~1.9 min to both products, so the
#: five-minute gap between the policies is unaffected.
DEFAULT_POLL_INTERVAL_MIN = 5.0

#: A frame older than this is not an anchor at all. Without the cap a gap
#: in the archive would silently anchor a cycle on a six-hour-old frame
#: and the run would look complete.
DEFAULT_MAX_ANCHOR_AGE_MIN = 60.0

#: Spacing of the history frames (both products publish every 10 min).
DEFAULT_STEP_MIN = 10.0
#: A frame is "on the grid" within a minute.
DEFAULT_TOLERANCE_S = 60.0


def _as_utc(when: datetime) -> datetime:
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class ProductLag:
    """Publication lag per product, in minutes after nominal frame time.

    ``flat`` overrides both — the escape hatch that reproduces a run made
    before this module existed, when the replay simulated one flat frame
    age for every frame (``--frame-age-min 14``).
    """

    fullrange_min: float = DEFAULT_LAG_FULLRANGE_MIN
    doppler_min: float = DEFAULT_LAG_DOPPLER_MIN
    flat: float | None = None

    def minutes(self, scan_type: str) -> float:
        if self.flat is not None:
            return float(self.flat)
        if scan_type == SCAN_TYPE_DOPPLER:
            return float(self.doppler_min)
        if scan_type == SCAN_TYPE_FULL_RANGE:
            return float(self.fullrange_min)
        raise ValueError(f"no publication lag for scan type {scan_type!r}")

    def as_dict(self) -> dict[str, Any]:
        return {
            SCAN_TYPE_FULL_RANGE: float(self.fullrange_min),
            SCAN_TYPE_DOPPLER: float(self.doppler_min),
            "flat": None if self.flat is None else float(self.flat),
        }


DEFAULT_LAG = ProductLag()


@dataclass(frozen=True)
class AnchorSelection:
    """The frame one cycle stands on, and what else was available.

    ``frame_age_min`` is ``now - timestamp``: the number the products'
    lead bookkeeping is corrected by and the wall clock the decision runs
    on (``generated_at = timestamp + frame_age_min``).
    """

    now: datetime
    policy: str
    scan_type: str
    timestamp: datetime
    path: Path
    frame_age_min: float
    fullrange_ts: datetime | None = None
    fullrange_path: Path | None = None
    doppler_ts: datetime | None = None
    doppler_path: Path | None = None
    #: freshest policy, fullRange anchor — the candidate behaved like the
    #: baseline for this instant. ``reason`` says why.
    degraded: bool = False
    #: the same-type history was not available and the instant fell all
    #: the way back to the fullRange policy.
    fallback: bool = False
    reason: str | None = None

    @property
    def key(self) -> tuple[str, datetime]:
        """Identity of the anchor — two instants sharing it are one cycle."""
        return (self.scan_type, self.timestamp)

    @property
    def generated_at(self) -> datetime:
        return self.timestamp + timedelta(minutes=self.frame_age_min)

    def with_now(self, now: datetime) -> "AnchorSelection":
        now = _as_utc(now)
        return replace(
            self, now=now,
            frame_age_min=(now - self.timestamp).total_seconds() / 60.0,
        )

    def as_fullrange(self, *, reason: str) -> "AnchorSelection | None":
        """The same instant, re-anchored on its newest fullRange frame."""
        if self.fullrange_ts is None or self.fullrange_path is None:
            return None
        return replace(
            self,
            scan_type=SCAN_TYPE_FULL_RANGE,
            timestamp=self.fullrange_ts,
            path=self.fullrange_path,
            frame_age_min=(self.now - self.fullrange_ts).total_seconds() / 60.0,
            degraded=True,
            fallback=True,
            reason=reason,
        )


def _newest_published(
    frames: Mapping[datetime, Path] | None,
    now: datetime,
    lag_min: float,
    max_age_min: float,
) -> tuple[datetime, Path] | None:
    """Newest frame that has been published by ``now`` and is not stale."""
    if not frames:
        return None
    best: tuple[datetime, Path] | None = None
    for ts, path in frames.items():
        ts = _as_utc(ts)
        age = (now - ts).total_seconds() / 60.0
        if age < lag_min - 1e-9:
            continue                      # not published yet
        if age > max_age_min + 1e-9:
            continue                      # too stale to anchor a cycle
        if best is None or ts > best[0]:
            best = (ts, Path(path))
    return best


def select_anchor(
    now: datetime,
    frames_by_type: Mapping[str, Mapping[datetime, Path]],
    *,
    policy: str = POLICY_FULLRANGE,
    lag: ProductLag = DEFAULT_LAG,
    max_age_min: float = DEFAULT_MAX_ANCHOR_AGE_MIN,
) -> AnchorSelection | None:
    """The frame a cycle running at ``now`` would stand on.

    ``frames_by_type`` is ``{scan_type: {timestamp: path}}`` — what
    :func:`product_pairs.frame_map` returns, or what a directory probe
    builds. A frame counts as available only once ``ts + lag <= now``:
    the archive knows a frame's nominal time, and simulating history
    without its publication lag is the mistake that made every past
    backtest optimistic by 13 minutes.

    Under :data:`POLICY_FULLRANGE` the anchor is the newest published
    fullRange frame — today's behaviour, with a real lag instead of a
    flat 14 minutes. Under :data:`POLICY_FRESHEST` it is the newer of the
    two products by *frame* time, ties going to fullRange (they can only
    tie if DMI ever publishes both on the same minute, which it does not).

    Returns ``None`` when no fullRange frame is available: it is the base
    layer of every anchor field (the fill outside doppler's range), so a
    cycle without one is not a cycle this module can plan.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown anchor policy {policy!r}; want one of {POLICIES}")
    now = _as_utc(now)
    full = _newest_published(
        frames_by_type.get(SCAN_TYPE_FULL_RANGE), now,
        lag.minutes(SCAN_TYPE_FULL_RANGE), max_age_min,
    )
    if full is None:
        return None
    dop = None
    if policy == POLICY_FRESHEST:
        dop = _newest_published(
            frames_by_type.get(SCAN_TYPE_DOPPLER), now,
            lag.minutes(SCAN_TYPE_DOPPLER), max_age_min,
        )

    if dop is not None and dop[0] > full[0]:
        scan_type, (ts, path) = SCAN_TYPE_DOPPLER, dop
        degraded, reason = False, None
    else:
        scan_type, (ts, path) = SCAN_TYPE_FULL_RANGE, full
        degraded = policy == POLICY_FRESHEST
        reason = None
        if degraded:
            reason = (
                "no doppler frame published" if dop is None
                else "the fullRange frame is the fresher of the two"
            )
    return AnchorSelection(
        now=now,
        policy=policy,
        scan_type=scan_type,
        timestamp=ts,
        path=path,
        frame_age_min=(now - ts).total_seconds() / 60.0,
        fullrange_ts=full[0],
        fullrange_path=full[1],
        doppler_ts=None if dop is None else dop[0],
        doppler_path=None if dop is None else dop[1],
        degraded=degraded,
        reason=reason,
    )


def _poll_grid(
    start: datetime, end: datetime, poll_interval_min: float,
) -> Iterable[datetime]:
    """Poll instants in ``[start, end]``, aligned to midnight UTC."""
    step = timedelta(minutes=poll_interval_min)
    midnight = start.replace(hour=0, minute=0, second=0, microsecond=0)
    n = int((start - midnight) / step)
    t = midnight + n * step
    while t < start:
        t += step
    while t <= end:
        yield t
        t += step


def decision_instants(
    start: datetime,
    end: datetime,
    frames_by_type: Mapping[str, Mapping[datetime, Path]],
    *,
    policy: str = POLICY_FULLRANGE,
    lag: ProductLag = DEFAULT_LAG,
    poll_interval_min: float = DEFAULT_POLL_INTERVAL_MIN,
    max_age_min: float = DEFAULT_MAX_ANCHOR_AGE_MIN,
) -> list[AnchorSelection]:
    """One selection per *new* anchor, over the poll grid in ``[start, end]``.

    The service polls every 5 min but recomputes only when a new frame has
    arrived (the runtime's no-new-frame fast path). So a poll that finds
    the anchor it already had is not a decision instant, and every
    selection returned here stands on a frame no earlier selection stood
    on — which also keeps ``radar_ts`` unique, the key every downstream
    consumer deduplicates decision rows on.

    With the measured lags this yields **one instant per ten minutes under
    both policies**, at the same wall-clock instants, differing only in
    which frame is underneath: fullRange at age 15 min, doppler at age 10.
    The two products publish together, so "freshest" does not double the
    decision rate — it makes each decision five minutes fresher.

    The first poll in ``[start, end]`` always yields a cycle: there is no
    previous anchor to compare it against, so it stands on whatever was
    already published and its frame is older than the steady state's. A
    day plan starts its walk at midnight, where that instant anchors on
    the previous day's last frame and is dropped by the day it belongs to.
    """
    start, end = _as_utc(start), _as_utc(end)
    if poll_interval_min > 0:
        candidates: Iterable[datetime] = _poll_grid(start, end, poll_interval_min)
    else:
        # No poll grid: the cycle is assumed to see a frame the instant it
        # is published. This is what the flat --frame-age-min model means,
        # and it is the only way to reproduce a pre-L3 run to the minute.
        eligible = (
            (SCAN_TYPE_FULL_RANGE,) if policy == POLICY_FULLRANGE
            else (SCAN_TYPE_FULL_RANGE, SCAN_TYPE_DOPPLER)
        )
        published: set[datetime] = set()
        for scan_type in eligible:
            for ts in (frames_by_type.get(scan_type) or {}):
                published.add(
                    _as_utc(ts) + timedelta(minutes=lag.minutes(scan_type))
                )
        candidates = sorted(t for t in published if start <= t <= end)
    out: list[AnchorSelection] = []
    seen: set[tuple[str, datetime]] = set()
    for now in candidates:
        selection = select_anchor(
            now, frames_by_type, policy=policy, lag=lag, max_age_min=max_age_min,
        )
        if selection is None or selection.key in seen:
            continue
        seen.add(selection.key)
        out.append(selection)
    return out


# ---------------------------------------------------------------------------
# The anchor field
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnchorField:
    """The dBZ field one cycle advects, and where it came from.

    ``doppler_mask`` is True exactly where the doppler frame supplied the
    value — inside its 120 km range, ``nodata`` excluded. It is what a
    later diagnostic (or the runtime's quality report) needs to say how
    much of a served field was the fresh product.
    """

    dbz: np.ndarray
    doppler_mask: np.ndarray
    scan_type: str
    anchor_ts: datetime
    fill_ts: datetime | None


def anchor_field(
    anchor_dbz: np.ndarray,
    *,
    scan_type: str,
    anchor_ts: datetime,
    fill_dbz: np.ndarray | None = None,
    fill_ts: datetime | None = None,
    harmonisation: Mapping[str, Any] | None = None,
    distance_km: np.ndarray | None = None,
    season: str | None = None,
) -> AnchorField:
    """The per-pixel freshest covering frame, as one dBZ grid.

    A fullRange anchor is returned untouched (a copy), because outside
    doppler's range there is nothing fresher and inside it the fullRange
    frame IS the anchor. A doppler anchor is harmonised onto fullRange's
    distribution first — never used raw, see the module docstring — and
    then, wherever doppler reports ``nodata`` (beyond 120 km, or a blanked
    sector), the value comes from the newest fullRange frame instead.

    ``undetect`` (``-inf``, an observed dry pixel) counts as coverage:
    doppler saying "nothing there" is data, and replacing it with a
    five-minute-older fullRange reading would be a step backwards.
    ``nodata`` (NaN) does not. Where neither product covers a pixel the
    result is NaN, exactly as a fullRange-only cycle already produces
    outside 240 km.
    """
    src = np.asarray(anchor_dbz, dtype=np.float32)
    if scan_type == SCAN_TYPE_FULL_RANGE:
        return AnchorField(
            dbz=src.astype(np.float32, copy=True),
            doppler_mask=np.zeros(src.shape, dtype=bool),
            scan_type=scan_type,
            anchor_ts=anchor_ts,
            fill_ts=None,
        )
    if scan_type != SCAN_TYPE_DOPPLER:
        raise ValueError(f"cannot anchor on scan type {scan_type!r}")
    if harmonisation is None or distance_km is None:
        raise ValueError(
            "a doppler anchor needs the L2 harmonisation table and a radar "
            "distance grid; raw doppler is 20-30 % short of fullRange's echo "
            "area at 20 dBZ and must never enter the chain unmapped"
        )
    mapped = apply_harmonisation(
        src, distance_km, season or season_of(anchor_ts), harmonisation,
    )
    field = AnchorField(
        dbz=mapped.astype(np.float32, copy=False),
        doppler_mask=~np.isnan(mapped),
        scan_type=scan_type,
        anchor_ts=anchor_ts,
        fill_ts=None,
    )
    return field if fill_dbz is None else fill_uncovered(field, fill_dbz, fill_ts)


def fill_uncovered(
    field: AnchorField,
    fill_dbz: np.ndarray | None,
    fill_ts: datetime | None = None,
) -> AnchorField:
    """Take a doppler field's uncovered pixels from an older, wider frame.

    A no-op on anything but a doppler field: a fullRange anchor has no
    "uncovered" pixels to take from anywhere, and its mask is all-False,
    so composing it against a fill would replace the whole frame.
    """
    if fill_dbz is None or field.scan_type != SCAN_TYPE_DOPPLER:
        return field
    fill = np.asarray(fill_dbz, dtype=np.float32)
    if fill.shape != field.dbz.shape:
        raise ValueError(
            f"fill frame {fill.shape} does not match anchor {field.dbz.shape}"
        )
    return replace(
        field,
        dbz=np.where(field.doppler_mask, field.dbz, fill).astype(np.float32),
        fill_ts=fill_ts,
    )


# ---------------------------------------------------------------------------
# The history: three frames of one product
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HistorySelection:
    """The input triple for the flow and the cascade, oldest frame first.

    ``selection`` is the anchor these frames belong to — not necessarily
    the one asked for: when a doppler anchor has no doppler triple behind
    it, the whole instant falls back to the fullRange policy and this
    carries the re-anchored selection, with ``fallback`` set.

    ``fill_timestamps`` / ``fill_paths`` are the fullRange partner of each
    doppler frame (the ``:x0`` frame five minutes earlier), used only by
    :data:`HISTORY_FILLED`. ``None`` in a slot means no partner is
    archived; that frame then keeps doppler's own coverage.
    """

    scan_type: str
    timestamps: tuple[datetime, ...]
    paths: tuple[Path, ...]
    fill_timestamps: tuple[datetime | None, ...]
    fill_paths: tuple[Path | None, ...]
    selection: AnchorSelection
    fallback: bool = False
    reason: str | None = None

    @property
    def step_min(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        return (
            self.timestamps[-1] - self.timestamps[-2]
        ).total_seconds() / 60.0


def _lookup(
    frames: Mapping[datetime, Path] | None,
    target: datetime,
    tolerance_s: float,
) -> tuple[datetime, Path] | None:
    """The archived frame at ``target``, within ``tolerance_s``."""
    if not frames:
        return None
    exact = frames.get(target)
    if exact is not None:
        return (target, Path(exact))
    best: tuple[float, datetime, Path] | None = None
    for ts, path in frames.items():
        ts = _as_utc(ts)
        delta = abs((ts - target).total_seconds())
        if delta <= tolerance_s and (best is None or delta < best[0]):
            best = (delta, ts, Path(path))
    return None if best is None else (best[1], best[2])


def _triple(
    frames: Mapping[datetime, Path] | None,
    newest: datetime,
    n_frames: int,
    step_min: float,
    tolerance_s: float,
) -> tuple[list[datetime], list[Path]] | None:
    stamps: list[datetime] = []
    paths: list[Path] = []
    for i in range(n_frames - 1, -1, -1):
        found = _lookup(
            frames, newest - timedelta(minutes=step_min * i), tolerance_s,
        )
        if found is None:
            return None
        stamps.append(found[0])
        paths.append(found[1])
    return stamps, paths


def anchor_history(
    selection: AnchorSelection,
    frames_by_type: Mapping[str, Mapping[datetime, Path]],
    *,
    n_frames: int = 3,
    step_min: float = DEFAULT_STEP_MIN,
    tolerance_s: float = DEFAULT_TOLERANCE_S,
) -> HistorySelection | None:
    """The same-type input triple behind ``selection``, or ``None``.

    Same-type, always: DECIDE-3 puts the flow on same-product pairs and
    DECIDE-4 keeps the STEPS timestep at 10 minutes, so a doppler anchor
    is backed by doppler frames at ``T``, ``T-10``, ``T-20`` and a
    fullRange anchor by fullRange frames at the same offsets. The two
    products are never interleaved along the time axis.

    When a doppler anchor has no complete doppler triple — a gap, or the
    start of an archive month that has only one product — the instant
    falls back to the **fullRange policy**: re-anchored on the newest
    fullRange frame (five minutes older, ``frame_age_min`` recomputed) and
    backed by the fullRange triple. The caller counts those; a run where
    they are common has not measured the candidate.

    ``None`` means not even the fullRange triple is there, and the caller
    should record the instant as an error rather than invent a dry frame.
    """
    frames = frames_by_type.get(selection.scan_type)
    found = _triple(frames, selection.timestamp, n_frames, step_min, tolerance_s)
    if found is not None:
        stamps, paths = found
        return HistorySelection(
            scan_type=selection.scan_type,
            timestamps=tuple(stamps),
            paths=tuple(paths),
            fill_timestamps=tuple(
                _fill_partner(frames_by_type, ts, step_min, tolerance_s)[0]
                if selection.scan_type == SCAN_TYPE_DOPPLER else None
                for ts in stamps
            ),
            fill_paths=tuple(
                _fill_partner(frames_by_type, ts, step_min, tolerance_s)[1]
                if selection.scan_type == SCAN_TYPE_DOPPLER else None
                for ts in stamps
            ),
            selection=selection,
        )
    if selection.scan_type != SCAN_TYPE_DOPPLER:
        return None
    reason = (
        f"no doppler triple at {selection.timestamp:%Y-%m-%dT%H:%MZ}; "
        "fell back to the fullRange policy"
    )
    degraded = selection.as_fullrange(reason=reason)
    if degraded is None:
        return None
    found = _triple(
        frames_by_type.get(SCAN_TYPE_FULL_RANGE), degraded.timestamp,
        n_frames, step_min, tolerance_s,
    )
    if found is None:
        return None
    stamps, paths = found
    return HistorySelection(
        scan_type=SCAN_TYPE_FULL_RANGE,
        timestamps=tuple(stamps),
        paths=tuple(paths),
        fill_timestamps=tuple(None for _ in stamps),
        fill_paths=tuple(None for _ in stamps),
        selection=degraded,
        fallback=True,
        reason=reason,
    )


def _fill_partner(
    frames_by_type: Mapping[str, Mapping[datetime, Path]],
    doppler_ts: datetime,
    step_min: float,
    tolerance_s: float,
) -> tuple[datetime | None, Path | None]:
    """The fullRange frame that covers what doppler at ``doppler_ts`` misses.

    The one five minutes earlier: the newest fullRange frame that is not
    *after* the doppler frame, so the fill is stale by half a step and
    never by more than one.
    """
    frames = frames_by_type.get(SCAN_TYPE_FULL_RANGE)
    if not frames:
        return (None, None)
    best: tuple[datetime, Path] | None = None
    oldest = doppler_ts - timedelta(minutes=step_min, seconds=tolerance_s)
    for ts, path in frames.items():
        ts = _as_utc(ts)
        if ts > doppler_ts or ts < oldest:
            continue
        if best is None or ts > best[0]:
            best = (ts, Path(path))
    return (None, None) if best is None else best


# ---------------------------------------------------------------------------
# Per-process caches: the two things that must not be rebuilt per frame
# ---------------------------------------------------------------------------

#: ``grid key -> distance-to-nearest-radar grid``. Building one for the
#: national grid is ~3.4 M pyproj inverse transforms; the grid never
#: moves, so a worker builds it once and every frame reuses it (~14 MB).
_DISTANCE_CACHE: dict[tuple, np.ndarray] = {}

#: ``path -> parsed harmonisation payload``, read once per worker.
_HARMONISATION_CACHE: dict[str, dict[str, Any]] = {}


def distance_km_grid(composite: Any) -> np.ndarray:
    """Memoised :func:`product_pairs.radar_distance_grid` for one geometry."""
    key = (
        tuple(composite.reflectivity_dbz.shape),
        composite.projection,
        float(composite.xscale_m),
        float(composite.yscale_m),
        tuple(sorted(composite.corners_lonlat.items())),
    )
    grid = _DISTANCE_CACHE.get(key)
    if grid is None:
        grid = radar_distance_grid(composite)
        _DISTANCE_CACHE[key] = grid
    return grid


def load_harmonisation(path: str | Path) -> dict[str, Any]:
    """Read and validate ``doppler_harmonisation.json``, once per worker.

    The schema version is checked here as well as inside
    :func:`product_pairs.apply_harmonisation`, so a run configured with a
    table it cannot apply fails at startup rather than 4,000 frames in.
    """
    key = str(path)
    cached = _HARMONISATION_CACHE.get(key)
    if cached is not None:
        return cached
    payload = json.loads(Path(path).read_text())
    version = int(payload.get("schema_version", 0))
    if version != HARMONISATION_SCHEMA_VERSION:
        raise ValueError(
            f"{path}: harmonisation schema version {version} != "
            f"{HARMONISATION_SCHEMA_VERSION}; refusing to run"
        )
    if not (payload.get("tables") or {}):
        raise ValueError(f"{path}: harmonisation table is empty")
    _HARMONISATION_CACHE[key] = payload
    return payload


def harmonisation_stamp(
    path: str | Path | None, payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Provenance of the map a run applied, for ``summary.json``.

    Enough to tell two runs apart without re-reading either table: the
    path, the schema version, when it was fitted, how many cells it holds
    and a digest of the file.
    """
    if path is None:
        return {"path": None}
    path = Path(path)
    out: dict[str, Any] = {"path": str(path)}
    try:
        raw = path.read_bytes()
    except OSError as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    out["sha256"] = hashlib.sha256(raw).hexdigest()[:16]
    out["bytes"] = len(raw)
    data = payload if payload is not None else json.loads(raw)
    meta = data.get("meta") or {}
    out["schema_version"] = data.get("schema_version")
    out["fitted_at"] = meta.get("generated")
    out["n_fit_triples"] = meta.get("n_fit_triples")
    out["tables"] = sorted((data.get("tables") or {}).keys())
    return out


def counts_template() -> dict[str, int]:
    """Zeroed anchor counters, summed across day workers by the caller."""
    return {
        "instants": 0,
        "doppler_anchored": 0,
        "fullrange_anchored": 0,
        "degraded": 0,
        "history_fallback": 0,
        "no_anchor": 0,
        "no_history": 0,
    }


def count_selection(counts: dict[str, int], history: HistorySelection) -> None:
    """Fold one planned cycle into the run's anchor counters."""
    selection = history.selection
    counts["instants"] += 1
    if selection.scan_type == SCAN_TYPE_DOPPLER:
        counts["doppler_anchored"] += 1
    else:
        counts["fullrange_anchored"] += 1
    if selection.degraded and not history.fallback:
        counts["degraded"] += 1
    if history.fallback:
        counts["history_fallback"] += 1


def sum_counts(into: dict[str, int], add: Mapping[str, int] | None) -> dict[str, int]:
    """Accumulate one worker's counters into the run total."""
    if add:
        for key, value in add.items():
            into[key] = into.get(key, 0) + int(value)
    return into


def frame_age_stats(ages: Sequence[float]) -> dict[str, float | None]:
    """Observed anchor ages over a run — the freshness the run actually got."""
    if not ages:
        return {"n": 0, "mean": None, "p50": None, "min": None, "max": None}
    arr = np.asarray(ages, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": round(float(arr.mean()), 2),
        "p50": round(float(np.median(arr)), 2),
        "min": round(float(arr.min()), 2),
        "max": round(float(arr.max()), 2),
    }
