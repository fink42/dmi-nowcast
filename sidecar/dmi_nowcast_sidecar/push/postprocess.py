"""The fitted post-processing model, as the running service reads it (H-P).

The served ``p_rain[L]`` is a per-lead isotonic map of the STEPS ensemble
fraction and nothing else, and with 16 members sharing one motion field
that fraction saturates: a 13 mm/h band 13 km upstream and a drizzle edge
that will die before it arrives both read "1.0 raw" and both come out at
the same calibrated ≈ 0.55. A model on the features the cycle already has
— the raw fraction at every lead, how much rain sits upstream along the
flow and how far away, how fast it moves, the season, the hour, the frame
age, the distance to a radar — separates them: leave-one-month-out at the
gauges it beat the curve by ΔBSS +0.14…+0.19 at every lead and every
season, and the shipped warning rule by ΔF1 +0.03…+0.06 at three of four
leads (``archive/l3_and_postprocess_20260911/``).

Two things live here.

:class:`PostprocessTable`
    The read side of ``postprocess.json``, following
    :class:`~dmi_nowcast_sidecar.push.thresholds.ThresholdTable` exactly —
    the file is replaced under a running process (by the nightly refit on
    the private instance, by the ``sync`` task on the public one), so a
    ``(mtime_ns, size)`` stamp is compared before each use and the
    document re-read only when it moved; a writer calls
    :meth:`~PostprocessTable.note_changed` so a same-second rewrite is
    still picked up; and every failure mode — missing file, unreadable
    file, a schema version this build does not know, JSON that is not a
    model — degrades to **inactive** plus one log line. Inactive means the
    engine decides on the curve-calibrated number, which is what it did
    before this existed.

:class:`CyclePostprocess`
    One cycle's answer for the points it serves: the feature row and the
    post-processed probability per lead, computed ONCE for the union of
    every subscription's point, the gauge-eval stations and home, and
    published beside the national products so the push fan-out and the
    scoreboard read the same numbers off the same frame.

The features themselves come from ``dmi_nowcast_core.postprocess`` — the
same extraction and the same row assembler the offline replay uses, so a
model fitted on replay rows is applied to live rows of identical shape.
``sidecar/tests/test_push_postprocess.py`` pins that parity.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import structlog

from dmi_nowcast_core import postprocess as core_postprocess
from dmi_nowcast_core.postprocess import PostprocessModel

_log = structlog.get_logger(__name__)

#: Where a probability the engine used came from. ``"postprocess"`` is the
#: fitted model, ``"curve"`` the served isotonic calibration — the
#: fallback whenever the model has nothing to say about a point.
ProbabilitySource = str

#: Coordinates are rounded to this many decimals before they are used as a
#: key. ~1e-6 deg is about 0.1 m: far finer than the 500 m composite pixel,
#: so two rows that round together really are the same pixel, and far
#: coarser than float noise, so a subscription's stored coordinate and the
#: same coordinate read back from SQLite cannot miss each other.
_KEY_DECIMALS = 6


def point_key(lat: float, lon: float) -> tuple[float, float]:
    """The identity of a served point: its rounded coordinate, nothing else.

    Deliberately NOT the subscription's endpoint or the station's id. The
    cycle computes features for a set of *places*; it has no business
    knowing whose they are, and an endpoint is a bearer capability that
    must not travel outside the push store (see ``push.service``).
    """
    return (round(float(lat), _KEY_DECIMALS), round(float(lon), _KEY_DECIMALS))


def _file_stamp(path: Path) -> tuple[int, int] | None:
    """``(mtime_ns, size)``, or None when the file is not there.

    Both halves, exactly as ``compute._file_stamp`` and
    ``push.thresholds._file_stamp``: a same-second rewrite on a filesystem
    with coarse timestamps would otherwise look unchanged.
    """
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


class PostprocessTable:
    """The fitted model, hot-reloaded from one file.

    Cheap to construct and does no I/O until :meth:`maybe_reload` (or
    :meth:`load`) is called, so an instance on a deployment that has never
    fitted one costs nothing.
    """

    def __init__(self, path: Path | str | None) -> None:
        self.path = None if path is None else Path(path)
        self._model: PostprocessModel | None = None
        self._stamp: tuple[int, int] | None = None
        self._loaded = False
        self._dirty = False

    # -- state -------------------------------------------------------------

    @property
    def model(self) -> PostprocessModel | None:
        """The loaded model, or None when there is no usable one."""
        return self._model

    @property
    def active(self) -> bool:
        """Can this table answer for anything at all?"""
        return self._model is not None and bool(self._model.models)

    @property
    def loaded(self) -> bool:
        """Has a load been attempted at all (successful or not)?"""
        return self._loaded

    @property
    def fitted_at_utc(self) -> str | None:
        """``fitted_at_utc`` of the loaded model; None without one."""
        if self._model is None:
            return None
        return self._model.fitted_at_utc or None

    @property
    def leads(self) -> list[int]:
        """The leads the loaded model carries a fit for."""
        if self._model is None:
            return []
        return sorted(int(lead) for lead in self._model.models)

    @property
    def design_leads(self) -> list[int]:
        """The leads whose raw ensemble fraction the design reads.

        The cycle needs this to know which ``raw_frac_<lead>`` columns to
        sample: a model fitted on five design leads scores nothing useful
        if the runtime only fills three of them.
        """
        if self._model is None:
            return []
        return sorted(int(lead) for lead in self._model.design_leads)

    # -- loading -----------------------------------------------------------

    def load(self) -> PostprocessModel | None:
        """Read and validate the file. Never raises; logs one line."""
        self._loaded = True
        self._dirty = False
        self._stamp = None if self.path is None else _file_stamp(self.path)
        self._model = None
        if self.path is None:
            _log.info("push_postprocess_missing", path=None)
            return None
        try:
            text = self.path.read_text(encoding="utf-8")
        except OSError:
            # Not an error: a deployment that has not fitted yet, or the
            # public instance before its first sync. The engine falls back
            # to the curve, which is the rule that shipped.
            _log.info("push_postprocess_missing", path=str(self.path))
            return None
        try:
            model = PostprocessModel.loads(text)
        except Exception as exc:  # noqa: BLE001 — every way a file can be junk
            _log.warning(
                "push_postprocess_unusable",
                path=str(self.path),
                error=f"{type(exc).__name__}: {exc}",
                note="falling back to the curve-calibrated probability",
            )
            return None
        if not model.models:
            _log.warning(
                "push_postprocess_unusable",
                path=str(self.path),
                error="the document carries no fitted lead",
            )
            return None
        self._model = model
        _log.info(
            "push_postprocess_loaded",
            path=str(self.path),
            leads=self.leads,
            design_leads=self.design_leads,
            fitted_at=self.fitted_at_utc,
        )
        return model

    def note_changed(self) -> None:
        """Ask for a re-read at the next :meth:`maybe_reload`.

        The sync task and the nightly refit call this after writing the
        file. It only sets a flag — like ``ThresholdTable.note_changed``,
        the swap happens at the one moment it is safe, which here is the
        start of a cycle rather than the middle of one.
        """
        self._dirty = True

    def maybe_reload(self) -> bool:
        """Re-read when the file moved (or a writer asked). True if it did.

        One ``stat`` per call, a JSON parse only when the stamp changed.
        """
        if self.path is None:
            if not self._loaded:
                self.load()
                return True
            return False
        stamp = _file_stamp(self.path)
        if self._loaded and not self._dirty and stamp == self._stamp:
            return False
        if self._loaded:
            _log.info(
                "push_postprocess_changed_on_disk",
                path=str(self.path), was=self._stamp, now=stamp,
            )
        self.load()
        return True

    # -- reading -----------------------------------------------------------

    def predict_table(
        self, features: Mapping[str, Any],
    ) -> dict[int, np.ndarray]:
        """``{lead: array}`` for a whole feature table. ``{}`` when inactive.

        One design matrix for every lead, built once — it does not depend
        on the lead being predicted. Never raises: a model that cannot
        score the columns it was handed leaves the engine on the curve,
        which is a worse probability but a working service.
        """
        if self._model is None:
            return {}
        try:
            return {
                int(lead): np.asarray(values, dtype=np.float64)
                for lead, values in self._model.predict(features).items()
            }
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "push_postprocess_predict_failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            return {}

    def predict(self, lead: int, features: Mapping[str, Any]) -> float | None:
        """One point, one lead → the post-processed probability, or None.

        ``features`` is the stored feature row (scalars, as
        ``postprocess.feature_row`` writes it); it is lifted to
        single-element columns so exactly the same transform runs as in
        the fit. ``None`` means the model has nothing to say — inactive,
        an unfitted lead, or a row it could not score.
        """
        if self._model is None or int(lead) not in self._model.models:
            return None
        columns = {
            name: np.asarray([value], dtype=np.float64)
            if not isinstance(value, str) else np.asarray([value])
            for name, value in features.items()
            if value is not None
        }
        if not columns:
            return None
        values = self.predict_table(columns).get(int(lead))
        if values is None or values.size == 0:
            return None
        out = float(values[0])
        return out if math.isfinite(out) else None


# ---------------------------------------------------------------------------
# One cycle's answer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CyclePostprocess:
    """Features and post-processed probabilities for one cycle's points.

    Built once per radar observation, inside the cycle worker, and
    published as one immutable object so a reader on another thread can
    never pair one frame's features with another frame's probabilities.
    The arrays are tiny — a few hundred points by twenty float32 columns —
    and hold no reference to any grid.

    ``radar_ts_utc`` is what a consumer checks before trusting it: the
    push fan-out and the gauge scoreboard both refuse to attribute one
    frame's numbers to another frame's timestamp.
    """

    radar_ts_utc: datetime
    generated_at_utc: datetime
    #: Rounded ``(lat, lon)`` per row, in row order.
    keys: tuple[tuple[float, float], ...]
    #: One stored feature row per point, in row order.
    rows: tuple[dict[str, Any], ...]
    #: lead → one probability (or None) per point, in row order.
    p_post: dict[int, tuple[float | None, ...]]
    #: ``fitted_at_utc`` of the model that produced ``p_post``; None when
    #: no model was active and ``p_post`` is empty.
    fitted_at_utc: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "_index", {key: i for i, key in enumerate(self.keys)},
        )

    @property
    def active(self) -> bool:
        """Did a model actually score this cycle?"""
        return bool(self.p_post)

    @property
    def leads(self) -> tuple[int, ...]:
        return tuple(sorted(self.p_post))

    def index_of(self, lat: float, lon: float) -> int | None:
        """Row index of a point, or None when the cycle did not serve it."""
        return getattr(self, "_index").get(point_key(lat, lon))

    def features(self, lat: float, lon: float) -> dict[str, Any] | None:
        """This point's stored feature row, or None when it was not served."""
        index = self.index_of(lat, lon)
        return None if index is None else self.rows[index]

    def probability(
        self, lat: float, lon: float, lead: int,
    ) -> float | None:
        """Post-processed P(rain within ``lead``) at a point.

        ``None`` whenever the model cannot speak for it — no model, a lead
        it was not fitted for, a point this cycle did not serve, or a row
        it could not score. The caller falls back to the served
        curve-calibrated probability and says so, per observation.
        """
        values = self.p_post.get(int(lead))
        if values is None:
            return None
        index = self.index_of(lat, lon)
        if index is None:
            return None
        return values[index]

    def columns(self, lat: float, lon: float) -> dict[str, Any]:
        """Everything about a point, as decision-row columns.

        The feature columns plus ``p_post_<lead>`` — exactly what
        ``station_eval`` appends to a decision row so the nightly fit sees
        live rows on the same footing as the replay's. Empty for a point
        the cycle did not serve.
        """
        index = self.index_of(lat, lon)
        if index is None:
            return {}
        out = dict(self.rows[index])
        for lead, values in sorted(self.p_post.items()):
            out[core_postprocess.post_column(lead)] = values[index]
        return out


def build_cycle_postprocess(
    table: PostprocessTable,
    *,
    radar_ts_utc: datetime,
    generated_at_utc: datetime,
    keys: Sequence[tuple[float, float]],
    grid_features: Mapping[str, np.ndarray],
    raw_fractions: Mapping[int, Sequence[float | None]],
    shared: Sequence[Mapping[str, Any]],
    station_radar_km: Sequence[float],
    leads: Sequence[int],
    season: str,
    hour_utc: int,
    frame_age_min: float,
) -> CyclePostprocess:
    """Assemble one cycle's feature rows and score them.

    ``grid_features`` is :func:`postprocess.station_features`' output for
    the whole point list, ``raw_fractions[lead][i]`` the UNcalibrated
    ensemble fraction already read at point *i*'s product pixel, and
    ``shared[i]`` the three columns the design reads that the decision
    schema already carries (``observed_mm_h``, ``eta_min``,
    ``intensity_mm_h``). All three are in the same row order as ``keys``.

    ``shared`` feeds the model but is NOT written into the stored feature
    row: those columns have a writer already, and duplicating them would
    give a row two sources for one number.

    Scoring is one design matrix for the whole table, not one per point:
    at a few hundred points the per-point path would be a few hundred
    numpy calls per cycle for the same answer.
    """
    wanted = [int(lead) for lead in leads]
    blank: tuple[float | None, ...] = (None,) * len(keys)
    fractions = {lead: raw_fractions.get(lead, blank) for lead in wanted}
    rows = tuple(
        core_postprocess.feature_row(
            grid_features,
            index,
            raw_fractions={lead: fractions[lead][index] for lead in wanted},
            leads=wanted,
            season=season,
            hour_utc=hour_utc,
            frame_age_min=frame_age_min,
            station_radar_km=station_radar_km[index],
        )
        for index in range(len(keys))
    )
    p_post: dict[int, tuple[float | None, ...]] = {}
    fitted_at: str | None = None
    if rows and table.active:
        columns = _columns_of([
            {**row, **dict(shared[index])} for index, row in enumerate(rows)
        ])
        predicted = table.predict_table(columns)
        if predicted:
            fitted_at = table.fitted_at_utc
            p_post = {
                int(lead): tuple(
                    core_postprocess.finite_or_none(value) for value in values
                )
                for lead, values in sorted(predicted.items())
            }
    return CyclePostprocess(
        radar_ts_utc=radar_ts_utc,
        generated_at_utc=generated_at_utc,
        keys=tuple(keys),
        rows=rows,
        p_post=p_post,
        fitted_at_utc=fitted_at,
    )


def _columns_of(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Feature rows → the column dict ``build_design`` reads.

    ``None`` becomes NaN, which is what "missing" means everywhere in the
    design; ``season`` stays a string column because the transform one-hots
    it. A column no row carries is simply absent, and ``build_design``
    treats an absent column exactly like a column of missing values.
    """
    names: list[str] = []
    for row in rows:
        for name in row:
            if name not in names:
                names.append(name)
    out: dict[str, Any] = {}
    for name in names:
        if name == "season":
            out[name] = np.asarray(
                [str(row.get(name) or "") for row in rows], dtype="<U8",
            )
        else:
            out[name] = np.asarray(
                [
                    np.nan if row.get(name) is None else float(row[name])
                    for row in rows
                ],
                dtype=np.float64,
            )
    return out


__all__ = [
    "CyclePostprocess",
    "PostprocessTable",
    "ProbabilitySource",
    "build_cycle_postprocess",
    "point_key",
]
