"""The fitted threshold table, as the running service reads it (Phase G).

A subscriber chooses a horizon. The probability that horizon warns at is
not theirs to choose: it is fitted nightly by
``dmi_nowcast_sidecar.threshold_sweep`` against the rain gauges and lands
in a small JSON file (``dmi_nowcast_core.push_thresholds`` is its
contract). This class is the read side of that file.

It follows the national-curves pattern in ``compute.CycleEngine`` exactly,
for the same reasons:

* the file is replaced under a running process — by the nightly fit on the
  private instance, by the ``sync`` task on the public one — so a
  ``(mtime_ns, size)`` stamp is compared before each fan-out and the
  document is re-read only when it moved;
* the sync task calls :meth:`note_changed` after writing, so a rewrite
  within the same second (or on a filesystem with coarse timestamps) is
  still picked up;
* and every failure mode degrades to the fallback rather than raising.
  Missing file, unreadable file, JSON that is not a threshold document —
  all of them mean "not fitted yet", which is the rule the site shipped
  before the fit existed, plus one log line.

:meth:`effective` answers with the number *and* where it came from, so the
subscribe response and the ``push_eval`` log line can both say which rule
a subscriber is actually on.
"""
from __future__ import annotations

from pathlib import Path

import structlog

from dmi_nowcast_core.push_thresholds import (
    DEFAULT_FALLBACK_THRESHOLD_PCT,
    effective_threshold,
    lead_pick,
    load_thresholds,
    onset_rule,
)

_log = structlog.get_logger(__name__)

#: Where :meth:`ThresholdTable.effective` says a number came from.
#: ``"table"`` is a fitted pick, ``"fallback"`` is the shipped default for
#: a lead the table cannot speak for. (A third source, ``"override"``,
#: belongs to the subscription rather than the table — see
#: ``push.service``.)
Source = str


def _file_stamp(path: Path) -> tuple[int, int] | None:
    """``(mtime_ns, size)``, or None when the file is not there.

    Both halves, exactly as ``compute._file_stamp``: a same-second rewrite
    on a filesystem with coarse timestamps would otherwise look unchanged.
    """
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


class ThresholdTable:
    """The fitted horizon → threshold table, hot-reloaded from one file.

    Cheap to construct and does no I/O until :meth:`maybe_reload` (or
    :meth:`load`) is called, so an instance with push disabled costs
    nothing.
    """

    def __init__(self, path: Path | str | None) -> None:
        self.path = None if path is None else Path(path)
        self._doc: dict | None = None
        self._stamp: tuple[int, int] | None = None
        self._loaded = False
        self._dirty = False

    # -- state -------------------------------------------------------------

    @property
    def document(self) -> dict | None:
        """The loaded document, or None when there is no usable one."""
        return self._doc

    @property
    def loaded(self) -> bool:
        """Has a load been attempted at all (successful or not)?"""
        return self._loaded

    @property
    def fitted_at_utc(self) -> str | None:
        """``fitted_at_utc`` of the loaded table; None without one."""
        if not isinstance(self._doc, dict):
            return None
        value = self._doc.get("fitted_at_utc")
        return value if isinstance(value, str) else None

    @property
    def fallback_threshold_pct(self) -> int:
        """The percent every unfitted lead warns at."""
        if isinstance(self._doc, dict):
            candidate = self._doc.get("fallback_threshold_pct")
            if (
                not isinstance(candidate, bool)
                and isinstance(candidate, int)
                and 0 < candidate < 100
            ):
                return candidate
        return DEFAULT_FALLBACK_THRESHOLD_PCT

    @property
    def leads(self) -> list[int]:
        """The leads the loaded table carries a usable pick for."""
        if not isinstance(self._doc, dict):
            return []
        leads = self._doc.get("leads")
        if not isinstance(leads, dict):
            return []
        out: list[int] = []
        for key in leads:
            if isinstance(key, str) and key.isdigit():
                if lead_pick(self._doc, key) is not None:
                    out.append(int(key))
        return sorted(out)

    # -- loading -----------------------------------------------------------

    def load(self) -> dict | None:
        """Read and validate the file. Never raises; logs one line."""
        self._loaded = True
        self._dirty = False
        self._stamp = None if self.path is None else _file_stamp(self.path)
        doc = load_thresholds(self.path)
        self._doc = doc
        if doc is None:
            _log.info(
                "push_thresholds_missing",
                path=None if self.path is None else str(self.path),
                fallback_threshold_pct=DEFAULT_FALLBACK_THRESHOLD_PCT,
            )
        else:
            _log.info(
                "push_thresholds_loaded",
                path=str(self.path),
                leads=self.leads,
                fitted_at=self.fitted_at_utc,
                fallback_threshold_pct=self.fallback_threshold_pct,
            )
        return doc

    def note_changed(self) -> None:
        """Ask for a re-read at the next :meth:`maybe_reload`.

        The sync task and the nightly fit call this after writing the
        file. It only sets a flag — like ``note_curves_changed``, the swap
        happens at the one moment it is safe, which here is the start of a
        fan-out rather than the middle of one.
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
                "push_thresholds_changed_on_disk",
                path=str(self.path),
                was=self._stamp,
                now=stamp,
            )
        self.load()
        return True

    # -- reading -----------------------------------------------------------

    def effective(self, lead_min: object) -> tuple[int, Source]:
        """``(percent, source)`` for one horizon. Total, never raises.

        ``source`` is ``"table"`` when the loaded document carries a
        usable pick for this lead and ``"fallback"`` otherwise — a lead
        the fit skipped, a lead with too little evidence, or no table at
        all. The number is always one a subscription can be evaluated
        against.
        """
        doc = self._doc
        threshold = effective_threshold(doc, lead_min)
        try:
            key = str(int(lead_min))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return threshold, "fallback"
        source: Source = "table" if lead_pick(doc, key) is not None else "fallback"
        return threshold, source

    def onset_rule(self, lead_min: object) -> tuple[int, int] | None:
        """``(onset_threshold_pct, single_threshold_pct)`` when this lead is
        on the onset AND rule (S11), else None. Total, never raises."""
        return onset_rule(self._doc, lead_min)

    def headline(
        self, lead_min: object, *, onset_active: bool = True,
    ) -> tuple[int, Source, int | None]:
        """``(percent, source, onset_percent)`` a subscriber is SHOWN.

        On the single rule it is :meth:`effective`. On the onset AND rule
        the one percent the site can show is the onset threshold — the
        p_post half may be 0 ("onset alone decides"), which as a headline
        would read "warns at 0 %". With no onset model loaded
        (``onset_active`` False) every observation falls back to the single
        rule, so that is what is shown.
        """
        threshold, source = self.effective(lead_min)
        rule = self.onset_rule(lead_min)
        if rule is None:
            return threshold, source, None
        onset, single = rule
        if not onset_active:
            return single, source, None
        return onset, source, onset

    def snapshot(
        self, leads: object, *, onset_active: bool = True,
    ) -> dict[str, dict]:
        """``{"20": {"threshold_pct": 45, "source": "table"}, ...}``.

        What ``GET /api/push/options`` serves: every offered horizon with
        the rule it is on right now, so the browser never has to guess and
        an operator can diff what is served against what was fitted. A lead
        on the onset AND rule (S11) also carries ``onset_threshold_pct`` and
        ``post_threshold_pct``; its ``threshold_pct`` is :meth:`headline`'s.
        """
        out: dict[str, dict] = {}
        for lead in leads:  # type: ignore[union-attr]
            threshold, source, onset = self.headline(
                lead, onset_active=onset_active,
            )
            row: dict = {"threshold_pct": threshold, "source": source}
            if onset is not None:
                row["onset_threshold_pct"] = onset
                row["post_threshold_pct"] = self.effective(lead)[0]
            out[str(int(lead))] = row
        return out


__all__ = ["ThresholdTable"]
