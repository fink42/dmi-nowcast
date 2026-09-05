"""The fitted push thresholds document — one knob for notifications.

A subscriber picks a horizon and nothing else. The probability threshold
that horizon warns at is not a preference, it is a measurement: for each
lead, ``scripts/sweep_thresholds.py`` replays the live push rule over
recorded decision rows at every threshold on a grid, scores the result
against the rain gauges, and keeps the threshold that maximises F1 with a
minimum useful lead. This module is the contract for the small JSON file
that carries those numbers from the fit to the service.

Shape, version 1::

    {"schema_version": 1,
     "fitted_at_utc": "2026-09-05T02:11:07+00:00",
     "objective": {"metric": "f1", "min_useful_lead_min": 5,
                   "plateau_frac": 0.95, "min_warnings": 30,
                   "rearm_after_min": 60, "persistence_obs": 1,
                   "tolerance_min": 10, "dry_min": 30},
     "window": {"from": ..., "to": ..., "days": 62, "stations": 97,
                "rows": 1841203},
     "fallback_threshold_pct": 40,
     "leads": {"20": {"threshold_pct": 45, "insufficient": false, ...}}}

Two rules make this safe to read at request time:

**A lead the table does not cover falls back.** Not an error, not a
refusal to warn — ``fallback_threshold_pct`` is the rule shipping today,
and a horizon nobody has fitted yet behaves exactly as the site behaved
before the fit existed.

**A lead with too little evidence is marked, not guessed.**
``insufficient`` is true when the sweep saw fewer than
``objective.min_warnings`` scored warnings across the whole threshold grid
at that lead. Its ``threshold_pct`` is null and
:func:`effective_threshold` falls back, because a threshold fitted on four
warnings is a story about four warnings.

:func:`validate_thresholds` is the schema check; it returns a list of
human-readable problems, empty when the document is sound, in the same
style as ``quality_report.validate_report``. :func:`effective_threshold`
never raises and never returns None: whatever it is handed, a
subscription gets a number to compare against.
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "SCHEMA_VERSION",
    "DEFAULT_FALLBACK_THRESHOLD_PCT",
    "OBJECTIVE_SPEC",
    "WINDOW_SPEC",
    "LEAD_SPEC",
    "validate_thresholds",
    "validate_leads_table",
    "effective_threshold",
    "load_thresholds",
]

#: The version the sidecar and the sweep agree on.
SCHEMA_VERSION = 1

#: The rule shipping before any fit existed, and the answer for every lead
#: the table cannot speak for.
DEFAULT_FALLBACK_THRESHOLD_PCT = 40

#: ``objective``: what was maximised, and the rule constants the replay ran
#: under. Stored so a served threshold can always be traced back to the
#: definition of "good" that produced it.
OBJECTIVE_SPEC: dict[str, type] = {
    "metric": str,
    "min_useful_lead_min": float,
    "plateau_frac": float,
    "min_warnings": int,
    "rearm_after_min": int,
    "persistence_obs": int,
    "tolerance_min": int,
    "dry_min": int,
}

#: ``window``: the evidence the fit stands on.
WINDOW_SPEC: dict[str, type] = {
    "from": str,
    "to": str,
    "days": int,
    "stations": int,
    "rows": int,
}

#: One lead's row. Counts are integers, rates are nullable floats: a rate
#: over an empty denominator is null, never 0.0.
LEAD_SPEC: dict[str, type] = {
    "threshold_pct": int,      # nullable — null with insufficient evidence
    "insufficient": bool,
    "f1": float,               # nullable
    "precision": float,        # nullable
    "recall": float,           # nullable
    "far": float,              # nullable
    "csi": float,              # nullable
    "warnings": int,
    "hits": int,
    "false_alarms": int,
    "misses": int,
    "late": int,
}

#: Which of :data:`LEAD_SPEC`'s fields may be null.
_NULLABLE_LEAD_FIELDS = frozenset(
    {"threshold_pct", "f1", "precision", "recall", "far", "csi"}
)


def _is_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _is_iso(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def _check_block(
    block: Any, spec: Mapping[str, type], where: str, problems: list[str],
    *, nullable: frozenset[str] = frozenset(),
) -> None:
    if not isinstance(block, dict):
        problems.append(f"{where}: expected an object, got {type(block).__name__}")
        return
    for key, kind in spec.items():
        if key not in block:
            problems.append(f"{where}.{key}: missing")
            continue
        value = block[key]
        if value is None:
            if key not in nullable:
                problems.append(f"{where}.{key}: must not be null")
            continue
        if kind is bool:
            if not isinstance(value, bool):
                problems.append(f"{where}.{key}: expected true or false")
        elif kind is str:
            if not isinstance(value, str) or not value.strip():
                problems.append(f"{where}.{key}: expected a non-empty string")
            elif key in ("from", "to") and not _is_iso(value):
                problems.append(f"{where}.{key}: not an ISO timestamp")
        elif kind is int:
            if isinstance(value, bool) or not isinstance(value, int):
                problems.append(f"{where}.{key}: expected an integer")
        elif not _is_number(value):
            problems.append(f"{where}.{key}: expected a finite number")


def _check_plateau(value: Any, where: str, problems: list[str]) -> None:
    """``[lo, hi]`` whole percents in (0, 100), ``lo <= hi``, or null."""
    if value is None:
        return
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        problems.append(f"{where}: expected [lo, hi] or null")
        return
    lo, hi = value
    for edge in (lo, hi):
        if isinstance(edge, bool) or not isinstance(edge, int):
            problems.append(f"{where}: expected whole-percent integers")
            return
        if not 0 < edge < 100:
            problems.append(f"{where}: {edge} out of range (0, 100)")
            return
    if lo > hi:
        problems.append(f"{where}: lo {lo} above hi {hi}")


def validate_thresholds(doc: Any) -> list[str]:
    """Structural problems with a thresholds document, as readable lines.

    An empty list means the document is one :func:`effective_threshold`
    can be trusted with. The checks are deliberately strict about the
    things a silent misread would turn into a wrong notification rule —
    the version, the lead keys, the percent ranges — and about the one
    contradiction the writer could produce: a lead marked ``insufficient``
    that still carries a threshold.
    """
    problems: list[str] = []
    if not isinstance(doc, dict):
        return [f"thresholds: expected an object, got {type(doc).__name__}"]

    if doc.get("schema_version") != SCHEMA_VERSION:
        problems.append(
            f"schema_version: expected {SCHEMA_VERSION}, "
            f"got {doc.get('schema_version')!r}",
        )
    if not _is_iso(doc.get("fitted_at_utc")):
        problems.append("fitted_at_utc: missing or not an ISO timestamp")

    _check_block(doc.get("objective"), OBJECTIVE_SPEC, "objective", problems)
    _check_block(doc.get("window"), WINDOW_SPEC, "window", problems)

    fallback = doc.get("fallback_threshold_pct")
    if isinstance(fallback, bool) or not isinstance(fallback, int):
        problems.append("fallback_threshold_pct: expected an integer")
    elif not 0 < fallback < 100:
        problems.append(
            f"fallback_threshold_pct: {fallback} out of range (0, 100)",
        )

    leads = doc.get("leads")
    if not isinstance(leads, dict):
        problems.append("leads: expected an object keyed by lead minutes")
        return problems
    problems.extend(validate_leads_table(leads))
    return problems


def validate_leads_table(leads: Mapping[str, Any], where: str = "leads") -> list[str]:
    """Problems with the ``leads`` table on its own.

    Split out because the quality report embeds this table verbatim as its
    ``thresholds.leads`` section: one definition of a well-formed lead row,
    checked identically wherever it travels.
    """
    problems: list[str] = []
    for key, entry in leads.items():
        at = f"{where}.{key}"
        if not (isinstance(key, str) and key.isdigit() and int(key) > 0):
            problems.append(f"{at}: key must be a positive whole minute")
            continue
        _check_block(
            entry, LEAD_SPEC, at, problems, nullable=_NULLABLE_LEAD_FIELDS,
        )
        if not isinstance(entry, dict):
            continue
        threshold = entry.get("threshold_pct")
        if (
            threshold is not None
            and not isinstance(threshold, bool)
            and isinstance(threshold, int)
            and not 0 < threshold < 100
        ):
            problems.append(f"{at}.threshold_pct: out of range (0, 100)")
        if entry.get("insufficient") is True and threshold is not None:
            problems.append(
                f"{at}: insufficient evidence cannot carry a threshold",
            )
        for name in ("plateau", "radar_plateau", "agrees_with_radar"):
            if name not in entry:
                # Null, never absent: "we did not measure this" is a value
                # the reader has to be able to see.
                problems.append(f"{at}.{name}: missing (use null, never absent)")
        _check_plateau(entry.get("plateau"), f"{at}.plateau", problems)
        _check_plateau(
            entry.get("radar_plateau"), f"{at}.radar_plateau", problems,
        )
        agrees = entry.get("agrees_with_radar")
        if agrees is not None and not isinstance(agrees, bool):
            problems.append(f"{at}.agrees_with_radar: expected a boolean or null")
    return problems


def effective_threshold(doc: Any, lead_min: Any) -> int:
    """The percent this lead warns at: its fitted pick, or the fallback.

    Total: every input has an answer. A lead the table does not carry, a
    lead whose evidence was too thin to fit, a document that is the wrong
    shape or is not a document at all — all of them return the fallback,
    which is the rule the site shipped before any of this existed. A
    notification rule that raises at request time is worse than one that
    is merely not yet tuned.
    """
    fallback = DEFAULT_FALLBACK_THRESHOLD_PCT
    if isinstance(doc, dict):
        candidate = doc.get("fallback_threshold_pct")
        if (
            not isinstance(candidate, bool)
            and isinstance(candidate, int)
            and 0 < candidate < 100
        ):
            fallback = candidate
    else:
        return fallback

    leads = doc.get("leads")
    if not isinstance(leads, dict):
        return fallback
    try:
        key = str(int(lead_min))
    except (TypeError, ValueError):
        return fallback
    entry = leads.get(key)
    if not isinstance(entry, dict) or entry.get("insufficient") is True:
        return fallback
    threshold = entry.get("threshold_pct")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, int)
        or not 0 < threshold < 100
    ):
        return fallback
    return threshold


def load_thresholds(path: Path | str | None) -> dict | None:
    """Read and validate a thresholds file; ``None`` if it is unusable.

    Unusable covers absent, unreadable, not JSON, and structurally wrong.
    The caller's job is then to say "not fitted" — the quality page nulls
    its section, the service uses the fallback — never to serve half a
    document.
    """
    if path is None:
        return None
    file = Path(path)
    if not file.is_file():
        return None
    try:
        doc = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if validate_thresholds(doc):
        return None
    return doc
