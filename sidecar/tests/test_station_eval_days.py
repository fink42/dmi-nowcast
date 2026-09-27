"""Review R5: live decision rows in day files read back as the month file did.

``station_eval`` used to merge every cycle into ``eval/YYYY/MM.parquet``
(read, anti-join on ``(radar_ts, station_id)``, sort, rewrite — a month
that grows through the month). It now merges into
``eval/YYYY/MM_DD.parquet``. The merge function is the same; only the
file changes. So the gate is on the READERS: the same sequence of cycles
written both ways must load identically through every loader the
nightly jobs and the quality page use — ``threshold_sweep.load_decisions``
(sweep, review, served rule's dedupe), ``decision_rows.load_probabilities``
(Layer B, gauge reliability, the refit) and the core quality report's
``_load_decisions`` / ``_live_window``.

Including the month R5 is deployed in: a pre-R5 month file plus day
files that re-write some of its keys (a correction lands later, and the
later write must win).
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from dmi_nowcast_core.quality_report import (
    QualityInputs,
    _live_window,
    _load_decisions,
)
from dmi_nowcast_sidecar.decision_rows import load_probabilities
from dmi_nowcast_sidecar.station_eval import append_rows
from dmi_nowcast_sidecar.threshold_sweep import load_decisions

T0 = datetime(2026, 9, 29, 20, 0, tzinfo=timezone.utc)
STATIONS = ["06180", "06120", "06031", "06074", "06119"]
ACTIONS = ["none", "none", "none", "notify", "already_raining", "all_clear"]


def _cycles(seed: int = 3, n: int = 60):
    """(radar_ts, rows) per cycle; every other cycle re-emits the last frame
    with a changed action and probability (the later write must win)."""
    rng = random.Random(seed)
    out = []
    for k in range(n):
        ts = T0 + timedelta(minutes=10 * (k // 2))
        rows = []
        for station in STATIONS:
            p = rng.random()
            rows.append({
                "radar_ts": ts,
                "generated_at": ts + timedelta(minutes=12 + (k % 2)),
                "station_id": station,
                "p_rain": p,
                "p_rain_10": p / 2, "p_rain_20": p / 1.5, "p_rain_30": p,
                "eta_min": rng.choice([None, 12.0, 25.0]),
                "intensity_mm_h": 1.0,
                "observed_mm_h": 0.0,
                "forecast_now_mm_h": 0.0,
                "action": rng.choice(ACTIONS),
                "armed_after": True,
                "streak_after": 0,
                "threshold_pct": 40,
            })
        out.append((ts, rows))
    return out


def _month_path(root: Path, ts: datetime) -> Path:
    return root / "stations" / "eval" / f"{ts.year:04d}" / f"{ts.month:02d}.parquet"


def _day_path(root: Path, ts: datetime) -> Path:
    return (
        root / "stations" / "eval" / f"{ts.year:04d}"
        / f"{ts.month:02d}_{ts.day:02d}.parquet"
    )


def _write(root: Path, cycles, *, day_from: int | None) -> None:
    for i, (ts, rows) in enumerate(cycles):
        use_day = day_from is not None and i >= day_from
        path = _day_path(root, ts) if use_day else _month_path(root, ts)
        append_rows(path, rows, (10, 20, 30))


def _assert_same(new: Path, old: Path) -> None:
    eval_new = new / "stations" / "eval"
    eval_old = old / "stations" / "eval"
    rows_new, leads_new, counts_new = load_decisions([eval_new])
    rows_old, leads_old, counts_old = load_decisions([eval_old])
    assert leads_new == leads_old
    # Raw rows read can differ (a key the month file and a day file both
    # hold is read twice and deduped); the rows that come out cannot.
    assert (
        counts_new["rows"] - counts_new["duplicates"]
        == counts_old["rows"] - counts_old["duplicates"]
    )
    key = lambda r: (r["radar_ts"], r["station_id"])  # noqa: E731
    assert sorted(rows_new, key=key) == sorted(rows_old, key=key)

    p_new = load_probabilities([eval_new], (10, 30))
    p_old = load_probabilities([eval_old], (10, 30))
    assert len(p_new["t"]) == len(p_old["t"])
    order_new = np.lexsort((p_new["station"], p_new["t"]))
    order_old = np.lexsort((p_old["station"], p_old["t"]))
    np.testing.assert_array_equal(p_new["t"][order_new], p_old["t"][order_old])
    for lead in (10, 30):
        np.testing.assert_array_equal(
            p_new["p"][lead][order_new], p_old["p"][lead][order_old],
        )

    now = T0 + timedelta(days=2)
    dec_new, _ = _load_decisions(QualityInputs(corpus_dir=new, now=now))
    dec_old, _ = _load_decisions(QualityInputs(corpus_dir=old, now=now))
    assert set(dec_new.columns) == set(dec_old.columns)
    for name in dec_old.columns:
        np.testing.assert_array_equal(dec_new.columns[name], dec_old.columns[name])
    for name, (codes, values) in dec_old.strings.items():
        new_codes, new_values = dec_new.strings[name]
        assert [new_values[c] for c in new_codes] == [values[c] for c in codes]
    assert _live_window(QualityInputs(corpus_dir=new, now=now)) == _live_window(
        QualityInputs(corpus_dir=old, now=now),
    )


def test_day_files_load_exactly_as_the_month_file(tmp_path: Path) -> None:
    cycles = _cycles()
    _write(tmp_path / "old", cycles, day_from=None)
    _write(tmp_path / "new", cycles, day_from=0)
    names = sorted(p.name for p in (tmp_path / "new").rglob("*.parquet"))
    assert names == ["09_29.parquet", "09_30.parquet"]
    _assert_same(tmp_path / "new", tmp_path / "old")


def test_the_month_r5_is_deployed_in_loads_as_the_old_code_would(
    tmp_path: Path,
) -> None:
    """Month file for the first cycles, day files after — overlapping keys.

    Cycle 21 is the re-emit of cycle 20's frame, so the first day-file
    write replaces a key that the month file holds; every loader must
    return the day file's row, as the old in-place merge would have.
    """
    cycles = _cycles(seed=5)
    _write(tmp_path / "old", cycles, day_from=None)
    _write(tmp_path / "new", cycles, day_from=21)
    assert _month_path(tmp_path / "new", T0).is_file()
    _assert_same(tmp_path / "new", tmp_path / "old")
