"""The scoreboard's per-row work, done once and without copying rows.

``_score_decisions`` used to parse every row's stamp four times, call
``slot_end_of`` per row twice, and copy every row into a new dict to hang
its gauge flag on. These pin the replacements to the originals.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from dmi_nowcast_core import quality_report as qr
from dmi_nowcast_core.warning_score import (
    raining_now_agreement,
    slot_end_of,
    slot_ends_of_us,
)

UTC = timezone.utc
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _stamps() -> list[datetime]:
    rng = np.random.default_rng(5)
    base = datetime(2026, 3, 29, 0, 0, tzinfo=UTC)
    out = [
        base + timedelta(microseconds=int(x))
        for x in rng.integers(0, 3 * 86_400_000_000, size=2_000)
    ]
    # The edges: exactly on a slot, a second after, a microsecond before,
    # on a minute that is not a slot, and before 1970.
    out += [
        base, base + timedelta(seconds=1), base - timedelta(microseconds=1),
        base + timedelta(minutes=3), base + timedelta(minutes=10),
        datetime(1969, 12, 31, 23, 55, 30, tzinfo=UTC),
    ]
    return out


def test_the_vectorised_slot_end_is_slot_end_of() -> None:
    stamps = _stamps()
    micros = np.array(
        [(s - EPOCH) // timedelta(microseconds=1) for s in stamps],
        dtype=np.int64,
    )
    got = slot_ends_of_us(micros)
    want = [
        (slot_end_of(s) - EPOCH) // timedelta(microseconds=1) for s in stamps
    ]
    assert got.tolist() == want


def test_the_reports_slot_ends_are_equal_and_hash_equal() -> None:
    stamps: list[datetime | None] = list(_stamps())
    stamps[3] = None
    got = qr._slot_ends(stamps)
    for stamp, end in zip(stamps, got):
        if stamp is None:
            assert end is None
            continue
        assert end == slot_end_of(stamp)
        assert hash(end) == hash(slot_end_of(stamp))
        assert end.tzinfo is not None


def test_the_gauge_view_scores_like_the_copied_row() -> None:
    rows = []
    wet = []
    base = datetime(2026, 6, 1, tzinfo=UTC)
    for i in range(40):
        rows.append({
            "generated_at": base + timedelta(minutes=10 * i, seconds=30),
            "station_id": "06180",
            "forecast_now_mm_h": None if i % 7 == 0 else float(i % 3),
            "observed_mm_h": float(i % 2),
        })
        wet.append(None if i % 11 == 0 else bool(i % 4 == 0))
    copied = [{**row, "gauge_wet": w} for row, w in zip(rows, wet)]
    viewed = [qr._WithGauge(row, w) for row, w in zip(rows, wet)]
    assert raining_now_agreement(viewed) == raining_now_agreement(copied)
    assert dict(viewed[1]) == copied[1]
    assert "gauge_wet" in viewed[0] and "nope" not in viewed[0]


def test_the_source_counts_reach_the_subscriber_rule() -> None:
    carried = {
        "threshold_pct": 22.0, "rows_post_stored": 5.0,
        "rows_onset_computed": 9.0,
    }
    served = {
        "rows_post_stored": 10, "rows_post_computed": 0,
        "rows_post_unfillable": 7,
        # This run read no onset column: the carried count must go.
        "rows_onset_stored": None, "rows_onset_computed": None,
        "rows_onset_unfillable": None,
    }
    out = qr._merge_served_rule(dict(carried), served)
    assert out["rows_post_stored"] == 10.0
    assert out["rows_post_computed"] == 0.0
    assert out["rows_post_unfillable"] == 7.0
    assert "rows_onset_computed" not in out
    assert out["threshold_pct"] == 22.0


def _write_day(path, rows) -> None:
    import pyarrow.parquet as pq

    from dmi_nowcast_core.warning_score import decision_table

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(decision_table(rows, leads_min=(30,)), path)


def _row(radar_ts, station, *, p=0.5, action="none", late=14, gen=True):
    return {
        "radar_ts": radar_ts,
        "generated_at": radar_ts + timedelta(minutes=late) if gen else None,
        "station_id": station,
        "p_rain": p, "eta_min": None if p < 0.3 else 20.0,
        "intensity_mm_h": 1.0, "observed_mm_h": 0.0,
        "forecast_now_mm_h": None, "action": action,
        "armed_after": True, "streak_after": 0, "p_rain_30": p,
    }


def test_the_columnar_load_is_the_row_dict_load(tmp_path) -> None:
    """Same rows, same values, same order, same counts as the dict loop."""
    import pyarrow.parquet as pq

    now = datetime(2026, 9, 27, tzinfo=UTC)
    base = datetime(2026, 9, 1, tzinfo=UTC)
    replay = tmp_path / "replay" / "decisions"
    _write_day(replay / "2026-09-01.parquet", [
        _row(base + timedelta(minutes=10 * i), s, p=0.1 * (i % 9),
             action="notify" if i % 5 == 0 else "none")
        for i in range(30) for s in ("06180", "06181")
    ])
    _write_day(replay / "2026-09-02.parquet", [
        # Duplicates of the first day's keys: the later file wins.
        _row(base + timedelta(minutes=10 * i), "06180", p=0.99)
        for i in range(5)
    ] + [_row(base + timedelta(minutes=10 * 40), "06182", gen=False)])
    live = tmp_path / "corpus" / "stations" / "eval" / "2026" / "09.parquet"
    _write_day(live, [
        _row(base + timedelta(minutes=10 * i), "06181", p=0.77)
        for i in range(10, 20)
    ] + [
        # Older than the live cutoff: dropped, never overwriting.
        _row(datetime(2026, 5, 1, tzinfo=UTC), "06180", p=0.01),
    ])
    inputs = qr.QualityInputs(
        replay_dir=tmp_path / "replay", corpus_dir=tmp_path / "corpus",
        live_days=90, now=now,
    )
    rows, counts = qr._load_decisions(inputs)

    # The row-at-a-time reference this replaced.
    cutoff = now - timedelta(days=90)
    merged: dict = {}
    ref_counts = {"replay": 0, "live": 0, "duplicates": 0}
    for path in sorted(replay.glob("*.parquet")):
        for row in qr._read_decision_parquet(path):
            ref_counts["replay"] += 1
            key = (row["radar_ts"], str(row["station_id"]))
            ref_counts["duplicates"] += key in merged
            merged[key] = row
    for row in qr._read_decision_parquet(live):
        stamp = row["generated_at"]
        if stamp is not None and stamp < cutoff:
            continue
        ref_counts["live"] += 1
        key = (row["radar_ts"], str(row["station_id"]))
        ref_counts["duplicates"] += key in merged
        merged[key] = row
    reference = sorted(
        merged.values(),
        key=lambda r: (
            r["generated_at"] or datetime.min.replace(tzinfo=UTC),
            str(r["station_id"]),
        ),
    )
    assert counts == ref_counts
    assert [dict(r) for r in rows] == reference
    assert pq  # imported for the writer
