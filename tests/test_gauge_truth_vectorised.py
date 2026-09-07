"""The vectorised gauge-truth load must equal the reference, exactly.

``warning_score.gauge_slot_amounts`` + ``warning_score.onsets`` are the
definition of a wet slot and of an onset. ``gauge_truth_vectorised`` is a
numpy rewrite of the loop that three consumers ran over them, and the
only thing that makes the rewrite safe is that it is checked against the
original on the cases that broke it: gaps in the reporting, DMI's trace
sentinel, two readings inside one slot, a slot straddling two month
partitions, a station that reports only the duration parameter.

The reference here is deliberately a *copy* of the loop the callers ran —
month by month, padded, unioning the onsets — rather than a call into one
of them, because the sidecar's copy of that loop is not importable from
this suite and the point is to pin the behaviour, not the caller.
"""
from __future__ import annotations

import os
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from dmi_nowcast_core.metobs import Observation
from dmi_nowcast_core.station_store import StationObsStore
from dmi_nowcast_core.warning_score import (
    DEFAULT_DRY_MIN,
    DEFAULT_GAUGE_PAD_MIN,
    DEFAULT_ONSET_MIN_MM,
    PRECIP_DUR_PARAM,
    PRECIP_PARAM,
    SLOT_MIN,
    gauge_slot_amounts,
    gauge_slots,
    gauge_truth_vectorised,
    onsets,
)

UTC = timezone.utc
PAD = timedelta(minutes=DEFAULT_GAUGE_PAD_MIN)


# ---------------------------------------------------------------------------
# The reference: what the three consumers did, month by month
# ---------------------------------------------------------------------------


def _months_between(start: datetime, end: datetime) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        out.append((year, month))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


def reference_truth(
    root: Path,
    stations: list[str],
    start: datetime,
    end: datetime,
    *,
    dry_min: int = DEFAULT_DRY_MIN,
    onset_min_mm: float = DEFAULT_ONSET_MIN_MM,
) -> tuple[dict[str, list[datetime]], dict[str, datetime], int, dict]:
    """``threshold_sweep.gauge_truth`` / ``quality_report._gauge_truth``.

    Onsets unioned over padded month windows, the last known slot per
    station, the known-slot tally (which double counts the pad overlaps,
    exactly as the shipped loops do), and every known slot's wet flag.
    """
    store = StationObsStore(root)
    onset_sets: dict[str, set[datetime]] = {}
    known_until: dict[str, datetime] = {}
    wet_at: dict[tuple[str, datetime], bool] = {}
    known_slots = 0
    for (year, month) in _months_between(start, end):
        month_start = datetime(year, month, 1, tzinfo=UTC) - PAD
        if month == 12:
            month_end = datetime(year + 1, 1, 1, tzinfo=UTC) + PAD
        else:
            month_end = datetime(year, month + 1, 1, tzinfo=UTC) + PAD
        table = store.read(
            month_start, month_end,
            [PRECIP_PARAM, PRECIP_DUR_PARAM], list(stations),
        )
        for station in stations:
            slots = gauge_slot_amounts(
                table, station, start_utc=month_start, end_utc=month_end,
            )
            if not slots:
                continue
            onset_sets.setdefault(station, set()).update(
                onsets(slots, dry_min, onset_min_mm=onset_min_mm),
            )
            for stamp, wet, _mm in slots:
                if wet is None:
                    continue
                known_slots += 1
                wet_at[(station, stamp)] = wet
                previous = known_until.get(station)
                if previous is None or stamp > previous:
                    known_until[station] = stamp
    return (
        {s: sorted(v) for s, v in onset_sets.items()},
        known_until,
        known_slots,
        wet_at,
    )


def assert_same(
    root, stations, start, end,
    *, dry_min=DEFAULT_DRY_MIN, onset_min_mm=DEFAULT_ONSET_MIN_MM,
):
    """Both implementations, every field compared."""
    want_onsets, want_known, want_slots, want_wet = reference_truth(
        root, stations, start, end,
        dry_min=dry_min, onset_min_mm=onset_min_mm,
    )
    truth = gauge_truth_vectorised(
        root, start, end, stations,
        dry_min=dry_min, onset_min_mm=onset_min_mm,
    )
    for station in stations:
        assert truth.onsets[station] == want_onsets.get(station, []), station
    assert truth.known_until == want_known
    assert truth.known_slots == want_slots
    for (station, stamp), wet in want_wet.items():
        assert truth.wet_at(station, stamp) is wet, (station, stamp)
    return truth


def write(root: Path, rows) -> None:
    StationObsStore(root).append(rows)


def obs(station, when, parameter=PRECIP_PARAM, value=0.0) -> Observation:
    return Observation(
        station_id=station, observed_utc=when,
        parameter_id=parameter, value=value,
    )


# ---------------------------------------------------------------------------
# Hand-made series
# ---------------------------------------------------------------------------

DAY = datetime(2026, 6, 10, 0, 0, tzinfo=UTC)


def _slot(n: int, base: datetime = DAY) -> datetime:
    return base + timedelta(minutes=SLOT_MIN * n)


def test_a_plain_dry_spell_then_rain_is_one_onset(tmp_path: Path) -> None:
    # Six dry slots and 0.4 mm: an onset under the shipped defaults, which
    # is what this case is pinned on.
    rows = [obs("06074", _slot(n), value=0.0) for n in range(6)]
    rows.append(obs("06074", _slot(6), value=0.4))
    write(tmp_path, rows)
    truth = assert_same(
        tmp_path, ["06074"], DAY, DAY + timedelta(hours=2),
    )
    assert truth.onsets["06074"] == [_slot(6)]


def test_rain_that_never_delivers_is_not_an_onset(tmp_path: Path) -> None:
    """The same shape, drizzling: 0.1 mm and then nothing more."""
    rows = [obs("06074", _slot(n), value=0.0) for n in range(6)]
    rows.append(obs("06074", _slot(6), value=0.1))
    rows.append(obs("06074", _slot(7), value=0.0))
    write(tmp_path, rows)
    truth = assert_same(tmp_path, ["06074"], DAY, DAY + timedelta(hours=2))
    assert truth.onsets["06074"] == []
    # …and it is the amount rule that dropped it, not the dry run.
    loose = assert_same(
        tmp_path, ["06074"], DAY, DAY + timedelta(hours=2), onset_min_mm=0.0,
    )
    assert loose.onsets["06074"] == [_slot(6)]


def test_an_unreported_slot_never_certifies_a_dry_spell(tmp_path: Path) -> None:
    # Slots 0,1 dry, slot 2 missing, slot 3 dry, slot 4 wet: only two known
    # dry slots stand behind the rain, so it is not an onset.
    rows = [obs("06074", _slot(n), value=0.0) for n in (0, 1, 3)]
    rows.append(obs("06074", _slot(4), value=1.0))
    write(tmp_path, rows)
    truth = assert_same(
        tmp_path, ["06074"], DAY, DAY + timedelta(hours=2), dry_min=30,
    )
    assert truth.onsets["06074"] == []


def test_the_trace_sentinel_is_below_the_threshold_not_a_negative_depth(
    tmp_path: Path,
) -> None:
    # DMI's "traces of precipitation" marker. It must read as 0.0 mm — dry
    # by the amount arm — and never subtract from the shower it belongs to.
    rows = [obs("06074", _slot(n), value=-0.1) for n in range(4)]
    rows.append(obs("06074", _slot(4), value=0.2))
    rows.append(obs("06074", _slot(5), value=0.3))
    write(tmp_path, rows)
    truth = assert_same(
        tmp_path, ["06074"], DAY, DAY + timedelta(hours=2), dry_min=30,
    )
    assert truth.onsets["06074"] == [_slot(4)]
    series = truth.series["06074"]
    assert series.wet_at(_slot(0)) is False
    assert [mm for _t, mm in series.onsets_with_amounts(30)] == [pytest.approx(0.5)]


def test_two_readings_in_one_slot_take_the_larger(tmp_path: Path) -> None:
    rows = [obs("06074", _slot(n), value=0.0) for n in range(4)]
    # 07:41 and 07:45 both belong to the slot ending 07:50.
    rows.append(obs("06074", _slot(4) - timedelta(minutes=9), value=0.0))
    rows.append(obs("06074", _slot(4) - timedelta(minutes=5), value=0.6))
    write(tmp_path, rows)
    truth = assert_same(
        tmp_path, ["06074"], DAY, DAY + timedelta(hours=2), dry_min=30,
    )
    assert truth.onsets["06074"] == [_slot(4)]


def test_the_duration_arm_alone_makes_a_slot_wet_and_known(
    tmp_path: Path,
) -> None:
    rows = [obs("06074", _slot(n), PRECIP_DUR_PARAM, 0.0) for n in range(4)]
    rows.append(obs("06074", _slot(4), PRECIP_DUR_PARAM, 2.0))
    write(tmp_path, rows)
    truth = assert_same(
        tmp_path, ["06074"], DAY, DAY + timedelta(hours=2),
        dry_min=30, onset_min_mm=0.0,
    )
    assert truth.onsets["06074"] == [_slot(4)]
    # No amount was ever reported, so the depth stays unknown — which the
    # amount test reads as zero millimetres, not as evidence of rain. Under
    # the shipped floor this wet slot is therefore not an onset at all.
    series = truth.series["06074"]
    assert series.onsets_with_amounts(30, onset_min_mm=0.2) == []


def test_a_slot_end_exactly_on_the_boundary_stays_in_its_own_slot(
    tmp_path: Path,
) -> None:
    # 07:20:00 IS the end of (07:10, 07:20]; 07:20:01 opens the next one.
    rows = [obs("06074", _slot(n), value=0.0) for n in range(4)]
    rows.append(obs("06074", _slot(4), value=0.9))
    rows.append(obs("06074", _slot(4) + timedelta(seconds=1), value=0.0))
    write(tmp_path, rows)
    truth = assert_same(
        tmp_path, ["06074"], DAY, DAY + timedelta(hours=2), dry_min=30,
    )
    assert truth.onsets["06074"] == [_slot(4)]


def test_an_onset_across_a_month_boundary_is_found_once(tmp_path: Path) -> None:
    # The dry spell is in June's partition, the rain in July's, and the
    # month-by-month reader only sees both because of the pad. The
    # vectorised grid spans them, and the two must agree.
    boundary = datetime(2026, 7, 1, 0, 0, tzinfo=UTC)
    rows = [
        obs("06074", boundary - timedelta(minutes=SLOT_MIN * n), value=0.0)
        for n in (1, 2, 3, 4)
    ]
    rows.append(obs("06074", boundary, value=0.0))
    rows.append(obs("06074", boundary + timedelta(minutes=SLOT_MIN), value=0.7))
    write(tmp_path, rows)
    truth = assert_same(
        tmp_path, ["06074"],
        boundary - timedelta(days=1), boundary + timedelta(days=1),
        dry_min=30,
    )
    assert truth.onsets["06074"] == [boundary + timedelta(minutes=SLOT_MIN)]


def test_several_stations_never_borrow_each_others_slots(
    tmp_path: Path,
) -> None:
    rows = []
    for n in range(4):
        rows.append(obs("06074", _slot(n), value=0.0))
        rows.append(obs("06079", _slot(n), value=5.0))
    rows.append(obs("06074", _slot(4), value=1.0))
    write(tmp_path, rows)
    truth = assert_same(
        tmp_path, ["06074", "06079"], DAY, DAY + timedelta(hours=2),
        dry_min=30,
    )
    assert truth.onsets["06074"] == [_slot(4)]
    # 06079 rained through the whole record: no dry spell, no onset.
    assert truth.onsets["06079"] == []


def test_a_station_the_archive_never_heard_of_scores_nothing(
    tmp_path: Path,
) -> None:
    write(tmp_path, [obs("06074", _slot(n), value=0.0) for n in range(4)])
    truth = assert_same(
        tmp_path, ["06074", "99999"], DAY, DAY + timedelta(hours=2),
    )
    # An entry, so a caller can iterate stations without a KeyError…
    assert truth.onsets["99999"] == []
    # …but no ``known_until``, which is the test the callers use to leave a
    # station out of the scored pool entirely.
    assert "99999" not in truth.known_until


def test_a_missing_partition_is_not_an_error(tmp_path: Path) -> None:
    truth = gauge_truth_vectorised(
        tmp_path, DAY, DAY + timedelta(days=1), ["06074"],
    )
    assert truth.onsets == {"06074": []}
    assert truth.known_until == {}
    assert truth.known_slots == 0


@pytest.mark.parametrize("onset_min_mm", [0.0, 0.2, 0.5])
@pytest.mark.parametrize("dry_min", [10, 30, 60, 120])
def test_every_dry_spell_length_matches_the_reference(
    tmp_path: Path, dry_min: int, onset_min_mm: float,
) -> None:
    rng = random.Random(4 + dry_min)
    rows = []
    for n in range(0, 24 * 6 * 3):
        roll = rng.random()
        if roll < 0.04:
            continue  # the station said nothing about this slot
        # Mostly dry, as Denmark is: a two-hour dry spell has to be
        # reachable or the longest variant would have nothing to find.
        value = 0.0 if roll < 0.93 else round(rng.uniform(0.0, 4.0), 1)
        rows.append(obs("06074", _slot(n), value=value))
        if rng.random() < 0.5:
            # The duration arm follows the amount, apart from the odd
            # drizzle that rounds to 0.0 mm and still wets a road.
            minutes = (
                float(rng.randint(1, 10)) if value >= 0.1
                else (1.0 if rng.random() < 0.05 else 0.0)
            )
            rows.append(obs("06074", _slot(n), PRECIP_DUR_PARAM, minutes))
    write(tmp_path, rows)
    truth = assert_same(
        tmp_path, ["06074"], DAY, DAY + timedelta(days=3),
        dry_min=dry_min, onset_min_mm=onset_min_mm,
    )
    assert truth.onsets["06074"], "the fixture should produce some onsets"


def test_a_random_month_of_several_stations_matches_the_reference(
    tmp_path: Path,
) -> None:
    """The whole shape at once: gaps, traces, duplicates, both arms."""
    rng = random.Random(11)
    stations = ["06074", "06079", "06081"]
    base = datetime(2026, 5, 28, tzinfo=UTC)
    rows = []
    for station in stations:
        for n in range(0, 6 * 24 * 8):
            when = base + timedelta(minutes=SLOT_MIN * n)
            roll = rng.random()
            if roll < 0.15:
                continue
            if roll < 0.2:
                value = -0.1  # trace
            elif roll < 0.75:
                value = 0.0
            else:
                value = round(rng.uniform(0.0, 6.0), 1)
            # Half the readings land off the slot end, which is where the
            # ceiling in ``slot_end_of`` earns its keep.
            offset = timedelta(seconds=rng.choice([0, -61, -299, -1]))
            rows.append(obs(station, when + offset, value=value))
            if rng.random() < 0.4:
                rows.append(obs(
                    station, when, PRECIP_DUR_PARAM,
                    float(rng.choice([0, 0, 1, 4, 10])),
                ))
    write(tmp_path, rows)
    truth = assert_same(
        tmp_path, stations, base, base + timedelta(days=8),
    )
    strict = sum(len(v) for v in truth.onsets.values())
    assert strict > 5, "the fixture should produce onsets under the shipped rule"
    # The same archive under the rule that shipped before the amount test:
    # strictly more onsets, and the two implementations agree on both.
    loose = assert_same(
        tmp_path, stations, base, base + timedelta(days=8),
        dry_min=30, onset_min_mm=0.0,
    )
    assert sum(len(v) for v in loose.onsets.values()) > strict


def test_an_alternative_onset_definition_costs_no_second_read(
    tmp_path: Path,
) -> None:
    """``onsets_for`` re-derives from the grid already in memory.

    The variant analysis (``scripts/onset_sensitivity.py``) asks five
    questions of one archive read; each must give what a fresh load under
    those parameters gives.
    """
    rng = random.Random(7)
    rows = []
    for n in range(6 * 24 * 4):
        roll = rng.random()
        if roll < 0.1:
            continue
        rows.append(obs(
            "06074", _slot(n),
            value=0.0 if roll < 0.72 else round(rng.uniform(0.0, 3.0), 1),
        ))
    write(tmp_path, rows)
    start, end = DAY, DAY + timedelta(days=4)
    truth = gauge_truth_vectorised(tmp_path, start, end, ["06074"])
    for dry_min in (30, 60, 120):
        for floor in (None, 0.2, 0.5):
            fresh = gauge_truth_vectorised(
                tmp_path, start, end, ["06074"],
                dry_min=dry_min, onset_min_mm=floor,
            )
            again = truth.onsets_for(dry_min, onset_min_mm=floor)
            assert [t for t, _mm in again["06074"]] == fresh.onsets["06074"]


def test_the_amount_test_spans_the_onset_slot_and_the_next(
    tmp_path: Path,
) -> None:
    rows = [obs("06074", _slot(n), value=0.0) for n in range(4)]
    rows.append(obs("06074", _slot(4), value=0.1))
    rows.append(obs("06074", _slot(5), value=0.3))
    write(tmp_path, rows)
    truth = gauge_truth_vectorised(tmp_path, DAY, DAY + timedelta(hours=2), ["06074"])
    series = truth.series["06074"]
    assert [t for t, _mm in series.onsets_with_amounts(30)] == [_slot(4)]
    assert series.onsets_with_amounts(30, onset_min_mm=0.4) != []
    assert series.onsets_with_amounts(30, onset_min_mm=0.5) == []


def test_a_failed_amount_test_still_resets_the_dry_run(tmp_path: Path) -> None:
    """It rained; the slot after it is not the start of a new event."""
    rows = [obs("06074", _slot(n), value=0.0) for n in range(4)]
    rows.append(obs("06074", _slot(4), value=0.1))   # wet, but only 0.1 mm
    rows.append(obs("06074", _slot(5), value=2.0))   # rain in earnest
    write(tmp_path, rows)
    truth = gauge_truth_vectorised(tmp_path, DAY, DAY + timedelta(hours=2), ["06074"])
    series = truth.series["06074"]
    # Without the amount test the first slot is the onset; with a floor it
    # cannot reach, the candidate is dropped AND the second slot is not
    # promoted, because the dry run behind it is zero.
    assert [t for t, _mm in series.onsets_with_amounts(30, onset_min_mm=0.0)] == [
        _slot(4),
    ]
    assert series.onsets_with_amounts(30, onset_min_mm=2.5) == []


def test_the_slot_list_matches_the_reference_grid(tmp_path: Path) -> None:
    rows = [obs("06074", _slot(n), value=float(n % 3)) for n in range(6)]
    write(tmp_path, rows)
    start, end = DAY, DAY + timedelta(hours=1)
    truth = gauge_truth_vectorised(tmp_path, start, end, ["06074"])
    series = truth.series["06074"]
    store = StationObsStore(tmp_path)
    month_start = datetime(2026, 6, 1, tzinfo=UTC) - PAD
    month_end = datetime(2026, 7, 1, tzinfo=UTC) + PAD
    table = store.read(
        month_start, month_end, [PRECIP_PARAM, PRECIP_DUR_PARAM], ["06074"],
    )
    want = gauge_slots(table, "06074", start_utc=month_start, end_utc=month_end)
    assert series.slots() == want


# ---------------------------------------------------------------------------
# The real archive, when this machine has one
# ---------------------------------------------------------------------------

#: A corpus root with ``stations/obs/YYYY/MM.parquet`` under it. Set it to
#: run the equality and benchmark checks against real DMI observations;
#: CI has no archive and skips them.
LOCAL_CORPUS = os.environ.get("DMI_NOWCAST_TEST_CORPUS")


def _local_month() -> tuple[Path, int, int]:
    if not LOCAL_CORPUS:
        pytest.skip("set DMI_NOWCAST_TEST_CORPUS to a corpus root to run this")
    root = Path(LOCAL_CORPUS)
    months = sorted((root / "stations" / "obs").glob("*/*.parquet"))
    if not months:
        pytest.skip(f"no station observations under {root}")
    newest = months[-1]
    return root, int(newest.parent.name), int(newest.stem)


@pytest.mark.slow
def test_a_real_month_matches_the_reference() -> None:
    """The same equality, on rows DMI actually published.

    Skipped without a local archive. It is the check that matters most —
    the hand-made series cannot reproduce what a hundred real stations do
    with late reports, duplicated instants and parameters that come and
    go — and it is far too slow for CI, because the reference side is the
    half-hour loop this whole change exists to remove.
    """
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    root, year, month = _local_month()
    path = StationObsStore(root).partition_path(year, month)
    column = pq.read_table(path, columns=["station_id"]).column("station_id")
    stations = sorted(pc.unique(column.combine_chunks()).to_pylist())[:12]
    start = datetime(year, month, 1, tzinfo=UTC)
    assert_same(root, stations, start, start + timedelta(days=1))


@pytest.mark.slow
def test_the_whole_local_corpus_loads_in_seconds() -> None:
    """The budget, measured: the full archive in well under a minute.

    The number this pins is not a micro-benchmark, it is the reason the
    nightly report was killed on the VM — the row-at-a-time load took
    about half an hour and 5.5 GB for what is here a few seconds and a
    few tens of megabytes. Skipped without a local archive.
    """
    import resource
    import time

    if not LOCAL_CORPUS:
        pytest.skip("set DMI_NOWCAST_TEST_CORPUS to a corpus root to run this")
    root = Path(LOCAL_CORPUS)
    months = sorted((root / "stations" / "obs").glob("*/*.parquet"))
    if len(months) < 2:
        pytest.skip("need at least two month partitions to be worth timing")
    first, last = months[0], months[-1]
    start = datetime(int(first.parent.name), int(first.stem), 1, tzinfo=UTC)
    end = datetime(int(last.parent.name), int(last.stem), 28, tzinfo=UTC)

    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    started = time.perf_counter()
    truth = gauge_truth_vectorised(root, start, end, None)
    elapsed = time.perf_counter() - started
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1 if os.uname().sysname == "Darwin" else 1024  # ru_maxrss units
    grew = (peak - before) * scale / (1024 * 1024)
    print(
        f"\n{len(truth.series)} station(s), {truth.known_slots} known slot(s), "
        f"{sum(len(v) for v in truth.onsets.values())} onset(s) "
        f"in {elapsed:.1f}s, RSS +{grew:.0f} MB"
    )
    assert elapsed < 60.0
    assert grew < 500.0
