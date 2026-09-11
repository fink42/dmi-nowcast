"""Gauge join: the verification instant, the slot lookup, and the wet rule.

The one thing that must not drift is *when* the gauge is read. The corpus
verifies a row at ``T + ceil((lead + frame_age)/timestep - 1e-9) *
timestep``; if the gauge were read at any other instant the two truths
would be measuring different weather and the whole comparison would be
noise. The first tests here pin that against the builder's own function.

Fully synthetic: a small corpus Parquet written with the builder's schema,
a small store, no network.
"""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import build_calibration_corpus as bcc  # noqa: E402  (after sys.path edit)
import join_gauge_truth as jgt  # noqa: E402
from dmi_nowcast_core.metobs import Observation  # noqa: E402
from dmi_nowcast_core.station_store import StationObsStore  # noqa: E402

EVENT = datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# The verification instant
# ---------------------------------------------------------------------------


def test_join_uses_the_builder_s_own_snap_function() -> None:
    """Not a copy — the same object, so it cannot drift."""
    assert jgt.snap_lead_min is bcc.snap_lead_min


@pytest.mark.parametrize("timestep", [5.0, 10.0])
@pytest.mark.parametrize("lead", [5, 10, 15, 20, 30, 45, 60])
@pytest.mark.parametrize("age", [0.0, 3.0, 10.0, 12.0, 15.0, 17.99738, 18.0])
def test_fallback_snap_matches_the_builder(lead: int, age: float, timestep: float) -> None:
    """The local fallback exists only for an import failure; it must be
    indistinguishable from the real thing, or a fallback run would verify
    against a different instant and nobody would notice."""
    assert jgt._snap_lead_min_fallback(lead + age, timestep) == bcc.snap_lead_min(
        lead + age, timestep,
    )


def test_verification_instant_matches_the_builder_s_formula() -> None:
    for lead in (5, 10, 20, 45, 60):
        for age in (0.0, 12.5, 17.0):
            expected = EVENT + timedelta(
                minutes=bcc.snap_lead_min(lead + age, 10.0)
            )
            assert jgt.verification_instant(EVENT, lead, age, 10.0) == expected


def test_frame_age_pushes_the_instant_onto_a_later_slot() -> None:
    """The point of the frame-age convention: a 5-minute lead read off a
    17-minute-old frame is verified at T+30, not T+10."""
    assert jgt.verification_instant(EVENT, 5, 17.0, 10.0) == EVENT + timedelta(minutes=30)
    assert jgt.verification_instant(EVENT, 5, 0.0, 10.0) == EVENT + timedelta(minutes=10)


def test_gauge_slot_is_the_identity_on_grid() -> None:
    on_grid = EVENT + timedelta(minutes=30)
    assert jgt.gauge_slot(on_grid) == on_grid


def test_gauge_slot_rounds_an_off_grid_instant() -> None:
    assert jgt.gauge_slot(EVENT + timedelta(minutes=32)) == EVENT + timedelta(minutes=30)
    assert jgt.gauge_slot(EVENT + timedelta(minutes=37)) == EVENT + timedelta(minutes=40)


# ---------------------------------------------------------------------------
# The wet rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mm, dur, expected, why",
    [
        (0.2, 0.0, 1, "amount alone is enough"),
        (0.1, None, 1, "the floor is inclusive, and a missing duration is fine"),
        (0.0, 3.0, 1, "duration alone rescues drizzle below the 0.1 mm floor"),
        (0.4, 5.0, 1, "both channels agree"),
        (0.0, 0.0, 0, "neither channel fires"),
        (0.0, None, 0, "no amount, no duration to fall back on"),
        (None, 5.0, None, "a missing AMOUNT is missing truth, duration or not"),
        (None, None, None, "nothing at all"),
    ],
)
def test_wet_rule(mm, dur, expected, why: str) -> None:
    assert jgt.wet_outcome(mm, dur, 0.1, 1.0) == expected, why


def test_trace_is_not_wet_by_amount_but_can_be_wet_by_duration() -> None:
    """DMI's -0.1 is "traces, less than 0.1 kg/m²": real precipitation
    that the gauge could not measure. It is never an amount — but the
    duration channel is exactly what such a slot is for."""
    assert jgt.wet_outcome(-0.1, 0.0, 0.1, 1.0) == 0
    assert jgt.wet_outcome(-0.1, None, 0.1, 1.0) == 0
    assert jgt.wet_outcome(-0.1, 2.0, 0.1, 1.0) == 1


def test_wet_thresholds_are_configurable() -> None:
    assert jgt.wet_outcome(0.3, None, 0.5, 1.0) == 0
    assert jgt.wet_outcome(0.6, None, 0.5, 1.0) == 1
    assert jgt.wet_outcome(0.0, 2.0, 0.1, 3.0) == 0
    assert jgt.wet_outcome(0.0, 3.0, 0.1, 3.0) == 1


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def _corpus_row(point_id: str, lead: int, *, age: float = 17.0,
                raw_prob: float = 0.5, outcome: int | None = 1) -> dict:
    return {
        "event_time": EVENT.isoformat(),
        "point_id": point_id,
        "lat": 55.47, "lon": 10.33, "region": "Denmark",
        "lead_min": lead, "raw_prob": raw_prob, "outcome": outcome,
        "sample_weight": 1.0, "frame_age_min": age, "error": None,
        "ensemble_size": 16, "n_cascade_levels": 6, "downsample_factor": 4,
        "threshold_mm_h": 0.5, "disc_radius_m": 1000.0, "detection_stat": "p90",
        "scan_type": "fullRange", "motion_method": "farneback",
        "timestep_min": 10.0, "n_timesteps": 8,
        "leads_min_csv": "5,10,20,30,45,60", "frame_age_range_csv": "12,18",
        "settings_hash": "deadbeefdeadbeef", "schema_version": 3,
    }


def _write_corpus(path: Path, rows: list[dict]) -> None:
    pq.write_table(pa.Table.from_pylist(rows, schema=bcc._parquet_schema()), path)


def _run_join(corpus: Path, corpus_dir: Path, out: Path, *extra: str):
    return subprocess.run(
        [sys.executable, str(REPO / "scripts" / "join_gauge_truth.py"),
         "--corpus", str(corpus), "--corpus-dir", str(corpus_dir),
         "--out", str(out), *extra],
        capture_output=True, text=True,
    )


def _end_to_end_store(corpus_dir: Path) -> None:
    """T+30 wet, T+40 dry, T+50 absent."""
    StationObsStore(corpus_dir).append([
        Observation("06126", EVENT + timedelta(minutes=30), "precip_past10min", 0.7),
        Observation("06126", EVENT + timedelta(minutes=30), "precip_dur_past10min", 6.0),
        Observation("06126", EVENT + timedelta(minutes=40), "precip_past10min", 0.0),
        Observation("06126", EVENT + timedelta(minutes=40), "precip_dur_past10min", 0.0),
        # T+50 is deliberately absent: lead 30 has no gauge truth.
    ])


def test_end_to_end_join(tmp_path: Path) -> None:
    """The default rule is cumulative, so lead 20 is a hit on the T+30
    shower its window covers — the same event ``p_rain[20]`` describes."""
    corpus_dir = tmp_path / "corpus"
    _end_to_end_store(corpus_dir)

    # Lead 5 @ age 17 ends at T+30; lead 20 @ age 17 at T+40; lead 30 at T+50.
    rows = [
        _corpus_row("06126", 5),
        _corpus_row("06126", 20),
        _corpus_row("06126", 30),
    ]
    corpus = tmp_path / "in.parquet"
    out = tmp_path / "out.parquet"
    _write_corpus(corpus, rows)

    rc = _run_join(corpus, corpus_dir, out)
    assert rc.returncode == 0, rc.stderr

    table = pq.read_table(out)
    # The readings stay the SNAPPED instant's, whatever the rule.
    assert table.column("gauge_mm").to_pylist() == [
        pytest.approx(0.7), pytest.approx(0.0), None,
    ]
    assert table.column("gauge_dur_min").to_pylist() == [
        pytest.approx(6.0), pytest.approx(0.0), None,
    ]
    # lead 5  → window {T+10, T+20, T+30}: wet at T+30.
    # lead 20 → window {T+10 … T+40}: still wet at T+30 — "rain within 20".
    # lead 30 → final slot T+50 missing: no truth.
    assert table.column("gauge_outcome").to_pylist() == [1, 1, None]
    assert table.column("gauge_outcome_rule").to_pylist() == ["within"] * 3


def test_outcome_rule_instant_is_the_old_single_slot_behaviour(tmp_path: Path) -> None:
    """Bit-for-bit the pre-2026-09-11 join: only the snapped slot counts,
    so the T+30 shower is a miss for a lead verified at T+40."""
    corpus_dir = tmp_path / "corpus"
    _end_to_end_store(corpus_dir)
    corpus = tmp_path / "in.parquet"
    out = tmp_path / "out.parquet"
    _write_corpus(corpus, [
        _corpus_row("06126", 5), _corpus_row("06126", 20), _corpus_row("06126", 30),
    ])

    rc = _run_join(corpus, corpus_dir, out, "--outcome-rule", "instant")
    assert rc.returncode == 0, rc.stderr
    table = pq.read_table(out)
    assert table.column("gauge_outcome").to_pylist() == [1, 0, None]
    assert table.column("gauge_mm").to_pylist() == [
        pytest.approx(0.7), pytest.approx(0.0), None,
    ]
    assert table.column("gauge_outcome_rule").to_pylist() == ["instant"] * 3
    # One slot per row, so the window amount IS the instant amount.
    assert table.column("gauge_mm_window").to_pylist() == [
        pytest.approx(0.7), pytest.approx(0.0), None,
    ]
    assert "rule=instant" in rc.stdout


def test_join_preserves_row_order_columns_and_types(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    StationObsStore(corpus_dir).append([
        Observation("06126", EVENT + timedelta(minutes=30), "precip_past10min", 0.3),
    ])
    rows = [
        _corpus_row("06126", 5, raw_prob=0.11),
        _corpus_row("06126", 20, raw_prob=0.22),
        _corpus_row("06126", 45, raw_prob=0.33),
    ]
    corpus = tmp_path / "in.parquet"
    out = tmp_path / "out.parquet"
    _write_corpus(corpus, rows)
    assert _run_join(corpus, corpus_dir, out).returncode == 0

    original = pq.read_table(corpus)
    joined = pq.read_table(out)
    assert joined.num_rows == original.num_rows
    # Every original column survives, unchanged, in the same order.
    assert joined.column_names[: original.num_columns] == original.column_names
    for name in original.column_names:
        assert joined.schema.field(name).type == original.schema.field(name).type
        assert joined.column(name).to_pylist() == original.column(name).to_pylist()
    # The three new columns, with the promised types.
    assert joined.column_names[original.num_columns:] == [
        "gauge_mm", "gauge_dur_min", "gauge_outcome",
        "gauge_mm_window", "gauge_outcome_rule",
    ]
    assert joined.schema.field("gauge_mm").type == pa.float32()
    assert joined.schema.field("gauge_dur_min").type == pa.float32()
    assert joined.schema.field("gauge_outcome").type == pa.int8()
    assert joined.schema.field("gauge_outcome").nullable
    assert joined.schema.field("gauge_mm_window").type == pa.float32()
    assert joined.schema.field("gauge_mm_window").nullable
    assert joined.schema.field("gauge_outcome_rule").type == pa.string()
    # The rule is in the file's metadata as well as on every row: a report
    # reading the schema alone must be able to say which event it scored.
    assert (joined.schema.metadata or {})[b"gauge_outcome_rule"] == b"within"


def test_missing_duration_alone_does_not_null_the_outcome(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    StationObsStore(corpus_dir).append([
        Observation("06126", EVENT + timedelta(minutes=30), "precip_past10min", 0.5),
    ])
    corpus = tmp_path / "in.parquet"
    out = tmp_path / "out.parquet"
    _write_corpus(corpus, [_corpus_row("06126", 5)])
    assert _run_join(corpus, corpus_dir, out).returncode == 0
    table = pq.read_table(out)
    assert table.column("gauge_dur_min").to_pylist() == [None]
    assert table.column("gauge_outcome").to_pylist() == [1]


def test_trace_slot_is_zero_mm_and_dry_without_duration(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    StationObsStore(corpus_dir).append([
        Observation("06126", EVENT + timedelta(minutes=30), "precip_past10min", -0.1),
    ])
    corpus = tmp_path / "in.parquet"
    out = tmp_path / "out.parquet"
    _write_corpus(corpus, [_corpus_row("06126", 5)])
    rc = _run_join(corpus, corpus_dir, out)
    assert rc.returncode == 0
    assert "traces_normalised=1" in rc.stdout
    table = pq.read_table(out)
    assert table.column("gauge_mm").to_pylist() == [pytest.approx(0.0)]
    assert table.column("gauge_outcome").to_pylist() == [0]


def test_wet_thresholds_flow_through_the_cli(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    StationObsStore(corpus_dir).append([
        Observation("06126", EVENT + timedelta(minutes=30), "precip_past10min", 0.3),
    ])
    corpus = tmp_path / "in.parquet"
    _write_corpus(corpus, [_corpus_row("06126", 5)])

    out_default = tmp_path / "a.parquet"
    assert _run_join(corpus, corpus_dir, out_default).returncode == 0
    assert pq.read_table(out_default).column("gauge_outcome").to_pylist() == [1]

    out_strict = tmp_path / "b.parquet"
    assert _run_join(corpus, corpus_dir, out_strict, "--wet-mm", "0.5").returncode == 0
    assert pq.read_table(out_strict).column("gauge_outcome").to_pylist() == [0]


def test_non_station_point_ids_join_to_nothing_and_say_so(tmp_path: Path) -> None:
    """The radar calibration corpus's point_ids are grid points, not
    stations, so a run against it must produce nulls and a clear warning
    rather than a crash or a silently empty file."""
    corpus_dir = tmp_path / "corpus"
    StationObsStore(corpus_dir).append([
        Observation("06126", EVENT + timedelta(minutes=30), "precip_past10min", 0.7),
    ])
    corpus = tmp_path / "in.parquet"
    out = tmp_path / "out.parquet"
    _write_corpus(corpus, [_corpus_row("fyn-centroid", 5), _corpus_row("bornholm-01", 20)])

    rc = _run_join(corpus, corpus_dir, out)
    assert rc.returncode == 0, rc.stderr
    assert "no corpus row matched a gauge slot" in rc.stdout
    table = pq.read_table(out)
    assert table.num_rows == 2
    assert table.column("gauge_outcome").to_pylist() == [None, None]
    assert table.column("gauge_mm").to_pylist() == [None, None]


def test_empty_gauge_archive_is_not_an_error(tmp_path: Path) -> None:
    corpus = tmp_path / "in.parquet"
    out = tmp_path / "out.parquet"
    _write_corpus(corpus, [_corpus_row("06126", 5)])
    rc = _run_join(corpus, tmp_path / "empty-corpus", out)
    assert rc.returncode == 0, rc.stderr
    assert pq.read_table(out).column("gauge_outcome").to_pylist() == [None]


def test_missing_corpus_file_exits_nonzero(tmp_path: Path) -> None:
    rc = _run_join(tmp_path / "nope.parquet", tmp_path, tmp_path / "out.parquet")
    assert rc.returncode == 2


def test_per_lead_counts_are_reported(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    StationObsStore(corpus_dir).append([
        Observation("06126", EVENT + timedelta(minutes=30), "precip_past10min", 0.7),
    ])
    corpus = tmp_path / "in.parquet"
    out = tmp_path / "out.parquet"
    _write_corpus(corpus, [_corpus_row("06126", 5), _corpus_row("06126", 20)])
    rc = _run_join(corpus, corpus_dir, out)
    assert "lead_min" in rc.stdout
    lines = [ln.split() for ln in rc.stdout.splitlines() if ln.strip().startswith(("5 ", "20 "))]
    counts = {int(ln[0]): (int(ln[2]), int(ln[3])) for ln in lines}
    assert counts == {5: (1, 0), 20: (0, 1)}


# ---------------------------------------------------------------------------
# --point-set: a union corpus holds gauge points AND radar grid points
#
# One STEPS run per event serves both point sets, so the station corpus and
# the national calibration corpus are now one build. Only the gauge points
# can join a gauge observation — the radar points are grid coordinates, not
# station ids — so the join takes the same --point-set selector the fit and
# the report do.
# ---------------------------------------------------------------------------


def test_point_set_selects_only_the_station_rows(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    store = StationObsStore(corpus_dir)
    store.append([
        Observation("06126", EVENT + timedelta(minutes=30), "precip_past10min", 0.7),
        Observation("06126", EVENT + timedelta(minutes=30), "precip_dur_past10min", 6.0),
    ])

    station_row = _corpus_row("06126", 5)
    station_row["point_set"] = "station_points"
    grid_row = _corpus_row("grid-0042", 5)
    grid_row["point_set"] = "calibration_points_v2"
    corpus = tmp_path / "union.parquet"
    out = tmp_path / "out.parquet"
    _write_corpus(corpus, [station_row, grid_row])

    proc = _run_join(corpus, corpus_dir, out, "--point-set", "station_points")
    assert proc.returncode == 0, proc.stderr
    table = pq.read_table(out)
    # Only the gauge point survives, and it joined.
    assert table.column("point_id").to_pylist() == ["06126"]
    assert table.column("point_set").to_pylist() == ["station_points"]
    assert table.column("gauge_outcome").to_pylist() == [1]
    assert "kept 1 rows" in proc.stdout


def test_point_set_all_keeps_every_row_as_before(tmp_path: Path) -> None:
    """Default behaviour is unchanged: non-station point_ids are kept with
    null gauge columns, exactly as a pre-union corpus behaved."""
    corpus_dir = tmp_path / "corpus"
    StationObsStore(corpus_dir).append([
        Observation("06126", EVENT + timedelta(minutes=30), "precip_past10min", 0.7),
        Observation("06126", EVENT + timedelta(minutes=30), "precip_dur_past10min", 6.0),
    ])
    rows = [_corpus_row("06126", 5), _corpus_row("grid-0042", 5)]
    rows[0]["point_set"] = "station_points"
    rows[1]["point_set"] = "calibration_points_v2"
    corpus = tmp_path / "union.parquet"
    out = tmp_path / "out.parquet"
    _write_corpus(corpus, rows)

    proc = _run_join(corpus, corpus_dir, out)
    assert proc.returncode == 0, proc.stderr
    table = pq.read_table(out)
    assert table.num_rows == 2
    assert table.column("gauge_outcome").to_pylist() == [1, None]


def test_point_set_on_a_corpus_without_the_column_is_refused(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    StationObsStore(corpus_dir)
    schema = pa.schema([
        f for f in bcc._parquet_schema() if f.name != "point_set"
    ])
    row = _corpus_row("06126", 5)
    row.pop("point_set", None)
    corpus = tmp_path / "old.parquet"
    pq.write_table(pa.Table.from_pylist([row], schema=schema), corpus)

    proc = _run_join(corpus, corpus_dir, tmp_path / "out.parquet",
                     "--point-set", "station_points")
    assert proc.returncode == 2
    assert "no point_set column" in proc.stderr


def test_unknown_point_set_is_refused_and_lists_what_exists(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    StationObsStore(corpus_dir)
    row = _corpus_row("06126", 5)
    row["point_set"] = "station_points"
    corpus = tmp_path / "union.parquet"
    _write_corpus(corpus, [row])

    proc = _run_join(corpus, corpus_dir, tmp_path / "out.parquet",
                     "--point-set", "not_a_set")
    assert proc.returncode == 2
    assert "station_points" in proc.stderr


# ---------------------------------------------------------------------------
# --outcome-rule: the gauge verdict must describe the SERVED event
#
# ``build_calibration_corpus`` has scored the radar truth cumulatively
# since 2026-09-09 — rain in ANY frame at T + j*step, j = 1..snapped/step —
# because the served probability is "rain WITHIN L". The gauge half went on
# verifying one instant, so the quality page's gauge reliability curve sat
# below the diagonal by construction. These tests pin the two rules to the
# same instants.
# ---------------------------------------------------------------------------


def _builder_offsets(lead: int, age: float, timestep: float) -> tuple[int, ...]:
    """The builder's own offset line, copied here as the reference.

    ``_gather_event_frames`` computes it inline, so it cannot be imported;
    this is that expression verbatim, and the test below holds
    ``window_offsets_min`` to it.
    """
    target = bcc.snap_lead_min(lead + age, timestep)
    step = int(round(timestep))
    return tuple(range(step, target, step)) + (target,)


@pytest.mark.parametrize("timestep", [5.0, 10.0])
@pytest.mark.parametrize("age", [0.0, 12.0, 17.0, 18.0])
@pytest.mark.parametrize("lead", [0, 5, 10, 15, 20, 30, 45, 60])
def test_window_offsets_are_the_builder_s_verification_instants(
    lead: int, age: float, timestep: float,
) -> None:
    assert jgt.window_offsets_min(lead, age, timestep) == _builder_offsets(
        lead, age, timestep,
    )


@pytest.mark.parametrize("lead", [0, 5, 10, 20, 45, 60])
@pytest.mark.parametrize("age", [0.0, 12.0, 17.5, 18.0])
def test_instant_rule_reads_only_the_snapped_instant(lead: int, age: float) -> None:
    offsets = jgt.window_offsets_min(lead, age, 10.0, jgt.OUTCOME_RULE_INSTANT)
    assert offsets == (bcc.snap_lead_min(lead + age, 10.0),)


def test_vectorised_snap_matches_the_scalar_one() -> None:
    """The join walks the window with numpy; a different rounding there
    would verify against a different instant than the builder did."""
    import numpy as np

    leads = [0, 5, 10, 15, 20, 30, 45, 60, 90]
    ages = [0.0, 3.0, 12.0, 15.0, 17.99738, 18.0]
    steps = [5.0, 10.0, 7.5]
    effective = np.array([lead + age for lead in leads for age in ages
                          for _ in steps], dtype=float)
    timestep = np.array([step for _ in leads for _ in ages for step in steps],
                        dtype=float)
    got = jgt.snap_lead_min_vec(effective, timestep)
    want = [bcc.snap_lead_min(e, s) for e, s in zip(effective, timestep)]
    assert got.tolist() == want


def test_vectorised_gauge_slot_matches_the_scalar_one() -> None:
    import numpy as np

    offsets = [0, 3, 5, 7, 10, 12, 17, 20, 32, 37, 40, 55, 80]
    instants = [EVENT + timedelta(minutes=off) for off in offsets]
    got = jgt.gauge_slot_sec_vec(
        np.array([int(i.timestamp()) for i in instants], dtype=np.int64)
    )
    want = [int(jgt.gauge_slot(i).timestamp()) for i in instants]
    assert got.tolist() == want


@pytest.mark.parametrize("wet_mm, wet_dur", [(0.1, 1.0), (0.5, 3.0)])
def test_vectorised_wet_rule_matches_the_scalar_one(wet_mm: float, wet_dur: float) -> None:
    import numpy as np

    readings = [None, -0.1, 0.0, 0.05, 0.1, 0.3, 0.6, 2.0]
    durations = [None, 0.0, 0.5, 1.0, 3.0, 10.0]
    raw = np.array(
        [np.nan if mm is None else mm for mm in readings for _ in durations],
        dtype=np.float32,
    )
    dur = np.array(
        [np.nan if d is None else d for _ in readings for d in durations],
        dtype=np.float32,
    )
    observed, wet, mm = jgt.slot_verdicts(raw, dur, wet_mm, wet_dur)
    want = [
        jgt.wet_outcome(a, b, wet_mm, wet_dur)
        for a in readings for b in durations
    ]
    got = [None if not ok else int(w) for ok, w in zip(observed, wet)]
    assert got == want
    # The trace sentinel is folded to 0.0 mm, never summed as an amount.
    assert mm[raw < 0.0].tolist() == [0.0] * int((raw < 0.0).sum())


def _wet_at_t20_store(corpus_dir: Path, station: str = "06126") -> None:
    """A station wet ONLY at T+20, dry at T+10/T+30/T+50, silent at T+40."""
    StationObsStore(corpus_dir).append([
        Observation(station, EVENT + timedelta(minutes=10), "precip_past10min", 0.0),
        Observation(station, EVENT + timedelta(minutes=10), "precip_dur_past10min", 0.0),
        Observation(station, EVENT + timedelta(minutes=20), "precip_past10min", 0.7),
        Observation(station, EVENT + timedelta(minutes=20), "precip_dur_past10min", 6.0),
        Observation(station, EVENT + timedelta(minutes=30), "precip_past10min", 0.0),
        Observation(station, EVENT + timedelta(minutes=30), "precip_dur_past10min", 0.0),
        # T+40 absent: an intermediate hole for lead 50, the FINAL slot for
        # lead 40.
        Observation(station, EVENT + timedelta(minutes=50), "precip_past10min", 0.0),
        Observation(station, EVENT + timedelta(minutes=50), "precip_dur_past10min", 0.0),
    ])


def _t20_rows() -> list[dict]:
    """Zero frame age, so lead L snaps to exactly T+L on the 10-min grid."""
    return [_corpus_row("06126", lead, age=0.0) for lead in (10, 20, 30, 40, 50)]


def test_within_scores_the_whole_window(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    _wet_at_t20_store(corpus_dir)
    corpus = tmp_path / "in.parquet"
    out = tmp_path / "out.parquet"
    _write_corpus(corpus, _t20_rows())

    rc = _run_join(corpus, corpus_dir, out)
    assert rc.returncode == 0, rc.stderr
    table = pq.read_table(out)

    # lead 10 → window {T+10}: ends BEFORE the shower, so a genuine 0.
    # lead 20 → {T+10, T+20}: the shower is inside it.
    # lead 30 → {T+10 … T+30}: still inside — "rain within 30" contains it.
    # lead 40 → final slot T+40 missing: no truth at all.
    # lead 50 → {T+10 … T+50} with T+40 skipped: the shower still counts.
    assert table.column("gauge_outcome").to_pylist() == [0, 1, 1, None, 1]
    assert table.column("gauge_outcome_rule").to_pylist() == ["within"] * 5


def test_instant_scores_only_the_snapped_slot(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    _wet_at_t20_store(corpus_dir)
    corpus = tmp_path / "in.parquet"
    out = tmp_path / "out.parquet"
    _write_corpus(corpus, _t20_rows())

    rc = _run_join(corpus, corpus_dir, out, "--outcome-rule", "instant")
    assert rc.returncode == 0, rc.stderr
    table = pq.read_table(out)
    # Only the lead whose snapped instant IS T+20 is a hit.
    assert table.column("gauge_outcome").to_pylist() == [0, 1, 0, None, 0]
    assert table.column("gauge_outcome_rule").to_pylist() == ["instant"] * 5
    assert pq.read_schema(out).metadata[b"gauge_outcome_rule"] == b"instant"


def test_missing_final_slot_nulls_the_row_under_both_rules(tmp_path: Path) -> None:
    """A lead is DEFINED by its snapped instant: a hole there is no truth,
    exactly as a missing verification frame is for the radar outcome."""
    corpus_dir = tmp_path / "corpus"
    _wet_at_t20_store(corpus_dir)
    corpus = tmp_path / "in.parquet"
    _write_corpus(corpus, [_corpus_row("06126", 40, age=0.0)])
    for extra in ((), ("--outcome-rule", "instant")):
        out = tmp_path / f"out{len(extra)}.parquet"
        assert _run_join(corpus, corpus_dir, out, *extra).returncode == 0
        table = pq.read_table(out)
        assert table.column("gauge_outcome").to_pylist() == [None]
        assert table.column("gauge_mm").to_pylist() == [None]


def test_missing_intermediate_slot_still_scores_and_is_counted(tmp_path: Path) -> None:
    """A 10-minute hole in the middle of a window must not throw the whole
    lead away — it can only ever hide rain, so it is skipped and counted."""
    corpus_dir = tmp_path / "corpus"
    _wet_at_t20_store(corpus_dir)
    corpus = tmp_path / "in.parquet"
    out = tmp_path / "out.parquet"
    # Lead 50's window is {T+10 … T+50}; T+40 is missing, T+50 is not.
    _write_corpus(corpus, [_corpus_row("06126", 50, age=0.0)])

    rc = _run_join(corpus, corpus_dir, out)
    assert rc.returncode == 0, rc.stderr
    assert pq.read_table(out).column("gauge_outcome").to_pylist() == [1]
    # The per-run summary names the gap rather than swallowing it.
    assert "intermediate_slots_missing=1" in rc.stdout
    assert "rows_with_a_gap=1" in rc.stdout
    assert "slots_scored=4" in rc.stdout


def test_window_amount_sums_the_observed_slots(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    _wet_at_t20_store(corpus_dir)
    corpus = tmp_path / "in.parquet"
    out = tmp_path / "out.parquet"
    _write_corpus(corpus, _t20_rows())
    assert _run_join(corpus, corpus_dir, out).returncode == 0
    table = pq.read_table(out)

    # gauge_mm stays the SNAPPED instant's reading …
    assert table.column("gauge_mm").to_pylist() == [
        pytest.approx(0.0), pytest.approx(0.7), pytest.approx(0.0),
        None, pytest.approx(0.0),
    ]
    # … while gauge_mm_window is the window's own amount. It is a READING,
    # not a verdict, so lead 40 keeps the 0.7 its three observed slots saw
    # even though its final slot — and therefore its outcome — is missing.
    assert table.column("gauge_mm_window").to_pylist() == [
        pytest.approx(0.0), pytest.approx(0.7), pytest.approx(0.7),
        pytest.approx(0.7), pytest.approx(0.7),
    ]


def test_window_amount_is_null_when_no_slot_was_observed(tmp_path: Path) -> None:
    corpus = tmp_path / "in.parquet"
    out = tmp_path / "out.parquet"
    _write_corpus(corpus, [_corpus_row("06126", 20, age=0.0)])
    assert _run_join(corpus, tmp_path / "empty-corpus", out).returncode == 0
    table = pq.read_table(out)
    assert table.column("gauge_mm_window").to_pylist() == [None]
    assert table.column("gauge_outcome").to_pylist() == [None]
    assert table.column("gauge_outcome_rule").to_pylist() == ["within"]


def test_dead_gauge_exclusion_still_applies_to_the_window_rule(tmp_path: Path) -> None:
    """A bucket stuck at zero must not grade anything, however many slots
    the cumulative window now reads. Its READINGS are still written out."""
    corpus_dir = tmp_path / "corpus"
    _wet_at_t20_store(corpus_dir)          # 06126 saw rain: the window is wet
    StationObsStore(corpus_dir).append([   # 06999 reported plenty, never wet
        Observation("06999", EVENT + timedelta(minutes=10), "precip_past10min", 0.0),
        Observation("06999", EVENT + timedelta(minutes=20), "precip_past10min", 0.0),
        Observation("06999", EVENT + timedelta(minutes=30), "precip_past10min", 0.0),
    ])
    corpus = tmp_path / "in.parquet"
    out = tmp_path / "out.parquet"
    _write_corpus(corpus, [
        _corpus_row("06126", 30, age=0.0), _corpus_row("06999", 30, age=0.0),
    ])

    rc = _run_join(corpus, corpus_dir, out, "--min-known-slots", "3")
    assert rc.returncode == 0, rc.stderr
    assert "dead gauge 06999" in rc.stdout
    table = pq.read_table(out)
    assert table.column("gauge_outcome").to_pylist() == [1, None]
    # The readings survive — they are what the station said.
    assert table.column("gauge_mm").to_pylist() == [
        pytest.approx(0.0), pytest.approx(0.0),
    ]
    assert table.column("gauge_mm_window").to_pylist() == [
        pytest.approx(0.7), pytest.approx(0.0),
    ]
    assert table.column("gauge_outcome_rule").to_pylist() == ["within", "within"]


def test_dead_gauge_rule_can_be_switched_off(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    _wet_at_t20_store(corpus_dir)
    StationObsStore(corpus_dir).append([
        Observation("06999", EVENT + timedelta(minutes=30), "precip_past10min", 0.0),
    ])
    corpus = tmp_path / "in.parquet"
    out = tmp_path / "out.parquet"
    _write_corpus(corpus, [_corpus_row("06999", 30, age=0.0)])
    rc = _run_join(corpus, corpus_dir, out, "--min-known-slots", "0")
    assert rc.returncode == 0, rc.stderr
    assert "dead gauge" not in rc.stdout
    assert pq.read_table(out).column("gauge_outcome").to_pylist() == [0]


def test_an_unknown_outcome_rule_is_refused(tmp_path: Path) -> None:
    corpus = tmp_path / "in.parquet"
    _write_corpus(corpus, [_corpus_row("06126", 20)])
    rc = _run_join(corpus, tmp_path / "corpus", tmp_path / "out.parquet",
                   "--outcome-rule", "eventually")
    assert rc.returncode != 0
    assert "outcome-rule" in rc.stderr


def test_a_corpus_with_no_rows_is_not_an_error(tmp_path: Path) -> None:
    corpus = tmp_path / "in.parquet"
    out = tmp_path / "out.parquet"
    _write_corpus(corpus, [])
    rc = _run_join(corpus, tmp_path / "corpus", out)
    assert rc.returncode == 0, rc.stderr
    table = pq.read_table(out)
    assert table.num_rows == 0
    assert "gauge_mm_window" in table.column_names
    assert "gauge_outcome_rule" in table.column_names


@pytest.mark.parametrize("timestep", [0.0, None, 0.4])
def test_an_unusable_timestep_joins_to_null_without_warnings(
    tmp_path: Path, timestep: float | None,
) -> None:
    """No timestep, no window: the row gets a null verdict rather than an
    exception or a numpy warning on stderr."""
    corpus_dir = tmp_path / "corpus"
    _wet_at_t20_store(corpus_dir)
    row = _corpus_row("06126", 20, age=0.0)
    row["timestep_min"] = timestep
    corpus = tmp_path / "in.parquet"
    out = tmp_path / "out.parquet"
    _write_corpus(corpus, [row])

    rc = _run_join(corpus, corpus_dir, out)
    assert rc.returncode == 0, rc.stderr
    assert rc.stderr.strip() == ""
    assert pq.read_table(out).column("gauge_outcome").to_pylist() == [None]
