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


def test_end_to_end_join(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    store = StationObsStore(corpus_dir)
    # Lead 5 @ age 17 verifies at T+30; lead 20 @ age 17 verifies at T+40.
    store.append([
        Observation("06126", EVENT + timedelta(minutes=30), "precip_past10min", 0.7),
        Observation("06126", EVENT + timedelta(minutes=30), "precip_dur_past10min", 6.0),
        Observation("06126", EVENT + timedelta(minutes=40), "precip_past10min", 0.0),
        Observation("06126", EVENT + timedelta(minutes=40), "precip_dur_past10min", 0.0),
        # T+50 is deliberately absent: lead 30 has no gauge truth.
    ])

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
    assert table.column("gauge_mm").to_pylist() == [
        pytest.approx(0.7), pytest.approx(0.0), None,
    ]
    assert table.column("gauge_dur_min").to_pylist() == [
        pytest.approx(6.0), pytest.approx(0.0), None,
    ]
    assert table.column("gauge_outcome").to_pylist() == [1, 0, None]


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
    ]
    assert joined.schema.field("gauge_mm").type == pa.float32()
    assert joined.schema.field("gauge_dur_min").type == pa.float32()
    assert joined.schema.field("gauge_outcome").type == pa.int8()
    assert joined.schema.field("gauge_outcome").nullable


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
