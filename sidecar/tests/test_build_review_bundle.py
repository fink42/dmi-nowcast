"""The bundle builder CLI (``scripts/build_review_bundle.py``).

The script lives at the repo root but needs core **plus** sidecar
(``review`` for the population, ``review_frames`` for the imagery), so it
is tested from this suite — the same reasoning as
``test_replay_warnings.py``.

Everything here runs through ``--fixture``, which synthesises a whole
bundle with no corpus, no parquet and no HDF5. That mode is not a testing
convenience bolted on: it is how the frontend is developed while the
corporate VPN blocks the VM, so it is worth testing in its own right.

The assertions concentrate on the places where a plausible-looking bundle
would quietly mislead a reviewer — a timestamp the browser cannot parse, a
frame claimed present that was never rendered, a rule that cannot be
reproduced — rather than on the shape of the documents, which
``dmi_nowcast_sidecar.review`` already pins from the inside.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import build_review_bundle as brb  # noqa: E402

_LOOKS_LIKE_A_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")


# ---------------------------------------------------------------------------
# One build, shared by the tests that only read it
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bundle(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("bundle")
    corpus = tmp_path_factory.mktemp("fixture-corpus")
    brb.main([
        "--fixture", "--seed", "4242",
        "--out-dir", str(out), "--fixture-corpus", str(corpus),
    ])
    return out


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _details(bundle: Path) -> list[dict]:
    return [_read(p) for p in sorted((bundle / "events").glob("*.json"))]


# ---------------------------------------------------------------------------
# The bundle is complete and self-contained
# ---------------------------------------------------------------------------

def test_fixture_mode_writes_a_whole_bundle(bundle: Path) -> None:
    for name in ("manifest.json", "events.json", "stations.json", "tags.json"):
        assert (bundle / name).is_file(), name
    index = _read(bundle / "events.json")
    assert index["events"], "a bundle with no events is not a bundle"
    for row in index["events"]:
        assert (bundle / row["detail"]).is_file(), row["event_id"]
    assert list((bundle / "frames").glob("*.png"))


def test_the_synthetic_archive_is_an_input_not_part_of_the_bundle(
    tmp_path: Path,
) -> None:
    """``--fixture`` must not ship its own scratch HDF5 to the reviewer.

    It is an input the builder happens to fabricate. Written inside the
    bundle it would be rsynced to the laptop and served by the review
    server alongside the real documents.
    """
    out, corpus = tmp_path / "b", tmp_path / "c"
    brb.main([
        "--fixture", "--out-dir", str(out), "--fixture-corpus", str(corpus),
    ])
    assert list(corpus.rglob("*.h5")), "the fixture archive should exist"
    assert not list(out.rglob("*.h5")), "…but not inside the bundle"


def test_every_timestamp_is_strict_iso_8601(bundle: Path) -> None:
    """``str(datetime)`` uses a SPACE separator, which some JS engines
    refuse. The failure mode is not an exception — it is 'NaN minutes
    before the onset' rendered in the timeline, which survives a demo.
    """
    # Documentation subtrees map COLUMN NAMES to prose, and one of the
    # columns is itself called `hour_utc`. Its value is a sentence, not an
    # instant, so the whole subtree is skipped rather than pattern-matched.
    skip = (".feature_doc", ".documentation", ".caveats", "_comment")

    def walk(node, path="") -> None:
        if any(part in path for part in skip):
            return
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, f"{path}[{i}]")
        elif isinstance(node, str) and path.endswith("_utc") and node:
            assert _LOOKS_LIKE_A_DATE.match(node), f"{path} is not a date: {node!r}"
            assert "T" in node, f"{path} is not ISO 8601: {node!r}"
            parsed = datetime.fromisoformat(node)
            assert parsed.tzinfo is not None, f"{path} has no offset: {node!r}"

    walk(_read(bundle / "manifest.json"), "manifest")
    walk(_read(bundle / "events.json"), "index")
    for detail in _details(bundle):
        walk(detail, "detail")


# ---------------------------------------------------------------------------
# Presence, and the difference between "no" and "don't know"
# ---------------------------------------------------------------------------

def test_a_rendered_bundle_reports_its_frames_present(bundle: Path) -> None:
    for detail in _details(bundle):
        assert detail["frames"], detail["event_id"]
        assert all(frame["present"] is True for frame in detail["frames"])


def test_no_frames_leaves_presence_unknown_rather_than_false(
    tmp_path: Path,
) -> None:
    """``--no-frames`` has not got the imagery; it has not proved it absent.

    False would say "we looked and there is no composite for that instant",
    which is a claim about the archive. None says "nobody rendered", which
    is the truth. The grid is null for the same reason.
    """
    out = tmp_path / "b"
    brb.main([
        "--fixture", "--no-frames", "--out-dir", str(out),
        "--fixture-corpus", str(tmp_path / "c"),
    ])
    assert _read(out / "manifest.json")["grid"] is None
    for detail in _details(out):
        assert all(frame["present"] is None for frame in detail["frames"])


# ---------------------------------------------------------------------------
# Provenance: can a reader tell whether to trust this?
# ---------------------------------------------------------------------------

def test_the_manifest_states_how_the_probability_was_obtained(
    bundle: Path,
) -> None:
    """The single field that decides whether the sample is honest.

    Out-of-fold means the post-processing model never saw the event it is
    judging. Anything else means it partly did, which biases the sample
    toward the model's residual failures.
    """
    rule = _read(bundle / "manifest.json")["rule"]
    assert rule["probability_provenance"]
    assert "held_out" in rule


def test_the_manifest_carries_the_sample_s_reproducibility(
    bundle: Path,
) -> None:
    sampling = _read(bundle / "manifest.json")["sampling"]
    assert sampling["seed"] == 4242
    assert sampling["population_hash"]


def test_the_same_seed_reproduces_the_same_events(tmp_path: Path) -> None:
    ids = []
    for run in ("a", "b"):
        out = tmp_path / run
        brb.main([
            "--fixture", "--seed", "99", "--out-dir", str(out),
            "--fixture-corpus", str(tmp_path / f"c{run}"), "--no-frames",
        ])
        index = _read(out / "events.json")
        ids.append([row["event_id"] for row in index["events"]])
    assert ids[0] == ids[1], "a seeded draw that moves cannot be compared"


def test_a_bundle_id_is_stable_for_a_stable_draw() -> None:
    """Annotations key on the bundle id, so a resumed build must not
    orphan a reviewer's judgements."""
    when = datetime(2026, 9, 16, tzinfo=timezone.utc)
    first = brb._bundle_id(7, "abc123", when)
    assert first == brb._bundle_id(7, "abc123", when)
    assert first != brb._bundle_id(8, "abc123", when)
    assert first != brb._bundle_id(7, "different", when)


def test_caveats_are_structured_even_when_the_builder_writes_prose(
    bundle: Path,
) -> None:
    """The UI ranks caveats by severity, so every one needs the shape."""
    for caveat in _read(bundle / "manifest.json")["caveats"]:
        assert set(caveat) >= {"code", "severity", "detail"}
        assert caveat["severity"] in ("low", "medium", "high")


# ---------------------------------------------------------------------------
# Neighbours
# ---------------------------------------------------------------------------

def test_neighbours_carry_a_position_so_the_map_can_place_them(
    bundle: Path,
) -> None:
    placed = 0
    for detail in _details(bundle):
        for entry in (detail.get("neighbours") or {}).get("stations") or []:
            assert "lat" in entry and "lon" in entry and "bearing_deg" in entry
            if entry["lat"] is not None:
                placed += 1
                assert -90 <= entry["lat"] <= 90
                assert -180 <= entry["lon"] <= 180
                assert 0 <= entry["bearing_deg"] < 360
    assert placed, "the fixture should place at least one neighbour"


def test_an_unplaceable_neighbour_gets_nulls_not_a_zero_position() -> None:
    """A dot at (0, 0) is in the Gulf of Guinea. Null draws nothing."""
    detail = {
        "station": {"lat": 55.0, "lon": 10.0},
        "neighbours": {"stations": [{"station_id": "nope"}]},
    }
    brb._place_neighbours(detail, {})
    entry = detail["neighbours"]["stations"][0]
    assert entry["lat"] is None and entry["lon"] is None
    assert entry["bearing_deg"] is None


def test_bearing_is_measured_from_the_station_toward_the_neighbour() -> None:
    """Due north is 0, due east 90 — the convention the UI renders as a
    compass direction, and the thing that separates a diverted cell from
    one that had already passed."""
    assert brb._bearing_deg(55.0, 10.0, 56.0, 10.0) == pytest.approx(0.0, abs=0.5)
    assert brb._bearing_deg(55.0, 10.0, 55.0, 11.0) == pytest.approx(90.0, abs=0.5)
    assert brb._bearing_deg(55.0, 10.0, 54.0, 10.0) == pytest.approx(180.0, abs=0.5)
    assert brb._bearing_deg(55.0, 10.0, 55.0, 9.0) == pytest.approx(270.0, abs=0.5)


# ---------------------------------------------------------------------------
# Refusals: fail fast, and say which flag
# ---------------------------------------------------------------------------

def test_lomo_without_its_fold_thresholds_is_refused_up_front() -> None:
    """Fitting a fold is a full sweep per month, so the thresholds are an
    input. Refused before any I/O, not deep inside rule validation."""
    with pytest.raises(SystemExit) as excinfo:
        brb.main([
            "--out-dir", "/tmp/never", "--corpus-dir", "/tmp/never",
            "--rule-source", "lomo", "--threshold-pct", "60",
        ])
    assert "--fold-thresholds" in str(excinfo.value)


def test_fold_thresholds_without_lomo_are_refused() -> None:
    with pytest.raises(SystemExit) as excinfo:
        brb.main([
            "--out-dir", "/tmp/never", "--corpus-dir", "/tmp/never",
            "--fold-thresholds", "/tmp/never.json", "--threshold-pct", "60",
        ])
    assert "lomo" in str(excinfo.value)


def test_a_missing_points_file_names_the_flag_that_fixes_it(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        brb.main([
            "--out-dir", str(tmp_path / "b"),
            "--corpus-dir", str(tmp_path / "empty"),
            "--threshold-pct", "60",
        ])
    message = str(excinfo.value)
    assert "station_points.json" in message and "--points" in message


def test_fold_thresholds_parse_from_year_month_keys(tmp_path: Path) -> None:
    path = tmp_path / "folds.json"
    path.write_text('{"2026-03": 60, "2026-04": {"threshold_pct": 65}}')
    assert brb._load_fold_thresholds(str(path)) == {
        (2026, 3): 60, (2026, 4): 65,
    }


def test_a_threshold_table_is_read_by_lead(tmp_path: Path) -> None:
    document = {"leads": {"30": {"threshold_pct": 60}, "45": {"threshold_pct": 70}}}
    assert brb._threshold_for_lead(document, 30, None) == 60
    assert brb._threshold_for_lead(document, 45, None) == 70
    with pytest.raises(SystemExit):
        brb._threshold_for_lead(document, 20, None)


def test_tree_labels_name_the_source_a_reviewer_will_read(tmp_path: Path) -> None:
    """``row_source`` answers "replay row or live row?", which matters
    because the live rows are the ones that lost their feature columns."""
    assert brb._tree_label(Path("/c/stations/eval")) == "live"
    assert brb._tree_label(
        Path("/c/stations/replay_postprocess/decisions"),
    ) == "replay_postprocess"


def test_naive_datetimes_are_refused_rather_than_assumed_utc() -> None:
    """A naive stamp would be off by an hour or two for exactly the summer
    convective events this review cares most about."""
    with pytest.raises(ValueError):
        brb._json_default(datetime(2026, 6, 1, 12, 0))
    assert brb._json_default(
        datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc),
    ) == "2026-06-01T12:00:00+00:00"

# ---------------------------------------------------------------------------
# The awkward shapes
# ---------------------------------------------------------------------------

def test_fixture_stress_forces_the_shapes_a_good_day_never_produces(
    tmp_path: Path,
) -> None:
    """A happy-path bundle proves nothing about the failure shapes.

    The frontend's drift alarm parses real producer output, so it can only
    cover shapes the producer actually emitted. Missing frames, their
    reasons, `present: false` and the event flags only appear when
    something went wrong — and those are precisely the ones a reviewer most
    needs rendered correctly, because a frozen repeat of the previous frame
    across a gap reads as "nothing changed" rather than "no picture".

    This also caught a real interface bug: `write_frames` reports missing
    frames as stamp STRINGS while `build_event` wants instants, which the
    happy path could never reveal.
    """
    out = tmp_path / "b"
    brb.main([
        "--fixture", "--fixture-stress", "--seed", "424242",
        "--out-dir", str(out), "--fixture-corpus", str(tmp_path / "c"),
    ])
    frames = _read(out / "manifest.json")["frames"]
    assert frames["missing"], "no frame went missing"
    assert frames["missing_reasons"], "a missing frame with no reason is a mystery"
    for stamp in frames["missing"]:
        assert stamp in frames["missing_reasons"]

    absent = 0
    flags: set[str] = set()
    for detail in _details(out):
        flags |= set(detail.get("flags") or [])
        absent += sum(1 for f in detail["frames"] if f["present"] is False)
    assert absent, "a missing composite must show as a frame absent, not present"
    assert flags, "the stress bundle should raise at least one event flag"


def test_every_event_flag_is_one_the_schema_documents(tmp_path: Path) -> None:
    """An undocumented flag is a badge the reviewer cannot interpret."""
    from dmi_nowcast_core import review_schema

    out = tmp_path / "b"
    brb.main([
        "--fixture", "--fixture-stress", "--seed", "424242",
        "--out-dir", str(out), "--fixture-corpus", str(tmp_path / "c"),
        "--no-frames",
    ])
    for detail in _details(out):
        for flag in detail.get("flags") or []:
            assert flag in review_schema.EVENT_FLAGS, flag
