"""The fitted push-thresholds contract (Phase G, G1).

Two things are under test and they are the two ways this file can hurt
somebody:

0. **The stability guard damps a refit.** The sweep runs nightly on a
   window that slides by a day; most refits differ by noise. A rule that
   changed under a subscriber every night for no measured reason would be
   worse than one that is merely a percent off.
1. **The validator rejects what the service must not read.** A wrong
   version, a lead key that is not a lead, a percentage outside (0, 100),
   a lead claiming both "insufficient evidence" and a threshold — each one
   would otherwise become a notification rule nobody chose.
2. **The reader is total.** ``effective_threshold`` is called at request
   time, so every input — a missing lead, a null pick, a document that is
   not a document — has to produce a number rather than an exception.

No files, no network: the document is a dict built here.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from dmi_nowcast_core.push_thresholds import (
    DEFAULT_FALLBACK_THRESHOLD_PCT,
    SCHEMA_VERSION,
    apply_stability_guard,
    effective_threshold,
    load_thresholds,
    validate_thresholds,
)


def _lead(**overrides) -> dict:
    row = {
        "threshold_pct": 45,
        "insufficient": False,
        "f1": 0.42,
        "precision": 0.48,
        "recall": 0.37,
        "far": 0.55,
        "csi": 0.27,
        "warnings": 210,
        "hits": 94,
        "false_alarms": 101,
        "misses": 150,
        "late": 15,
        "plateau": [40, 55],
        "radar_plateau": [35, 60],
        "agrees_with_radar": True,
    }
    row.update(overrides)
    return row


def _doc(**overrides) -> dict:
    doc = {
        "schema_version": SCHEMA_VERSION,
        "fitted_at_utc": "2026-09-05T02:11:07+00:00",
        "objective": {
            "metric": "f1",
            "min_useful_lead_min": 5.0,
            "plateau_frac": 0.95,
            "min_warnings": 30,
            "rearm_after_min": 60,
            "persistence_obs": 1,
            "tolerance_min": 10,
            "dry_min": 30,
        },
        "window": {
            "from": "2026-07-01T00:00:00+00:00",
            "to": "2026-09-01T00:00:00+00:00",
            "days": 62,
            "stations": 97,
            "rows": 1841203,
        },
        "fallback_threshold_pct": 40,
        "leads": {
            "20": _lead(threshold_pct=50, plateau=[45, 55]),
            "30": _lead(),
            "60": _lead(
                threshold_pct=None, insufficient=True, f1=None,
                precision=None, recall=None, far=None, csi=None,
                warnings=0, hits=0, false_alarms=0, misses=0, late=0,
                plateau=None, radar_plateau=None, agrees_with_radar=None,
            ),
        },
    }
    doc.update(overrides)
    return doc


# ---------------------------------------------------------------------------
# The validator
# ---------------------------------------------------------------------------


class TestValidator:
    def test_a_well_formed_document_has_no_problems(self) -> None:
        assert validate_thresholds(_doc()) == []

    def test_a_wrong_version_is_a_problem(self) -> None:
        problems = validate_thresholds(_doc(schema_version=2))
        assert any("schema_version" in p for p in problems)

    def test_a_non_document_is_rejected_whole(self) -> None:
        assert validate_thresholds("40 %")
        assert validate_thresholds(None)

    def test_a_missing_objective_field_is_caught(self) -> None:
        doc = _doc()
        del doc["objective"]["min_useful_lead_min"]
        assert any(
            "objective.min_useful_lead_min" in p
            for p in validate_thresholds(doc)
        )

    def test_a_bad_timestamp_is_caught(self) -> None:
        assert any(
            "fitted_at_utc" in p
            for p in validate_thresholds(_doc(fitted_at_utc="last Tuesday"))
        )

    @pytest.mark.parametrize("fallback", [0, 100, 140, "40", True, None])
    def test_the_fallback_must_be_a_whole_percent(self, fallback) -> None:
        assert any(
            "fallback_threshold_pct" in p
            for p in validate_thresholds(_doc(fallback_threshold_pct=fallback))
        )

    def test_a_lead_key_that_is_not_a_lead_is_caught(self) -> None:
        problems = validate_thresholds(_doc(leads={"soon": _lead()}))
        assert any("positive whole minute" in p for p in problems)

    def test_a_threshold_out_of_range_is_caught(self) -> None:
        doc = _doc(leads={"30": _lead(threshold_pct=140)})
        assert any("threshold_pct" in p for p in validate_thresholds(doc))

    def test_insufficient_evidence_cannot_carry_a_threshold(self) -> None:
        """The one contradiction the writer could produce."""
        doc = _doc(leads={"30": _lead(insufficient=True, threshold_pct=45)})
        assert any("insufficient" in p for p in validate_thresholds(doc))

    def test_a_missing_lead_field_is_caught(self) -> None:
        row = _lead()
        del row["late"]
        assert any(
            "leads.30.late" in p
            for p in validate_thresholds(_doc(leads={"30": row}))
        )

    def test_a_count_may_not_be_null(self) -> None:
        doc = _doc(leads={"30": _lead(hits=None)})
        assert any("leads.30.hits" in p for p in validate_thresholds(doc))

    def test_a_rate_may_be_null(self) -> None:
        """An unmeasurable rate is null; that is the honest value."""
        doc = _doc(leads={"30": _lead(far=None, csi=None)})
        assert validate_thresholds(doc) == []

    @pytest.mark.parametrize(
        "plateau", [[55, 40], [40], [40, 140], ["40", "55"], 40],
    )
    def test_a_malformed_plateau_is_caught(self, plateau) -> None:
        doc = _doc(leads={"30": _lead(plateau=plateau)})
        assert any("plateau" in p for p in validate_thresholds(doc))

    def test_a_plateau_written_as_absent_is_caught(self) -> None:
        row = _lead()
        del row["radar_plateau"]
        assert any(
            "radar_plateau" in p
            for p in validate_thresholds(_doc(leads={"30": row}))
        )

    def test_agreement_must_be_a_boolean_or_null(self) -> None:
        doc = _doc(leads={"30": _lead(agrees_with_radar="mostly")})
        assert any("agrees_with_radar" in p for p in validate_thresholds(doc))


# ---------------------------------------------------------------------------
# The reader
# ---------------------------------------------------------------------------


class TestEffectiveThreshold:
    def test_a_fitted_lead_returns_its_pick(self) -> None:
        doc = _doc()
        assert effective_threshold(doc, 20) == 50
        assert effective_threshold(doc, 30) == 45

    def test_a_lead_the_table_does_not_carry_falls_back(self) -> None:
        assert effective_threshold(_doc(), 45) == 40

    def test_an_insufficient_lead_falls_back(self) -> None:
        assert effective_threshold(_doc(), 60) == 40

    def test_the_documents_own_fallback_is_honoured(self) -> None:
        assert effective_threshold(_doc(fallback_threshold_pct=35), 45) == 35

    def test_a_string_lead_is_read_as_a_number(self) -> None:
        assert effective_threshold(_doc(), "30") == 45

    @pytest.mark.parametrize(
        "junk", [None, "thresholds", 42, {}, {"leads": "none"}],
    )
    def test_a_document_that_is_not_a_document_still_answers(self, junk) -> None:
        assert effective_threshold(junk, 30) == DEFAULT_FALLBACK_THRESHOLD_PCT

    def test_a_lead_that_is_not_a_lead_still_answers(self) -> None:
        assert effective_threshold(_doc(), "soon") == 40
        assert effective_threshold(_doc(), None) == 40

    def test_a_nonsense_threshold_in_the_table_falls_back(self) -> None:
        doc = _doc(leads={"30": _lead(threshold_pct=0)})
        assert effective_threshold(doc, 30) == 40


# ---------------------------------------------------------------------------
# Loading from disk
# ---------------------------------------------------------------------------


class TestLoad:
    def test_a_good_file_round_trips(self, tmp_path: Path) -> None:
        path = tmp_path / "push_thresholds.json"
        path.write_text(json.dumps(_doc()))
        loaded = load_thresholds(path)
        assert loaded is not None
        assert effective_threshold(loaded, 30) == 45

    def test_an_absent_file_is_none(self, tmp_path: Path) -> None:
        assert load_thresholds(tmp_path / "nothing.json") is None
        assert load_thresholds(None) is None

    def test_a_broken_file_is_none_rather_than_half_a_rule(
        self, tmp_path: Path,
    ) -> None:
        broken = tmp_path / "broken.json"
        broken.write_text("{not json")
        assert load_thresholds(broken) is None

        invalid = tmp_path / "invalid.json"
        invalid.write_text(json.dumps(_doc(schema_version=99)))
        assert load_thresholds(invalid) is None


# ---------------------------------------------------------------------------
# The stability guard
# ---------------------------------------------------------------------------


class TestStabilityGuard:
    """What a nightly refit is allowed to change, and what it is not.

    The fixture's ``_doc()`` has three leads and they cover the three
    interesting shapes: 20 carries 50 %, 30 carries 45 %, 60 has no pick
    at all.
    """

    def test_without_a_previous_table_every_pick_is_a_first_fit(self) -> None:
        out = apply_stability_guard(_doc(), None)
        assert out["leads"]["20"]["threshold_pct"] == 50
        assert out["leads"]["20"]["guard"] == "first_fit"
        assert out["leads"]["30"]["guard"] == "first_fit"
        # The unfittable lead is marked as such, not promoted.
        assert out["leads"]["60"]["guard"] == "insufficient"
        assert out["leads"]["60"]["threshold_pct"] is None

    def test_a_big_well_evidenced_move_is_taken(self) -> None:
        previous = _doc(leads={"30": _lead(threshold_pct=30)})
        new = _doc(leads={"30": _lead(threshold_pct=45, warnings=210)})
        out = apply_stability_guard(new, previous)
        assert out["leads"]["30"]["threshold_pct"] == 45
        assert out["leads"]["30"]["guard"] == "changed"

    def test_a_small_move_keeps_the_served_value(self) -> None:
        """44 → 45 is one point of noise, not a new rule."""
        previous = _doc(leads={"30": _lead(threshold_pct=44, f1=0.40)})
        new = _doc(leads={"30": _lead(threshold_pct=45, warnings=500)})
        out = apply_stability_guard(new, previous)
        row = out["leads"]["30"]
        assert row["threshold_pct"] == 44
        assert row["guard"] == "kept_previous"
        # The rejected candidate is recorded, and the counts described are
        # the served threshold's, not the candidate's.
        assert row["candidate_threshold_pct"] == 45
        assert row["f1"] == 0.40

    def test_a_thinly_evidenced_move_keeps_the_served_value(self) -> None:
        """Far enough away, but fitted on 12 warnings."""
        previous = _doc(leads={"30": _lead(threshold_pct=30)})
        new = _doc(leads={"30": _lead(threshold_pct=60, warnings=12)})
        out = apply_stability_guard(new, previous)
        assert out["leads"]["30"]["threshold_pct"] == 30
        assert out["leads"]["30"]["guard"] == "kept_previous"
        assert out["leads"]["30"]["candidate_threshold_pct"] == 60

    def test_the_thresholds_are_configurable(self) -> None:
        previous = _doc(leads={"30": _lead(threshold_pct=44)})
        new = _doc(leads={"30": _lead(threshold_pct=45, warnings=12)})
        loose = apply_stability_guard(
            new, previous, min_delta_pct=1, min_warnings=10,
        )
        assert loose["leads"]["30"]["threshold_pct"] == 45
        assert loose["leads"]["30"]["guard"] == "changed"

    def test_a_lead_that_stops_being_fittable_falls_back(self) -> None:
        """No pick tonight means the fallback, not a stale number.

        Keeping the old threshold would mean serving a rule while the
        document says ``insufficient``, and ``validate_thresholds``
        forbids exactly that contradiction.
        """
        previous = _doc(leads={"60": _lead(threshold_pct=55)})
        out = apply_stability_guard(_doc(), previous)
        row = out["leads"]["60"]
        assert row["insufficient"] is True
        assert row["threshold_pct"] is None
        assert row["guard"] == "insufficient"
        assert effective_threshold(out, 60) == DEFAULT_FALLBACK_THRESHOLD_PCT

    def test_a_guarded_document_is_still_a_valid_document(self) -> None:
        previous = _doc(leads={"20": _lead(threshold_pct=48), "30": _lead()})
        out = apply_stability_guard(_doc(), previous)
        assert validate_thresholds(out) == []
        # Every lead of the new fit survives, and only those.
        assert set(out["leads"]) == {"20", "30", "60"}

    def test_the_new_document_is_not_mutated(self) -> None:
        new = _doc()
        previous = _doc(leads={"30": _lead(threshold_pct=44)})
        out = apply_stability_guard(new, previous)
        assert out["leads"]["30"]["threshold_pct"] == 44
        assert new["leads"]["30"]["threshold_pct"] == 45
        assert "guard" not in new["leads"]["30"]
        # Everything outside ``leads`` is the new fit's.
        assert out["fitted_at_utc"] == new["fitted_at_utc"]
        assert out["window"] == new["window"]

    def test_a_previous_document_that_is_junk_is_a_first_fit(self) -> None:
        for junk in (None, "40 %", {}, {"leads": "later"}):
            out = apply_stability_guard(_doc(), junk)
            assert out["leads"]["30"]["threshold_pct"] == 45
            assert out["leads"]["30"]["guard"] == "first_fit"

    def test_a_document_with_no_leads_is_returned_unchanged(self) -> None:
        broken = {"schema_version": SCHEMA_VERSION}
        assert apply_stability_guard(broken, _doc()) is broken
