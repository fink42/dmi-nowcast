"""The review bundle's frozen contract (``dmi_nowcast_core.review_schema``).

This module is imported by four separate consumers — the bundle builder,
the frame writer, the review server and the browser's ``schema.ts`` — so
the tests here are less about behaviour than about pinning the things a
later edit could quietly break: the vocabulary's shape, the fact that a
code cannot be renamed without a version bump, and the invariants the
server's validation depends on.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from dmi_nowcast_core import review_schema as rs


def test_tag_codes_are_unique_across_groups():
    """A code in two groups would make ``tags_for_class`` lossy.

    ``tags_for_class('hit')`` merges the false-alarm and miss dictionaries,
    so a code present in both would silently take the second group's
    description — and the hit control group is exactly where both lists are
    offered at once.
    """
    fa = set(rs.FALSE_ALARM_TAGS)
    miss = set(rs.MISS_TAGS)
    common = set(rs.COMMON_TAGS)
    assert not fa & miss
    assert not fa & common
    assert not miss & common


def test_every_tag_has_a_nonempty_description():
    """A code with no description is a code nobody can apply consistently."""
    for group in (rs.FALSE_ALARM_TAGS, rs.MISS_TAGS, rs.COMMON_TAGS, rs.VERDICTS):
        for code, text in group.items():
            assert text.strip(), f"{code} has no description"


def test_tag_codes_are_prefixed_by_their_group():
    """``fa_`` / ``miss_`` prefixes let the export group without a lookup."""
    assert all(code.startswith("fa_") for code in rs.FALSE_ALARM_TAGS)
    assert all(code.startswith("miss_") for code in rs.MISS_TAGS)


def test_all_tag_codes_is_the_union_the_server_validates_against():
    codes = rs.all_tag_codes()
    assert codes == set(rs.FALSE_ALARM_TAGS) | set(rs.MISS_TAGS) | set(rs.COMMON_TAGS)
    # The server rejects anything outside this set with a 422; a typo'd
    # constant here would silently widen that gate.
    assert "fa_virga_or_aloft" in codes
    assert "miss_disarmed_rearm" in codes
    assert "definitely_not_a_tag" not in codes


def test_non_mechanism_tags_are_real_codes():
    """The aggregation excludes these from the mechanism ranking.

    If one were misspelled it would silently stay IN the ranking, and
    "threshold marginal" — a coin flip, not a mechanism — would look like
    the leading explanation for false alarms.
    """
    assert rs.NON_MECHANISM_TAGS <= rs.all_tag_codes()


@pytest.mark.parametrize("event_class", rs.OUTCOME_CLASSES)
def test_every_outcome_class_offers_tags(event_class):
    offered = rs.tags_for_class(event_class)
    assert offered
    assert set(offered) <= rs.all_tag_codes()
    # The common tags are offered whichever side the anchor sits on.
    assert set(rs.COMMON_TAGS) <= set(offered)


def test_warning_side_classes_get_false_alarm_causes():
    for event_class in ("false_alarm", "late"):
        offered = set(rs.tags_for_class(event_class))
        assert set(rs.FALSE_ALARM_TAGS) <= offered
        assert not set(rs.MISS_TAGS) & offered


def test_onset_side_classes_get_miss_causes():
    for event_class in ("miss", "miss_late", "uncovered"):
        offered = set(rs.tags_for_class(event_class))
        assert set(rs.MISS_TAGS) <= offered
        assert not set(rs.FALSE_ALARM_TAGS) & offered


def test_hit_control_group_gets_both_cause_lists():
    """The control group exists to supply a base rate.

    A base rate is only useful if the same vocabulary is available: if
    ``fa_cell_died`` describes 40 % of the hits too, it explains nothing
    about the false alarms — and the only way to find that out is to let a
    reviewer apply it to a hit.
    """
    offered = set(rs.tags_for_class("hit"))
    assert set(rs.FALSE_ALARM_TAGS) <= offered
    assert set(rs.MISS_TAGS) <= offered


def test_tags_document_is_json_shaped_and_complete():
    doc = rs.tags_document()
    assert doc["vocab_version"] == rs.REVIEW_VOCAB_VERSION
    assert [g["group"] for g in doc["tag_groups"]] == ["false_alarm", "miss", "common"]

    documented = {
        tag["code"] for group in doc["tag_groups"] for tag in group["tags"]
    }
    assert documented == rs.all_tag_codes()
    assert {v["code"] for v in doc["verdicts"]} == set(rs.VERDICTS)

    # Every class the sampler can draw must be answerable from the shipped
    # document alone — the browser renders the bundle's vocabulary, not
    # whatever the frontend happened to be compiled with.
    assert set(doc["classes"]) == set(rs.OUTCOME_CLASSES)

    import json

    json.dumps(doc)  # must round-trip; no tuples, no sets, no datetimes


def test_pending_is_never_a_drawable_class():
    """``pending`` means "ask again later", not "judge me".

    DMI backfills late station reports, so a pending label can still move.
    Drawing one would ask a human to adjudicate evidence that does not
    exist yet.
    """
    assert "pending" not in rs.OUTCOME_CLASSES


def test_uncovered_is_drawable_as_a_control():
    """Nothing else in the system ever audits the coverage rule.

    ``uncovered`` onsets are silently removed from POD's denominator. If
    none is ever reviewed, that removal is never checked.
    """
    assert "uncovered" in rs.OUTCOME_CLASSES


def test_warning_side_classes_are_the_ones_anchored_on_a_send():
    assert rs.WARNING_SIDE_CLASSES == {"false_alarm", "late", "hit"}
    assert rs.WARNING_SIDE_CLASSES <= set(rs.OUTCOME_CLASSES)


def test_intensity_bands_are_contiguous_and_cover_everything():
    """A gap between bands would drop events out of the sampling frame."""
    edges = [(lo, hi) for _, lo, hi in rs.INTENSITY_BANDS]
    assert edges[0][0] == 0.0
    assert edges[-1][1] == float("inf")
    for (_, hi), (lo, _) in zip(edges, edges[1:]):
        assert hi == lo


def test_strata_keys_match_the_index_fields_they_read():
    """Every stratum must be filterable in the browser.

    The index row is all the browser has for filtering, so a stratum that
    is not an index field could be sampled on but never inspected.
    """
    for key in rs.STRATA_KEYS:
        field = "class" if key == "outcome_class" else key
        assert field in rs.INDEX_FIELDS, key


def test_index_fields_carry_what_the_filters_need():
    required = {
        "event_id", "class", "control", "station_id", "anchor_utc",
        "dual_truth", "flags", "detail",
    }
    assert required <= set(rs.INDEX_FIELDS)
    assert len(set(rs.INDEX_FIELDS)) == len(rs.INDEX_FIELDS)


def test_event_flags_all_have_an_explanation():
    for code, text in rs.EVENT_FLAGS.items():
        assert text.strip(), f"{code} has no explanation"


def test_frame_window_is_padded_past_the_decision_window():
    """A composite is 13-24 min old when a cycle stands on it.

    Without the pad, a decision at the left edge of the decision window
    refers to a frame the bundle does not contain, and the reviewer scrubs
    to the moment that matters and finds a blank map.
    """
    assert rs.DEFAULT_FRAME_PAD_MIN > 0
    assert rs.DEFAULT_WINDOW_MIN == 90


def test_radar_disc_matches_the_production_geometry():
    """The second opinion must be read the way the service reads its own.

    1 km disc, p90 — the same geometry ``sample.sample_disc`` uses for the
    "raining now" number the engine acts on. A different radius here would
    make the radar verdict answer a slightly different question than the
    decision it is being used to judge.
    """
    assert rs.RADAR_DISC_RADIUS_M == 1000.0


def test_neighbour_radius_is_wider_than_a_shower():
    """20 km exceeds a Danish summer shower's diameter, deliberately.

    A wet neighbour says rain existed in the area — NOT that it rained at
    the event's station. The radius is chosen so that claim is the honest
    one; the UI must label it that way.
    """
    assert rs.NEIGHBOUR_RADIUS_KM >= 15.0


def test_event_id_is_stable_sortable_and_filename_safe():
    when = datetime(2026, 6, 12, 13, 40, tzinfo=timezone.utc)
    eid = rs.event_id("false_alarm", "06074", when)
    assert eid == "fa-06074-20260612T1340Z"
    # Stability matters because annotations are keyed on it: a reviewer's
    # judgements must survive a bundle rebuild that adds events.
    assert rs.event_id("false_alarm", "06074", when) == eid
    assert "/" not in eid and " " not in eid


@pytest.mark.parametrize("event_class", rs.OUTCOME_CLASSES)
def test_event_id_abbreviations_are_distinct_per_class(event_class):
    when = datetime(2026, 6, 12, 13, 40, tzinfo=timezone.utc)
    ids = {
        cls: rs.event_id(cls, "06074", when) for cls in rs.OUTCOME_CLASSES
    }
    assert len(set(ids.values())) == len(rs.OUTCOME_CLASSES)
    assert ids[event_class].endswith("-06074-20260612T1340Z")
