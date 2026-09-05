"""Scoring push warnings against gauge observations (Phase F, F3).

Pure functions only: no radar, no store, no pyarrow (the schema helper is
exercised behind an importorskip). Gauge rows are hand-written dicts in
exactly the shape ``StationObsStore.read`` returns, so this suite pins the
scoring contract without depending on the store being built.

The cases that matter are the boundaries — a slot that is unknown rather
than dry, an onset after *exactly* three dry slots, an onset landing on
the last second of the tolerance window, and two warnings competing for
one onset — because every one of them is a way for a scoreboard to
flatter itself.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from dmi_nowcast_core.warning_score import (
    DECISION_COLUMNS,
    DEFAULT_PRODUCT_LEADS_MIN,
    decision_columns,
    p_rain_column,
    parse_p_rain_column,
    per_lead_columns,
    PRECIP_DUR_PARAM,
    PRECIP_PARAM,
    gauge_slots,
    onsets,
    pooled_summary,
    raining_now_agreement,
    score_warnings,
    slot_end_of,
)

T0 = datetime(2026, 9, 5, 6, 0, tzinfo=timezone.utc)


def _row(station: str, minutes: int, param: str, value: float | None) -> dict:
    return {
        "station_id": station,
        "observed_utc": T0 + timedelta(minutes=minutes),
        "parameter_id": param,
        "value": value,
    }


def _amount(station: str, minutes: int, mm: float | None) -> dict:
    return _row(station, minutes, PRECIP_PARAM, mm)


def _dur(station: str, minutes: int, m: float | None) -> dict:
    return _row(station, minutes, PRECIP_DUR_PARAM, m)


# ---------------------------------------------------------------------------
# slot_end_of
# ---------------------------------------------------------------------------


def test_slot_end_keeps_an_exact_boundary() -> None:
    # 06:10:00 IS the end of (06:00, 06:10]; it must not roll to 06:20.
    assert slot_end_of(T0 + timedelta(minutes=10)) == T0 + timedelta(minutes=10)


def test_slot_end_rounds_an_interior_instant_up() -> None:
    assert slot_end_of(T0 + timedelta(minutes=3, seconds=1)) == T0 + timedelta(
        minutes=10
    )
    assert slot_end_of(T0 + timedelta(seconds=1)) == T0 + timedelta(minutes=10)


def test_slot_end_rejects_naive_datetimes() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        slot_end_of(datetime(2026, 9, 5, 6, 0))


# ---------------------------------------------------------------------------
# gauge_slots
# ---------------------------------------------------------------------------


def test_gauge_slots_wet_by_amount_and_by_duration() -> None:
    rows = [
        _amount("A", 10, 0.0), _dur("A", 10, 0.0),      # dry
        _amount("A", 20, 0.1), _dur("A", 20, 0.0),      # wet: amount arm
        _amount("A", 30, 0.0), _dur("A", 30, 1.0),      # wet: duration arm
        _amount("A", 40, 0.09), _dur("A", 40, 0.0),     # below both arms
    ]
    slots = gauge_slots(rows, "A")
    assert [wet for _, wet in slots] == [False, True, True, False]
    assert [ts for ts, _ in slots] == [
        T0 + timedelta(minutes=m) for m in (10, 20, 30, 40)
    ]


def test_gauge_slots_trace_sentinel_is_below_the_amount_threshold() -> None:
    # DMI encodes "trace, less than 0.1 kg/m²" as -0.1. It is not a
    # negative depth and it is not 0.1 mm of rain.
    slots = gauge_slots([_amount("A", 10, -0.1)], "A")
    assert slots == [(T0 + timedelta(minutes=10), False)]


def test_gauge_slots_marks_an_unreported_slot_unknown_not_dry() -> None:
    rows = [_amount("A", 10, 0.0), _amount("A", 30, 0.0)]
    slots = gauge_slots(rows, "A")
    assert [wet for _, wet in slots] == [False, None, False]


def test_gauge_slots_null_value_leaves_the_slot_unknown() -> None:
    assert gauge_slots([_amount("A", 10, None)], "A") == []
    slots = gauge_slots([_amount("A", 10, None)], "A",
                        start_utc=T0, end_utc=T0 + timedelta(minutes=10))
    assert [wet for _, wet in slots] == [None, None]


def test_gauge_slots_one_reported_parameter_is_enough_to_decide() -> None:
    # A station reporting only the duration is still scoreable: "nothing
    # was said" is unknown, "one arm said no" is dry.
    slots = gauge_slots([_dur("A", 10, 0.0)], "A")
    assert slots == [(T0 + timedelta(minutes=10), False)]


def test_gauge_slots_filters_by_station_and_ignores_other_parameters() -> None:
    rows = [
        _amount("A", 10, 5.0),
        _amount("B", 10, 0.0),
        _row("A", 20, "temp_dry", 12.3),
    ]
    # The unrelated parameter neither wets a slot nor extends the grid.
    assert gauge_slots(rows, "A") == [(T0 + timedelta(minutes=10), True)]
    assert gauge_slots(rows, "B") == [(T0 + timedelta(minutes=10), False)]


def test_gauge_slots_pins_the_grid_to_an_explicit_window() -> None:
    slots = gauge_slots(
        [_amount("A", 20, 1.0)], "A",
        start_utc=T0, end_utc=T0 + timedelta(minutes=40),
    )
    assert [wet for _, wet in slots] == [None, None, True, None, None]


def test_gauge_slots_empty_table_is_empty() -> None:
    assert gauge_slots([], "A") == []


def test_gauge_slots_accepts_an_arrow_table() -> None:
    pa = pytest.importorskip("pyarrow")
    table = pa.table({
        "station_id": ["A", "A"],
        "observed_utc": pa.array(
            [T0 + timedelta(minutes=10), T0 + timedelta(minutes=20)],
            type=pa.timestamp("us", tz="UTC"),
        ),
        "parameter_id": [PRECIP_PARAM, PRECIP_PARAM],
        "value": pa.array([0.0, 0.3], type=pa.float32()),
    })
    assert [wet for _, wet in gauge_slots(table, "A")] == [False, True]


# ---------------------------------------------------------------------------
# onsets
# ---------------------------------------------------------------------------


def _slots(pattern: str) -> list[tuple[datetime, bool | None]]:
    """``"ddw?"`` → dry, dry, wet, unknown — one character per slot."""
    mapping = {"d": False, "w": True, "?": None}
    return [
        (T0 + timedelta(minutes=10 * (i + 1)), mapping[ch])
        for i, ch in enumerate(pattern)
    ]


def test_onset_after_exactly_three_dry_slots() -> None:
    got = onsets(_slots("dddw"))
    assert got == [T0 + timedelta(minutes=40)]


def test_no_onset_after_only_two_dry_slots() -> None:
    assert onsets(_slots("ddw")) == []


def test_wet_at_the_very_start_is_not_an_onset() -> None:
    # No evidence about what came before the record.
    assert onsets(_slots("wwww")) == []


def test_only_the_first_wet_slot_of_a_spell_is_an_onset() -> None:
    assert onsets(_slots("dddwww")) == [T0 + timedelta(minutes=40)]


def test_a_second_spell_needs_its_own_dry_run() -> None:
    assert onsets(_slots("dddwdddw")) == [
        T0 + timedelta(minutes=40), T0 + timedelta(minutes=80),
    ]
    # Only two dry slots between the spells → the second is not an onset.
    assert onsets(_slots("dddwddw")) == [T0 + timedelta(minutes=40)]


def test_unknown_slot_resets_the_dry_run() -> None:
    # Missing data can never certify a dry spell.
    assert onsets(_slots("dd?dw")) == []
    assert onsets(_slots("dd?dddw")) == [T0 + timedelta(minutes=70)]


def test_a_hole_in_the_grid_resets_the_dry_run() -> None:
    slots = _slots("ddd")
    slots.append((T0 + timedelta(minutes=200), True))  # a seam, not a sequence
    assert onsets(slots) == []


def test_dry_min_is_configurable() -> None:
    assert onsets(_slots("dw"), dry_min=10) == [T0 + timedelta(minutes=20)]
    assert onsets(_slots("ddddw"), dry_min=50) == []


# ---------------------------------------------------------------------------
# score_warnings
# ---------------------------------------------------------------------------


def _at(minutes: float) -> datetime:
    return T0 + timedelta(minutes=minutes)


def test_hit_inside_the_window_carries_the_lead_error() -> None:
    res = score_warnings([(_at(0), 20.0)], [_at(30)])
    assert res.summary["hits"] == 1
    assert res.summary["false_alarms"] == 0
    assert res.summary["misses"] == 0
    assert res.summary["pod"] == 1.0
    assert res.summary["far"] == 0.0
    # ETA 20 min, rain 30 min out: the rain came LATER than promised, so
    # the warning was early → negative under the eta − actual convention.
    assert res.warnings[0].lead_error_min == pytest.approx(-10.0)
    assert res.warnings[0].outcome == "hit"
    assert res.onsets[0].outcome == "hit"


def test_onset_on_the_last_second_of_the_tolerance_is_a_hit() -> None:
    res = score_warnings([(_at(0), 30.0)], [_at(40)])
    assert res.summary["hits"] == 1


def test_onset_one_minute_past_the_tolerance_is_a_false_alarm_and_a_miss() -> None:
    res = score_warnings([(_at(0), 30.0)], [_at(41)])
    assert res.summary == {
        **res.summary,
        "hits": 0, "false_alarms": 1, "misses": 1, "far": 1.0, "pod": 0.0,
    }
    assert res.warnings[0].outcome == "false_alarm"
    assert res.onsets[0].outcome == "miss"


def test_onset_at_the_send_instant_does_not_count() -> None:
    # The window is (sent, sent+lead+tol]: rain already falling when the
    # warning went out is not something the warning predicted.
    res = score_warnings([(_at(0), 5.0)], [_at(0)])
    assert res.summary["hits"] == 0
    assert res.summary["false_alarms"] == 1


def test_one_onset_is_claimed_by_at_most_one_warning_earliest_first() -> None:
    res = score_warnings([(_at(10), 20.0), (_at(0), 30.0)], [_at(35)])
    assert res.summary["hits"] == 1
    assert res.summary["false_alarms"] == 1
    hit = [w for w in res.warnings if w.outcome == "hit"][0]
    assert hit.sent_utc == _at(0)          # the earlier warning wins
    assert res.summary["n_onsets"] == 1
    assert res.summary["misses"] == 0


def test_two_onsets_in_one_window_leave_the_second_unclaimed() -> None:
    res = score_warnings([(_at(0), 10.0)], [_at(5), _at(35)])
    assert res.summary["hits"] == 1
    assert res.summary["misses"] == 1
    assert [o.outcome for o in res.onsets] == ["hit", "miss"]


def test_a_miss_is_an_onset_no_warning_preceded() -> None:
    res = score_warnings([], [_at(120)])
    assert res.summary["warnings"] == 0
    assert res.summary["misses"] == 1
    assert res.summary["pod"] == 0.0
    assert res.summary["far"] is None      # no warnings → FAR undefined


def test_counts_always_balance() -> None:
    warnings = [(_at(0), 20.0), (_at(90), 25.0), (_at(300), 30.0)]
    onset_times = [_at(25), _at(200)]
    res = score_warnings(warnings, onset_times)
    s = res.summary
    assert s["hits"] + s["false_alarms"] == s["warnings"] == 3
    assert s["hits"] + s["misses"] == s["n_onsets"] == 2


def test_lead_error_quantiles_are_linear_interpolated() -> None:
    # Rain 1..5 min later than the ETA → errors of −1..−5, so sorted the
    # population is −5..−1: p25 = −4, p50 = −3, p75 = −2 (numpy's rule).
    warnings = []
    onset_times = []
    for i, late in enumerate((1, 2, 3, 4, 5)):
        sent = _at(i * 200)
        warnings.append((sent, 20.0))
        onset_times.append(sent + timedelta(minutes=20 + late))
    q = score_warnings(warnings, onset_times).summary["lead_error_min"]
    assert q == {"p25": -4.0, "p50": -3.0, "p75": -2.0, "n": 5}


def test_a_warning_that_fired_too_late_scores_positive() -> None:
    """The failure that matters: less lead than the notification promised."""
    # Promised rain in 30 min; it was falling 12 min later.
    res = score_warnings([(_at(0), 30.0)], [_at(12)])
    assert res.warnings[0].lead_error_min == pytest.approx(18.0)
    assert res.summary["lead_error_min"]["p50"] == pytest.approx(18.0)


def test_a_warning_without_an_eta_still_scores_but_adds_no_lead_error() -> None:
    res = score_warnings([(_at(0), None)], [_at(20)])
    assert res.summary["hits"] == 1
    assert res.warnings[0].lead_error_min is None
    assert res.summary["lead_error_min"]["n"] == 0
    assert res.summary["lead_error_min"]["p50"] is None


def test_summary_echoes_the_definition_it_was_produced_under() -> None:
    s = score_warnings([], [], lead_min=45, tolerance_min=5, dry_min=60).summary
    assert (s["lead_min"], s["tolerance_min"], s["dry_min"]) == (45, 5, 60)


def test_pooled_summary_recomputes_rather_than_averages() -> None:
    busy = score_warnings([(_at(0), 20.0)] , [_at(30)])            # 1 hit
    quiet = score_warnings([(_at(0), 20.0), (_at(500), 20.0)], []) # 2 FAs
    pooled = pooled_summary([busy, quiet], lead_min=30)
    assert pooled["warnings"] == 3
    assert pooled["hits"] == 1
    assert pooled["false_alarms"] == 2
    assert pooled["far"] == pytest.approx(2 / 3)
    assert pooled["lead_error_min"]["n"] == 1
    assert pooled["lead_min"] == 30


# ---------------------------------------------------------------------------
# raining_now_agreement
# ---------------------------------------------------------------------------


def _decision(minutes: int, forecast: float | None, observed: float | None) -> dict:
    return {
        "generated_at": _at(minutes),
        "station_id": "A",
        "forecast_now_mm_h": forecast,
        "observed_mm_h": observed,
    }


def test_agreement_scores_both_series_against_the_same_slot() -> None:
    slots = {"A": _slots("wwdd")}          # 06:10, 06:20 wet; 06:30, 06:40 dry
    rows = [
        _decision(5, 1.0, 1.0),            # slot 06:10 wet — both right
        _decision(15, 0.0, 2.0),           # slot 06:20 wet — forecast misses
        _decision(25, 1.0, 0.0),           # slot 06:30 dry — forecast cries wolf
        _decision(35, 0.0, 0.0),           # slot 06:40 dry — both right
    ]
    out = raining_now_agreement(rows, slots)
    assert out["n_scored"] == 4
    assert out["gauge_wet_rate"] == 0.5
    assert out["forecast_now"] == {
        "n": 4, "agreement": 0.5, "pod": 0.5, "far": 0.5,
        "hits": 1, "misses": 1, "false_alarms": 1, "correct_negatives": 1,
    }
    assert out["observed"]["agreement"] == 1.0
    assert out["observed"]["far"] == 0.0


def test_agreement_threshold_is_inclusive() -> None:
    out = raining_now_agreement(
        [_decision(5, 0.5, 0.49)], {"A": _slots("w")}, threshold_mm_h=0.5,
    )
    assert out["forecast_now"]["hits"] == 1
    assert out["observed"]["misses"] == 1


def test_agreement_skips_rows_whose_slot_is_unknown() -> None:
    out = raining_now_agreement([_decision(5, 1.0, 1.0)], {"A": _slots("?")})
    assert out["n_rows"] == 1
    assert out["n_scored"] == 0
    assert out["gauge_wet_rate"] is None
    assert out["forecast_now"]["n"] == 0


def test_agreement_counts_each_series_separately_when_one_is_missing() -> None:
    # A cycle whose observed reduction failed still contributes its
    # forecast sample — the two series report their own n.
    out = raining_now_agreement([_decision(5, 1.0, None)], {"A": _slots("w")})
    assert out["forecast_now"]["n"] == 1
    assert out["observed"]["n"] == 0
    assert out["observed"]["agreement"] is None


def test_agreement_accepts_gauge_truth_on_the_row() -> None:
    row = {**_decision(5, 1.0, 0.0), "gauge_wet": True}
    out = raining_now_agreement([row])
    assert out["n_scored"] == 1
    assert out["forecast_now"]["hits"] == 1
    assert out["observed"]["misses"] == 1


def test_agreement_with_no_rows_is_all_none() -> None:
    out = raining_now_agreement([], {})
    assert out["n_rows"] == 0
    assert out["forecast_now"]["agreement"] is None
    assert out["observed"]["pod"] is None


# ---------------------------------------------------------------------------
# The shared decision row
# ---------------------------------------------------------------------------


def test_decision_schema_matches_the_declared_columns() -> None:
    pytest.importorskip("pyarrow")
    from dmi_nowcast_core.warning_score import decision_schema

    schema = decision_schema()
    assert tuple(schema.names) == decision_columns()
    # The base columns stay first and in order — readers pin them.
    assert tuple(schema.names)[:len(DECISION_COLUMNS)] == DECISION_COLUMNS
    # Every forecast field must be nullable: None from the sampler means
    # "off coverage", which must never round-trip as a dry 0.0.
    for name in ("p_rain", "eta_min", "intensity_mm_h", "observed_mm_h",
                 "forecast_now_mm_h"):
        assert schema.field(name).nullable


# ---------------------------------------------------------------------------
# Per-lead probability columns
# ---------------------------------------------------------------------------


def test_p_rain_column_names_round_trip() -> None:
    assert p_rain_column(30) == "p_rain_30"
    assert parse_p_rain_column("p_rain_45") == 45
    # The rule's-lead column is NOT a per-lead column, and neither is junk.
    assert parse_p_rain_column("p_rain") is None
    assert parse_p_rain_column("p_rain_x") is None
    assert parse_p_rain_column("observed_mm_h") is None


def test_decision_columns_appends_one_column_per_served_lead() -> None:
    assert decision_columns() == DECISION_COLUMNS + tuple(
        f"p_rain_{lead}" for lead in DEFAULT_PRODUCT_LEADS_MIN
    )
    assert decision_columns([30, 5, 5]) == DECISION_COLUMNS + (
        "p_rain_5", "p_rain_30",
    )


def test_per_lead_columns_maps_the_sampler_dict() -> None:
    # A nodata lead stays None — a lead with no probability is unknown,
    # never 0 %.
    assert per_lead_columns({10: 0.25, 30: None}) == {
        "p_rain_10": 0.25, "p_rain_30": None,
    }
    assert per_lead_columns(None) == {}
    assert per_lead_columns({}) == {}


def test_decision_schema_types_the_per_lead_columns_nullable_float32() -> None:
    pa = pytest.importorskip("pyarrow")
    from dmi_nowcast_core.warning_score import decision_schema

    schema = decision_schema([10, 30])
    for name in ("p_rain_10", "p_rain_30"):
        assert schema.field(name).type == pa.float32()
        assert schema.field(name).nullable


def _base_row() -> dict:
    return {
        "radar_ts": T0,
        "generated_at": T0 + timedelta(minutes=14),
        "station_id": "06180",
        "p_rain": 0.7,
        "eta_min": 20.0,
        "intensity_mm_h": 1.0,
        "observed_mm_h": 0.0,
        "forecast_now_mm_h": 0.0,
        "action": "notify",
        "armed_after": False,
        "streak_after": 1,
    }


def test_align_fills_an_old_file_that_predates_the_per_lead_columns() -> None:
    """A parquet with only the base columns must still read."""
    pytest.importorskip("pyarrow")
    from dmi_nowcast_core.warning_score import (
        align_decision_table,
        decision_table,
    )

    old = decision_table([_base_row()], leads_min=())
    assert tuple(old.schema.names) == DECISION_COLUMNS
    aligned = align_decision_table(old)
    assert tuple(aligned.schema.names) == decision_columns()
    row = aligned.to_pylist()[0]
    assert row["p_rain"] == pytest.approx(0.7)      # untouched
    assert row["p_rain_30"] is None                 # unknown, not 0.0


def test_align_keeps_a_lead_the_file_has_but_the_caller_did_not_ask_for() -> None:
    pytest.importorskip("pyarrow")
    from dmi_nowcast_core.warning_score import (
        align_decision_table,
        decision_leads_in,
        decision_table,
    )

    wide = decision_table(
        [{**_base_row(), "p_rain_5": 0.1, "p_rain_30": 0.7}], leads_min=(5, 30),
    )
    aligned = align_decision_table(wide, leads_min=(30,))
    assert decision_leads_in(aligned) == (5, 30)     # nothing dropped
    assert aligned.to_pylist()[0]["p_rain_5"] == pytest.approx(0.1)


def test_concat_unions_files_written_under_different_lead_sets() -> None:
    """The report producer reads days that straddle the schema change."""
    pytest.importorskip("pyarrow")
    from dmi_nowcast_core.warning_score import (
        concat_decision_tables,
        decision_table,
    )

    old = decision_table([_base_row()], leads_min=())
    new = decision_table(
        [{**_base_row(), "p_rain_30": 0.7}], leads_min=(30,),
    )
    merged = concat_decision_tables([old, new], leads_min=())
    assert merged.num_rows == 2
    assert tuple(merged.schema.names) == DECISION_COLUMNS + ("p_rain_30",)
    assert [r["p_rain_30"] for r in merged.to_pylist()] == [None, pytest.approx(0.7)]


def test_concat_of_nothing_is_an_empty_typed_table() -> None:
    pytest.importorskip("pyarrow")
    from dmi_nowcast_core.warning_score import concat_decision_tables

    empty = concat_decision_tables([], leads_min=(30,))
    assert empty.num_rows == 0
    assert tuple(empty.schema.names) == DECISION_COLUMNS + ("p_rain_30",)


# ---------------------------------------------------------------------------
# The pending boundary: not grading a promise that has not come due
# ---------------------------------------------------------------------------
#
# The defect this fixes, verbatim from the first live report: two warnings
# sent at 12:25Z appeared as false alarms in a report generated at 12:26Z.
# The gauge had said nothing about 12:25-13:05 yet, so the only thing that
# scoring measured was how recently the report was built.


def _at(minutes: int) -> datetime:
    return T0 + timedelta(minutes=minutes)


def test_a_warning_whose_window_is_still_open_is_pending_not_a_false_alarm() -> None:
    # Sent at T0+60; window closes at T0+100 (lead 30 + tolerance 10).
    # The gauge has only reported to T0+65.
    result = score_warnings(
        [(_at(60), 20.0)], [], lead_min=30, tolerance_min=10,
        known_until=_at(65),
    )
    assert [w.outcome for w in result.warnings] == ["pending"]
    summary = result.summary
    assert summary["pending"] == 1
    assert summary["n_sent"] == 1
    # Excluded from every rate, not counted as a failure.
    assert summary["warnings"] == 0
    assert summary["hits"] == 0
    assert summary["false_alarms"] == 0
    assert summary["far"] is None


def test_the_same_warning_is_a_false_alarm_once_the_window_has_closed() -> None:
    result = score_warnings(
        [(_at(60), 20.0)], [], lead_min=30, tolerance_min=10,
        known_until=_at(140),
    )
    assert [w.outcome for w in result.warnings] == ["false_alarm"]
    assert result.summary["pending"] == 0
    assert result.summary["false_alarms"] == 1
    assert result.summary["far"] == 1.0


def test_without_known_until_nothing_is_pending() -> None:
    """A closed historical window scores exactly as it always did."""
    result = score_warnings([(_at(60), 20.0)], [], lead_min=30, tolerance_min=10)
    assert [w.outcome for w in result.warnings] == ["false_alarm"]
    assert result.summary["pending"] == 0
    assert result.summary["warnings"] == result.summary["n_sent"] == 1


def test_a_claimed_onset_settles_a_warning_even_with_its_window_open() -> None:
    """A confirmed hit is never demoted to pending.

    Demoting it would take the hit out of POD's numerator while leaving
    its onset in the denominator — turning a hit into a miss, which is
    worse than the bug the rule fixes.
    """
    result = score_warnings(
        [(_at(60), 20.0)], [_at(80)], lead_min=30, tolerance_min=10,
        known_until=_at(85),   # window closes at 100; the onset is in hand
    )
    assert [w.outcome for w in result.warnings] == ["hit"]
    assert [o.outcome for o in result.onsets] == ["hit"]
    assert result.summary["pending"] == 0
    assert result.summary["pod"] == 1.0


def test_an_onset_near_the_edge_is_pending_not_a_miss() -> None:
    """DMI backfills late reports; a slot at the edge can still move."""
    result = score_warnings(
        [], [_at(95)], lead_min=30, tolerance_min=10, known_until=_at(100),
    )
    assert [o.outcome for o in result.onsets] == ["pending"]
    summary = result.summary
    assert summary["pending_onsets"] == 1
    assert summary["misses"] == 0
    assert summary["pod"] is None


def test_an_onset_well_inside_the_known_window_is_a_miss() -> None:
    result = score_warnings(
        [], [_at(40)], lead_min=30, tolerance_min=10, known_until=_at(100),
    )
    assert [o.outcome for o in result.onsets] == ["miss"]
    assert result.summary["misses"] == 1
    assert result.summary["pod"] == 0.0


def test_the_counts_still_add_up_with_pending_in_the_mix() -> None:
    """hits + false_alarms == warnings, and hits + misses == scored onsets."""
    result = score_warnings(
        [
            (_at(0), 20.0),     # claims the onset at 20  → hit
            (_at(60), 20.0),    # nothing follows, window closed → false alarm
            (_at(306), 20.0),   # window open at known_until → pending
        ],
        # 305 is before the last warning was sent, so nothing can claim
        # it, and it sits inside the tolerance of the edge → pending.
        [_at(20), _at(200), _at(305)],
        lead_min=30, tolerance_min=10, known_until=_at(310),
    )
    s = result.summary
    assert [w.outcome for w in result.warnings] == [
        "hit", "false_alarm", "pending",
    ]
    assert [o.outcome for o in result.onsets] == ["hit", "miss", "pending"]
    assert s["hits"] + s["false_alarms"] == s["warnings"] == 2
    assert s["pending"] == 1 and s["n_sent"] == 3
    assert s["hits"] + s["misses"] == s["n_onsets"] - s["pending_onsets"] == 2
    assert s["pod"] == 0.5
    assert s["far"] == 0.5


def test_pooled_summary_keeps_pending_out_of_the_national_rates() -> None:
    settled = score_warnings(
        [(_at(0), 20.0)], [_at(20)], lead_min=30, tolerance_min=10,
        known_until=_at(400),
    )
    fresh = score_warnings(
        [(_at(300), 20.0)], [], lead_min=30, tolerance_min=10,
        known_until=_at(305),
    )
    pooled = pooled_summary([settled, fresh])
    assert pooled["hits"] == 1
    assert pooled["pending"] == 1
    assert pooled["warnings"] == 1
    assert pooled["false_alarms"] == 0
    # One station's still-open window must not drag the national FAR up.
    assert pooled["far"] == 0.0
    assert pooled["pod"] == 1.0


def test_a_pending_warning_contributes_no_lead_error() -> None:
    result = score_warnings(
        [(_at(300), 20.0)], [], lead_min=30, tolerance_min=10,
        known_until=_at(305),
    )
    assert result.summary["lead_error_min"]["n"] == 0
    assert result.summary["lead_error_min"]["p50"] is None


# ---------------------------------------------------------------------------
# Coverage: an onset is only a miss where a decision could have caught it
# ---------------------------------------------------------------------------
#
# The defect this fixes, from the first served report: five warnings, two
# hits, and 2 088 "misses" — gauge onsets from a backfilled archive running
# back to December, scored against decision rows that existed only for that
# day. POD came out 0.001, which measured the archive's depth and nothing
# about the service.


def _cover(*minutes: int, extend: float = 40.0) -> list:
    from dmi_nowcast_core.warning_score import coverage_runs

    return coverage_runs([_at(m) for m in minutes], extend_min=extend)


def test_coverage_runs_merges_frames_within_two_cycles() -> None:
    # 10-minute frames: one continuous run, ending a lead window later.
    runs = _cover(0, 10, 20, 30, extend=40.0)
    assert runs == [(_at(0), _at(70))]


def test_coverage_runs_tolerates_one_missed_cycle() -> None:
    """A 20-minute gap is a hiccup; the rain either side was watched."""
    assert _cover(0, 20, 40, extend=0.0) == [(_at(0), _at(40))]


def test_coverage_runs_breaks_on_a_longer_gap() -> None:
    """A three-hour hole is an outage, and nothing in it was watched."""
    runs = _cover(0, 10, 190, 200, extend=40.0)
    assert runs == [(_at(0), _at(50)), (_at(190), _at(240))]


def test_coverage_runs_of_nothing_covers_nothing() -> None:
    from dmi_nowcast_core.warning_score import coverage_runs

    assert coverage_runs([]) == []


def test_coverage_runs_dedupes_and_sorts() -> None:
    assert _cover(30, 0, 10, 0, 20, extend=0.0) == [(_at(0), _at(30))]


def test_an_onset_a_week_before_the_first_decision_is_not_a_miss() -> None:
    week_before = T0 - timedelta(days=7)
    result = score_warnings(
        [], [week_before], lead_min=30, tolerance_min=10,
        coverage=_cover(0, 10, 20),
    )
    assert [o.outcome for o in result.onsets] == ["uncovered"]
    summary = result.summary
    assert summary["uncovered_onsets"] == 1
    assert summary["misses"] == 0
    # No scored onsets at all, so POD is "not measured", never 0.0.
    assert summary["pod"] is None


def test_an_onset_inside_a_covered_run_with_no_warning_is_a_miss() -> None:
    result = score_warnings(
        [], [_at(15)], lead_min=30, tolerance_min=10,
        coverage=_cover(0, 10, 20), known_until=_at(500),
    )
    assert [o.outcome for o in result.onsets] == ["miss"]
    assert result.summary["misses"] == 1
    assert result.summary["uncovered_onsets"] == 0
    assert result.summary["pod"] == 0.0


def test_a_three_hour_gap_between_rows_does_not_bridge() -> None:
    """Rain in the hole is nobody's to catch; rain in either run is."""
    result = score_warnings(
        [],
        [_at(5), _at(100), _at(195)],   # run 1, the gap, run 2
        lead_min=30, tolerance_min=10,
        coverage=_cover(0, 10, 190, 200),
        known_until=_at(500),
    )
    assert [o.outcome for o in result.onsets] == ["miss", "uncovered", "miss"]
    assert result.summary["misses"] == 2
    assert result.summary["uncovered_onsets"] == 1


def test_the_run_tail_covers_the_promise_the_last_frame_made() -> None:
    """The final frame promises the next lead window; that is in scope."""
    result = score_warnings(
        [], [_at(55)], lead_min=30, tolerance_min=10,
        coverage=_cover(0, 10, 20), known_until=_at(500),
    )
    # Run ends at 20 + 40 = 60, so an onset at 55 is inside it.
    assert [o.outcome for o in result.onsets] == ["miss"]


def test_a_claimed_onset_is_a_hit_even_outside_coverage() -> None:
    """The claim is its own evidence and no coverage rule unmakes it.

    Demoting it would drop a hit from POD's numerator while its onset
    stayed in the denominator — the same trap the pending rule avoids.
    """
    result = score_warnings(
        [(_at(20), 20.0)], [_at(55)], lead_min=30, tolerance_min=10,
        coverage=[(_at(0), _at(30))],   # deliberately short: 55 is outside
        known_until=_at(500),
    )
    assert [w.outcome for w in result.warnings] == ["hit"]
    assert [o.outcome for o in result.onsets] == ["hit"]
    assert result.summary["pod"] == 1.0
    assert result.summary["uncovered_onsets"] == 0


def test_without_coverage_every_onset_is_scored_as_before() -> None:
    result = score_warnings([], [T0 - timedelta(days=7)], lead_min=30)
    assert [o.outcome for o in result.onsets] == ["miss"]
    assert result.summary["uncovered_onsets"] == 0
    assert result.summary["coverage_runs"] == 0


def test_the_counts_add_up_with_coverage_pending_and_misses_together() -> None:
    result = score_warnings(
        [(_at(0), 20.0), (_at(60), 20.0)],
        [T0 - timedelta(days=7), _at(20), _at(150), _at(495)],
        lead_min=30, tolerance_min=10,
        coverage=_cover(0, 10, 20, 60, 70, 150, 160, 490, 500),
        known_until=_at(500),
    )
    s = result.summary
    assert [o.outcome for o in result.onsets] == [
        "uncovered", "hit", "miss", "pending",
    ]
    assert s["hits"] + s["misses"] == (
        s["n_onsets"] - s["pending_onsets"] - s["uncovered_onsets"]
    ) == 2
    assert s["pod"] == 0.5


def test_pooled_summary_pools_uncovered_onsets_out_of_the_rates() -> None:
    watched = score_warnings(
        [(_at(0), 20.0)], [_at(20)], lead_min=30, tolerance_min=10,
        coverage=_cover(0, 10, 20), known_until=_at(500),
    )
    archive_only = score_warnings(
        [], [T0 - timedelta(days=7), T0 - timedelta(days=6)],
        lead_min=30, tolerance_min=10, coverage=_cover(0, 10, 20),
    )
    pooled = pooled_summary([watched, archive_only])
    assert pooled["hits"] == 1
    assert pooled["uncovered_onsets"] == 2
    assert pooled["misses"] == 0
    # The archive's depth must not drag the national POD toward zero.
    assert pooled["pod"] == 1.0


def test_raining_now_agreement_only_ever_looks_at_decision_rows() -> None:
    """Confirmation, not a change: it iterates rows, never the slot grid.

    A station with a year of gauge slots and two decision rows scores two
    slots — the ones a decision was made in — and nothing else.
    """
    slots = [
        (T0 + timedelta(minutes=10 * k), k % 2 == 0) for k in range(-500, 500)
    ]
    rows = [
        {"station_id": "06180", "generated_at": _at(0),
         "forecast_now_mm_h": 1.0, "observed_mm_h": 1.0},
        {"station_id": "06180", "generated_at": _at(10),
         "forecast_now_mm_h": 0.0, "observed_mm_h": 0.0},
    ]
    out = raining_now_agreement(rows, {"06180": slots})
    assert out["n_rows"] == 2
    assert out["n_scored"] == 2
    assert out["forecast_now"]["n"] == 2


# ---------------------------------------------------------------------------
# The minimum useful lead: hit, or late?
# ---------------------------------------------------------------------------


def test_the_default_makes_nothing_late() -> None:
    """Rain thirty seconds after the warning still scores as a hit at 0.0.

    The historical numbers on the quality page were produced without this
    knob, so the scorer's own default has to leave them untouched.
    """
    res = score_warnings([(_at(0), 5.0)], [_at(0.5)])
    assert res.summary["late"] == 0
    assert res.summary["hits"] == 1
    assert res.summary["min_useful_lead_min"] == 0.0
    assert res.warnings[0].outcome == "hit"


def test_exactly_the_minimum_useful_lead_is_still_a_hit() -> None:
    """The boundary is inclusive: five minutes is five minutes of warning."""
    res = score_warnings([(_at(0), 5.0)], [_at(5)], min_useful_lead_min=5.0)
    assert res.summary["hits"] == 1
    assert res.summary["late"] == 0
    assert res.warnings[0].outcome == "hit"
    assert res.onsets[0].outcome == "hit"


def test_a_shorter_lead_is_late_and_lands_on_the_recall_side() -> None:
    res = score_warnings([(_at(0), 5.0)], [_at(4)], min_useful_lead_min=5.0)
    s = res.summary
    assert res.warnings[0].outcome == "late"
    assert res.onsets[0].outcome == "miss_late"
    assert (s["hits"], s["late"], s["false_alarms"], s["misses"]) == (0, 1, 0, 0)
    # The rain came and nobody was usefully told: recall counts it, and
    # POD stays equal to recall by construction.
    assert s["recall"] == 0.0
    assert s["pod"] == s["recall"]
    # Precision never saw it: no warning was wrong, so its denominator is
    # empty and the rate is undefined rather than zero.
    assert s["precision"] is None
    assert s["f1"] is None
    assert s["csi"] == 0.0


def test_a_late_warning_is_not_a_false_alarm_so_far_is_not_one_minus_precision() -> None:
    """The documented asymmetry, in one case.

    One warning whose rain lands two minutes later (late) and one warning
    nothing followed (a false alarm). FAR counts both in its denominator;
    precision counts only the second.
    """
    res = score_warnings(
        [(_at(0), 10.0), (_at(600), 10.0)], [_at(2)],
        min_useful_lead_min=5.0,
    )
    s = res.summary
    assert [w.outcome for w in res.warnings] == ["late", "false_alarm"]
    assert (s["hits"], s["late"], s["false_alarms"], s["warnings"]) == (0, 1, 1, 2)
    assert s["far"] == pytest.approx(0.5)
    assert s["precision"] == 0.0
    assert s["far"] != 1.0 - s["precision"]


def test_a_late_claim_still_consumes_its_onset() -> None:
    """Two warnings, one onset: the first claims it even though it is late.

    Letting the second warning re-claim the onset as a hit would let a
    rule that only ever fires at the last moment launder its lates into
    hits by sending twice.
    """
    res = score_warnings(
        [(_at(0), 10.0), (_at(1), 10.0)], [_at(3)], min_useful_lead_min=5.0,
    )
    assert [w.outcome for w in res.warnings] == ["late", "false_alarm"]
    assert res.summary["n_onsets"] == 1
    assert res.summary["misses"] == 0


def test_the_counts_balance_with_late_in_the_mix() -> None:
    """``hits + misses + late == n_onsets − pending − uncovered``."""
    warnings = [(_at(0), 20.0), (_at(200), 20.0)]
    onset_times = [_at(2), _at(225), _at(600)]
    s = score_warnings(
        warnings, onset_times, known_until=_at(2000), min_useful_lead_min=5.0,
    ).summary
    assert (s["hits"], s["late"], s["misses"]) == (1, 1, 1)
    assert s["hits"] + s["misses"] + s["late"] == (
        s["n_onsets"] - s["pending_onsets"] - s["uncovered_onsets"]
    )
    assert s["hits"] + s["late"] + s["false_alarms"] == s["warnings"]


def test_a_late_warning_keeps_its_lead_error_but_stays_out_of_the_quantiles() -> None:
    """The spread answers "when we warned in time, how close was the ETA?"."""
    res = score_warnings(
        [(_at(0), 30.0), (_at(600), 30.0)], [_at(2), _at(625)],
        min_useful_lead_min=5.0,
    )
    late = res.warnings[0]
    assert late.outcome == "late"
    assert late.lead_error_min == pytest.approx(28.0)
    spread = res.summary["lead_error_min"]
    assert spread["n"] == 1
    assert spread["p50"] == pytest.approx(5.0)


def test_f_scores_are_the_textbook_harmonic_means() -> None:
    from dmi_nowcast_core.warning_score import skill_scores

    skill = skill_scores(hits=3, false_alarms=1, misses=1, late=2)
    assert skill["precision"] == pytest.approx(0.75)
    assert skill["recall"] == pytest.approx(0.5)
    assert skill["f1"] == pytest.approx(0.6)
    assert skill["f_beta"]["1"] == skill["f1"]
    # β < 1 leans on precision, β > 1 on recall.
    assert skill["f_beta"]["0.5"] == pytest.approx(0.6818181818, rel=1e-9)
    assert skill["f_beta"]["2"] == pytest.approx(0.5357142857, rel=1e-9)
    # Late sits on the miss side of CSI: 3 / (3 hits + 1 miss + 2 late
    # + 1 false alarm).
    assert skill["csi"] == pytest.approx(3 / 7)


def test_undefined_rates_are_none_not_zero() -> None:
    from dmi_nowcast_core.warning_score import skill_scores

    nothing = skill_scores(hits=0, false_alarms=0, misses=0, late=0)
    assert nothing["precision"] is None
    assert nothing["recall"] is None
    assert nothing["f1"] is None
    assert nothing["csi"] is None
    # Both rates defined but zero: still no harmonic mean to report.
    empty = skill_scores(hits=0, false_alarms=1, misses=1, late=0)
    assert empty["precision"] == 0.0
    assert empty["recall"] == 0.0
    assert empty["f1"] is None


def test_pooled_summary_pools_the_late_column_and_recomputes_the_scores() -> None:
    prompt = score_warnings(
        [(_at(0), 20.0)], [_at(20)], min_useful_lead_min=5.0,
    )                                                   # 1 hit
    tardy = score_warnings(
        [(_at(0), 20.0), (_at(600), 20.0)], [_at(2)], min_useful_lead_min=5.0,
    )                                                   # 1 late, 1 false alarm
    pooled = pooled_summary([prompt, tardy], lead_min=30)
    assert (pooled["hits"], pooled["late"], pooled["false_alarms"]) == (1, 1, 1)
    assert pooled["precision"] == pytest.approx(0.5)     # 1 / (1 + 1)
    assert pooled["recall"] == pytest.approx(0.5)        # 1 / (1 + 0 + 1)
    assert pooled["pod"] == pooled["recall"]
    assert pooled["f1"] == pytest.approx(0.5)
    assert pooled["csi"] == pytest.approx(1 / 3)         # 1 / (1 + 0 + 1 + 1)
    assert pooled["far"] == pytest.approx(1 / 3)         # 1 wrong of 3 sent
    assert pooled["lead_error_min"]["n"] == 1
