"""The neighbour-gauge features, and the random-point validation helpers.

The ``ng_*`` block is the approximation that stands in for the one feature
family a subscriber cannot have — the gauge under their feet. Everything
about it is a claim about GEOMETRY (which neighbours count, and how the
motion decides), and geometry is exactly the kind of thing that is easy to
get backwards and impossible to notice: an upstream/downstream sign error
would still produce a model, still produce a skill number, and still be
worth nothing. So the layouts here are synthetic and exact — gauges placed
at whole kilometres around a point, a motion with a bearing chosen by hand
— and every assertion is arithmetic rather than "roughly".

The direction convention under test, once, in words: ``bulk_dir_deg`` is
the compass bearing the rain is heading TOWARD (0 = north, 90 = east; see
this module's docstring and the ``bulk_dir_deg`` catalogue entry). A gauge
is therefore UPSTREAM when it lies in the direction the rain is coming
FROM, i.e. opposite the bearing.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from dmi_nowcast_core import postprocess as pp

NOW = datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc)
HOME = (56.0, 10.0)


def _offset(east_km: float, north_km: float = 0.0) -> tuple[float, float]:
    """A coordinate ``east_km`` east and ``north_km`` north of :data:`HOME`."""
    return (
        HOME[0] + north_km / pp.KM_PER_DEG_LAT,
        HOME[1] + east_km / pp.KM_PER_DEG_LON,
    )


def _slots(
    mm_by_age: dict[int, float] | None = None,
    *,
    dry_mm: float = 0.0,
    minutes: int = 420,
    known: bool = True,
) -> list[tuple[datetime, bool | None, float | None]]:
    """A contiguous slot grid ending at ``NOW``.

    ``mm_by_age`` maps "minutes before NOW at which the slot ENDS" to the
    amount in it; every other slot carries ``dry_mm``. ``known=False``
    gives a gauge that reported nothing at all.

    Note which ages are VISIBLE: at the default 10-minute availability lag
    the horizon is ``NOW - 10``, so the slot ending at age 0 has not
    reached the service yet and the freshest readable one is at age 10.
    Every fixture below that wants rain the model may see puts it at 10 or
    older, and :class:`TestTheVisibilityHorizon` is the one that puts it
    at 0 on purpose.
    """
    amounts = dict(mm_by_age or {})
    out = []
    for age in range(minutes, -1, -10):
        end = NOW - timedelta(minutes=age)
        if not known:
            out.append((end, None, None))
            continue
        mm = float(amounts.get(age, dry_mm))
        out.append((end, mm >= 0.1, mm))
    return out


def _features(
    points, slots, coords, *, bulk_kmh=40.0, bulk_dir_deg=90.0, **kwargs,
) -> dict[str, np.ndarray]:
    return pp.neighbour_gauge_features(
        points, slots, coords,
        now_utc=NOW, bulk_kmh=bulk_kmh, bulk_dir_deg=bulk_dir_deg, **kwargs,
    )


# ---------------------------------------------------------------------------
# The sign of "upstream"
# ---------------------------------------------------------------------------


class TestTheMotionFrame:
    """Three gauges, one motion, and only one of them upstream.

    ``W`` is 20 km west, ``E`` 20 km east, ``N`` 20 km north; the rain is
    heading east at 40 km/h. Under the convention the only gauge whose
    rain can reach the point is ``W`` — it is 30 minutes upstream — ``E``
    is behind the point and ``N`` is off the track by 20 km, which is more
    than the corridor's half-width.
    """

    COORDS = {
        "W": _offset(-20.0), "E": _offset(20.0), "N": _offset(0.0, 20.0),
    }

    def _slots(self) -> dict:
        # All three are raining hard, so anything that shows up in the
        # upstream block got there by geometry and not by being the only
        # wet gauge in the country.
        return {sid: _slots({10: 2.0, 20: 2.0, 30: 2.0}) for sid in self.COORDS}

    def test_the_upstream_gauge_is_the_one_the_rain_comes_from(self) -> None:
        got = _features([HOME], self._slots(), self.COORDS)
        assert got["ng_frame_ok"][0] == 1.0
        # 20 km at 40 km/h is 30 minutes: the first bin, and only it.
        assert got["ng_up_count_t30"][0] == 1.0
        assert got["ng_up_count_t60"][0] == 0.0
        assert got["ng_up_count_t120"][0] == 0.0
        assert got["ng_upwet_tau_min"][0] == pytest.approx(30.0, abs=1e-3)
        assert got["ng_upwet_cross_km"][0] == pytest.approx(0.0, abs=1e-6)

    def test_reversing_the_motion_reverses_which_gauge_counts(self) -> None:
        """The same layout with the rain heading west picks ``E`` instead.

        This is the assertion that actually pins the sign: a convention
        error that made "upstream" mean "downstream" would pass the test
        above by symmetry of the fixture and fail this one.
        """
        east = _features([HOME], self._slots(), self.COORDS, bulk_dir_deg=90.0)
        west = _features([HOME], self._slots(), self.COORDS, bulk_dir_deg=270.0)
        assert east["ng_up_count_t30"][0] == west["ng_up_count_t30"][0] == 1.0
        # Both find exactly one gauge at tau = 30 min, and the two cannot
        # be the same gauge — so drop E from the layout and watch the
        # eastward case lose nothing while the westward case loses all.
        without_e = {k: v for k, v in self.COORDS.items() if k != "E"}
        slots = {k: v for k, v in self._slots().items() if k != "E"}
        assert _features([HOME], slots, without_e, bulk_dir_deg=90.0)[
            "ng_up_count_t30"
        ][0] == 1.0
        assert _features([HOME], slots, without_e, bulk_dir_deg=270.0)[
            "ng_up_count_t30"
        ][0] == 0.0

    def test_a_downstream_gauge_reaches_the_vicinity_block_and_nothing_else(
        self,
    ) -> None:
        """``E`` is 20 km away and raining, and says so only without direction."""
        coords = {"E": self.COORDS["E"]}
        slots = {"E": _slots({10: 2.0, 20: 2.0, 30: 2.0})}
        got = _features([HOME], slots, coords)
        assert got["ng_up_count_t30"][0] == 0.0
        assert math.isnan(got["ng_upwet_tau_min"][0])
        assert math.isnan(got["ng_up_mm_max_t30"][0])
        # ...but the direction-free block still knows it is raining 20 km away.
        assert got["ng_near_km"][0] == pytest.approx(20.0, abs=0.05)
        assert got["ng_count_20km"][0] == 1.0
        assert got["ng_wet_share_20km"][0] == 1.0
        assert got["ng_near_min_since_wet"][0] == pytest.approx(10.0, abs=1e-6)

    def test_the_corridor_has_a_hard_cross_track_edge(self) -> None:
        """A gauge upstream but too far off the track does not count.

        Two gauges 20 km upstream, one 14 km off the track and one 16 km
        off: the corridor's half-width is 15 km, so exactly one is inside.
        """
        coords = {
            "IN": _offset(-20.0, 14.0), "OUT": _offset(-20.0, 16.0),
        }
        slots = {sid: _slots({10: 3.0}) for sid in coords}
        got = _features([HOME], slots, coords)
        assert got["ng_up_count_t30"][0] == 1.0
        assert got["ng_upwet_cross_km"][0] == pytest.approx(14.0, abs=0.05)

    def test_travel_time_puts_each_gauge_in_its_own_bin(self) -> None:
        """One gauge per bin, at 30 km/h: 10, 20 and 40 km upstream."""
        coords = {
            "A": _offset(-10.0),   # tau = 20 min  -> (0, 30]
            "B": _offset(-20.0),   # tau = 40 min  -> (30, 60]
            "C": _offset(-40.0),   # tau = 80 min  -> (60, 120]
            "D": _offset(-59.0),   # tau = 118 min -> (60, 120], still inside
        }
        slots = {
            "A": _slots({10: 1.0}), "B": _slots({10: 2.0}),
            "C": _slots({10: 3.0}), "D": _slots({10: 4.0}),
        }
        got = _features([HOME], slots, coords, bulk_kmh=30.0)
        assert got["ng_up_count_t30"][0] == 1.0
        assert got["ng_up_count_t60"][0] == 1.0
        assert got["ng_up_count_t120"][0] == 2.0
        # The bins are disjoint: each maximum is its own bin's gauge.
        assert got["ng_up_mm_max_t30"][0] == pytest.approx(1.0)
        assert got["ng_up_mm_max_t60"][0] == pytest.approx(2.0)
        assert got["ng_up_mm_max_t120"][0] == pytest.approx(4.0)
        # The nearest wet gauge by arrival time is the nearest one.
        assert got["ng_upwet_tau_min"][0] == pytest.approx(20.0, abs=1e-3)
        assert got["ng_upwet_mm_30"][0] == pytest.approx(1.0)

    def test_a_gauge_beyond_the_radius_is_out_even_when_it_is_upstream(
        self,
    ) -> None:
        coords = {"FAR": _offset(-70.0)}
        slots = {"FAR": _slots({10: 9.0})}
        got = _features([HOME], slots, coords, bulk_kmh=60.0)
        assert got["ng_up_count_t30"][0] == 0.0
        # ...and it is not the nearest-gauge block's business either way:
        # ng_near_km is pure geometry and reports it.
        assert got["ng_near_km"][0] == pytest.approx(70.0, abs=0.2)
        assert got["ng_count_20km"][0] == 0.0
        assert math.isnan(got["ng_wet_share_20km"][0])

    def test_the_cross_track_weight_leans_on_the_gauge_in_the_middle(
        self,
    ) -> None:
        """The weighted mean is not the plain mean when the gauges differ.

        Two gauges the same 20 km upstream: one on the track with 1 mm,
        one 10 km off with 5 mm. Weights are 1 and 1/(1 + 10/5) = 1/3, so
        the weighted mean is (1 + 5/3) / (4/3) = 2.0 against a plain mean
        of 3.0.
        """
        coords = {"ON": _offset(-20.0), "OFF": _offset(-20.0, 10.0)}
        slots = {"ON": _slots({10: 1.0}), "OFF": _slots({10: 5.0})}
        got = _features([HOME], slots, coords)
        assert got["ng_up_mm_max_t30"][0] == pytest.approx(5.0)
        assert got["ng_up_mm_wmean_t30"][0] == pytest.approx(2.0, abs=1e-3)
        assert got["ng_up_wet_share_t30"][0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# frame_ok: what survives when the motion does not
# ---------------------------------------------------------------------------


class TestTheFrameFallback:
    COORDS = {"W": _offset(-15.0)}

    def _slots(self) -> dict:
        return {"W": _slots({10: 4.0, 20: 4.0})}

    @pytest.mark.parametrize(
        "speed, bearing",
        [
            (4.0, 90.0),            # below NG_MIN_SPEED_KMH
            (0.0, 90.0),
            (40.0, float("nan")),   # no direction
            (float("nan"), 90.0),
            (None, None),
        ],
    )
    def test_an_unusable_motion_nulls_the_upstream_block(
        self, speed, bearing,
    ) -> None:
        got = _features(
            [HOME], self._slots(), self.COORDS,
            bulk_kmh=speed, bulk_dir_deg=bearing,
        )
        assert got["ng_frame_ok"][0] == 0.0
        for name, _definition in pp.SCALAR_FEATURE_COLUMNS_NG:
            if name.startswith(("ng_up_", "ng_upwet_")):
                assert math.isnan(got[name][0]), name

    def test_the_vicinity_block_is_still_filled(self) -> None:
        got = _features(
            [HOME], self._slots(), self.COORDS, bulk_kmh=1.0, bulk_dir_deg=90.0,
        )
        assert got["ng_frame_ok"][0] == 0.0
        assert got["ng_near_km"][0] == pytest.approx(15.0, abs=0.05)
        assert got["ng_count_20km"][0] == 1.0
        assert got["ng_wet_share_20km"][0] == 1.0
        assert got["ng_near_mm_60"][0] == pytest.approx(8.0)

    def test_the_speed_floor_is_the_documented_one(self) -> None:
        at = _features(
            [HOME], self._slots(), self.COORDS,
            bulk_kmh=pp.NG_MIN_SPEED_KMH, bulk_dir_deg=90.0,
        )
        below = _features(
            [HOME], self._slots(), self.COORDS,
            bulk_kmh=pp.NG_MIN_SPEED_KMH - 0.01, bulk_dir_deg=90.0,
        )
        assert at["ng_frame_ok"][0] == 1.0
        assert below["ng_frame_ok"][0] == 0.0


# ---------------------------------------------------------------------------
# Availability: the model may not read what the service had not been told
# ---------------------------------------------------------------------------


class TestTheVisibilityHorizon:
    """The same rule and the same horizon as ``station_gauge_features``."""

    COORDS = {"W": _offset(-20.0)}

    def test_a_slot_inside_the_lag_is_invisible(self) -> None:
        """Rain in the newest slot cannot be read at a ten-minute lag.

        One wet slot, ending AT the decision instant, with dry slots
        behind it. At the default lag the horizon is ``NOW - 10``, that
        slot ends after it, and the upstream gauge reads exactly as it
        would if the shower had not happened: known, dry, no wet neighbour
        anywhere. Drop the lag to zero and the very same input says the
        opposite.
        """
        slots = {"W": _slots({0: 6.0})}
        hidden = _features([HOME], slots, self.COORDS)
        assert hidden["ng_up_count_t30"][0] == 1.0     # known...
        assert hidden["ng_up_mm_max_t30"][0] == pytest.approx(0.0)  # ...and dry
        assert hidden["ng_up_wet_share_t30"][0] == pytest.approx(0.0)
        assert math.isnan(hidden["ng_upwet_tau_min"][0])
        assert hidden["ng_wet_share_20km"][0] == pytest.approx(0.0)
        seen = _features([HOME], slots, self.COORDS, lag_min=0.0)
        assert seen["ng_up_mm_max_t30"][0] == pytest.approx(6.0)
        assert seen["ng_upwet_mm_30"][0] == pytest.approx(6.0)
        assert seen["ng_upwet_tau_min"][0] == pytest.approx(30.0, abs=1e-3)

    def test_the_millimetre_window_is_measured_from_the_horizon(self) -> None:
        """30 visible minutes means the three slots before the horizon.

        Slots ending 10, 20, 30 and 40 minutes before NOW carry 1 mm each.
        At a 10-minute lag the horizon is NOW - 10, so the visible last
        half hour is the slots ending at -10, -20 and -30: 3 mm, not 4.
        """
        slots = {"W": _slots({10: 1.0, 20: 1.0, 30: 1.0, 40: 1.0})}
        got = _features([HOME], slots, self.COORDS)
        assert got["ng_up_mm_max_t30"][0] == pytest.approx(3.0)
        assert got["ng_upwet_mm_30"][0] == pytest.approx(3.0)
        # ...and the 60-minute vicinity window catches all four.
        assert got["ng_near_mm_60"][0] == pytest.approx(4.0)

    def test_the_horizon_is_the_own_gauge_block_s_horizon(self) -> None:
        """One availability rule, read at the point and at its neighbour.

        The ``g_*`` block and the ``ng_*`` block are two readings of the
        same archive under the same lag, and a model trained on one and a
        neighbour described by the other would be comparing a gauge that
        may see the last ten minutes against one that may not. Rather than
        assert the rule twice, this walks the boundary: for every lag, what
        ``station_gauge_features`` says about a gauge AT the point must be
        what ``neighbour_gauge_features`` says about the same gauge 20 km
        upstream of it.
        """
        slots = _slots({0: 1.0, 10: 1.0, 20: 1.0, 30: 1.0})
        for lag in (0.0, 10.0, 20.0, 35.0):
            own = pp.station_gauge_features([slots], now_utc=NOW, lag_min=lag)
            neighbour = _features(
                [HOME], {"W": slots}, self.COORDS, lag_min=lag,
            )
            # ``ng_up_mm_max_t30`` over a single upstream gauge IS that
            # gauge's ``g_mm_30``; ``ng_upwet_*`` is not the comparison,
            # because it exists only where a neighbour is actually wet.
            assert neighbour["ng_up_mm_max_t30"][0] == pytest.approx(
                own["g_mm_30"][0], abs=1e-5,
            ), lag
            assert neighbour["ng_near_min_since_wet"][0] == pytest.approx(
                own["g_min_since_wet"][0], abs=1e-5,
            ), lag

    def test_minutes_since_wet_is_measured_from_the_decision_instant(
        self,
    ) -> None:
        """The lag is part of the age, exactly as in ``g_min_since_wet``."""
        slots = {"W": _slots({40: 1.0})}
        got = _features([HOME], slots, self.COORDS)
        assert got["ng_near_min_since_wet"][0] == pytest.approx(40.0)

    def test_a_gauge_dry_all_window_is_capped_not_null(self) -> None:
        got = _features([HOME], {"W": _slots()}, self.COORDS)
        assert got["ng_near_min_since_wet"][0] == pytest.approx(
            pp.GAUGE_SINCE_CAP_MIN,
        )
        assert got["ng_wet_share_20km"][0] == 0.0
        assert got["ng_count_20km"][0] == 1.0

    def test_a_silent_gauge_is_null_and_out_of_every_denominator(self) -> None:
        got = _features(
            [HOME], {"W": _slots(known=False)}, self.COORDS,
        )
        assert math.isnan(got["ng_near_min_since_wet"][0])
        assert math.isnan(got["ng_near_mm_60"][0])
        assert got["ng_count_20km"][0] == 0.0
        assert math.isnan(got["ng_wet_share_20km"][0])
        assert got["ng_up_count_t30"][0] == 0.0
        # The geometry is still geometry: a silent gauge is still a gauge.
        assert got["ng_near_km"][0] == pytest.approx(20.0, abs=0.05)


# ---------------------------------------------------------------------------
# Leave-self-out
# ---------------------------------------------------------------------------


class TestSelfExclusion:
    """The reason the block means the same thing at a gauge and at an address."""

    COORDS = {"HOME": HOME, "W": _offset(-20.0)}

    def _slots(self) -> dict:
        # The point's own gauge is soaked; the neighbour is dry.
        return {"HOME": _slots({10: 50.0, 20: 50.0}), "W": _slots()}

    def test_the_point_s_own_gauge_never_appears(self) -> None:
        got = _features(
            [HOME], self._slots(), self.COORDS, exclude_self=["HOME"],
        )
        assert got["ng_near_km"][0] == pytest.approx(20.0, abs=0.05)
        assert got["ng_near_mm_60"][0] == pytest.approx(0.0)
        assert got["ng_wet_share_20km"][0] == 0.0
        assert got["ng_count_20km"][0] == 1.0
        assert math.isnan(got["ng_upwet_tau_min"][0])

    def test_a_gauge_at_the_point_is_excluded_without_being_named(self) -> None:
        """The belt to ``exclude_self``'s braces.

        A caller that resolved a coordinate to no station id at all — the
        live cycle's ordinary case — still cannot read a gauge standing on
        the point back as its own neighbour.
        """
        got = _features([HOME], self._slots(), self.COORDS, exclude_self=None)
        assert got["ng_near_km"][0] == pytest.approx(20.0, abs=0.05)
        assert got["ng_wet_share_20km"][0] == 0.0

    def test_a_gauge_just_outside_the_self_radius_is_a_neighbour(self) -> None:
        coords = {"NEXT_DOOR": _offset(1.0)}
        got = _features([HOME], {"NEXT_DOOR": _slots({10: 3.0})}, coords)
        assert got["ng_near_km"][0] == pytest.approx(1.0, abs=0.02)
        assert got["ng_count_20km"][0] == 1.0

    def test_exclude_self_must_line_up_with_the_points(self) -> None:
        with pytest.raises(ValueError, match="one entry per point"):
            _features(
                [HOME, HOME], self._slots(), self.COORDS, exclude_self=["HOME"],
            )


# ---------------------------------------------------------------------------
# Shape, vectorisation and the digested table
# ---------------------------------------------------------------------------


class TestTheProducerItself:
    def test_every_catalogue_column_comes_back_as_float32(self) -> None:
        coords = {"W": _offset(-20.0)}
        got = _features([HOME, _offset(5.0)], {"W": _slots({10: 1.0})}, coords)
        assert set(got) == {
            name for name, _definition in pp.SCALAR_FEATURE_COLUMNS_NG
        }
        for name, values in got.items():
            assert values.dtype == np.float32, name
            assert values.shape == (2,), name

    def test_points_are_answered_independently(self) -> None:
        """Two points, one gauge, one call — and each gets its own geometry."""
        coords = {"W": _offset(-20.0)}
        slots = {"W": _slots({10: 4.0})}
        together = _features([HOME, _offset(-10.0)], slots, coords)
        alone = [
            _features([HOME], slots, coords),
            _features([_offset(-10.0)], slots, coords),
        ]
        for name in together:
            for index in range(2):
                a, b = together[name][index], alone[index][name][0]
                assert (math.isnan(a) and math.isnan(b)) or a == pytest.approx(b)
        # The second point is 10 km from the gauge, the first 20 km.
        assert together["ng_near_km"][0] == pytest.approx(20.0, abs=0.05)
        assert together["ng_near_km"][1] == pytest.approx(10.0, abs=0.05)

    def test_the_digested_table_gives_the_same_answer_as_the_mapping(
        self,
    ) -> None:
        """``GaugeSlotTable`` is a performance choice, never a different number."""
        coords = {"W": _offset(-20.0), "E": _offset(15.0)}
        slots = {"W": _slots({10: 2.0, 20: 1.0}), "E": _slots({30: 4.0})}
        table = pp.GaugeSlotTable.from_slots(slots)
        direct = _features([HOME], slots, coords)
        prepared = _features([HOME], table, coords)
        for name in direct:
            a, b = direct[name][0], prepared[name][0]
            assert (math.isnan(a) and math.isnan(b)) or a == pytest.approx(b)

    def test_no_gauges_at_all_is_an_empty_answer_not_a_crash(self) -> None:
        got = _features([HOME], {}, {})
        assert got["ng_frame_ok"][0] == 1.0
        assert got["ng_count_20km"][0] == 0.0
        assert math.isnan(got["ng_near_km"][0])

    def test_a_gauge_with_no_coordinate_is_simply_not_a_neighbour(self) -> None:
        got = _features([HOME], {"GHOST": _slots({10: 9.0})}, {})
        assert math.isnan(got["ng_near_km"][0])
        assert got["ng_count_20km"][0] == 0.0


# ---------------------------------------------------------------------------
# The random-point validation helpers
# ---------------------------------------------------------------------------


class TestOwnGaugeMasking:
    def test_every_own_gauge_column_becomes_unknown(self) -> None:
        features = {
            "g_mm_10": np.array([1.0, 2.0]),
            "g_mm_30": np.array([1.0, 2.0]),
            "g_mm_60": np.array([1.0, 2.0]),
            "g_min_since_wet": np.array([10.0, 20.0]),
            "g_dry_60": np.array([1.0, 0.0]),
            "g_known": np.array([1.0, 1.0]),
            "ng_near_km": np.array([12.0, 8.0]),
            "raw_frac_20": np.array([0.4, 0.6]),
        }
        masked = pp.mask_own_gauge(features)
        for name in pp.OWN_GAUGE_COLUMNS:
            assert np.all(np.isnan(masked[name])), name
        assert np.all(masked["g_known"] == 0.0)
        # ...and nothing else moved. The neighbour block in particular is
        # leave-self-out already, so masking it would throw away the one
        # gauge signal an address really does get.
        assert np.array_equal(masked["ng_near_km"], features["ng_near_km"])
        assert np.array_equal(masked["raw_frac_20"], features["raw_frac_20"])
        # The input is untouched — the caller still needs it for the truth
        # side (the dry subset).
        assert features["g_known"][0] == 1.0


class TestStationGroups:
    def _coords(self, n: int = 20) -> dict[str, tuple[float, float]]:
        return {
            f"s{index:02d}": (56.0, 8.0 + index * 0.3) for index in range(n)
        }

    def test_the_split_is_deterministic_and_balanced(self) -> None:
        groups = pp.station_groups(self._coords(), groups=5)
        assert groups == pp.station_groups(self._coords(), groups=5)
        counts = np.bincount(list(groups.values()), minlength=5)
        assert counts.tolist() == [4, 4, 4, 4, 4]

    def test_every_group_spans_the_country(self) -> None:
        """Interleaved by longitude, so a group is a comb and not a region."""
        coords = self._coords()
        groups = pp.station_groups(coords, groups=5)
        for index in range(5):
            lons = [
                coords[sid][1] for sid, g in groups.items() if g == index
            ]
            assert max(lons) - min(lons) > 4.0

    def test_an_unknown_station_gets_its_own_group(self) -> None:
        groups = {"a": 0, "b": 1}
        codes = pp.group_codes(np.array(["a", "b", "zz"]), groups)
        assert codes.tolist() == [0, 1, -1]


class TestFoldPlans:
    def test_one_axis_is_leave_one_month_out(self) -> None:
        month = np.array([1, 1, 2, 2, 3])
        plan = pp.FoldPlan.by_month(month)
        assert plan.folds() == [(1,), (2,), (3,)]
        assert plan.test_mask((2,)).tolist() == [False, False, True, True, False]
        assert plan.train_mask((2,)).tolist() == [True, True, False, False, True]

    def test_two_axes_hold_out_the_month_and_the_place_together(self) -> None:
        """The assertion the whole protocol rests on.

        A row may train a fold only when it differs on BOTH axes. A row
        that shares the month, or shares the group, is used by neither
        side — which is why 50 folds cost less training data each than 10
        do, and why the number that comes out is about a station the model
        has never seen in a month it has never seen.
        """
        month = np.array([1, 1, 2, 2, 3, 3])
        group = np.array([0, 1, 0, 1, 0, 1])
        plan = pp.FoldPlan.by_month_and_group(month, group)
        assert len(plan.folds()) == 6
        test = plan.test_mask((2, 0))
        train = plan.train_mask((2, 0))
        assert test.tolist() == [False, False, True, False, False, False]
        # month 1 group 1 and month 3 group 1: different month AND group.
        assert train.tolist() == [False, True, False, False, False, True]
        # No row is in both, and the rows in neither are the shared-axis ones.
        assert not (test & train).any()
        assert int((~test & ~train).sum()) == 3

    def test_no_fold_ever_trains_on_its_own_month_or_group(self) -> None:
        rng = np.random.default_rng(4)
        month = rng.integers(0, 10, 500)
        group = rng.integers(0, 5, 500)
        plan = pp.FoldPlan.by_month_and_group(month, group)
        assert len(plan.folds()) == 50
        for fold in plan.folds():
            train = plan.train_mask(fold)
            assert not (month[train] == fold[0]).any()
            assert not (group[train] == fold[1]).any()

    def test_a_station_the_grid_dropped_never_gets_a_fold(self) -> None:
        """``-1`` is "not one of ours", not a sixth group.

        Those rows come from a station the dead-gauge rule excluded; they
        have no gradable outcome, so a fold of their own would fit a model
        per month to predict nothing. They still sit in every other fold's
        training set, where they are dropped for the same reason.
        """
        month = np.array([1, 1, 1, 2, 2, 2])
        group = np.array([0, 1, -1, 0, 1, -1])
        plan = pp.FoldPlan.by_month_and_group(month, group)
        assert plan.folds() == [(1, 0), (1, 1), (2, 0), (2, 1)]
        # ...and the unmatched rows are still available to train with.
        assert plan.train_mask((1, 0)).tolist() == [
            False, False, False, False, True, True,
        ]

    def test_the_label_names_both_axes(self) -> None:
        plan = pp.FoldPlan.by_month_and_group(
            np.array([2026 * 12 + 5]), np.array([3]),
        )
        assert plan.label((2026 * 12 + 5, 3)) == "2026-06 x g3"


class TestDistanceBins:
    def test_rows_land_in_the_bin_their_distance_names(self) -> None:
        km = np.array([0.0, 9.9, 10.0, 25.0, 31.0, np.nan])
        bins = pp.distance_bins(km)
        assert list(bins) == ["0-10 km", "10-20 km", "20-30 km", "30+ km"]
        assert bins["0-10 km"].tolist() == [True, True, False, False, False, False]
        assert bins["10-20 km"].tolist() == [False, False, True, False, False, False]
        assert bins["20-30 km"].tolist() == [False, False, False, True, False, False]
        assert bins["30+ km"].tolist() == [False, False, False, False, True, False]

    def test_a_row_with_no_distance_is_in_no_bin(self) -> None:
        bins = pp.distance_bins(np.array([np.nan, np.nan]))
        assert not any(mask.any() for mask in bins.values())


class TestRandomPointWeights:
    """A square country with a known answer, then the real one.

    A 2 degree x 2 degree box with one gauge in the middle: every random
    point's distance to it is computable by hand, so the sampler and the
    kilometre grid can be checked against arithmetic rather than against
    themselves.
    """

    SQUARE = ((
        (9.0, 55.0), (11.0, 55.0), (11.0, 57.0), (9.0, 57.0),
    ),)

    def test_the_sampler_stays_inside_the_outline(self) -> None:
        lat, lon = pp.random_points_in(self.SQUARE, n=5000, seed=1)
        assert lat.size == 5000
        assert lat.min() >= 55.0 and lat.max() <= 57.0
        assert lon.min() >= 9.0 and lon.max() <= 11.0

    def test_the_draw_is_uniform_by_area_not_by_latitude(self) -> None:
        """Equal-area sampling puts slightly more points in the south."""
        lat, _lon = pp.random_points_in(self.SQUARE, n=200_000, seed=2)
        south = float((lat < 56.0).mean())
        # sin-weighted, the southern half of a 55-57 box holds ~50.2 %.
        assert 0.500 < south < 0.506

    def test_a_point_outside_every_ring_is_rejected(self) -> None:
        inside = pp.points_inside(
            np.array([10.0, 12.0]), np.array([56.0, 56.0]), self.SQUARE,
        )
        assert inside.tolist() == [True, False]

    def test_the_weights_match_the_geometry_they_stand_on(self) -> None:
        """One gauge in the middle of the box, and the bins counted by hand.

        Inside a circle of radius r around the gauge the area is pi r^2;
        the box is 2 degrees = ~222 km tall and ~124 km wide at 56 N, so
        the 10 km circle is entirely inside it and its share is exactly
        pi * 100 / (222 * 124).
        """
        got = pp.random_point_distance_weights(
            {"mid": (56.0, 10.0)}, self.SQUARE, n=200_000, seed=3,
        )
        height = 2.0 * pp.KM_PER_DEG_LAT
        width = 2.0 * pp.KM_PER_DEG_LON
        expected = math.pi * 100.0 / (height * width)
        assert got["weights"][0] == pytest.approx(expected, rel=0.03)
        assert sum(got["weights"]) == pytest.approx(1.0)
        assert got["n"] == 200_000
        assert got["stations"] == 1

    def test_it_is_reproducible_from_the_seed(self) -> None:
        kwargs = {"outline": self.SQUARE, "n": 20_000}
        a = pp.random_point_distance_weights({"mid": (56.0, 10.0)}, seed=7, **kwargs)
        b = pp.random_point_distance_weights({"mid": (56.0, 10.0)}, seed=7, **kwargs)
        c = pp.random_point_distance_weights({"mid": (56.0, 10.0)}, seed=8, **kwargs)
        assert a["counts"] == b["counts"]
        assert a["counts"] != c["counts"]

    def test_the_gauge_distribution_is_nearest_OTHER_gauge(self) -> None:
        """What a training row's ``ng_near_km`` actually holds."""
        coords = {
            "a": (56.0, 10.0),
            "b": (56.0, 10.0 + 20.0 / pp.KM_PER_DEG_LON),
            "c": (56.0, 10.0 + 50.0 / pp.KM_PER_DEG_LON),
        }
        got = pp.gauge_distance_weights(coords)
        assert got["n"] == 3
        # a->b 20, b->a 20, c->b 30.
        assert got["km"]["p50"] == pytest.approx(20.0, abs=0.1)
        assert got["km"]["max"] == pytest.approx(30.0, abs=0.1)

    def test_denmark_is_closer_to_a_gauge_than_a_gauge_is_to_another(
        self,
    ) -> None:
        """The finding the whole re-weighting exists to expose.

        The gauge network is roughly a lattice, so a gauge's nearest
        NEIGHBOUR is about a lattice step away while a random point is
        about half a step from the nearest node. The archive's rows are
        therefore drawn from FARTHER-from-a-gauge conditions than a
        subscriber lives in, and the re-weighting moves the answer toward
        the close bins rather than away from them.
        """
        from dmi_nowcast_core.denmark_outline import DENMARK_OUTLINE

        coords = {
            f"s{index}": (55.0 + 0.3 * (index % 7), 8.5 + 0.55 * (index // 7))
            for index in range(70)
        }
        random_point = pp.random_point_distance_weights(
            coords, DENMARK_OUTLINE, n=50_000, seed=0,
        )
        at_gauge = pp.gauge_distance_weights(coords)
        assert random_point["km"]["p50"] < at_gauge["km"]["p50"]


class TestTheExpectation:
    WEIGHTS = {"0-10 km": 0.5, "10-20 km": 0.45, "20-30 km": 0.04, "30+ km": 0.01}

    def test_it_is_the_weighted_mean_of_the_bins(self) -> None:
        by_bin = {"0-10 km": 0.2, "10-20 km": 0.1, "20-30 km": 0.0, "30+ km": -0.1}
        got = pp.random_point_expectation(by_bin, self.WEIGHTS)
        assert got["value"] == pytest.approx(
            0.5 * 0.2 + 0.45 * 0.1 + 0.04 * 0.0 + 0.01 * -0.1,
        )
        assert got["covered"] == pytest.approx(1.0)

    def test_an_empty_bin_is_dropped_and_the_reach_says_so(self) -> None:
        by_bin = {"0-10 km": None, "10-20 km": 0.1}
        got = pp.random_point_expectation(by_bin, self.WEIGHTS)
        # Renormalised over the bins that have a value...
        assert got["value"] == pytest.approx(0.1)
        # ...and the reader is told it stands on 45 % of the country.
        assert got["covered"] == pytest.approx(0.45)

    def test_nothing_to_average_is_None_rather_than_zero(self) -> None:
        got = pp.random_point_expectation({"0-10 km": None}, self.WEIGHTS)
        assert got["value"] is None
        assert got["covered"] == pytest.approx(0.0)
