"""The combined-rule study (``scripts/combined_rule_study.py``): rule columns,
the engine equivalence of the indicator column, the picker and LOMO."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import combined_rule_study as study  # noqa: E402
import eta_revision_study as ers  # noqa: E402

NAN = float("nan")


def test_rule_columns():
    po = np.array([0.10, 0.26, 0.26, NAN, 0.50])
    pp = np.array([0.70, 0.20, 0.60, 0.90, NAN])
    col = study.rule_column
    np.testing.assert_array_equal(col(po, pp, "served", 0, 60), [1, 0, 1, 1, NAN])
    np.testing.assert_array_equal(col(po, pp, "onset", 26, 0), [0, 1, 1, NAN, 1])
    # Combined rules skip a row with either input missing.
    np.testing.assert_array_equal(col(po, pp, "or", 26, 60), [1, 1, 1, NAN, NAN])
    np.testing.assert_array_equal(col(po, pp, "and", 26, 60), [0, 0, 1, NAN, NAN])
    # 0.5 * 0.26 + 0.5 * 0.60 = 0.43
    np.testing.assert_array_equal(col(po, pp, "blend", 0.5, 43), [0, 0, 1, NAN, NAN])


def test_threshold_edge_is_the_engines_comparison():
    # p == pct / 100 fires (>=), exactly as push.engine.evaluate compares.
    p = np.array([0.35, 0.34999999])
    np.testing.assert_array_equal(study.rule_column(p, p, "served", 0, 35), [1, 0])


def test_indicator_column_replays_like_the_probability():
    """The engine only reads ``over``: p at thr and the indicator at 50 %
    produce the same pushes, including skips of null rows and re-arm."""
    rng = np.random.default_rng(1)
    n = 200
    radar = np.arange(n, dtype=np.int64) * 10 * ers.US_PER_MIN
    radar[120:] += 60 * ers.US_PER_MIN  # a coverage gap
    p = rng.uniform(0, 1, n)
    p[rng.uniform(size=n) < 0.1] = NAN
    arrays = {"radar": radar, "gen": radar + 5 * ers.US_PER_MIN, "p30": p,
              "eta": np.where(rng.uniform(size=n) < 0.5, 15.0, NAN),
              "intensity": np.full(n, NAN), "observed": np.where(rng.uniform(size=n) < 0.1, 2.0, 0.0),
              "forecast": np.full(n, NAN)}
    direct = ers.replay_pushes(arrays, 30, 45)
    ind = study.rule_column(p, p, "served", 0, 45)
    via = ers.replay_pushes({**arrays, "p30": ind}, 30, 50)
    assert direct == via
    assert len(direct[0]) > 0


def _counts(h, late, fa, miss):
    return np.array([h, late, fa, 0, miss])


def test_pick_cell_prefers_beating_served_on_both():
    served = _counts(20, 0, 80, 60)  # P 0.20, R 0.25
    train = {
        ("x", "hiF1_lowR"): _counts(30, 0, 20, 91),   # P .60 R .248: fails recall
        ("x", "both"): _counts(22, 0, 70, 58),        # P .239 R .275: beats both
        ("x", "both_lower_f1"): _counts(21, 0, 79, 59),
    }
    assert study.pick_cell(train, served) == (("x", "both"), True)
    # No cell beats served on both: plain max F1.
    train.pop(("x", "both"))
    train.pop(("x", "both_lower_f1"))
    train[("x", "worse")] = _counts(10, 0, 90, 70)
    assert study.pick_cell(train, served) == (("x", "hiF1_lowR"), False)


def test_lomo_scores_the_held_out_month_with_the_training_pick():
    # Four days: months 0, 0, 1, 1. Cell A is good in month 0, B in month 1.
    day_month = np.array([0, 0, 1, 1])
    served = np.array([_counts(5, 0, 5, 5)] * 4)
    a = np.array([_counts(9, 0, 1, 1)] * 2 + [_counts(1, 0, 9, 9)] * 2)
    b = np.array([_counts(1, 0, 9, 9)] * 2 + [_counts(9, 0, 1, 1)] * 2)
    daily = {(30, "served", 0, 45): served, (30, "or", 1, 0): a, (30, "or", 2, 0): b}
    res = study.lomo(daily, (30, "served", 0, 45), [(30, "or", 1, 0), (30, "or", 2, 0)], day_month)
    # Held-out month 0 is picked on month 1 (B wins there) and scored with B.
    assert res["picks"][0]["cell"] == ["or", 2, 0]
    assert res["picks"][1]["cell"] == ["or", 1, 0]
    np.testing.assert_array_equal(res["daily"], np.array([_counts(1, 0, 9, 9)] * 4))


def test_bootstrap_identical_rules_give_zero_delta():
    d = np.array([_counts(3, 1, 4, 2), _counts(1, 0, 2, 5), _counts(0, 0, 1, 1)])
    ci = study.bootstrap_delta(d, d, n=50)
    assert ci["d_precision_ci"] == [0.0, 0.0]
    assert ci["d_recall_ci"] == [0.0, 0.0]
