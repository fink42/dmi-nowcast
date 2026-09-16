"""The tree post-processor: the export, the numpy evaluator, and the budget.

Two halves, and they skip differently.

* **The evaluator**, on hand-built trees. No LightGBM, no fit — a tree
  whose right answer is arithmetic, so a wrong split rule or a wrong
  missing-value convention fails here with a number a reader can check by
  hand. These run everywhere.
* **The export**, against LightGBM itself. ``booster.dump_model()`` is
  the source of truth for what this module writes, so the only test worth
  having is the one that scores the SAME rows through LightGBM's
  ``predict`` and through the numpy evaluator and demands they agree.
  These need LightGBM, which lives only in ``.venv-fit`` (see
  ``scripts/fit_postprocess.py``) and is skipped anywhere else::

      .venv-fit/bin/python -m pytest tests/test_postprocess_trees.py

The timing budget is pinned too, and generously: the live cycle scores
~110 points at four leads inside a radar cycle, and the nightly
evaluation scores the whole archive. A change that makes the evaluator
ten times slower is a change worth noticing, and the thresholds here are
loose enough that a loaded laptop does not fail the suite.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pytest

from dmi_nowcast_core import postprocess_trees as pt

needs_lightgbm = pytest.mark.skipif(
    not pt.lightgbm_available(),
    reason="LightGBM is only in .venv-fit; run that interpreter to include these",
)


# ---------------------------------------------------------------------------
# The evaluator, on trees whose answer is arithmetic
# ---------------------------------------------------------------------------


def _stump(
    feature: int,
    threshold: float,
    left_value: float,
    right_value: float,
    *,
    default_left: bool = True,
    missing: int = pt.MISSING_NAN,
) -> pt.Tree:
    """One split, two leaves. Node 0 is the split, 1 left, 2 right."""
    return pt.Tree(
        feature=np.array([feature, -1, -1], dtype=np.int32),
        threshold=np.array([threshold, 0.0, 0.0], dtype=np.float64),
        left=np.array([1, -1, -1], dtype=np.int32),
        right=np.array([2, -1, -1], dtype=np.int32),
        value=np.array([0.0, left_value, right_value], dtype=np.float64),
        default_left=np.array([default_left, False, False], dtype=bool),
        missing=np.array([missing, 0, 0], dtype=np.int8),
    )


def test_a_row_goes_left_when_the_value_is_at_or_below_the_threshold() -> None:
    """LightGBM's rule is ``x <= threshold``, and the boundary is left."""
    ensemble = pt.TreeEnsemble(trees=(_stump(0, 2.0, -1.0, +1.0),), n_features=1)
    design = np.array([[1.9], [2.0], [2.1]])
    assert ensemble.raw_score(design).tolist() == [-1.0, -1.0, 1.0]


def test_the_base_score_is_added_once_not_once_per_tree() -> None:
    trees = (_stump(0, 0.0, 1.0, 1.0), _stump(0, 0.0, 10.0, 10.0))
    ensemble = pt.TreeEnsemble(trees=trees, base_score=0.5, n_features=1)
    assert ensemble.raw_score(np.zeros((1, 1))).tolist() == [11.5]


def test_an_empty_ensemble_is_its_base_score() -> None:
    """A single-class fold: no tree to grow, and the constant says so."""
    ensemble = pt.TreeEnsemble(trees=(), base_score=-2.0, n_features=3)
    out = pt.TreeEnsemble(trees=(), base_score=-2.0, n_features=3).predict_proba(
        np.zeros((4, 3)),
    )
    assert ensemble.raw_score(np.zeros((4, 3))).tolist() == [-2.0] * 4
    assert out == pytest.approx([1 / (1 + np.exp(2.0))] * 4)


class TestMissingValues:
    """All three of LightGBM's conventions, because a dump can hold all three."""

    def test_nan_takes_the_default_side(self) -> None:
        left = pt.TreeEnsemble(
            trees=(_stump(0, 0.0, -1.0, +1.0, default_left=True),), n_features=1,
        )
        right = pt.TreeEnsemble(
            trees=(_stump(0, 0.0, -1.0, +1.0, default_left=False),), n_features=1,
        )
        design = np.array([[np.nan]])
        assert left.raw_score(design).tolist() == [-1.0]
        assert right.raw_score(design).tolist() == [+1.0]

    def test_zero_missing_sends_zero_to_the_default_side_too(self) -> None:
        """``missing_type == "Zero"``: |x| <= 1e-35 counts as absent."""
        tree = _stump(
            0, -5.0, -1.0, +1.0, default_left=True, missing=pt.MISSING_ZERO,
        )
        ensemble = pt.TreeEnsemble(trees=(tree,), n_features=1)
        # 0.0 is above the threshold and would go right on value alone.
        design = np.array([[0.0], [np.nan], [-4.0], [-6.0]])
        assert ensemble.raw_score(design).tolist() == [-1.0, -1.0, 1.0, -1.0]

    def test_none_missing_reads_a_nan_as_zero(self) -> None:
        """A column with no NaN in training still has to answer for one."""
        tree = _stump(
            0, -1.0, -1.0, +1.0, default_left=True, missing=pt.MISSING_NONE,
        )
        ensemble = pt.TreeEnsemble(trees=(tree,), n_features=1)
        # NaN -> 0.0, which is above -1.0, so right — NOT the default side.
        assert ensemble.raw_score(np.array([[np.nan]])).tolist() == [+1.0]


def test_a_deeper_tree_walks_every_level() -> None:
    """Two levels, four leaves; each quadrant has its own value."""
    tree = pt.Tree(
        feature=np.array([0, 1, 1, -1, -1, -1, -1], dtype=np.int32),
        threshold=np.array([0.0, 0.0, 0.0, 0, 0, 0, 0], dtype=np.float64),
        left=np.array([1, 3, 5, -1, -1, -1, -1], dtype=np.int32),
        right=np.array([2, 4, 6, -1, -1, -1, -1], dtype=np.int32),
        value=np.array([0, 0, 0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        default_left=np.zeros(7, dtype=bool),
        missing=np.full(7, pt.MISSING_NAN, dtype=np.int8),
    )
    ensemble = pt.TreeEnsemble(trees=(tree,), n_features=2)
    design = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=float)
    assert ensemble.raw_score(design).tolist() == [1.0, 2.0, 3.0, 4.0]
    assert tree.depth == 3


def test_the_row_chunking_does_not_change_the_answer(monkeypatch) -> None:
    """The walker budget is a memory knob, never a numerical one.

    Not bit-identical, and it does not need to be: the leaf values are
    summed in a different pairwise order when the chunks differ, which
    costs the last couple of ulps. 1e-12 is four orders of magnitude
    tighter than anything a probability is read to.
    """
    rng = np.random.default_rng(5)
    design = rng.normal(size=(200, 3))
    trees = tuple(
        _stump(i % 3, float(rng.normal()), float(rng.normal()), float(rng.normal()))
        for i in range(40)
    )
    ensemble = pt.TreeEnsemble(trees=trees, n_features=3)
    whole = ensemble.raw_score(design)
    monkeypatch.setattr(pt, "WALKER_BUDGET", 41)  # one row per chunk
    assert pt.TreeEnsemble(trees=trees, n_features=3).raw_score(
        design,
    ) == pytest.approx(whole, abs=1e-12)


def test_a_design_of_the_wrong_width_is_refused() -> None:
    ensemble = pt.TreeEnsemble(trees=(_stump(0, 0.0, 1.0, 2.0),), n_features=4)
    with pytest.raises(ValueError, match="4"):
        ensemble.raw_score(np.zeros((2, 3)))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_the_json_round_trip_is_exact() -> None:
    rng = np.random.default_rng(11)
    trees = tuple(
        _stump(
            i % 4, float(rng.normal()), float(rng.normal()), float(rng.normal()),
            default_left=bool(i % 2),
            missing=(pt.MISSING_NAN, pt.MISSING_ZERO, pt.MISSING_NONE)[i % 3],
        )
        for i in range(12)
    )
    ensemble = pt.TreeEnsemble(
        trees=trees, base_score=0.25, n_features=4,
        feature_names=("a", "b", "c", "d"), params={"max_depth": 5},
    )
    restored = pt.TreeEnsemble.loads(ensemble.dumps())
    design = rng.normal(size=(50, 4))
    design[rng.uniform(size=(50, 4)) < 0.2] = np.nan
    assert restored.raw_score(design) == pytest.approx(
        ensemble.raw_score(design), abs=0.0,
    )
    assert restored.feature_names == ensemble.feature_names
    assert restored.params == ensemble.params
    assert restored.base_score == ensemble.base_score


def test_a_trees_block_from_another_schema_version_is_refused() -> None:
    payload = pt.TreeEnsemble(trees=(_stump(0, 0.0, 1.0, 2.0),)).to_json()
    payload["schema_version"] = pt.TREES_SCHEMA_VERSION + 1
    with pytest.raises(pt.TreeExportError, match="schema_version"):
        pt.TreeEnsemble.from_json(payload)


def test_a_categorical_split_is_refused_at_export() -> None:
    """Every design column here is numeric; a categorical split is a bug."""
    with pytest.raises(pt.TreeExportError, match="numerical"):
        pt.Tree.from_dump({
            "split_index": 0, "split_feature": 0, "threshold": "1||2",
            "decision_type": "==", "default_left": True, "missing_type": "None",
            "left_child": {"leaf_value": 1.0},
            "right_child": {"leaf_value": 2.0},
        })


def test_a_multiclass_or_averaged_booster_is_refused() -> None:
    dump = {
        "objective": "binary sigmoid:1", "num_class": 1,
        "average_output": True, "tree_info": [],
    }
    with pytest.raises(pt.TreeExportError, match="average_output"):
        pt.TreeEnsemble.from_lightgbm(dump)
    with pytest.raises(pt.TreeExportError, match="not binary"):
        pt.TreeEnsemble.from_lightgbm({
            "objective": "regression", "num_class": 1, "tree_info": [],
        })


# ---------------------------------------------------------------------------
# The budget
# ---------------------------------------------------------------------------


def _big_ensemble(n_trees: int = 300, n_features: int = 40) -> pt.TreeEnsemble:
    rng = np.random.default_rng(7)
    trees = []
    for _ in range(n_trees):
        # A depth-5 balanced tree: 31 internal nodes, 32 leaves.
        internal, leaves = 31, 32
        total = internal + leaves
        feature = np.full(total, -1, dtype=np.int32)
        left = np.full(total, -1, dtype=np.int32)
        right = np.full(total, -1, dtype=np.int32)
        feature[:internal] = rng.integers(0, n_features, size=internal)
        left[:internal] = np.arange(1, internal * 2, 2)
        right[:internal] = np.arange(2, internal * 2 + 1, 2)
        trees.append(pt.Tree(
            feature=feature,
            threshold=rng.normal(size=total),
            left=left, right=right,
            value=rng.normal(size=total) * 0.05,
            default_left=rng.uniform(size=total) < 0.5,
            missing=np.full(total, pt.MISSING_NAN, dtype=np.int8),
        ))
    return pt.TreeEnsemble(
        trees=tuple(trees), n_features=n_features,
    )


def test_a_live_cycle_scores_inside_its_budget() -> None:
    """~110 points at four leads, in well under a radar cycle's slack."""
    ensemble = _big_ensemble()
    rng = np.random.default_rng(3)
    design = rng.normal(size=(110, 40))
    ensemble.raw_score(design[:1])  # build the flat table once, as a live
    started = time.perf_counter()    # process would have on its first cycle
    for _ in range(4):
        ensemble.predict_proba(design)
    elapsed = time.perf_counter() - started
    assert elapsed < 0.05, f"110 rows x 4 leads took {elapsed * 1000:.1f} ms"


def test_the_nightly_evaluation_scores_the_archive_inside_its_budget() -> None:
    """450 000 rows — one nightly pass over the replay tree."""
    ensemble = _big_ensemble()
    rng = np.random.default_rng(4)
    design = rng.normal(size=(450_000, 40))
    started = time.perf_counter()
    ensemble.predict_proba(design)
    elapsed = time.perf_counter() - started
    assert elapsed < 60.0, f"450k rows took {elapsed:.1f} s"


# ---------------------------------------------------------------------------
# Parity with LightGBM itself
# ---------------------------------------------------------------------------


def _fitted(n: int = 4000, k: int = 12, seed: int = 0):
    import lightgbm as lgb  # noqa: F401 — imported for the skip to be honest

    rng = np.random.default_rng(seed)
    design = rng.normal(size=(n, k))
    # Missing in the first half of the columns and nowhere else, so the
    # dump carries BOTH a "NaN" missing type and a "None" one and the
    # evaluator has to know both.
    holes = rng.uniform(size=(n, k)) < 0.1
    holes[:, k // 2:] = False
    design[holes] = np.nan
    filled = np.nan_to_num(design)
    z = 0.9 * filled[:, 0] - 1.2 * filled[:, 1] * filled[:, 2] + 0.4 * filled[:, 3]
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-z))).astype(np.float64)
    result = pt.fit_trees(
        design, y,
        params={"n_estimators": 60, "max_depth": 4, "min_data_in_leaf": 20},
        feature_names=[f"c{i}" for i in range(k)], seed=1,
    )
    return design, y, result


@needs_lightgbm
def test_the_numpy_evaluator_matches_lightgbms_own_predict() -> None:
    """The one test that makes the export trustworthy.

    ``dump_model()`` is what the export reads, so the only way to know the
    reading is right is to score the same rows both ways. 1e-6 is the
    plan's tolerance; the observed difference is float noise.
    """
    design, _y, result = _fitted()
    theirs = result["booster"].predict(design)
    mine = result["ensemble"].predict_proba(design)
    assert float(np.max(np.abs(mine - theirs))) < 1e-6


@needs_lightgbm
def test_parity_survives_the_json_round_trip() -> None:
    design, _y, result = _fitted(seed=2)
    theirs = result["booster"].predict(design)
    restored = pt.TreeEnsemble.loads(result["ensemble"].dumps())
    assert float(np.max(np.abs(restored.predict_proba(design) - theirs))) < 1e-6


@needs_lightgbm
def test_the_export_carries_the_feature_names_and_the_parameters() -> None:
    _design, _y, result = _fitted(seed=3)
    ensemble = result["ensemble"]
    assert ensemble.feature_names == tuple(f"c{i}" for i in range(12))
    assert ensemble.n_features == 12
    assert ensemble.n_trees == 60
    assert ensemble.params["num_boost_round"] == 60
    assert ensemble.base_score == 0.0
    # Readable as plain JSON by anything, not only by this module.
    payload = json.loads(ensemble.dumps())
    assert payload["trees"][0]["feature"][0] >= 0


@needs_lightgbm
def test_a_single_class_fold_comes_back_as_a_constant() -> None:
    """The same answer ``fit_logistic`` gives: a base score, and it says so."""
    rng = np.random.default_rng(6)
    design = rng.normal(size=(500, 4))
    result = pt.fit_trees(design, np.zeros(500), seed=1)
    assert result["n_trees"] == 0
    assert "single-class" in result["message"]
    probability = result["ensemble"].predict_proba(design)
    assert float(probability.max()) < 0.01


@needs_lightgbm
def test_a_monotone_constraint_is_honoured_by_the_exported_ensemble() -> None:
    """The lead feature only ever pushes the answer up.

    Fitted on rows where the constrained feature is NEGATIVELY related to
    the outcome, so an unconstrained ensemble would certainly learn a
    decreasing response and the test would fail if the constraint were
    dropped on the way to LightGBM.
    """
    rng = np.random.default_rng(8)
    n, k = 6000, 3
    design = rng.normal(size=(n, k))
    design[:, 2] = rng.choice([20.0, 30.0, 45.0, 60.0], size=n)
    z = 0.8 * design[:, 0] - 0.06 * design[:, 2]
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-z))).astype(np.float64)

    free = pt.fit_trees(design, y, params={"n_estimators": 80}, seed=1)
    tied = pt.fit_trees(
        design, y, params={"n_estimators": 80}, seed=1,
        monotone_constraints=[0, 0, 1],
    )
    probe = np.tile(design[:200], (1, 1))
    curves_free, curves_tied = [], []
    for lead in (20.0, 30.0, 45.0, 60.0):
        probe[:, 2] = lead
        curves_free.append(free["ensemble"].predict_proba(probe).mean())
        curves_tied.append(tied["ensemble"].predict_proba(probe).mean())
    assert curves_free != sorted(curves_free)      # the data says "down"
    assert curves_tied == sorted(curves_tied)      # the constraint says "up"


def test_a_constraint_of_the_wrong_length_is_refused() -> None:
    if not pt.lightgbm_available():
        pytest.skip("LightGBM is only in .venv-fit")
    with pytest.raises(ValueError, match="monotone_constraints"):
        pt.fit_trees(
            np.zeros((200, 3)), np.r_[np.zeros(100), np.ones(100)],
            monotone_constraints=[1, 0],
        )
