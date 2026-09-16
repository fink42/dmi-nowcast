"""Gradient-boosted trees for the post-processor, fitted offline, served in numpy.

Why a second model family at all
--------------------------------
The logistic in :mod:`dmi_nowcast_core.postprocess` is additive in the
design columns: it can say "further upstream is worse" and "faster is
better" but not "further upstream stops mattering once the bulk motion is
this slow". The splines and the hand-written interactions of the v2 design
buy back some of that, one guess at a time. A boosted ensemble finds the
same shape without anyone guessing which pair to multiply.

Why the split between fitting and serving
-----------------------------------------
The sidecar image carries numpy, scipy and pyarrow. It does **not** carry
LightGBM or scikit-learn, and it is not going to: the serving container is
rebuilt on every deploy and it is the thing a radar cycle blocks on. So
the fit happens offline — ``scripts/fit_postprocess.py``, in the
``.venv-fit`` virtualenv described in its docstring — and what ships is a
**JSON description of the trees** that this module evaluates in pure
numpy.

    LightGBM  --dump_model()-->  :class:`TreeEnsemble`  --to_json()-->
    postprocess.json  --loads()-->  :meth:`TreeEnsemble.predict_proba`

``booster.dump_model()`` is the source of truth for the export, and
``tests/test_postprocess_trees.py`` pins the numpy evaluator against
LightGBM's own ``predict`` to 1e-6 on the same rows. If those two ever
disagree the export is wrong, not the evaluator.

The split rule, exactly as LightGBM implements it
-------------------------------------------------
A numerical node sends a row **left** when ``x <= threshold``. Missing is
where the subtlety is, and the dump tells us which of three conventions
each node was built under:

``missing_type == "NaN"``
    NaN goes to the node's ``default_left`` side.
``missing_type == "Zero"``
    NaN *and* values within ``1e-35`` of zero go to the default side.
``missing_type == "None"``
    No missing handling was built into the node; a NaN arriving at serving
    time is read as 0.0 and compared normally — which is what LightGBM's
    own predictor does.

All three are implemented because all three can appear in one dump: a
column with no NaN in the training rows gets ``"None"`` and would still
see a NaN from a live cycle that could not compute it.

Conventions
-----------
* numpy in, numpy out, vectorised over rows **and** trees: the whole
  forest is walked together, one numpy pass per tree level. Measured on a
  300-tree depth-5 ensemble over a 40-column design: 450 000 rows in
  ~13 s (the nightly evaluation), 110 rows × 4 leads in ~12 ms (a live
  cycle). Walking one tree at a time costs the same for the big case and
  twelve times as much for the small one, which is the one a radar cycle
  waits on.
* The raw score is the plain **sum of leaf values** plus ``base_score``.
  LightGBM folds its ``boost_from_average`` init score into the first
  tree's leaves, so ``base_score`` is 0.0 for an ensemble exported from a
  booster — the field exists so a document can carry one anyway.
* Categorical splits are refused at export time. Every column of this
  project's design is numeric, including the one-hots, and a silently
  mis-evaluated categorical split would be a wrong probability rather than
  an error.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "TREES_SCHEMA_VERSION",
    "MISSING_NONE",
    "MISSING_ZERO",
    "MISSING_NAN",
    "ZERO_THRESHOLD",
    "Tree",
    "TreeEnsemble",
    "DEFAULT_TREE_PARAMS",
    "lightgbm_available",
    "fit_trees",
]

#: Version of the ``trees`` block this module writes and reads.
TREES_SCHEMA_VERSION = 1

#: ``missing_type`` codes, as stored in the JSON. Integers rather than the
#: dump's strings: there is one per node and a 300-tree ensemble has tens
#: of thousands of them.
MISSING_NONE = 0
MISSING_ZERO = 1
MISSING_NAN = 2

_MISSING_FROM_DUMP = {
    "None": MISSING_NONE,
    "Zero": MISSING_ZERO,
    "NaN": MISSING_NAN,
}

#: LightGBM's ``kZeroThreshold``: what counts as zero for a ``"Zero"``
#: missing type.
ZERO_THRESHOLD = 1e-35


class TreeExportError(ValueError):
    """A booster this module refuses to export (or a document it refuses)."""


@dataclass(frozen=True, eq=False)
class Tree:
    """One regression tree, as flat node arrays with the root at index 0.

    A leaf is a node whose ``feature`` is ``-1``; its ``value`` is the leaf
    output and its ``left``/``right`` are ``-1``. An internal node's
    ``value`` is unused. Flat arrays rather than nested dicts because the
    evaluator indexes them with a whole column of node ids at once.
    """

    feature: np.ndarray       # int32, -1 at a leaf
    threshold: np.ndarray     # float64
    left: np.ndarray          # int32
    right: np.ndarray         # int32
    value: np.ndarray         # float64, the leaf output
    default_left: np.ndarray  # bool
    missing: np.ndarray       # int8, one of the MISSING_* codes

    @property
    def n_nodes(self) -> int:
        return int(self.feature.size)

    @property
    def depth(self) -> int:
        """Longest root-to-leaf path, in nodes. Reported, not relied on.

        Bounded by the node count so a malformed document — a child index
        pointing back up the tree — comes back with a finite number rather
        than hanging the process that read it.
        """
        depth = 0
        stack = [(0, 1)]
        budget = 2 * self.n_nodes + 2
        while stack and budget > 0:
            budget -= 1
            node, level = stack.pop()
            if not 0 <= node < self.n_nodes or self.feature[node] < 0:
                depth = max(depth, level)
                continue
            stack.append((int(self.left[node]), level + 1))
            stack.append((int(self.right[node]), level + 1))
        return depth

    # -- persistence --------------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        return {
            "feature": [int(v) for v in self.feature],
            "threshold": [float(v) for v in self.threshold],
            "left": [int(v) for v in self.left],
            "right": [int(v) for v in self.right],
            "value": [float(v) for v in self.value],
            "default_left": [bool(v) for v in self.default_left],
            "missing": [int(v) for v in self.missing],
        }

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> "Tree":
        feature = np.asarray(raw["feature"], dtype=np.int32)
        size = feature.size
        return cls(
            feature=feature,
            threshold=np.asarray(raw["threshold"], dtype=np.float64),
            left=np.asarray(raw["left"], dtype=np.int32),
            right=np.asarray(raw["right"], dtype=np.int32),
            value=np.asarray(raw["value"], dtype=np.float64),
            default_left=np.asarray(raw["default_left"], dtype=bool),
            missing=np.asarray(
                raw.get("missing", np.full(size, MISSING_NAN)), dtype=np.int8,
            ),
        )

    @classmethod
    def from_dump(cls, structure: Mapping[str, Any]) -> "Tree":
        """One ``tree_structure`` of ``booster.dump_model()``, flattened."""
        feature: list[int] = []
        threshold: list[float] = []
        left: list[int] = []
        right: list[int] = []
        value: list[float] = []
        default_left: list[bool] = []
        missing: list[int] = []

        def add(node: Mapping[str, Any]) -> int:
            index = len(feature)
            if "leaf_value" in node:
                feature.append(-1)
                threshold.append(0.0)
                left.append(-1)
                right.append(-1)
                value.append(float(node["leaf_value"]))
                default_left.append(False)
                missing.append(MISSING_NONE)
                return index
            decision = str(node.get("decision_type", "<="))
            if decision != "<=":
                raise TreeExportError(
                    f"node {node.get('split_index')} splits on "
                    f"'{decision}'; only numerical '<=' splits are exported"
                )
            kind = str(node.get("missing_type", "None"))
            if kind not in _MISSING_FROM_DUMP:
                raise TreeExportError(f"unknown missing_type {kind!r}")
            feature.append(int(node["split_feature"]))
            threshold.append(float(node["threshold"]))
            left.append(-1)
            right.append(-1)
            value.append(0.0)
            default_left.append(bool(node.get("default_left", False)))
            missing.append(_MISSING_FROM_DUMP[kind])
            left[index] = add(node["left_child"])
            right[index] = add(node["right_child"])
            return index

        add(structure)
        return cls(
            feature=np.asarray(feature, dtype=np.int32),
            threshold=np.asarray(threshold, dtype=np.float64),
            left=np.asarray(left, dtype=np.int32),
            right=np.asarray(right, dtype=np.int32),
            value=np.asarray(value, dtype=np.float64),
            default_left=np.asarray(default_left, dtype=bool),
            missing=np.asarray(missing, dtype=np.int8),
        )


#: How many (row, tree) walkers one evaluation chunk may carry. The whole
#: forest is walked at once — one numpy pass per tree LEVEL rather than
#: per tree — which is what keeps a 110-point live cycle under a
#: millisecond instead of tens of them; the budget is what keeps a
#: 450 000-row nightly evaluation from asking for 135 million of them at
#: once. 2 M walkers is ~16 MB of int64 plus a handful of temporaries.
WALKER_BUDGET = 2_000_000


class _FlatForest:
    """Every tree's nodes concatenated into one table, children re-indexed.

    Built once per ensemble and cached. The point is the evaluator's
    Python overhead: walking 300 trees one at a time is 300 × depth numpy
    calls whatever the row count, which at a live cycle's ~110 rows is
    almost all of the runtime. Walking the whole forest at once is
    ``depth`` calls over ``rows × trees`` elements, so the overhead is
    paid once and the useful work is identical.
    """

    __slots__ = (
        "feature", "threshold", "left", "right", "value", "default_left",
        "missing", "roots", "n_trees", "depth", "max_nodes",
    )

    def __init__(self, trees: Sequence[Tree]) -> None:
        offsets: list[int] = []
        total = 0
        for tree in trees:
            offsets.append(total)
            total += tree.n_nodes
        self.n_trees = len(trees)
        self.roots = np.asarray(offsets, dtype=np.int64)
        self.feature = np.concatenate(
            [t.feature for t in trees]
        ).astype(np.int64, copy=False)
        self.threshold = np.concatenate([t.threshold for t in trees])
        self.value = np.concatenate([t.value for t in trees])
        self.default_left = np.concatenate([t.default_left for t in trees])
        self.missing = np.concatenate([t.missing for t in trees])
        left = np.concatenate([
            np.where(t.left < 0, 0, t.left.astype(np.int64) + offset)
            for t, offset in zip(trees, offsets)
        ]).astype(np.int64, copy=False)
        right = np.concatenate([
            np.where(t.right < 0, 0, t.right.astype(np.int64) + offset)
            for t, offset in zip(trees, offsets)
        ]).astype(np.int64, copy=False)
        self.left, self.right = left, right
        self.depth = max((t.depth for t in trees), default=1)
        self.max_nodes = max((t.n_nodes for t in trees), default=1)


@dataclass(frozen=True, eq=False)
class TreeEnsemble:
    """A boosted ensemble, and the only thing about it the service needs.

    ``raw_score`` is ``base_score + Σ leaf value``; ``predict_proba`` is
    its logistic. Nothing here knows about leads, calibration or gauges —
    :class:`~dmi_nowcast_core.postprocess.LeadModel` owns those.
    """

    trees: tuple[Tree, ...]
    base_score: float = 0.0
    n_features: int = 0
    #: Informational: the design column names the ensemble was fitted on,
    #: in order. The model document carries the authoritative copy; this
    #: one makes a stray ``trees`` block readable on its own.
    feature_names: tuple[str, ...] = ()
    #: The LightGBM parameters the fit ran under, for the report.
    params: dict[str, Any] | None = None

    @property
    def n_trees(self) -> int:
        return len(self.trees)

    @property
    def n_nodes(self) -> int:
        return sum(tree.n_nodes for tree in self.trees)

    # -- evaluation ---------------------------------------------------------

    def _forest(self) -> "_FlatForest | None":
        """The cached flat node table; None for an empty ensemble."""
        if not self.trees:
            return None
        forest = self.__dict__.get("_flat")
        if forest is None:
            forest = _FlatForest(self.trees)
            object.__setattr__(self, "_flat", forest)
        return forest

    def raw_score(self, design: np.ndarray) -> np.ndarray:
        """Sum of the leaf values, plus the base score. One value per row.

        Vectorised over rows AND trees: every (row, tree) pair walks down
        one level per numpy pass, in chunks of :data:`WALKER_BUDGET` pairs,
        and a pair that has reached a leaf drops out of the next pass.
        ``design`` is the RAW design matrix — NaN means missing and each
        node's own ``missing_type`` decides where it goes.
        """
        matrix = np.ascontiguousarray(design, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError(f"design must be 2-D, got {matrix.ndim}-D")
        if self.n_features and matrix.shape[1] != self.n_features:
            raise ValueError(
                f"design has {matrix.shape[1]} columns, the ensemble was "
                f"fitted on {self.n_features}"
            )
        n = matrix.shape[0]
        out = np.full(n, float(self.base_score), dtype=np.float64)
        forest = self._forest()
        if forest is None or n == 0:
            return out
        trees = forest.n_trees
        chunk = max(1, WALKER_BUDGET // max(trees, 1))
        feature, left, right = forest.feature, forest.left, forest.right
        threshold, default_left = forest.threshold, forest.default_left
        missing, value = forest.missing, forest.value
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            width = stop - start
            node = np.repeat(forest.roots, width)
            rows = np.tile(np.arange(start, stop, dtype=np.int64), trees)
            live = np.arange(node.size, dtype=np.int64)
            # One pass per level. Bounded by the deepest tree's node count
            # so a document whose children point back up terminates.
            for _ in range(forest.max_nodes + 1):
                if live.size == 0:
                    break
                here = node[live]
                column = feature[here]
                internal = column >= 0
                if not internal.all():
                    live = live[internal]
                    if live.size == 0:
                        break
                    here = here[internal]
                    column = column[internal]
                values = matrix[rows[live], column]
                filled = np.nan_to_num(values, nan=0.0)
                kind = missing[here]
                to_default = (np.isnan(values) & (kind != MISSING_NONE)) | (
                    (kind == MISSING_ZERO) & (np.abs(filled) <= ZERO_THRESHOLD)
                )
                go_left = np.where(
                    to_default, default_left[here], filled <= threshold[here],
                )
                node[live] = np.where(go_left, left[here], right[here])
            out[start:stop] += value[node].reshape(trees, width).sum(axis=0)
        return out

    def predict_proba(self, design: np.ndarray) -> np.ndarray:
        """The ensemble's probability per row — LightGBM's ``predict``."""
        return _sigmoid(self.raw_score(design))

    # -- persistence --------------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": TREES_SCHEMA_VERSION,
            "base_score": float(self.base_score),
            "n_features": int(self.n_features),
            "n_trees": self.n_trees,
            "feature_names": list(self.feature_names),
            "params": dict(self.params or {}),
            "trees": [tree.to_json() for tree in self.trees],
        }

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> "TreeEnsemble":
        version = int(raw.get("schema_version", TREES_SCHEMA_VERSION))
        if version != TREES_SCHEMA_VERSION:
            raise TreeExportError(
                f"trees block schema_version {version}, expected "
                f"{TREES_SCHEMA_VERSION}"
            )
        return cls(
            trees=tuple(Tree.from_json(entry) for entry in raw["trees"]),
            base_score=float(raw.get("base_score", 0.0)),
            n_features=int(raw.get("n_features", 0)),
            feature_names=tuple(str(v) for v in raw.get("feature_names") or ()),
            params=dict(raw.get("params") or {}) or None,
        )

    def dumps(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_json(), indent=indent)

    @classmethod
    def loads(cls, text: str) -> "TreeEnsemble":
        return cls.from_json(json.loads(text))

    # -- export -------------------------------------------------------------

    @classmethod
    def from_lightgbm(
        cls,
        booster: Any,
        *,
        feature_names: Sequence[str] = (),
        params: Mapping[str, Any] | None = None,
    ) -> "TreeEnsemble":
        """Export a trained LightGBM booster (or its ``dump_model()`` dict).

        The dump is the source of truth. LightGBM folds the
        ``boost_from_average`` init score into the first tree's leaves, so
        the exported ``base_score`` is 0.0 and the sum of leaf values is
        the raw score — which is exactly what the parity test checks.
        """
        dump = booster if isinstance(booster, dict) else booster.dump_model()
        objective = str(dump.get("objective", ""))
        if not objective.startswith("binary"):
            raise TreeExportError(
                f"objective {objective!r} is not binary; this evaluator "
                "only knows the logistic link"
            )
        if int(dump.get("num_class", 1)) != 1:
            raise TreeExportError("multiclass boosters are not exportable")
        if bool(dump.get("average_output", False)):
            raise TreeExportError(
                "average_output boosters (random forest mode) are not "
                "exportable; the raw score would not be a plain leaf sum"
            )
        trees = tuple(
            Tree.from_dump(info["tree_structure"]) for info in dump["tree_info"]
        )
        names = tuple(
            str(v) for v in (feature_names or dump.get("feature_names") or ())
        )
        width = len(names) if names else int(dump.get("max_feature_idx", -1)) + 1
        return cls(
            trees=trees,
            base_score=0.0,
            n_features=width,
            feature_names=names,
            params=dict(params or {}) or None,
        )


def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


# ---------------------------------------------------------------------------
# The fit (offline only)
# ---------------------------------------------------------------------------

#: Regularisation for ~450 000 rows at a ~10 % base rate. Shallow and slow
#: on purpose: the signal this is asked to add on top of the raw ensemble
#: fraction is a handful of interactions, not a memorised archive.
DEFAULT_TREE_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "num_leaves": 31,
    "lambda_l2": 1.0,
}


def lightgbm_available() -> bool:
    """Is LightGBM importable in this interpreter?

    False in the sidecar image and in the project's own ``.venv`` — both
    deliberately. Only the offline fit venv (``.venv-fit``) has it, and
    the tests that need it skip on this.
    """
    try:
        import lightgbm  # noqa: F401
    except Exception:  # noqa: BLE001 — a broken install is "not available"
        return False
    return True


def fit_trees(
    x: np.ndarray,
    y: np.ndarray,
    *,
    params: Mapping[str, Any] | None = None,
    feature_names: Sequence[str] = (),
    seed: int = 0,
    threads: int = 0,
    monotone_constraints: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Fit a binary LightGBM ensemble and return it already exported.

    ``monotone_constraints`` is one of ``+1`` / ``0`` / ``-1`` per column,
    LightGBM's own encoding. The caller that uses it is the shared-lead
    fit, which constrains the response to be non-decreasing in the lead:
    "rain within 60 min" contains "rain within 45 min", so an ensemble
    that learned otherwise would have learned noise. Constraining during
    training is strictly better than repairing the output afterwards — the
    repair hides the mistake, the constraint stops it being made — and the
    serving path does both anyway.

    ``x`` is the RAW design matrix — no standardisation and no imputation.
    Trees are invariant to a monotone rescaling of a column, and LightGBM
    learns a default direction per split for the missing values, which is
    strictly more than the training-mean imputation the logistic has to
    make do with.

    Returns ``{"ensemble", "n", "base_rate", "params", "n_trees"}``, the
    same shape :func:`~dmi_nowcast_core.postprocess.fit_logistic` returns
    so the caller's bookkeeping is the same either way.

    Raises :class:`ImportError` when LightGBM is not installed — this runs
    in ``.venv-fit`` and nowhere else.
    """
    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "LightGBM is required to FIT a tree post-processor and is "
            "deliberately absent from the sidecar image and the project "
            "venv. Create the fit venv:\n"
            "  uv venv --python .venv/bin/python .venv-fit\n"
            "  uv pip install --python .venv-fit/bin/python lightgbm "
            "scikit-learn\n"
            "and run the fit with .venv-fit/bin/python."
        ) from exc

    design = np.asarray(x, dtype=np.float64)
    outcome = np.asarray(y, dtype=np.float64).reshape(-1)
    if design.ndim != 2:
        raise ValueError(f"x must be 2-D, got {design.ndim}-D")
    if outcome.size != design.shape[0]:
        raise ValueError(f"x has {design.shape[0]} rows and y has {outcome.size}")
    if design.shape[0] == 0:
        raise ValueError("no rows to fit")
    if not np.all((outcome == 0.0) | (outcome == 1.0)):
        raise ValueError("y must be 0/1")

    settings = dict(DEFAULT_TREE_PARAMS)
    settings.update(dict(params or {}))
    rounds = int(settings.pop("n_estimators", DEFAULT_TREE_PARAMS["n_estimators"]))
    base_rate = float(outcome.mean())
    names = [str(v) for v in feature_names] or [
        f"f{i}" for i in range(design.shape[1])
    ]

    if base_rate <= 0.0 or base_rate >= 1.0:
        # A single-class fold has no tree to grow. An empty ensemble plus a
        # constant base score is the honest answer, and it matches what
        # ``fit_logistic`` does in the same situation.
        clipped = min(
            max(base_rate, 1.0 / (outcome.size + 2.0)),
            1.0 - 1.0 / (outcome.size + 2.0),
        )
        return {
            "ensemble": TreeEnsemble(
                trees=(),
                base_score=float(np.log(clipped / (1.0 - clipped))),
                n_features=design.shape[1],
                feature_names=tuple(names),
                params=dict(settings),
            ),
            "n": int(outcome.size),
            "base_rate": base_rate,
            "n_trees": 0,
            "params": dict(settings),
            "message": "single-class training fold; base score only",
        }

    constraints = list(monotone_constraints or ())
    if constraints and len(constraints) != design.shape[1]:
        raise ValueError(
            f"monotone_constraints has {len(constraints)} entries for "
            f"{design.shape[1]} design columns"
        )
    train_params = {
        "objective": "binary",
        "verbose": -1,
        "seed": int(seed),
        "num_threads": int(threads),
        **({"monotone_constraints": constraints} if any(constraints) else {}),
        # Bit-for-bit reproducibility across runs of the same rows. The
        # report has to be reproducible or "ΔBSS +0.02" is not a claim
        # anyone can check.
        "deterministic": True,
        "force_row_wise": True,
        **settings,
    }
    dataset = lgb.Dataset(
        design, label=outcome, feature_name=names, free_raw_data=True,
    )
    booster = lgb.train(train_params, dataset, num_boost_round=rounds)
    ensemble = TreeEnsemble.from_lightgbm(
        booster, feature_names=names,
        params={"num_boost_round": rounds, **settings},
    )
    return {
        "ensemble": ensemble,
        "booster": booster,
        "n": int(outcome.size),
        "base_rate": base_rate,
        "n_trees": ensemble.n_trees,
        "params": {"num_boost_round": rounds, **settings},
        "message": "",
    }
