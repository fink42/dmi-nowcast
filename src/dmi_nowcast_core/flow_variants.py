"""The Phase H4 motion candidates: the H-c grid, LK, multi-frame, oracle.

``variants.py`` owns the registry and the four entries that describe the
*service* (``bulk``, ``confidence``/``production``, ``persistence``).
This module owns the entries that exist only to be screened against them
on Layer A, and registers itself on import — ``variants`` imports it at
the bottom of the file, so ``list_variants()`` sees everything however a
caller reached the registry.

What is here, and why each one (``forecast_skill_plan.md`` §3):

* **``farneback_w{W}_l{L}_p{P}``** — H-c's parameter sweep. Production
  has never scored its own Farnebäck settings; ``winsize=31, levels=3,
  poly_n=7`` is inherited, not measured. 24 cells: ``W ∈ {21, 41, 51,
  61}`` × ``L ∈ {3, 4, 5}`` × ``P ∈ {5, 7}``.
* **``lucaskanade``** — pysteps' ``dense_lucaskanade``, the literature's
  reference method and, per H-F, immune *by construction* to the stall
  that motivated this phase: it tracks sparse corners and interpolates,
  so a featureless echo interior gets a neighbour's vector rather than a
  spurious zero.
* **``median3``** — H-b's multi-frame estimate: the per-pixel median of
  three consecutive same-type pair estimates. A stall on one pair is
  outvoted by two; a stall on all three is real.
* **``oracle``** — not a candidate. The flow measured from the pair that
  *straddles the target time*, i.e. the motion that actually happened.
  It is the ceiling any amount of motion tuning could reach at +10, and
  an upper bound beyond it, so a candidate's gain is readable as a share
  of what is available rather than as a bare CSI delta.

**Every entry routes its raw estimate through**
``estimate_motion(..., flow=(vy, vx), completion="confidence")``. That is
the whole point of the module: completion, sanitising and the ±30 px
clip are then bit-for-bit identical across all of them, and the only
thing that differs between two Layer A runs is the estimator. A
candidate that did its own completion would be measuring two changes at
once, which the protocol's "identical cases" rule exists to prevent.

Deviations from the plan text, both deliberate, both agreed in the H4
brief:

* the plan's H-c grid reads ``winsize ∈ {21, 31, 41, 51}``, but 31 is
  ``confidence`` itself (registering it again would score the baseline
  twice under two names, and ``register_variant`` refuses the shadow
  anyway), and H-F asks for ``winsize=61`` to be "folded into the H-c
  grid". So the swept set is ``{21, 41, 51, 61}``, and 31 stays the
  named baseline the others are compared against.
* H-b says "six frames retained"; ``median3`` uses **four** — three
  pairs need four frames, and the sixth frame in the plan's sentence is
  the live cycle's retention buffer, not an input to the median. There
  is no acceleration term, as H-b requires.

The extra-frame interface
-------------------------
``median3`` and ``oracle`` need frames the two-frame ``FlowVariant``
signature does not carry. Rather than widen the signature for every
entry, a variant *declares* what it wants as attributes on the callable
(:data:`~dmi_nowcast_core.variants.NEEDS_HISTORY_ATTR` /
:data:`~dmi_nowcast_core.variants.NEEDS_FUTURE_ATTR`) and the harness
passes ``history_dbz`` / ``future_dbz`` only to the entries that declare
them. Existing entries are untouched and keep working unchanged. See
``variants.variant_requirements``.
"""
from __future__ import annotations

import numpy as np

from .dense_flow import DEFAULT_FILL_DBZ, _to_uint8, dense_flow, estimate_motion
from .variants import (
    MAX_PX_PER_FRAME,
    SUPPORT_THRESHOLD_MM_H,
    register_variant,
)

__all__ = [
    "FARNEBACK_LEVELS",
    "FARNEBACK_POLY_N",
    "FARNEBACK_WINSIZES",
    "FarnebackVariant",
    "POLY_SIGMA_FOR_POLY_N",
    "farneback_grid",
    "farneback_name",
    "lucaskanade_flow",
    "median3_flow",
    "oracle_flow",
]

#: The frame spacing every entry assumes, in minutes. Only the *diagnostic*
#: half of :func:`~dmi_nowcast_core.dense_flow.estimate_motion` uses it —
#: the field itself stays in px per frame — but it is spelled out here so
#: the number matches ``variants.confidence_flow`` exactly rather than by
#: coincidence.
DT_MIN = 10.0


def _completed(
    prev_dbz: np.ndarray,
    curr_dbz: np.ndarray,
    rain_now_mm_h: np.ndarray,
    flow: tuple[np.ndarray, np.ndarray],
    *,
    pixel_km: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One raw estimate → the served completion, sanitise and clip.

    The single place any candidate in this module turns an estimate into
    a field. ``completion="confidence"`` is the live policy; passing the
    estimate in through ``flow=`` rather than letting ``estimate_motion``
    call ``dense_flow`` itself is what lets a candidate change the
    estimator and *nothing else*.
    """
    motion = estimate_motion(
        prev_dbz, curr_dbz, rain_now_mm_h,
        pixel_km=pixel_km, dt_min=DT_MIN,
        support_threshold_mm_h=SUPPORT_THRESHOLD_MM_H,
        completion="confidence", max_px_per_frame=MAX_PX_PER_FRAME,
        flow=flow,
    )
    return motion.vy, motion.vx


# --------------------------------------------------------------------------
# H-c: the Farnebäck parameter grid
# --------------------------------------------------------------------------
#: Swept ``winsize``. 31 is missing on purpose: it is ``confidence``.
FARNEBACK_WINSIZES = (21, 41, 51, 61)

#: Swept pyramid depth. Production runs 3.
FARNEBACK_LEVELS = (3, 4, 5)

#: Swept polynomial neighbourhood. Production runs 7.
FARNEBACK_POLY_N = (5, 7)

#: OpenCV's recommended ``poly_sigma`` for each ``poly_n``, from the
#: ``calcOpticalFlowFarneback`` documentation. Not swept independently:
#: the two are a matched pair (sigma is the width of the Gaussian that
#: weights the polynomial fit over a ``poly_n``-wide neighbourhood), and
#: varying them separately would double the grid to measure a knob whose
#: sensible value is a function of the other one.
POLY_SIGMA_FOR_POLY_N = {5: 1.1, 7: 1.5}


def farneback_name(winsize: int, levels: int, poly_n: int) -> str:
    """Registry name for one grid cell: ``farneback_w41_l4_p7``."""
    return f"farneback_w{int(winsize)}_l{int(levels)}_p{int(poly_n)}"


class FarnebackVariant:
    """One cell of the H-c grid: Farnebäck at fixed parameters.

    A class rather than a closure because the registry's contract asks
    for picklable entries (the harness runs one process per day under
    ``ProcessPoolExecutor``) and because the parameters are then
    readable off the object in a report — ``repr`` prints them.

    ``needs_history`` / ``needs_future`` are declared explicitly, at
    their default values, so the grid documents that it is a plain
    two-frame estimator rather than relying on the harness's ``getattr``
    fallback to say so.
    """

    needs_history = 0
    needs_future = False

    def __init__(self, winsize: int, levels: int, poly_n: int) -> None:
        if int(poly_n) not in POLY_SIGMA_FOR_POLY_N:
            raise ValueError(
                f"poly_n {poly_n} has no recommended poly_sigma; known: "
                f"{sorted(POLY_SIGMA_FOR_POLY_N)}"
            )
        self.winsize = int(winsize)
        self.levels = int(levels)
        self.poly_n = int(poly_n)
        self.poly_sigma = POLY_SIGMA_FOR_POLY_N[self.poly_n]
        self.name = farneback_name(self.winsize, self.levels, self.poly_n)

    def __call__(
        self,
        prev_dbz: np.ndarray,
        curr_dbz: np.ndarray,
        rain_now_mm_h: np.ndarray,
        *,
        pixel_km: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        flow = dense_flow(
            prev_dbz, curr_dbz,
            winsize=self.winsize, levels=self.levels,
            poly_n=self.poly_n, poly_sigma=self.poly_sigma,
        )
        return _completed(
            prev_dbz, curr_dbz, rain_now_mm_h, flow, pixel_km=pixel_km,
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"FarnebackVariant(winsize={self.winsize}, levels={self.levels}, "
            f"poly_n={self.poly_n}, poly_sigma={self.poly_sigma})"
        )


def farneback_grid() -> dict[str, FarnebackVariant]:
    """The 24 H-c cells, keyed by registry name, in a stable order."""
    return {
        (cell := FarnebackVariant(w, ell, p)).name: cell
        for w in FARNEBACK_WINSIZES
        for ell in FARNEBACK_LEVELS
        for p in FARNEBACK_POLY_N
    }


# --------------------------------------------------------------------------
# the literature's reference estimator
# --------------------------------------------------------------------------
#: Destination-grid chunking for ``idwinterp2d``. Memory only — the
#: result is bit-identical to pysteps' default of 4. See
#: :func:`lucaskanade_flow` for the measurement that chose 16.
LK_NCHUNKS = 16


def lucaskanade_flow(
    prev_dbz: np.ndarray,
    curr_dbz: np.ndarray,
    rain_now_mm_h: np.ndarray,
    *,
    pixel_km: float,
) -> tuple[np.ndarray, np.ndarray]:
    """pysteps' ``dense_lucaskanade``, then the production completion.

    Sparse Shi–Tomasi corners on the previous frame, pyramidal
    Lucas–Kanade to track them into the current one, outlier rejection
    and declustering on the resulting sparse vectors, then inverse
    distance weighting onto the full grid. All at pysteps' own defaults
    (``fd_method="shitomasi"``, ``interp_method="idwinterp2d"``,
    ``nr_std_outlier=3``, ``k_outlier=30``, ``size_opening=3``,
    ``decl_scale=20``) — the point of having it is that it is the
    published method, so tuning it here would defeat the comparison.

    **One non-default: ``nchunks=16``.** It is a memory knob, not an
    algorithm parameter — upstream documents it as "split and process
    the destination grid in nchunks, useful for large grids to limit the
    memory footprint", and the chunks partition the OUTPUT grid, so the
    result is bit-identical (measured, and pinned by
    ``tests/test_flow_variants.py``). It has to be raised because the
    default of 4 is sized for a 500×500 European composite, not for the
    DMI 1728×1984: ``idwinterp2d`` materialises a
    ``(pixels_in_chunk, k=20)`` distance and index pair plus the gathered
    values, which at the default is a **1.0 GB** transient. Measured on
    the native grid: nchunks=4 → 3.05 s / 1008 MB, nchunks=16 → 2.99 s /
    340 MB, nchunks=64 → 3.00 s / 172 MB. Two batch workers under the
    VM's 5 GB cgroup cap cannot afford 2 GB of transient for a number
    that does not change, and 16 buys the reduction for nothing.

    **The input is the same image Farnebäck sees.** ``dense_flow``
    normalises dBZ to uint8 over :data:`~dmi_nowcast_core.dense_flow.
    DBZ_RANGE` after filling NaN/−inf with the grid floor, and that
    quantisation is part of what either estimator has to work with. LK
    gets the identical array (as float32, because pysteps rescales and
    casts to uint8 itself and would otherwise quantise twice), so a
    difference between the two entries is the estimator and not the
    preprocessing.

    **Convention.** ``dense_lucaskanade`` returns ``(2, m, n)`` with
    ``[0]`` the x- (column, eastward) component and ``[1]`` the y- (row,
    southward) component, in px per timestep — the same units and the
    same sense as ``dense_flow``'s ``(vy, vx)``, only transposed in
    order. The swap happens here and ``tests/test_flow_variants.py``
    pins it against a known displacement; get it wrong and the field
    advects the rain sideways, which no aggregate score would explain.

    **Zero-motion case.** With no trackable corner (a dry or perfectly
    flat frame) pysteps returns a zero field, which then goes through
    completion like any other estimate: the far field relaxes toward a
    bulk of zero, so the result is the persistence field. That is the
    honest answer — nothing was measurable — and it is what the stall
    diagnostic will report.
    """
    from ._vendor.pysteps_motion import dense_lucaskanade

    prev = _to_uint8(np.asarray(prev_dbz), DEFAULT_FILL_DBZ)
    curr = _to_uint8(np.asarray(curr_dbz), DEFAULT_FILL_DBZ)
    # float32, not uint8: ``track_features`` and the Shi-Tomasi detector
    # each rescale their input to 0-255 and cast, so handing them uint8
    # would round twice. The VALUES are already the uint8 levels.
    stack = np.stack([prev, curr]).astype(np.float32)
    del prev, curr
    field = dense_lucaskanade(stack, interp_kwargs={"nchunks": LK_NCHUNKS})
    del stack
    vx = np.ascontiguousarray(field[0], dtype=np.float32)
    vy = np.ascontiguousarray(field[1], dtype=np.float32)
    del field
    return _completed(prev_dbz, curr_dbz, rain_now_mm_h, (vy, vx), pixel_km=pixel_km)


# --------------------------------------------------------------------------
# H-b: the multi-frame median
# --------------------------------------------------------------------------
#: Extra OLDER frames ``median3`` wants beyond ``prev``.
MEDIAN3_HISTORY = 2


def _median_of_three(fields: list[np.ndarray]) -> np.ndarray:
    """Per-pixel median of exactly three same-shaped float32 grids.

    ``overwrite_input=True`` because ``fields`` is a private stack this
    function owns: on the native 1728×1984 composite each estimate is
    13.7 MB, and letting numpy partition in place saves a fourth copy
    per component on a VM that runs two of these side by side under a
    5 GB cgroup cap.
    """
    stack = np.stack(fields)
    out = np.median(stack, axis=0, overwrite_input=True)
    del stack
    return out.astype(np.float32)


def median3_flow(
    prev_dbz: np.ndarray,
    curr_dbz: np.ndarray,
    rain_now_mm_h: np.ndarray,
    *,
    pixel_km: float,
    history_dbz: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """H-b: per-pixel median of three consecutive pair estimates.

    Four frames, three overlapping pairs — ``(h0, h1)``, ``(h1, prev)``,
    ``(prev, curr)`` — each through stock Farnebäck (production's
    ``winsize=31, levels=3, poly_n=7``, so the ONLY difference from
    ``confidence`` is the combination), then a per-pixel median, then the
    shared completion.

    Why the median and not a mean: the failure it targets is a *stall*,
    a pixel whose estimate collapses to zero because its neighbourhood
    had nothing to match on that particular pair. That is a one-sided
    error, so a mean over three pairs still carries a third of it, while
    a median discards it outright as long as the other two pairs saw
    structure. The symmetric case — genuinely stationary rain — reads
    zero on all three and survives the median unchanged.

    **No acceleration term** (H-b, explicitly). Three estimates could be
    differenced into an acceleration and extrapolated, but a second
    derivative from three noisy 10-minute samples is noise, and it would
    make this candidate two changes rather than one.

    ``history_dbz`` is ``[oldest, …, frame before prev]`` — the order the
    harness supplies, chronological, so ``history_dbz[-1]`` is always the
    frame immediately before ``prev`` whatever ``needs_history`` says.
    Only the last :data:`MEDIAN3_HISTORY` entries are used; a longer list
    is accepted (a future entry may want more) and the extra oldest
    frames are ignored.
    """
    if history_dbz is None or len(history_dbz) < MEDIAN3_HISTORY:
        have = 0 if history_dbz is None else len(history_dbz)
        raise ValueError(
            f"the 'median3' variant needs {MEDIAN3_HISTORY} frames older than "
            f"prev_dbz, passed as history_dbz=[oldest, ..., before_prev]; got "
            f"{have}. The harness supplies them when the variant declares "
            f"needs_history; a direct caller has to pass them itself."
        )
    history = [np.asarray(f) for f in history_dbz[-MEDIAN3_HISTORY:]]
    shape = np.asarray(curr_dbz).shape
    for i, frame in enumerate(history):
        if frame.shape != shape:
            raise ValueError(
                f"history_dbz[{i}] has shape {frame.shape}, expected {shape}"
            )

    frames = [*history, np.asarray(prev_dbz), np.asarray(curr_dbz)]
    vys: list[np.ndarray] = []
    vxs: list[np.ndarray] = []
    for older, newer in zip(frames, frames[1:]):
        vy, vx = dense_flow(older, newer)
        vys.append(np.asarray(vy, dtype=np.float32))
        vxs.append(np.asarray(vx, dtype=np.float32))

    flow = (_median_of_three(vys), _median_of_three(vxs))
    del vys, vxs
    return _completed(prev_dbz, curr_dbz, rain_now_mm_h, flow, pixel_km=pixel_km)


median3_flow.needs_history = MEDIAN3_HISTORY  # type: ignore[attr-defined]
median3_flow.needs_future = False  # type: ignore[attr-defined]


# --------------------------------------------------------------------------
# the ceiling
# --------------------------------------------------------------------------
def oracle_flow(
    prev_dbz: np.ndarray,
    curr_dbz: np.ndarray,
    rain_now_mm_h: np.ndarray,
    *,
    pixel_km: float,
    future_dbz: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The motion that actually happened: Farnebäck on ``(curr, future)``.

    **Not a candidate to ship, and it cannot be.** It reads a frame that
    does not exist when the forecast is made. It is registered so that
    Layer A can answer the question every parameter sweep needs and none
    of its cells can: how much of the remaining error is motion at all?

    At +10 the estimate is exactly the displacement being forecast, so
    the score is the ceiling for any motion tuning — an H-c cell that
    lands close to it says the grid is exhausted and the loss is in the
    growth/decay the advection does not model, not in the flow. Beyond
    +10 it is an upper bound rather than a ceiling: knowing the first
    step's true motion does not make the *next* steps' motion known, so
    a candidate can in principle exceed it on a decelerating system,
    and in practice does not.

    Everything downstream of the estimate is the production path, same
    as every other entry here — so the gap to ``confidence`` is the
    estimator's error and nothing else.
    """
    if future_dbz is None:
        raise ValueError(
            "the 'oracle' variant needs the frame AFTER curr_dbz, passed as "
            "future_dbz=<next frame>; got None. It reads the future on "
            "purpose — it is a Layer A ceiling, not a forecast — and the "
            "harness supplies the frame when a variant declares needs_future."
        )
    future = np.asarray(future_dbz)
    shape = np.asarray(curr_dbz).shape
    if future.shape != shape:
        raise ValueError(
            f"future_dbz has shape {future.shape}, expected {shape}"
        )
    flow = dense_flow(curr_dbz, future)
    return _completed(prev_dbz, curr_dbz, rain_now_mm_h, flow, pixel_km=pixel_km)


oracle_flow.needs_history = 0  # type: ignore[attr-defined]
oracle_flow.needs_future = True  # type: ignore[attr-defined]


# --------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------
def _register_all() -> None:
    """Add every candidate to the registry. Idempotent by refusal.

    ``register_variant`` raises on a duplicate name, which is what makes
    this safe to call exactly once — at import — and loud if a second
    module ever tries to claim ``lucaskanade``.
    """
    for name, cell in farneback_grid().items():
        register_variant(name, cell)
    register_variant("lucaskanade", lucaskanade_flow)
    register_variant("median3", median3_flow)
    register_variant("oracle", oracle_flow)


_register_all()
