"""Registry of motion-field variants for the Phase H Layer A harness.

Why a registry rather than a flag in the harness: Phase H queues several
independent candidates that all change exactly one thing — the completed
flow field the deterministic advection and STEPS both consume (H-c's
Farnebäck parameter grid, H-F's confidence-gated completion, H-b's
multi-frame median, and ``dense_lucaskanade`` as the literature's
reference method). If each of those edited the harness, two candidates
could never be scored on identical cases, which is the one thing the
plan's protocol insists on. Instead every candidate adds an entry here
and the harness keeps a single ``--variant`` switch.

**The interface.** A variant is a callable::

    make_flow(prev_dbz, curr_dbz, rain_now_mm_h, *, pixel_km) -> (vy, vx)

* ``prev_dbz`` / ``curr_dbz`` — reflectivity (dBZ) on the native grid, in
  time order, exactly the arrays ``compute.py`` hands to ``dense_flow``.
  NaN (nodata) and −inf (undetect) are the caller's, not cleaned first.
* ``rain_now_mm_h`` — rain rate for ``curr_dbz``, the echo support that
  motion completion needs.
* ``pixel_km`` — grid spacing, 0.5 on the DMI 500 m composite.
* returns ``(vy, vx)`` float32 in **pixels per frame**, already
  completed, sanitised and clipped: the field a consumer can advect with
  directly, with no NaN and no silent unit change. Positive ``vy`` is
  southward, positive ``vx`` eastward, as in ``dense_flow``.

A variant must be a module-level function (or otherwise picklable): the
harness runs one process per day under ``ProcessPoolExecutor``.

**Extra frames (added for H4).** Two Phase H candidates cannot be
expressed on a two-frame signature: ``median3`` combines three pair
estimates and ``oracle`` reads the frame after ``curr``. Rather than
widen the signature for all thirty-odd entries — which would make every
caller fetch frames that no entry it uses will look at — a variant
*declares* what it needs, as attributes on the callable::

    make_flow.needs_history = 2      # extra OLDER frames, beyond ``prev``
    make_flow.needs_future = True    # the frame AFTER ``curr``

and the harness then passes, **as keyword arguments and only to the
entries that declared them**::

    history_dbz=[oldest, ..., the frame before prev]   # len == needs_history
    future_dbz=<the frame after curr>

Both default to "not needed" (:func:`variant_requirements` reads them
with ``getattr``), so every entry written before this existed keeps
working untouched, and a caller that does not implement the extension
can still run any variant that declares nothing. A variant that declares
a need must raise a clear ``ValueError`` when the frames are absent
rather than silently degrade — a Layer A run that quietly scored
``oracle`` as ``confidence`` would be worse than one that crashed.

New entries go through :func:`register_variant` at import time of the
module that defines them, or straight into ``_VARIANTS`` here when they
belong to the core library. The Phase H candidates live in
:mod:`dmi_nowcast_core.flow_variants`, imported at the bottom of this
file so ``list_variants()`` is complete however the registry was reached.

**Beyond Layer A (H4, 2026-09-13).** The registry is no longer the
harness's alone: ``dense_flow.estimate_motion(flow_variant=NAME)``
resolves it too, which is how a Layer A winner reaches the gauge replay
(``replay_warnings.py --flow-variant``) and the live cycle
(``forecast.flow_variant``) on the *identical* field — the only thing
that makes a Layer A screening result transferable. Two consequences for
anyone adding an entry:

* the field an entry returns is what gets served. ``estimate_motion``
  wraps it without re-completing anything, so an entry that skipped the
  shared completion would ship an incomparable field, not just score as
  one;
* :func:`forecast_variants` / :func:`check_forecast_variant` gate that
  path on the declared frame needs. ``oracle`` and ``median3`` are
  refused there — the harness still scores both, because it alone can
  feed them.
"""
from __future__ import annotations

from typing import NamedTuple, Protocol

import numpy as np

from .dense_flow import complete_flow, dense_flow, estimate_motion

__all__ = [
    "COMPLETION_ATTR",
    "FlowVariant",
    "MAX_PX_PER_FRAME",
    "NEEDS_FUTURE_ATTR",
    "NEEDS_HISTORY_ATTR",
    "PRODUCTION_VARIANT",
    "SUPPORT_THRESHOLD_MM_H",
    "VariantRequirements",
    "check_forecast_variant",
    "forecast_variants",
    "get_variant",
    "list_variants",
    "bulk_flow",
    "confidence_flow",
    "persistence_flow",
    "production_flow",
    "register_variant",
    "variant_completion",
    "variant_requirements",
]

#: Displacement clip, copied from ``compute.py::_MAX_PX_PER_FRAME``. 30 px
#: per 10-min frame on the 500 m grid is 90 km/h — beyond any Danish storm
#: motion, so anything above it is an optical-flow artefact.
MAX_PX_PER_FRAME = 30.0

#: Echo support for ``complete_flow``, copied from the live
#: ``ForecastConfig.rain_threshold_mm_h``.
SUPPORT_THRESHOLD_MM_H = 0.5


#: Attribute a variant sets to ask for extra OLDER frames (an int, 0 = none).
NEEDS_HISTORY_ATTR = "needs_history"

#: Attribute a variant sets to ask for the frame after ``curr`` (a bool).
NEEDS_FUTURE_ATTR = "needs_future"

#: Attribute a variant sets to declare WHICH ``complete_flow`` policy its
#: returned field already went through (``"bulk"`` or ``"confidence"``).
#: Only :func:`~dmi_nowcast_core.dense_flow.estimate_motion`'s
#: ``flow_variant=`` dispatch reads it, to fill
#: ``MotionEstimate.completion`` with the truth rather than with whatever
#: the caller asked for. Default ``"confidence"``, which is what
#: ``flow_variants._completed`` — the one helper every H4 candidate routes
#: through — applies.
COMPLETION_ATTR = "completion_policy"


class FlowVariant(Protocol):
    """Callable signature every registered variant satisfies.

    The two extra-frame keywords are part of the protocol but optional on
    both sides: a variant that does not declare a need never receives
    them, so it need not accept them, and every entry written before the
    declaration existed still type-checks against this.
    """

    def __call__(
        self,
        prev_dbz: np.ndarray,
        curr_dbz: np.ndarray,
        rain_now_mm_h: np.ndarray,
        *,
        pixel_km: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        ...


class VariantRequirements(NamedTuple):
    """What extra frames one variant asked for.

    ``history`` counts frames OLDER than ``prev``; ``future`` is the one
    frame after ``curr``. A caller reads this once per run and fetches
    accordingly — the point of a declaration rather than a try/except is
    that the harness can decide a case is unscoreable *before* parsing
    anything, and count the skip.
    """

    history: int
    future: bool


def variant_requirements(make_flow: FlowVariant) -> VariantRequirements:
    """Read a variant's extra-frame declaration, with the defaults applied.

    Deliberately forgiving on the way in and strict on the way out: any
    entry lacking the attributes (which is most of them) reads as
    ``(0, False)``, but a declared ``needs_history`` that is negative or
    not an integer is a bug in the entry, not a shrug, so it raises here
    where the name is still in hand rather than as an IndexError inside
    a worker three hours into a run.
    """
    raw_history = getattr(make_flow, NEEDS_HISTORY_ATTR, 0)
    try:
        history = int(raw_history)
    except (TypeError, ValueError):
        raise TypeError(
            f"{make_flow!r}.{NEEDS_HISTORY_ATTR} must be an int, got "
            f"{raw_history!r}"
        ) from None
    if history < 0:
        raise ValueError(
            f"{make_flow!r}.{NEEDS_HISTORY_ATTR} must be >= 0, got {history}"
        )
    return VariantRequirements(
        history=history,
        future=bool(getattr(make_flow, NEEDS_FUTURE_ATTR, False)),
    )


def variant_completion(make_flow: FlowVariant) -> str:
    """Which ``complete_flow`` policy the entry's returned field went through.

    Read off :data:`COMPLETION_ATTR`, defaulting to ``"confidence"``: every
    H4 candidate goes through ``flow_variants._completed``, which is the
    served policy. Only :func:`bulk_flow` (the pre-H-F field) and
    :func:`persistence_flow` (which completes nothing at all — a zero field
    has nothing to relax) declare ``"bulk"``.

    It exists so ``MotionEstimate.completion`` can report what actually
    ran when the estimate came from the registry rather than from
    ``estimate_motion``'s own sequence. Labelling a variant's field with
    the caller's requested policy would be the one lie the whole
    single-entry-point design exists to prevent.
    """
    value = str(getattr(make_flow, COMPLETION_ATTR, "confidence"))
    if value not in ("bulk", "confidence"):
        raise ValueError(
            f"{make_flow!r}.{COMPLETION_ATTR} must be 'bulk' or "
            f"'confidence', got {value!r}"
        )
    return value


def bulk_flow(
    prev_dbz: np.ndarray,
    curr_dbz: np.ndarray,
    rain_now_mm_h: np.ndarray,
    *,
    pixel_km: float,
) -> tuple[np.ndarray, np.ndarray]:
    """The pre-H-F production field (``flow_completion: bulk``) — the Layer A baseline.

    Mirrors ``sidecar/dmi_nowcast_sidecar/compute.py::_compute_sync`` step
    for step, and the order is load-bearing:

    1. ``dense_flow`` (OpenCV Farnebäck, stock ``winsize=31, levels=3,
       poly_n=7``) on **dBZ**, not rain rate.
    2. ``complete_flow`` on the *raw* estimate — before the ``nan_to_num``,
       because completion gives a non-finite pixel the bulk vector
       (weight 0) rather than the 0 px/frame that ``nan_to_num`` would
       freeze in.
    3. only then ``nan_to_num`` and the ±``MAX_PX_PER_FRAME`` clip.

    Production applies completion ahead of the sanitise for the same
    reason, so that both consumers — the deterministic overlay advection
    and the STEPS velocity — see the identical completed field.
    """
    vy, vx = dense_flow(prev_dbz, curr_dbz)
    vy, vx = complete_flow(
        vy, vx, rain_now_mm_h,
        pixel_km=pixel_km,
        support_threshold_mm_h=SUPPORT_THRESHOLD_MM_H,
    )
    vy = np.nan_to_num(vy, nan=0.0).astype(np.float32)
    vx = np.nan_to_num(vx, nan=0.0).astype(np.float32)
    np.clip(vy, -MAX_PX_PER_FRAME, MAX_PX_PER_FRAME, out=vy)
    np.clip(vx, -MAX_PX_PER_FRAME, MAX_PX_PER_FRAME, out=vx)
    return vy, vx


def confidence_flow(
    prev_dbz: np.ndarray,
    curr_dbz: np.ndarray,
    rain_now_mm_h: np.ndarray,
    *,
    pixel_km: float,
) -> tuple[np.ndarray, np.ndarray]:
    """The H-F hotfix field: robust bulk + confidence-gated completion.

    What the sidecar serves since ``forecast.flow_completion`` defaulted to
    ``confidence`` (2026-09-08): the same Farnebäck estimate as
    :func:`bulk_flow`, but ``complete_flow`` relaxes low-texture ON-echo
    pixels toward a bulk vector taken as the median over high-texture wet
    pixels — see ``dense_flow.estimate_motion`` and
    ``archive/flow_stall_20260908`` in the private repo for why. Routed
    through ``estimate_motion`` so the harness and the runtime cannot
    drift; ``dt_min`` only feeds the diagnostic, not the field.
    """
    motion = estimate_motion(
        prev_dbz, curr_dbz, rain_now_mm_h,
        pixel_km=pixel_km, dt_min=10.0,
        support_threshold_mm_h=SUPPORT_THRESHOLD_MM_H,
        completion="confidence", max_px_per_frame=MAX_PX_PER_FRAME,
    )
    return motion.vy, motion.vx


def persistence_flow(
    prev_dbz: np.ndarray,
    curr_dbz: np.ndarray,
    rain_now_mm_h: np.ndarray,
    *,
    pixel_km: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Zero motion: advecting with it reproduces the current observation.

    A variant rather than a special case in the harness, so the Eulerian
    persistence baseline goes through the same advection, the same ×4
    reduction and the same masking as every candidate. If it did not, a
    difference between persistence and a candidate could be an artefact
    of two code paths rather than of the physics.
    """
    zeros = np.zeros(np.asarray(curr_dbz).shape, dtype=np.float32)
    return zeros, zeros.copy()


#: ``bulk_flow`` IS the legacy completion, and ``persistence_flow``
#: completes nothing at all (a zero field has nothing to relax toward a
#: bulk of zero), so neither went through the confidence gate. Declared so
#: ``MotionEstimate.completion`` reports the policy that ran rather than
#: the one the caller asked for. See :func:`variant_completion`.
bulk_flow.completion_policy = "bulk"  # type: ignore[attr-defined]
persistence_flow.completion_policy = "bulk"  # type: ignore[attr-defined]


#: The registry name the sidecar's default ``forecast.flow_completion``
#: corresponds to. ``production`` is an alias of this entry; a sidecar test
#: pins the two together so the harness's "production" can never quietly
#: mean a field the service no longer serves.
PRODUCTION_VARIANT = "confidence"

#: Kept as a name: the served field, whatever :data:`PRODUCTION_VARIANT` says.
production_flow = confidence_flow

_VARIANTS: dict[str, FlowVariant] = {
    "bulk": bulk_flow,
    "confidence": confidence_flow,
    "production": production_flow,
    "persistence": persistence_flow,
}


def register_variant(name: str, make_flow: FlowVariant) -> None:
    """Add a variant. Refuses to shadow an existing name.

    Silent replacement is the one failure mode that would be invisible in
    a report: a run labelled ``production`` scoring something else.
    """
    if name in _VARIANTS:
        raise ValueError(f"variant {name!r} is already registered")
    _VARIANTS[name] = make_flow


def get_variant(name: str) -> FlowVariant:
    """Look up a variant by name, with the available names in the error."""
    try:
        return _VARIANTS[name]
    except KeyError:
        raise KeyError(
            f"unknown flow variant {name!r}; known: {', '.join(list_variants())}"
        ) from None


def list_variants() -> tuple[str, ...]:
    """Registered variant names, sorted, for CLI help and reports."""
    return tuple(sorted(_VARIANTS))


def forecast_variants() -> tuple[str, ...]:
    """Names a FORECAST can be made with, sorted.

    Every entry that needs nothing but the frame pair a live cycle and the
    warning replay actually hold: no frames older than ``prev`` and, above
    all, no frame after ``curr``. So ``median3`` (four frames) and
    ``oracle`` (the future) are absent, and everything else is present.

    The Layer A harness deliberately does NOT use this — it fetches the
    extra frames and scores those two entries on purpose. This is the list
    for callers that cannot.
    """
    return tuple(
        name for name in list_variants()
        if variant_requirements(_VARIANTS[name]) == VariantRequirements(0, False)
    )


def check_forecast_variant(name: str) -> str:
    """Validate a ``--flow-variant`` / ``forecast.flow_variant`` name.

    Returns ``name`` unchanged, or raises :class:`ValueError` naming the
    usable entries. Three distinct refusals, each with its own message,
    because the three have different fixes:

    * an unknown name is a typo;
    * ``oracle`` reads the frame AFTER the one being forecast. It is the
      Layer A ceiling, not a forecast, and a run that quietly served or
      replayed it would report skill nothing could deliver — the single
      most misleading outcome in this whole registry;
    * ``median3`` and any future multi-frame entry need frames neither the
      cycle nor the replay carries, so they would have to degrade
      silently, and a silent degradation is how a candidate gets shipped
      on a baseline's score.

    ``ValueError`` rather than ``KeyError`` throughout, so an argparse
    ``p.error`` and a pydantic validator can both surface the message
    as-is.
    """
    try:
        make_flow = get_variant(name)
    except KeyError as exc:
        raise ValueError(str(exc.args[0])) from None
    req = variant_requirements(make_flow)
    usable = ", ".join(forecast_variants())
    if req.future:
        raise ValueError(
            f"flow variant {name!r} reads the frame AFTER the one being "
            "forecast — it is a Layer A ceiling, not a forecast, and cannot "
            f"be served or replayed. Usable variants: {usable}"
        )
    if req.history:
        raise ValueError(
            f"flow variant {name!r} needs {req.history} frame(s) older than "
            "the pair a cycle holds; neither the sidecar nor the warning "
            "replay carries them, and degrading silently would score a "
            f"candidate as its baseline. Usable variants: {usable}"
        )
    return name


# The Phase H candidates register themselves on import. This sits at the
# bottom because ``flow_variants`` imports ``register_variant`` and the
# two shared constants from here, so the names have to exist first; the
# circular pair resolves either way round because neither module touches
# the other at import time beyond that.
from . import flow_variants as _flow_variants  # noqa: E402,F401
