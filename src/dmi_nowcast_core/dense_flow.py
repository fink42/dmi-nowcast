"""Dense optical flow.

Plan §4.3 / §6.5: equivalent to the per-pixel motion estimate DMI uses for
their own radar forecast on dmi.dk, but with a backend hierarchy that lets
us pick the best available implementation at runtime.

Backends, tried in order:

1. **OpenCV Farnebäck** (preferred). Fastest (~50-150 ms on 1728×1984) and
   most robust on sparse fields. Used in pysteps and DMI's own pipeline.
   Not installable on HA OS today — see `manifest.json` for why.
2. **scikit-image iterative Lucas-Kanade** (`optical_flow_ilk`). Pure
   numpy/scipy, peer-reviewed, multi-resolution pyramid. Slower (~1-3 s)
   but quality is within a CSI point of Farnebäck on real radar data.
3. **Raise `DenseFlowUnavailable`**. Caller (``coordinator.py``) falls back
   to FFT phase correlation in ``motion.py`` for a single mean-motion
   vector — fine on uniform frontal flow, breaks on patchy convective.

We run optical flow on **dBZ**, not rain rate. dBZ is log-scaled and
roughly continuous; rain rate has a long heavy tail that the polynomial
expansion in Farnebäck and the LK gradient estimate both handle poorly.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

_LOGGER = logging.getLogger(__name__)


class DenseFlowUnavailable(RuntimeError):
    """Raised when no dense-flow backend (OpenCV or scikit-image) is installed.

    Caught by ``coordinator.py`` to fall back to mean motion.
    """

DEFAULT_FILL_DBZ = -32.0  # the DMI grid floor (raw=0 → -32 dBZ after offset)
DBZ_RANGE = (-32.0, 60.0)  # for uint8 normalization


def dense_flow(
    prev_dbz: np.ndarray,
    curr_dbz: np.ndarray,
    *,
    fill_value: float = DEFAULT_FILL_DBZ,
    pyr_scale: float = 0.5,
    levels: int = 3,
    winsize: int = 31,
    iterations: int = 5,
    poly_n: int = 7,
    poly_sigma: float = 1.5,
    ilk_radius: int = 7,
    ilk_num_warp: int = 8,
    ilk_downsample: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(vy, vx)`` displacement fields in pixels per frame.

    Convention: a positive ``vy`` means the field has moved downward (south)
    between ``prev_dbz`` and ``curr_dbz``; positive ``vx`` is rightward (east).
    NaN / -inf in the inputs are replaced with ``fill_value``.

    The OpenCV branch normalises to uint8 over ``DBZ_RANGE`` (Farnebäck is
    most reliable on uint8). The scikit-image branch keeps float32 and
    downsamples by ``ilk_downsample`` (default 2×) for speed — flow is
    rescaled back to original resolution before returning, so callers see
    the same per-pixel array shape regardless of backend.

    OpenCV defaults: ``winsize=31, poly_n=7`` favour stable estimates over
    fine-grained detail. For convective showers with rapid motion, larger
    windows trade temporal resolution for robustness.

    scikit-image defaults: ``ilk_radius=7`` (per-pixel window), ``num_warp=8``
    (good convergence for ≤20 px displacement), ``ilk_downsample=2`` (4×
    speed-up; sufficient resolution for our 500 m/pixel grid).
    """
    if prev_dbz.shape != curr_dbz.shape:
        raise ValueError(f"shape mismatch: prev {prev_dbz.shape} vs curr {curr_dbz.shape}")

    # Branch 1: OpenCV Farnebäck.
    try:
        import cv2
    except ImportError:
        cv2 = None  # type: ignore[assignment]
    if cv2 is not None:
        prev = _to_uint8(prev_dbz, fill_value)
        curr = _to_uint8(curr_dbz, fill_value)
        flow = cv2.calcOpticalFlowFarneback(
            prev, curr, None,
            pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, 0,
        )
        # cv2 flow: shape (H, W, 2); [..., 0] = vx, [..., 1] = vy.
        return flow[..., 1], flow[..., 0]

    # Branch 2: scikit-image TV-L1 dense flow.
    try:
        from skimage.registration import optical_flow_tvl1
    except ImportError:
        optical_flow_tvl1 = None  # type: ignore[assignment]
    if optical_flow_tvl1 is not None:
        return _dense_flow_skimage(
            prev_dbz, curr_dbz, fill_value=fill_value,
            downsample=ilk_downsample,
        )

    raise DenseFlowUnavailable(
        "No dense-flow backend installed; need opencv-python-headless or "
        "scikit-image. coordinator.py will fall back to mean motion."
    )


def _dense_flow_skimage(
    prev_dbz: np.ndarray,
    curr_dbz: np.ndarray,
    *,
    fill_value: float,
    downsample: int,
) -> tuple[np.ndarray, np.ndarray]:
    """scikit-image TV-L1 backend with optional spatial downsampling.

    Pure numpy/scipy under the hood; works on HA OS where opencv is not
    installable. Downsampling cuts runtime by ``downsample**2`` while
    keeping per-frame motion accurate to within a sub-pixel — the grid is
    500 m/pixel and the radar's effective resolution is closer to 1 km,
    so a 2× downsample loses nothing meaningful.

    Why TV-L1 and not Iterative LK: the iterative Lucas-Kanade method
    (``optical_flow_ilk``) is sensitive to window radius relative to
    feature size and gave directionally-wrong results on real DMI data
    (motion direction varied by 90° depending on downsample factor;
    disagreed with opencv Farnebäck reference). TV-L1's variational
    formulation handles large displacements and complex motion fields
    more reliably — verified to match opencv's direction within a few
    percent on the same frames.
    """
    from skimage.registration import optical_flow_tvl1

    prev = _to_float32_filled(prev_dbz, fill_value)
    curr = _to_float32_filled(curr_dbz, fill_value)
    h, w = prev.shape
    if downsample > 1:
        # Drop every Nth row/col rather than a Gaussian blur+resize — keeps
        # this cheap, and TV-L1 has its own smoothing internally.
        prev_ds = prev[::downsample, ::downsample]
        curr_ds = curr[::downsample, ::downsample]
    else:
        prev_ds, curr_ds = prev, curr

    # scikit-image's optical_flow_tvl1(reference, moving): returns shape
    # (2, H, W) where [0]=row (vy), [1]=col (vx). The flow describes how
    # ``reference`` would need to displace to land at ``moving``, which is
    # exactly the prev→curr convention we want. Defaults work well on
    # radar data; explicit dtype keeps numbers in float32.
    flow_ds = optical_flow_tvl1(
        prev_ds, curr_ds,
        dtype=np.float32,
    )

    if downsample > 1:
        # Upsample flow back to original grid resolution. NEAREST is fine for
        # advection — the flow field is smooth enough that bilinear would
        # only marginally change results, and NEAREST is much cheaper.
        vy = np.repeat(np.repeat(flow_ds[0], downsample, axis=0), downsample, axis=1)[:h, :w]
        vx = np.repeat(np.repeat(flow_ds[1], downsample, axis=0), downsample, axis=1)[:h, :w]
    else:
        vy, vx = flow_ds[0], flow_ds[1]
    return vy.astype(np.float32), vx.astype(np.float32)


def _to_uint8(arr: np.ndarray, fill_value: float) -> np.ndarray:
    a = np.nan_to_num(arr, nan=fill_value, posinf=fill_value, neginf=fill_value)
    lo, hi = DBZ_RANGE
    scaled = np.clip((a - lo) / (hi - lo), 0.0, 1.0) * 255.0
    return scaled.astype(np.uint8)


def _to_float32_filled(arr: np.ndarray, fill_value: float) -> np.ndarray:
    return np.nan_to_num(arr, nan=fill_value, posinf=fill_value, neginf=fill_value).astype(np.float32)


#: Box-mean window for :func:`flow_confidence`, in native pixels (15.5 km on
#: the 500 m grid). Deliberately the same order as Farnebäck's own
#: ``winsize=31``: the question the energy answers is "did the estimator's
#: window contain anything to match", so it has to be measured over that
#: window, not over a single pixel's gradient.
DEFAULT_CONFIDENCE_WINDOW_PX = 31

#: Percentile of the on-echo gradient energy at which the confidence weight
#: saturates: a pixel at or above it keeps its own estimate, one at zero
#: energy takes the bulk vector outright. 40 came out of the 2026-09-08
#: morning case (``archive/flow_stall_20260908/``), where the stalled
#: interior sat well below the 40th percentile of the shield's texture.
DEFAULT_CONFIDENCE_PERCENTILE = 40.0

#: Percentile of on-echo gradient energy above which a pixel's velocity is
#: trusted enough to vote in :func:`robust_bulk`. Higher than the gating
#: percentile on purpose: the bulk vector is a single number for the whole
#: domain, so it can afford to be picky, while the gate has to hand every
#: pixel *some* weight.
DEFAULT_TEXTURE_PERCENTILE = 60.0

#: Minimum number of high-texture wet pixels :func:`robust_bulk` needs
#: before it prefers their median to the rain-weighted mean. 200 pixels is
#: 50 km² on the 500 m grid — a couple of convective cells, enough for a
#: median to mean anything and small enough that a mostly-smooth shield
#: still qualifies (the evening case had tens of thousands).
DEFAULT_ROBUST_BULK_MIN_PIXELS = 200

#: A wet pixel moving slower than this is counted as "stalled" by
#: :func:`estimate_motion`'s diagnostic. 5 km/h is the threshold the
#: 2026-09-08 evidence file uses, so the served share is comparable with
#: the numbers in it.
STALL_SPEED_KMH = 5.0

#: Per-pixel clip on the flow, in px per frame (~90 km/h at 500 m / 10 min).
#: Mirrors ``compute._MAX_PX_PER_FRAME``; a dense-flow backend that
#: extrapolates wildly over dry pixels must not turn STEPS into noise.
DEFAULT_MAX_PX_PER_FRAME = 30.0


def flow_confidence(
    dbz_now: np.ndarray,
    *,
    window_px: int = DEFAULT_CONFIDENCE_WINDOW_PX,
    fill_value: float = DEFAULT_FILL_DBZ,
) -> np.ndarray:
    """Local gradient energy of the image the flow estimator actually saw.

    Why this exists (``archive/flow_stall_20260908/README.md``): Farnebäck
    is a *local* method. Inside a broad, flat or speckled echo its
    polynomial expansion has no coherent structure to match and the
    least-squares displacement collapses toward zero — on the 2026-09-08
    17:40Z stratiform shield, 55 % of the wet pixels in a 160 km box around
    Odense carried < 5 km/h while the shield translated at 25-28 km/h. The
    estimate is not merely noisy there, it is systematically zero, and
    nothing downstream could tell the difference between "measured, and it
    really is stationary" and "nothing to measure". This is that missing
    signal: where the image is featureless the energy is ~0, where there
    are edges and cells it is large.

    **Computed on the uint8 image, not on the float dBZ.** Farnebäck runs
    on ``_to_uint8(dbz, fill_value)`` (see :func:`dense_flow`), so that is
    the image whose texture decides whether the estimate has anything to
    track: the same clip at ``DBZ_RANGE``, the same quantisation to 256
    levels, the same flat plateau wherever the composite reads nodata.
    Measuring the energy on the float field instead would credit gradients
    the estimator never saw (below the quantisation step, or outside the
    normalisation range).

    Method: Sobel x/y (``ksize=3``) on that image as float32, then
    ``gx² + gy²``, then a box mean over ``window_px`` — one gradient pair
    and one separable box filter over the grid, ~40 ms on the native
    1728×1984 composite.

    Parameters
    ----------
    dbz_now:
        The *second* frame of the pair the flow was estimated from (the
        anchor the forecast is advected from).
    window_px:
        Side of the box-mean window, in pixels.
    fill_value:
        NaN / ±inf replacement, as in :func:`dense_flow`.

    Returns
    -------
    float32, same shape, **raw energy** (≥ 0, uncalibrated units of
    squared uint8 level per pixel). Callers turn it into a weight by
    comparing it against a percentile of its own on-echo distribution —
    the absolute scale depends on the composite's dynamic range and is not
    comparable between frames, but the *ranking* within a frame is exactly
    what "is there anything here to track" needs.
    """
    if int(window_px) < 1:
        raise ValueError(f"window_px must be >= 1, got {window_px}")
    window = int(window_px)
    img = _to_uint8(np.asarray(dbz_now), fill_value)

    try:
        import cv2
    except ImportError:
        cv2 = None  # type: ignore[assignment]

    if cv2 is not None:
        src = img.astype(np.float32)
        del img
        gx = cv2.Sobel(src, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE)
        gy = cv2.Sobel(src, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REPLICATE)
        del src
        # In-place: the native grid is 14 MB per float32 copy and the
        # sidecar runs under a memory cap.
        np.square(gx, out=gx)
        np.square(gy, out=gy)
        np.add(gx, gy, out=gx)
        del gy
        energy = cv2.boxFilter(
            gx, cv2.CV_32F, (window, window),
            normalize=True, borderType=cv2.BORDER_REPLICATE,
        )
        del gx
    else:
        from scipy.ndimage import sobel, uniform_filter

        src = img.astype(np.float32)
        del img
        gx = sobel(src, axis=1, mode="nearest")
        gy = sobel(src, axis=0, mode="nearest")
        del src
        np.square(gx, out=gx)
        np.square(gy, out=gy)
        np.add(gx, gy, out=gx)
        del gy
        energy = uniform_filter(gx, size=window, mode="nearest").astype(np.float32)
        del gx

    # A box mean of non-negative values is non-negative; the clamp only
    # removes filter round-off so callers can divide by a percentile of it
    # without ever producing a negative weight.
    np.maximum(energy, np.float32(0.0), out=energy)
    return energy


#: Default e-folding distance for motion-field completion, in km.
#:
#: Two scales bracket this. Below: Farnebäck's own reach. Its polynomial
#: expansion uses ``winsize=31`` over a 3-level pyramid, so a pixel more
#: than ~15 px (7.5 km) from any echo has echo inside its window only at
#: the coarse levels, where all that survives is the large-scale motion the
#: bulk vector already carries. Above: the advection distance we have to
#: cross. A 60-min lead at 30-60 km/h is 30-60 km of travel, and the weight
#: has to be small over most of *that* path or the far field never reaches
#: bulk speed — an e-fold comparable to the travel distance leaves the rain
#: crawling, which is the bug this whole function exists to fix.
#:
#: 10 km (20 px on the 500 m grid) sits between them: weight 0.78 at 5 km
#: from the echo (local structure kept where the estimate is real), 0.05 at
#: 30 km (bulk where it is not). Measured over a 60-min lead on the
#: synthetic of ``tests/test_flow_completion.py``, varying how far the
#: estimate reaches beyond the echo (6-25 px halo): a 5 km e-fold restores
#: 89-99 % of the observed rain mass and 94-100 % of the flow-implied
#: displacement, 10 km gives 69-96 % / 86-98 %, and 25 km only 44-88 % /
#: 75-95 % — i.e. a 25 km e-fold leaves a sizeable part of the barrier
#: standing, while 5 km discards near-echo structure a two-system day
#: needs. The raw field, for scale: 29-77 % / 67-91 %.
DEFAULT_EFOLD_KM = 10.0

#: Pixels of full-weight halo around the echo. The Farnebäck estimate stays
#: meaningful just outside the echo edge (the polynomial window straddles
#: it), and the radar's effective resolution is ~2 px anyway.
DEFAULT_SUPPORT_DILATION_PX = 3


def rain_weighted_bulk(
    vy: np.ndarray,
    vx: np.ndarray,
    rain_mm_h: np.ndarray,
    *,
    support_threshold_mm_h: float = 0.5,
) -> tuple[float, float] | None:
    """Rain-weighted mean flow over echo pixels, in px per frame.

    The bulk vector :func:`complete_flow` has relaxed the far field toward
    since R5. Rain weighting (rather than a flat mean) keeps a few drizzle
    pixels at the domain edge from out-voting the main band.

    Returns ``None`` when there is no echo carrying a usable velocity —
    the caller's "no bulk to relax toward" case. Split out of
    :func:`complete_flow` so :func:`estimate_motion` can report the same
    number it used, and so :func:`robust_bulk` has one fallback rather
    than a copy of this arithmetic.
    """
    vy_arr = np.asarray(vy, dtype=np.float32)
    vx_arr = np.asarray(vx, dtype=np.float32)
    rain = np.asarray(rain_mm_h, dtype=np.float32)

    finite_v = np.isfinite(vy_arr) & np.isfinite(vx_arr)
    support = np.isfinite(rain) & (rain >= support_threshold_mm_h)
    weights = np.where(support & finite_v, rain, np.float32(0.0))
    w_sum = float(weights.sum())
    if not np.isfinite(w_sum) or w_sum <= 0.0:
        return None
    vy_clean = np.where(finite_v, vy_arr, np.float32(0.0))
    vx_clean = np.where(finite_v, vx_arr, np.float32(0.0))
    return (
        float((vy_clean * weights).sum() / w_sum),
        float((vx_clean * weights).sum() / w_sum),
    )


def robust_bulk(
    vy: np.ndarray,
    vx: np.ndarray,
    rain_mm_h: np.ndarray,
    energy: np.ndarray,
    *,
    support_threshold_mm_h: float = 0.5,
    texture_percentile: float = DEFAULT_TEXTURE_PERCENTILE,
    min_pixels: int = DEFAULT_ROBUST_BULK_MIN_PIXELS,
) -> tuple[float, float]:
    """Bulk storm motion from the wet pixels that had something to track.

    Why not :func:`rain_weighted_bulk` (evidence:
    ``archive/flow_stall_20260908/``): on the 2026-09-08 evening shield the
    stalled pixels were the *majority* of the echo, so the rain-weighted
    mean was dragged down to 15-17 km/h while the textured pixels — the
    only ones whose estimate means anything — reported 25-28 km/h. A mean
    over a bimodal population where one mode is a measurement artefact is
    not an estimate of anything. Two changes fix that: take the **median**
    (breakdown point 50 %) and take it only over pixels whose local
    gradient energy is at or above ``texture_percentile`` of the on-echo
    distribution.

    Parameters
    ----------
    vy, vx:
        Flow in px per frame, as estimated (pre-completion).
    rain_mm_h:
        Rain rate on the same grid; NaN is nodata. Defines the echo.
    energy:
        :func:`flow_confidence` output on the same grid. Non-finite
        entries are treated as "no texture" and never qualify.
    support_threshold_mm_h:
        Rain rate a pixel needs to count as echo.
    texture_percentile:
        Percentile of the on-echo energy a pixel must reach to vote.
    min_pixels:
        Below this many qualifying pixels the median is not worth having;
        fall back to :func:`rain_weighted_bulk` over all echo pixels.

    Returns
    -------
    ``(bulk_vy, bulk_vx)`` in px per frame. ``(0.0, 0.0)`` when there is no
    echo at all — mirroring :func:`complete_flow`'s degenerate case, where
    "no bulk" and "zero bulk" are the same statement.
    """
    vy_arr = np.asarray(vy, dtype=np.float32)
    vx_arr = np.asarray(vx, dtype=np.float32)
    rain = np.asarray(rain_mm_h, dtype=np.float32)
    nrg = np.asarray(energy, dtype=np.float32)
    if vy_arr.shape != vx_arr.shape or vy_arr.shape != rain.shape:
        raise ValueError("vy, vx, rain_mm_h must all have the same shape")
    if nrg.shape != vy_arr.shape:
        raise ValueError(
            f"energy shape {nrg.shape} does not match the flow {vy_arr.shape}"
        )

    finite_v = np.isfinite(vy_arr) & np.isfinite(vx_arr)
    support = np.isfinite(rain) & (rain >= support_threshold_mm_h)
    wet = support & finite_v
    if not wet.any():
        return (0.0, 0.0)

    def _fallback() -> tuple[float, float]:
        weighted = rain_weighted_bulk(
            vy_arr, vx_arr, rain, support_threshold_mm_h=support_threshold_mm_h,
        )
        return weighted if weighted is not None else (0.0, 0.0)

    wet_energy = nrg[wet]
    wet_energy = wet_energy[np.isfinite(wet_energy)]
    if wet_energy.size == 0:
        return _fallback()
    threshold = float(np.percentile(wet_energy, float(texture_percentile)))
    del wet_energy
    # ``> 0`` as well as ``>= threshold``: a pixel with no gradient at all
    # in its window had literally nothing to track, so its zero velocity is
    # an artefact and never votes — whatever the percentile says. It has to
    # be said explicitly, because the percentile of a badly stalled frame
    # can itself BE zero (a majority of flat pixels), and ``>= 0`` would
    # then re-admit exactly the pixels this function exists to exclude.
    # NaN energy compares False on both, so it is excluded by construction.
    with np.errstate(invalid="ignore"):
        textured = wet & (nrg >= np.float32(threshold)) & (nrg > np.float32(0.0))
    if int(textured.sum()) < int(min_pixels):
        return _fallback()
    return float(np.median(vy_arr[textured])), float(np.median(vx_arr[textured]))


def _confidence_weight(
    energy: np.ndarray, wet: np.ndarray, percentile: float,
) -> np.ndarray:
    """``clip(energy / thr, 0, 1)``, ``thr`` = ``percentile`` of on-echo energy.

    The on-echo weight of :func:`complete_flow`'s confidence path. The
    threshold is taken over the *wet* pixels of this frame only: energy is
    an uncalibrated squared-gradient sum whose scale moves with the
    composite's dynamic range, so the only meaningful reference is the
    frame's own echo. ``thr <= 0`` — a frame with no gradient anywhere on
    the echo — means nothing was measurable, and the weight is 0 (bulk
    motion everywhere), never a division by zero.
    """
    wet_energy = energy[wet]
    wet_energy = wet_energy[np.isfinite(wet_energy)]
    threshold = (
        float(np.percentile(wet_energy, float(percentile)))
        if wet_energy.size else 0.0
    )
    del wet_energy
    if not np.isfinite(threshold) or threshold <= 0.0:
        return np.zeros(energy.shape, dtype=np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = energy / np.float32(threshold)
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    np.clip(out, np.float32(0.0), np.float32(1.0), out=out)
    return out


def complete_flow(
    vy: np.ndarray,
    vx: np.ndarray,
    rain_mm_h: np.ndarray,
    *,
    pixel_km: float,
    support_threshold_mm_h: float = 0.5,
    efold_km: float = DEFAULT_EFOLD_KM,
    dilation_px: float = DEFAULT_SUPPORT_DILATION_PX,
    confidence: np.ndarray | None = None,
    confidence_percentile: float = DEFAULT_CONFIDENCE_PERCENTILE,
    texture_percentile: float = DEFAULT_TEXTURE_PERCENTILE,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill in motion where optical flow has no echo to track.

    Farnebäck (and TV-L1) estimate motion from image gradients. We fill
    nodata/undetect with a flat ``DEFAULT_FILL_DBZ`` before the estimate, so
    everywhere without echo is a featureless plateau and the returned flow
    there is *exactly zero* — on a real DMI composite, 71 % of the grid came
    back with ``|v| < 0.5`` px/frame, with the median dropping from ~25
    px/frame near the echo to ~0 beyond 40-60 px. Advecting with that field
    (however good the integrator) makes rain stall along a stationary line
    ~20-30 km ahead of the echo, because the destination pixels ahead of the
    rain have nowhere to come *from*.

    So we blend the estimated field toward the bulk storm motion with
    distance from the echo::

        v_completed = w·v + (1 - w)·v_bulk,   w = exp(-d / τ)

    where ``d`` is the distance in pixels to the nearest echo pixel (minus a
    ``dilation_px`` full-weight halo) and ``τ = efold_km / pixel_km``.
    ``v_bulk`` is the rain-weighted mean of the flow over echo pixels — the
    same statistic ``compute._disc_motion`` takes around home, but global
    and without the disc. On the echo ``w = 1``: the estimated field is
    untouched, including its shear and rotation. Far from it the field tends
    to uniform storm motion, which is the honest prior — it is what a human
    reading a radar loop extrapolates with.

    Changed 2026-09-08 — the ``confidence`` path
    --------------------------------------------
    The paragraph above assumed the estimate is trustworthy *wherever there
    is echo*. It is not. Evidence, both cases measured and reproduced with
    this code: ``archive/flow_stall_20260908/README.md``. Inside a broad,
    flat or speckled shield Farnebäck has nothing to match and returns
    near-zero displacement, so ``w = 1`` on the echo preserved a stall
    covering 55 % of the wet pixels around Odense on the 17:40Z cycle —
    the front advanced, the interior stayed, and STEPS inherited the same
    field. The rain-weighted bulk was itself dragged to 15-17 km/h by that
    stalled majority, against the 25-28 km/h the textured pixels reported.

    Passing ``confidence`` (a :func:`flow_confidence` energy grid) changes
    two things, and nothing else:

    1. the relaxation target becomes :func:`robust_bulk` — the median over
       high-texture wet pixels — instead of the rain-weighted mean, so the
       stalled pixels cannot vote on where the far field is heading;
    2. **on** the dilated support the weight is no longer a flat 1 but
       ``c = clip(energy / thr, 0, 1)``, with ``thr`` the
       ``confidence_percentile`` of the energy over wet pixels, so a
       textured pixel keeps its own estimate and a featureless one is
       handed the bulk vector. ``thr <= 0`` (an all-flat frame) means
       nothing was measurable anywhere, and ``c`` is 0 throughout.

    **Off** the support the behaviour is untouched: the same
    ``exp(-d / τ)`` handover, now toward the robust bulk.

    With ``confidence=None`` this function is bit-for-bit what it was
    before that date (pinned by ``tests/test_flow_confidence.py``).

    Parameters
    ----------
    vy, vx:
        Flow in pixels per frame, as returned by :func:`dense_flow`.
    rain_mm_h:
        Rain rate on the same grid (NaN = nodata) — the echo support.
    pixel_km:
        Grid spacing in km (0.5 on the DMI 500 m composite).
    support_threshold_mm_h:
        Rain rate a pixel needs to count as echo. Use the same detection
        threshold the rest of the pipeline uses (config default 0.5 mm/h).
    efold_km:
        Distance over which the estimated flow relaxes to bulk motion.
    dilation_px:
        Full-weight halo around the echo, in pixels.
    confidence:
        Optional :func:`flow_confidence` energy grid, same shape. ``None``
        keeps the pre-2026-09-08 behaviour exactly.
    confidence_percentile:
        Percentile of on-echo energy at which the on-echo weight saturates
        at 1. Ignored when ``confidence is None``.
    texture_percentile:
        Forwarded to :func:`robust_bulk`. Ignored when ``confidence is
        None``.

    Returns
    -------
    ``(vy, vx)`` float32, same shape. **No-echo edge case**: with no pixel
    above the threshold there is no bulk vector to relax toward, so the
    input is returned unchanged rather than pulled to zero.
    """
    vy_arr = np.asarray(vy, dtype=np.float32)
    vx_arr = np.asarray(vx, dtype=np.float32)
    rain = np.asarray(rain_mm_h, dtype=np.float32)
    if vy_arr.shape != vx_arr.shape or vy_arr.shape != rain.shape:
        raise ValueError("vy, vx, rain_mm_h must all have the same shape")
    if pixel_km <= 0:
        raise ValueError(f"pixel_km must be > 0, got {pixel_km}")

    nrg: np.ndarray | None = None
    if confidence is not None:
        nrg = np.asarray(confidence, dtype=np.float32)
        if nrg.shape != vy_arr.shape:
            raise ValueError(
                f"confidence shape {nrg.shape} does not match the flow {vy_arr.shape}"
            )

    finite_v = np.isfinite(vy_arr) & np.isfinite(vx_arr)
    support = np.isfinite(rain) & (rain >= support_threshold_mm_h)

    # Bulk vector. ``rain_weighted_bulk`` also answers the degenerate
    # question — no echo carrying a usable velocity means no vector to
    # relax toward — so it runs on both paths and its ``None`` is the
    # single "nothing to do here" gate.
    weighted = rain_weighted_bulk(
        vy_arr, vx_arr, rain, support_threshold_mm_h=support_threshold_mm_h,
    )
    if weighted is None:
        return vy_arr.copy(), vx_arr.copy()
    if nrg is None:
        bulk_vy, bulk_vx = weighted
    else:
        bulk_vy, bulk_vx = robust_bulk(
            vy_arr, vx_arr, rain, nrg,
            support_threshold_mm_h=support_threshold_mm_h,
            texture_percentile=texture_percentile,
        )
    vy_clean = np.where(finite_v, vy_arr, np.float32(0.0))
    vx_clean = np.where(finite_v, vx_arr, np.float32(0.0))

    tau_px = float(efold_km) / float(pixel_km)
    if tau_px <= 0:
        # Degenerate e-folding: bulk motion everywhere off the echo.
        weight = support.astype(np.float32)
        on_support = support
    else:
        distance_px = distance_to_support(support)
        # The dilation is expressed through the distance field: everything
        # within ``dilation_px`` of an echo pixel keeps weight 1.
        np.subtract(distance_px, np.float32(max(0.0, dilation_px)), out=distance_px)
        np.maximum(distance_px, np.float32(0.0), out=distance_px)
        # "On the dilated support" is defined as the clamped distance
        # field's zero set rather than as ``d <= dilation_px``, so it is by
        # construction EXACTLY the region whose weight is 1 — OpenCV's
        # distance transform is a 5×5-mask approximation, and a pixel that
        # reads 3.0001 must not fall out of the gate on a rounding error.
        on_support = distance_px <= np.float32(0.0) if nrg is not None else None
        weight = np.exp(-distance_px / np.float32(tau_px))
        del distance_px

    if nrg is not None:
        weight = np.where(on_support, _confidence_weight(
            nrg, support & finite_v, confidence_percentile,
        ), weight)

    # Non-finite input velocities have no information to preserve: they take
    # the bulk vector outright (weight 0), not a NaN.
    weight = np.where(finite_v, weight, np.float32(0.0))
    out_vy = weight * vy_clean + (np.float32(1.0) - weight) * np.float32(bulk_vy)
    out_vx = weight * vx_clean + (np.float32(1.0) - weight) * np.float32(bulk_vx)
    return out_vy.astype(np.float32), out_vx.astype(np.float32)


def distance_to_support(support: np.ndarray) -> np.ndarray:
    """Euclidean distance (in pixels) from every pixel to the nearest True.

    Public because two consumers need the same notion of "how far is this
    pixel from any echo": :func:`complete_flow`'s relaxation weight, and
    ``national.motion_grids_kmh``'s nodata mask.

    OpenCV's ``distanceTransform`` when available (~10 ms on the native
    1728×1984 grid), else scipy's exact EDT. Note the inversion: OpenCV
    measures the distance from each *non-zero* pixel to the nearest *zero*
    one, so the echo mask goes in as the zeros.
    """
    try:
        import cv2
    except ImportError:
        cv2 = None  # type: ignore[assignment]
    if cv2 is not None:
        src = (~support).astype(np.uint8)
        return cv2.distanceTransform(src, cv2.DIST_L2, 5).astype(np.float32)
    from scipy.ndimage import distance_transform_edt

    return distance_transform_edt(~support).astype(np.float32)


def mean_flow(
    vy: np.ndarray,
    vx: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[float, float]:
    """Mean (vy, vx) over pixels in ``mask`` (or all pixels if ``mask`` is None).

    Useful for a "single mean motion vector" comparison against the Phase 2
    phase-correlation baseline.
    """
    if mask is None:
        return float(np.mean(vy)), float(np.mean(vx))
    return float(np.mean(vy[mask])), float(np.mean(vx[mask]))


def stalled_share(
    vy: np.ndarray,
    vx: np.ndarray,
    rain_mm_h: np.ndarray,
    *,
    pixel_km: float,
    dt_min: float,
    support_threshold_mm_h: float = 0.5,
    stall_kmh: float = STALL_SPEED_KMH,
) -> float:
    """Share of wet pixels whose *estimated* speed is below ``stall_kmh``.

    The H-F stall diagnostic, on the raw estimate rather than the completed
    field — completion is what the fix changes, so measuring after it would
    only ever report success. Comparable with
    ``archive/flow_stall_20260908/README.md``, which uses the same 5 km/h
    cut: production read 47-61 % on the 2026-09-08 evening shield and 11 %
    on the more textured morning band, against ~0 % for a field that is
    actually moving everywhere it rains.

    Returns 0.0 on a dry composite — no wet pixels, nothing stalled.
    """
    if pixel_km <= 0:
        raise ValueError(f"pixel_km must be > 0, got {pixel_km}")
    if dt_min <= 0:
        raise ValueError(f"dt_min must be > 0, got {dt_min}")
    rain = np.asarray(rain_mm_h, dtype=np.float32)
    wet = np.isfinite(rain) & (rain >= support_threshold_mm_h)
    n_wet = int(wet.sum())
    if n_wet == 0:
        return 0.0
    # Only the wet pixels are materialised: a full-grid hypot would be
    # another 14 MB float32 on the native composite for a scalar answer.
    to_kmh = np.float32(float(pixel_km) * 60.0 / float(dt_min))
    speed = np.hypot(
        np.asarray(vy, dtype=np.float32)[wet],
        np.asarray(vx, dtype=np.float32)[wet],
    )
    speed *= to_kmh
    return float(np.count_nonzero(speed < np.float32(stall_kmh)) / n_wet)


@dataclass(frozen=True)
class MotionEstimate:
    """One cycle's motion field, plus what it took to get there.

    ``vy``/``vx`` are the field every consumer advects with (completed,
    NaN-free, clipped). ``vy_raw``/``vx_raw`` are the same estimate before
    completion, sanitised the same way — kept because the served motion
    arrows and the stall diagnostic ask about what was *measured*, not what
    was filled in. They are two extra native-grid float32 arrays (~14 MB
    each on the DMI composite); callers under a memory cap should drop the
    ``MotionEstimate`` once they have taken what they need.
    """

    vy: np.ndarray
    vx: np.ndarray
    vy_raw: np.ndarray
    vx_raw: np.ndarray
    #: Bulk storm motion the completion relaxed toward, in px per frame.
    bulk_vy: float
    bulk_vx: float
    #: Share of wet pixels whose RAW estimate is below ``STALL_SPEED_KMH``.
    #: A property of the *estimator*, so it is unchanged by ``completion``
    #: and stays high on a stratiform shield however the field is
    #: completed — which is exactly what makes it the health signal for
    #: the defect (``archive/flow_stall_20260908/``).
    stalled_share: float
    #: The same share on the COMPLETED field — the one the forecast is
    #: actually advected with. This is the number the evidence file's
    #: variant tables report (production 11 %, texture-gated 0 % on the
    #: morning band); it is what says the fix reached the advection.
    stalled_share_completed: float
    #: Which completion policy produced ``vy``/``vx``.
    completion: Literal["bulk", "confidence"]


def estimate_motion(
    prev_dbz: np.ndarray,
    curr_dbz: np.ndarray,
    rain_now_mm_h: np.ndarray,
    *,
    pixel_km: float,
    dt_min: float,
    support_threshold_mm_h: float = 0.5,
    completion: Literal["bulk", "confidence"] = "confidence",
    confidence_window_px: int = DEFAULT_CONFIDENCE_WINDOW_PX,
    confidence_percentile: float = DEFAULT_CONFIDENCE_PERCENTILE,
    texture_percentile: float = DEFAULT_TEXTURE_PERCENTILE,
    max_px_per_frame: float = DEFAULT_MAX_PX_PER_FRAME,
    flow: tuple[np.ndarray, np.ndarray] | None = None,
) -> MotionEstimate:
    """Estimate → complete → sanitise, once, for every consumer.

    The runtime (``compute.py``), the calibration-corpus builder and the
    warning replay all have to produce the *same* motion field or the
    curves calibrate a forecast nobody serves. Before this function they
    each spelled the sequence out inline and had already drifted in small
    ways (``posinf``/``neginf`` handling, where the clip sits). One entry
    point makes that impossible.

    ``completion="bulk"`` reproduces the pre-2026-09-08 sequence exactly:
    estimate, :func:`complete_flow` on the raw estimate, then nan→0 and a
    ±``max_px_per_frame`` clip. ``completion="confidence"`` adds the H-F
    hotfix — a :func:`flow_confidence` grid, a :func:`robust_bulk` target
    and the on-echo gate (see :func:`complete_flow`).

    Parameters
    ----------
    prev_dbz, curr_dbz:
        The frame pair, in dBZ. Ignored when ``flow`` is supplied.
    rain_now_mm_h:
        Rain rate of ``curr_dbz``'s frame — the echo support.
    pixel_km, dt_min:
        Grid spacing in km and the measured inter-frame interval in
        minutes. Only the diagnostics are in physical units; ``vy``/``vx``
        stay in px per frame, as every consumer expects.
    completion:
        ``"bulk"`` (legacy) or ``"confidence"`` (H-F).
    flow:
        Pre-computed ``(vy, vx)`` raw estimate. Callers with their own
        fallback (a uniform phase-correlation shift when no dense-flow
        backend is installed) pass it here so the completion, sanitising
        and diagnostics still go through one code path.

    Raises
    ------
    DenseFlowUnavailable
        From :func:`dense_flow`, when ``flow`` is None and no backend is
        installed. Callers catch it and retry with a ``flow=`` fallback.
    """
    if completion not in ("bulk", "confidence"):
        raise ValueError(f"completion must be 'bulk' or 'confidence', got {completion!r}")
    if pixel_km <= 0:
        raise ValueError(f"pixel_km must be > 0, got {pixel_km}")
    if dt_min <= 0:
        raise ValueError(f"dt_min must be > 0, got {dt_min}")

    vy, vx = dense_flow(prev_dbz, curr_dbz) if flow is None else flow
    rain = np.asarray(rain_now_mm_h, dtype=np.float32)
    max_px = np.float32(max_px_per_frame)

    # Sanitised *raw* estimate: the served motion arrows' reference and the
    # stall diagnostic's input. Same nan→0 and clip as the completed field,
    # so wherever the estimate is kept the two agree exactly.
    vy_raw = np.nan_to_num(vy, nan=0.0).astype(np.float32)
    vx_raw = np.nan_to_num(vx, nan=0.0).astype(np.float32)
    np.clip(vy_raw, -max_px, max_px, out=vy_raw)
    np.clip(vx_raw, -max_px, max_px, out=vx_raw)

    energy: np.ndarray | None = None
    if completion == "confidence":
        energy = flow_confidence(curr_dbz, window_px=confidence_window_px)
        bulk = robust_bulk(
            vy, vx, rain, energy,
            support_threshold_mm_h=support_threshold_mm_h,
            texture_percentile=texture_percentile,
        )
    else:
        weighted = rain_weighted_bulk(
            vy, vx, rain, support_threshold_mm_h=support_threshold_mm_h,
        )
        bulk = weighted if weighted is not None else (0.0, 0.0)

    # Completion BEFORE the sanitise/clip, as the runtime has always done
    # it, so both consumers (the deterministic advection and the STEPS
    # velocity) get the completed field.
    out_vy, out_vx = complete_flow(
        vy, vx, rain,
        pixel_km=pixel_km,
        support_threshold_mm_h=support_threshold_mm_h,
        confidence=energy,
        confidence_percentile=confidence_percentile,
        texture_percentile=texture_percentile,
    )
    del energy
    out_vy = np.nan_to_num(out_vy, nan=0.0).astype(np.float32)
    out_vx = np.nan_to_num(out_vx, nan=0.0).astype(np.float32)
    np.clip(out_vy, -max_px, max_px, out=out_vy)
    np.clip(out_vx, -max_px, max_px, out=out_vx)

    return MotionEstimate(
        vy=out_vy,
        vx=out_vx,
        vy_raw=vy_raw,
        vx_raw=vx_raw,
        bulk_vy=float(bulk[0]),
        bulk_vx=float(bulk[1]),
        stalled_share=stalled_share(
            vy_raw, vx_raw, rain,
            pixel_km=pixel_km, dt_min=dt_min,
            support_threshold_mm_h=support_threshold_mm_h,
        ),
        stalled_share_completed=stalled_share(
            out_vy, out_vx, rain,
            pixel_km=pixel_km, dt_min=dt_min,
            support_threshold_mm_h=support_threshold_mm_h,
        ),
        completion=completion,
    )


