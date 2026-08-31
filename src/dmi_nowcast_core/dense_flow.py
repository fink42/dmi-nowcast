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


def complete_flow(
    vy: np.ndarray,
    vx: np.ndarray,
    rain_mm_h: np.ndarray,
    *,
    pixel_km: float,
    support_threshold_mm_h: float = 0.5,
    efold_km: float = DEFAULT_EFOLD_KM,
    dilation_px: float = DEFAULT_SUPPORT_DILATION_PX,
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

    finite_v = np.isfinite(vy_arr) & np.isfinite(vx_arr)
    support = np.isfinite(rain) & (rain >= support_threshold_mm_h)

    # Bulk vector: rain-weighted mean over echo pixels with a usable
    # velocity. Rain weighting (rather than a flat mean) keeps a few
    # drizzle pixels at the domain edge from out-voting the main band.
    weights = np.where(support & finite_v, rain, np.float32(0.0))
    w_sum = float(weights.sum())
    if not np.isfinite(w_sum) or w_sum <= 0.0:
        return vy_arr.copy(), vx_arr.copy()
    vy_clean = np.where(finite_v, vy_arr, np.float32(0.0))
    vx_clean = np.where(finite_v, vx_arr, np.float32(0.0))
    bulk_vy = float((vy_clean * weights).sum() / w_sum)
    bulk_vx = float((vx_clean * weights).sum() / w_sum)

    tau_px = float(efold_km) / float(pixel_km)
    if tau_px <= 0:
        # Degenerate e-folding: bulk motion everywhere off the echo.
        weight = support.astype(np.float32)
    else:
        distance_px = distance_to_support(support)
        # The dilation is expressed through the distance field: everything
        # within ``dilation_px`` of an echo pixel keeps weight 1.
        np.subtract(distance_px, np.float32(max(0.0, dilation_px)), out=distance_px)
        np.maximum(distance_px, np.float32(0.0), out=distance_px)
        weight = np.exp(-distance_px / np.float32(tau_px))

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


