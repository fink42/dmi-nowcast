"""Mean motion estimation via FFT phase correlation.

Phase 2 baseline. Computes a single ``(dy, dx)`` pixel displacement between two
consecutive frames, suitable for the Lagrangian-mean-motion baseline. Phase 3
will swap in pysteps' dense Lucas–Kanade / Farnebäck once that's installable.

No pysteps dependency — pure numpy.fft — so this works on macOS arm64 without
the OpenMP workaround needed for the pysteps source build.

Convention: ``dy, dx = phase_correlation_shift(prev, curr)`` means the rain
field in ``curr`` is ``prev`` shifted by ``(dy, dx)`` pixels (down-positive,
right-positive). To predict the field at ``t+Δ`` from the current field, advect
by ``(dy, dx) * (Δ / dt_between_frames)``.
"""
from __future__ import annotations

import numpy as np


def phase_correlation_shift(prev: np.ndarray, curr: np.ndarray) -> tuple[float, float]:
    """Return the integer-pixel ``(dy, dx)`` shift from ``prev`` to ``curr``.

    NaN / -inf pixels are treated as zeros (no contribution to the correlation).
    The peak of the inverse cross-power spectrum gives the displacement.
    """
    if prev.shape != curr.shape:
        raise ValueError(f"shape mismatch: prev {prev.shape} vs curr {curr.shape}")
    a = np.nan_to_num(prev, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    b = np.nan_to_num(curr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0, 0.0  # no signal to correlate

    A = np.fft.fft2(a)
    B = np.fft.fft2(b)
    # F[curr ⊗ prev] = conj(A) · B; peak at the displacement curr = shift(prev, dy, dx).
    cps = np.conj(A) * B
    cps /= np.maximum(np.abs(cps), 1e-12)
    r = np.fft.ifft2(cps).real
    peak = np.unravel_index(np.argmax(r), r.shape)
    h, w = a.shape
    dy = peak[0] - h if peak[0] > h // 2 else peak[0]
    dx = peak[1] - w if peak[1] > w // 2 else peak[1]
    return float(dy), float(dx)
