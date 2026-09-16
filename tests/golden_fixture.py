"""The rows the v1 pin stands on. Self-contained on purpose.

Shared by the golden GENERATOR (run once against the pre-v2 module) and
by the test that compares against it, so the pin cannot drift when some
other fixture in the suite is edited.
"""
from __future__ import annotations

import numpy as np

GOLDEN_LEADS = (20, 30)
GOLDEN_DESIGN_LEADS = (10, 20, 30)


def golden_row() -> dict:
    """One decision row with every v1 source column set to a known value."""
    values = {
        "raw_frac_10": 0.25, "raw_frac_20": 0.5, "raw_frac_30": 0.75,
        "observed_mm_h": 0.0, "obs_max_5km_mm_h": 1.0,
        "up_max_20km_mm_h": 3.0, "up_max_40km_mm_h": 3.0,
        "up_dist_km": 12.0, "up_wet_frac_40km": 0.2,
        "eta_min": 25.0, "intensity_mm_h": 2.0,
        "bulk_kmh": 30.0, "bulk_dir_deg": 270.0,
        "local_speed_kmh": 28.0, "stalled_share": 0.02,
        "frame_age_min": 15.0, "station_radar_km": 40.0,
        "hour_utc": 6.0, "season": "summer",
    }
    return {
        key: (
            np.array([value], dtype="<U8") if isinstance(value, str)
            else np.array([value], dtype=np.float64)
        )
        for key, value in values.items()
    }


def golden_dataset(seed: int = 17) -> dict:
    """Rows whose outcome rises with upstream rain and falls with distance.

    Four months of twelve days, six stations, ten minutes apart — enough
    for the leave-one-month-out folds and for every season to appear. The
    ensemble fraction is distance-blind and saturating, which is the
    defect the post-processor exists for and the reason a fit on these
    rows has a signal to find.
    """
    n_months, n_days, n_instants, n_stations = 4, 12, 20, 6
    rng = np.random.default_rng(seed)
    n = n_months * n_days * n_instants * n_stations
    month_index = np.repeat(np.arange(n_months), n_days * n_instants * n_stations)
    day_in_month = np.tile(
        np.repeat(np.arange(n_days), n_instants * n_stations), n_months,
    )
    instant = np.tile(
        np.repeat(np.arange(n_instants), n_stations), n_months * n_days,
    )
    station = np.tile(np.arange(n_stations), n_months * n_days * n_instants)
    base = np.datetime64("2026-01-01T06:00:00")
    t = (
        base.astype("datetime64[s]").astype(np.int64)
        + month_index.astype(np.int64) * 31 * 86400
        + day_in_month.astype(np.int64) * 86400
        + instant.astype(np.int64) * 600
    )
    up_max = np.exp(rng.normal(0.2, 0.9, size=n))
    up_dist = rng.uniform(0.0, 45.0, size=n)
    no_echo = up_dist > 40.0
    up_dist = np.where(no_echo, np.nan, up_dist)
    up_max = np.where(no_echo, 0.0, up_max)
    log_max = np.log1p(up_max)
    log_dist = np.log1p(np.nan_to_num(up_dist, nan=40.0))
    probability = 1.0 / (1.0 + np.exp(-(1.4 + 1.2 * log_max - 1.3 * log_dist)))
    y = {
        lead: (
            rng.uniform(size=n) < np.clip(probability * (lead / 30.0), 0, 1)
        ).astype(np.float64)
        for lead in GOLDEN_LEADS
    }
    agreement = 1.0 / (1.0 + np.exp(-(-1.4 + 1.3 * log_max)))
    fraction = np.clip(
        np.round(agreement * 1.6 * 16.0) / 16.0 + rng.normal(0, 0.03, size=n),
        0, 1,
    )
    features = {
        "up_max_40km_mm_h": up_max,
        "up_max_20km_mm_h": np.where(
            np.nan_to_num(up_dist, nan=99) <= 20, up_max, 0.0,
        ),
        "up_dist_km": up_dist,
        "up_wet_frac_40km": np.clip(up_max / 10.0, 0, 1),
    }
    for lead in GOLDEN_LEADS:
        features[f"raw_frac_{lead}"] = fraction
    return {
        "features": features,
        "truth": {
            lead: (y[lead], np.ones(n, dtype=bool)) for lead in GOLDEN_LEADS
        },
        "t": t,
        "station": station,
        "n": n,
    }
