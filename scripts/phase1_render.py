"""Phase 1 visual verification: render the latest DMI composite as PNG.

Compare side-by-side with radar.dmi.dk to confirm projection + Z–R look right.

Run:  .venv/bin/python scripts/phase1_render.py [--lat LAT --lon LON]

Saves ``reports/phase1_composite.png``. Marks the home location and a 1 km disc;
prints disc statistics so you can sanity-check magnitudes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dmi_nowcast_core.fetch import download, list_latest  # noqa: E402
from dmi_nowcast_core.geo import CompositeGeo  # noqa: E402
from dmi_nowcast_core.parse import parse_composite  # noqa: E402
from dmi_nowcast_core.sample import sample_disc  # noqa: E402
from dmi_nowcast_core.transform import dbz_to_rain_rate  # noqa: E402

ROOT = Path(__file__).parent.parent
CACHE = ROOT / ".recon_cache"
REPORTS = ROOT / "reports"

# Copenhagen Central — placeholder home. Real one comes from HA config later.
DEFAULT_LON = 12.5645
DEFAULT_LAT = 55.6726


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lat", type=float, default=DEFAULT_LAT)
    parser.add_argument("--lon", type=float, default=DEFAULT_LON)
    parser.add_argument("--scan-type", choices=["fullRange", "doppler"], default="fullRange")
    parser.add_argument("--radius-km", type=float, default=1.0)
    args = parser.parse_args()

    print("Fetching latest composite…")
    features = list_latest(limit=4, scan_type=args.scan_type)
    if not features:
        print("ERROR: no features returned", file=sys.stderr)
        return 1
    feature = features[0]
    path = download(feature, CACHE)
    print(f"  {feature.datetime_utc.isoformat()}  {feature.scan_type}  {path.name}")

    composite = parse_composite(path)
    geo = CompositeGeo(composite)
    rain = dbz_to_rain_rate(composite.reflectivity_dbz, zr_a=composite.zr_a, zr_b=composite.zr_b)
    stats = sample_disc(rain, geo, args.lon, args.lat, radius_m=args.radius_km * 1000.0)
    print(
        f"Disc @ ({args.lat:.4f}, {args.lon:.4f}) r={args.radius_km} km: "
        f"max={stats.max_mm_h:.3f} mm/h, mean={stats.mean_mm_h:.3f}, "
        f"p90={stats.p90_mm_h:.3f} ({stats.n_valid}/{stats.n_pixels_in_disc} valid pixels)"
    )

    idx = geo.lonlat_to_grid(args.lon, args.lat)
    home_row, home_col = idx.row, idx.col

    REPORTS.mkdir(parents=True, exist_ok=True)
    output = REPORTS / "phase1_composite.png"
    plot(rain, composite, geo, home_row, home_col, args.radius_km, stats, output)
    print(f"Saved → {output.relative_to(ROOT)}")
    return 0


def plot(rain, composite, geo, home_row, home_col, radius_km, stats, output):
    h, w = rain.shape
    fig, ax = plt.subplots(figsize=(10, 9), constrained_layout=True)

    # Use a log scale; clip at 0.01 mm/h to keep zeros transparent.
    display = np.where(np.isfinite(rain) & (rain > 0.01), rain, np.nan)
    im = ax.imshow(
        display,
        origin="upper",
        cmap="turbo",
        norm=LogNorm(vmin=0.05, vmax=50.0),
        interpolation="nearest",
    )

    # Home marker + disc
    radius_px = (radius_km * 1000.0) / composite.xscale_m
    ax.add_patch(Circle((home_col, home_row), radius_px, fill=False, edgecolor="white", linewidth=2))
    ax.add_patch(Circle((home_col, home_row), radius_px, fill=False, edgecolor="black", linewidth=1, linestyle="--"))
    ax.plot(home_col, home_row, "wo", markersize=6, markeredgecolor="black")

    # Zoom to ~150 km box around home so structure is visible.
    zoom_px = (150_000.0) / composite.xscale_m
    ax.set_xlim(home_col - zoom_px, home_col + zoom_px)
    ax.set_ylim(home_row + zoom_px, home_row - zoom_px)  # inverted for row-down
    ax.set_aspect("equal")

    title = (
        f"DMI composite — {composite.timestamp_utc.isoformat()} ({composite.quantity})\n"
        f"Disc @ home, r={radius_km} km: "
        f"max={stats.max_mm_h:.2f} mm/h, mean={stats.mean_mm_h:.2f}, p90={stats.p90_mm_h:.2f}"
    )
    ax.set_title(title)
    ax.set_xlabel("col (east →)")
    ax.set_ylabel("row (south →)")
    fig.colorbar(im, ax=ax, label="rain rate (mm/h, log)")

    fig.savefig(output, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
