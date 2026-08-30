"""Side-by-side comparison of skimage TV-L1 vs opencv Farnebäck dense
optical flow on real DMI radar data.

Picks ``--n-scenarios`` historical events from the cached radar archive
where rain is "imminent at Odense" (dry at T, wet within 20 min) plus
the latest live frames as a 5th scenario. For each scenario:

- Compute the per-pixel motion field with each algorithm
- Advect the latest radar frame forward at +5, +10, +15, +20, +25, +30 min
- Render an animated PNG per (scenario, algorithm) pair
- Also render the OBSERVED evolution (ground truth) so the user can
  judge which algorithm best matches reality

A small static HTML page glues the artifacts together: a 5-row grid
where each row shows (truth | TV-L1 | Farnebäck) for one scenario.

Usage::

    python scripts/compare_flow_methods.py \\
        --output /tmp/flow_compare --port 8766 --seed 42

Outputs land in ``/tmp/flow_compare/`` and a small HTTP server serves
the dashboard on ``http://localhost:8766``.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dmi_nowcast_core.advect import advect_field  # noqa: E402
from dmi_nowcast_core.geo import CompositeGeo  # noqa: E402
from dmi_nowcast_core.parse import parse_composite  # noqa: E402
from dmi_nowcast_core.render import LoopFrame, render_loop_png  # noqa: E402
from dmi_nowcast_core.sample import sample_disc  # noqa: E402
from dmi_nowcast_core.transform import dbz_to_rain_rate  # noqa: E402

_LOGGER = logging.getLogger("compare")

# Default reference point: the geometric centroid of Fyn (see
# scripts/build_calibration_points.py). Override with --lat/--lon.
REFERENCE_LAT = 55.33
REFERENCE_LON = 10.32
DISC_RADIUS_M = 1000.0
RAIN_THRESHOLD_MM_H = 0.5  # "clearly raining" — higher than the integration
                            # detection floor so we pick scenarios with
                            # meaningful rain reaching the reference point
FORECAST_LEADS_MIN = (5, 10, 15, 20, 25, 30)
FRAME_FN_RE = re.compile(r"dk\.com\.(\d{12})\.500_max\.h5$")


# ---------------------------------------------------------------------------
# Scenario selection
# ---------------------------------------------------------------------------


def _parse_archive_filename(path: Path) -> datetime | None:
    m = FRAME_FN_RE.search(path.name)
    if m is None:
        return None
    s = m.group(1)
    return datetime(
        int(s[0:4]), int(s[4:6]), int(s[6:8]),
        int(s[8:10]), int(s[10:12]),
        tzinfo=timezone.utc,
    )


def _build_odense_wet_index(
    archive_dir: Path, *, sample_every_n: int = 1,
) -> dict[datetime, bool]:
    """Walk the cached archive, classify each frame as wet/dry at Odense.

    ``sample_every_n=1`` checks every frame; larger values skip frames for
    speed (e.g. 3 = every third frame, ~5x faster, fine for finding
    candidates).
    """
    files = sorted(p for p in archive_dir.glob("dk.com.*.h5"))
    timestamps = [(p, _parse_archive_filename(p)) for p in files]
    timestamps = [(p, t) for (p, t) in timestamps if t is not None]
    timestamps.sort(key=lambda x: x[1])
    _LOGGER.info("classifying %d archive frames at Odense (1km disc, ≥ %.1f mm/h)",
                 len(timestamps[::sample_every_n]), RAIN_THRESHOLD_MM_H)

    index: dict[datetime, bool] = {}
    geo_cache = None
    for i, (path, ts) in enumerate(timestamps):
        if i % sample_every_n != 0:
            continue
        try:
            c = parse_composite(path)
        except Exception:  # noqa: BLE001
            continue
        if geo_cache is None:
            geo_cache = CompositeGeo(c)
        rain = dbz_to_rain_rate(c.reflectivity_dbz, zr_a=c.zr_a, zr_b=c.zr_b)
        stats = sample_disc(rain, geo_cache, REFERENCE_LON, REFERENCE_LAT, radius_m=DISC_RADIUS_M)
        is_wet = bool(np.isfinite(stats.max_mm_h) and stats.max_mm_h >= RAIN_THRESHOLD_MM_H)
        index[ts] = is_wet
        if i % 200 == 0:
            wet_so_far = sum(index.values())
            _LOGGER.info("  scanned %d/%d (%d wet at Odense so far)",
                         i + 1, len(timestamps), wet_so_far)
    return index


def _find_imminent_rain_events(
    wet_index: dict[datetime, bool], *, lead_min: int = 20, tol_min: int = 5,
) -> list[datetime]:
    """Events T where Odense is dry at T-tol and wet at T+lead_min (±tol)."""
    sorted_ts = sorted(wet_index.keys())
    by_ts = {t: wet_index[t] for t in sorted_ts}
    events: list[datetime] = []
    for t in sorted_ts:
        if by_ts.get(t):
            continue  # already wet at T → not "imminent"
        # Look for ANY wet frame within [T+lead_min-tol, T+lead_min+tol]
        lo = t + timedelta(minutes=lead_min - tol_min)
        hi = t + timedelta(minutes=lead_min + tol_min)
        future_wet = any(
            (lo <= s <= hi) and by_ts.get(s, False)
            for s in sorted_ts
        )
        if future_wet:
            events.append(t)
    return events


# ---------------------------------------------------------------------------
# Per-scenario rendering
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    label: str
    event_time: datetime
    frames_in: list[Path]    # 2 frames for dense flow input (T-10, T)
    frames_truth: list[tuple[int, Path]]  # sparse: (lead_min, path); may be empty


def _find_frame(archive_dir: Path, target: datetime, tol_min: int = 5) -> Path | None:
    """Find the cached frame closest to ``target`` within ``tol_min``."""
    best, best_dt = None, timedelta(minutes=tol_min + 1)
    for p in archive_dir.glob("dk.com.*.h5"):
        ts = _parse_archive_filename(p)
        if ts is None:
            continue
        d = abs(ts - target)
        if d < best_dt:
            best, best_dt = p, d
    return best


def _gather_scenario_frames(
    event_time: datetime, archive_dir: Path,
) -> tuple[list[Path], list[tuple[int, Path]]] | None:
    """Two input frames + however many verification frames are cached.

    Returns ``(input_frames, [(lead_min, path), ...])`` — the truth list is
    sparse (only the leads that have cached verification frames). Caller
    handles missing leads gracefully so an event with patchy verification
    coverage still produces useful forecast-side animations.
    """
    prev = _find_frame(archive_dir, event_time - timedelta(minutes=10))
    curr = _find_frame(archive_dir, event_time)
    if prev is None or curr is None:
        return None
    truth_paths: list[tuple[int, Path]] = []
    for lead in FORECAST_LEADS_MIN:
        t = _find_frame(archive_dir, event_time + timedelta(minutes=lead), tol_min=3)
        if t is not None:
            truth_paths.append((lead, t))
    # Require at least ONE verification frame so the truth column isn't empty.
    if not truth_paths:
        return None
    return [prev, curr], truth_paths


def _flow_tvl1(prev_dbz: np.ndarray, curr_dbz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Force the skimage TV-L1 path even if cv2 is available."""
    from skimage.registration import optical_flow_tvl1

    def _fill(a):
        return np.nan_to_num(a, nan=-32.0, posinf=-32.0, neginf=-32.0).astype(np.float32)
    prev = _fill(prev_dbz)
    curr = _fill(curr_dbz)
    # Match the integration's downsample
    DS = 2
    prev_ds = prev[::DS, ::DS]
    curr_ds = curr[::DS, ::DS]
    flow = optical_flow_tvl1(prev_ds, curr_ds, dtype=np.float32)
    vy = np.repeat(np.repeat(flow[0], DS, axis=0), DS, axis=1)[:prev.shape[0], :prev.shape[1]]
    vx = np.repeat(np.repeat(flow[1], DS, axis=0), DS, axis=1)[:prev.shape[0], :prev.shape[1]]
    return np.clip(vy, -30, 30).astype(np.float32), np.clip(vx, -30, 30).astype(np.float32)


def _flow_farneback(prev_dbz: np.ndarray, curr_dbz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Opencv Farnebäck. Raises if cv2 isn't installed."""
    import cv2
    DBZ_LO, DBZ_HI = -32.0, 60.0

    def _to_u8(a):
        f = np.nan_to_num(a, nan=DBZ_LO, posinf=DBZ_LO, neginf=DBZ_LO).astype(np.float32)
        return (np.clip((f - DBZ_LO) / (DBZ_HI - DBZ_LO), 0, 1) * 255).astype(np.uint8)
    flow = cv2.calcOpticalFlowFarneback(
        _to_u8(prev_dbz), _to_u8(curr_dbz), None,
        0.5, 3, 31, 5, 7, 1.5, 0,
    )
    vy, vx = flow[..., 1], flow[..., 0]
    return np.clip(vy, -30, 30).astype(np.float32), np.clip(vx, -30, 30).astype(np.float32)


def _render_animation_for_scenario(
    name: str, frames: list[LoopFrame], composite_ref, geo, out_path: Path,
) -> None:
    """Render an animated PNG using the existing radar overlay machinery."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    png = render_loop_png(
        frames,
        composite=composite_ref,
        geo=geo,
        home_lat=REFERENCE_LAT, home_lon=REFERENCE_LON,
        radius_km=1.0,
        zoom_km=100.0, output_px=500,
        now_stats_subline=name,
    )
    out_path.write_bytes(png)


def _render_scenario(scenario: Scenario, output_dir: Path) -> dict:
    """Produce three animated PNGs for one scenario: truth, TV-L1, Farnebäck."""
    composites_in = [parse_composite(p) for p in scenario.frames_in]
    truth_pairs = [(lead, parse_composite(p)) for (lead, p) in scenario.frames_truth]

    composite_now = composites_in[-1]
    composite_prev = composites_in[0]
    geo = CompositeGeo(composite_now)
    rain_now = dbz_to_rain_rate(
        composite_now.reflectivity_dbz, zr_a=composite_now.zr_a, zr_b=composite_now.zr_b,
    )

    dt_min = (composite_now.timestamp_utc - composite_prev.timestamp_utc).total_seconds() / 60.0
    dt_min = max(dt_min, 1.0)

    # Truth animation (the actual observed rain at each lead that was cached)
    truth_loop = [
        LoopFrame(rain_rate_mm_h=rain_now, timestamp_utc=composite_now.timestamp_utc,
                  label="obs", kind="observed", duration_ms=1200),
    ]
    for lead, c in truth_pairs:
        rr = dbz_to_rain_rate(c.reflectivity_dbz, zr_a=c.zr_a, zr_b=c.zr_b)
        truth_loop.append(LoopFrame(
            rain_rate_mm_h=rr, timestamp_utc=c.timestamp_utc,
            label=f"+{lead} min", kind="observed", duration_ms=600,
        ))

    out: dict[str, str] = {}
    base_label = scenario.event_time.strftime("%Y-%m-%d %H:%M UTC")
    _render_animation_for_scenario(
        f"truth — {base_label}", truth_loop, composite_now, geo,
        output_dir / f"{scenario.label}_truth.png",
    )
    out["truth"] = f"{scenario.label}_truth.png"

    for method_name, flow_fn in (("tvl1", _flow_tvl1), ("farneback", _flow_farneback)):
        try:
            vy, vx = flow_fn(composite_prev.reflectivity_dbz, composite_now.reflectivity_dbz)
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning("%s flow failed for %s: %s", method_name, scenario.label, exc)
            out[method_name] = None
            continue
        loop = [
            LoopFrame(rain_rate_mm_h=rain_now, timestamp_utc=composite_now.timestamp_utc,
                      label="obs", kind="observed", duration_ms=1200),
        ]
        for lead in FORECAST_LEADS_MIN:
            field = advect_field(rain_now, vy, vx, horizon_minutes=float(lead), dt_minutes=dt_min)
            loop.append(LoopFrame(
                rain_rate_mm_h=field,
                timestamp_utc=composite_now.timestamp_utc + timedelta(minutes=lead),
                label=f"+{lead} min", kind="forecast", duration_ms=600,
            ))
        # Speed/direction caption per method
        # Disc-area rain-weighted to mirror what the integration shows.
        from dmi_nowcast_core.transform import dbz_to_rain_rate as _drr  # alias
        rain_curr = _drr(composite_now.reflectivity_dbz, zr_a=composite_now.zr_a, zr_b=composite_now.zr_b)
        home_grid = geo.lonlat_to_grid(REFERENCE_LON, REFERENCE_LAT)
        hr, hc = int(round(home_grid.row)), int(round(home_grid.col))
        search_px = int(round(120_000 / composite_now.xscale_m))
        rs = slice(max(0, hr - search_px), min(rain_curr.shape[0], hr + search_px + 1))
        cs = slice(max(0, hc - search_px), min(rain_curr.shape[1], hc + search_px + 1))
        lr, lvy, lvx = rain_curr[rs, cs], vy[rs, cs], vx[rs, cs]
        weights = np.where(
            np.isfinite(lr) & (lr > 0.1) & np.isfinite(lvy) & np.isfinite(lvx),
            lr, 0.0,
        )
        w_sum = float(weights.sum())
        if w_sum > 0:
            dvy = float((lvy * weights).sum() / w_sum) / max(dt_min, 1.0)
            dvx = float((lvx * weights).sum() / w_sum) / max(dt_min, 1.0)
        else:
            dvy = dvx = 0.0
        # Same subline as integration
        subline = (
            f"{method_name.upper()}  ·  disc dy={dvy:+.2f} px/min  dx={dvx:+.2f} px/min"
        )
        out_path = output_dir / f"{scenario.label}_{method_name}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        png = render_loop_png(
            loop, composite=composite_now, geo=geo,
            home_lat=REFERENCE_LAT, home_lon=REFERENCE_LON,
            radius_km=1.0, zoom_km=100.0, output_px=500,
            now_stats_subline=subline,
            motion_dy_per_min=dvy, motion_dx_per_min=dvx,
        )
        out_path.write_bytes(png)
        out[method_name] = f"{scenario.label}_{method_name}.png"

    out["event_time"] = scenario.event_time.isoformat()
    out["label"] = scenario.label
    return out


# ---------------------------------------------------------------------------
# Tiny HTTP dashboard
# ---------------------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    server_version = "FlowCompare/0.1"
    output_dir: Path = Path(".")
    manifest_path: Path = Path(".")

    def do_GET(self):  # noqa: N802
        if self.path in ("/", "/index.html"):
            self._serve_index()
            return
        # Serve files under output_dir
        rel = self.path.lstrip("/")
        target = self.output_dir / rel
        if target.is_file():
            self.send_response(200)
            ct = "image/png" if target.suffix == ".png" else "application/octet-stream"
            self.send_header("Content-Type", ct)
            data = target.read_bytes()
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):  # noqa: A002
        return  # quiet

    def _serve_index(self):
        manifest = json.loads(self.manifest_path.read_text())
        rows_html = []
        for s in manifest["scenarios"]:
            cells = []
            for method in ("truth", "tvl1", "farneback"):
                file = s.get(method)
                if file:
                    cells.append(f'<td><img src="/{file}" alt="{method}"></td>')
                else:
                    cells.append(f'<td><em>—</em></td>')
            rows_html.append(
                f'<tr><th>{s["label"]}<br><small>{s["event_time"][:16]}</small></th>'
                + "".join(cells) + "</tr>"
            )
        body = f"""<!doctype html>
<html><head>
<meta charset="utf-8">
<title>Flow comparison</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; padding: 16px; background:#fafafa; }}
  table {{ border-collapse: collapse; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  th, td {{ padding: 8px; text-align: center; border-bottom: 1px solid #eee; vertical-align: middle; }}
  th {{ background: #f4f4f4; font-weight: 600; min-width: 90px; }}
  img {{ width: 380px; height: 380px; display: block; }}
  small {{ color: #888; font-weight: normal; }}
  h1 {{ margin: 0 0 4px 0; }}
  .legend {{ color: #555; margin-bottom: 16px; font-size: 0.9em; }}
</style>
</head><body>
<h1>Optical-flow comparison · Odense, 1 km disc</h1>
<div class="legend">
  Each row is one scenario. Columns: <b>truth</b> (observed evolution),
  <b>tvl1</b> (skimage TV-L1, what HA uses), <b>farneback</b>
  (opencv reference, dev only). Animations play automatically and loop.
  Generated at {manifest.get("generated_at", "?")}.
</div>
<table>
  <thead>
    <tr><th>Scenario</th><th>Truth</th><th>TV-L1</th><th>Farnebäck</th></tr>
  </thead>
  <tbody>
    {''.join(rows_html)}
  </tbody>
</table>
</body></html>"""
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", type=Path, default=Path("radar_archive"))
    ap.add_argument("--output", type=Path, default=Path("/tmp/flow_compare"))
    ap.add_argument("--n-scenarios", type=int, default=4,
                    help="historical scenarios; latest live is added on top")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--no-server", action="store_true",
                    help="generate artifacts only, don't launch the dashboard")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args.output.mkdir(parents=True, exist_ok=True)

    _LOGGER.info("Step 1: classify Odense wet/dry across archive")
    wet_index = _build_odense_wet_index(args.archive, sample_every_n=2)
    n_wet = sum(wet_index.values())
    _LOGGER.info("  %d archive frames classified, %d wet at Odense", len(wet_index), n_wet)

    _LOGGER.info("Step 2: find rain-imminent events (dry→wet within 20 min)")
    imminent = _find_imminent_rain_events(wet_index, lead_min=20, tol_min=5)
    _LOGGER.info("  %d candidate events", len(imminent))
    if len(imminent) < args.n_scenarios:
        _LOGGER.warning("only %d candidates, using all of them", len(imminent))

    rng = random.Random(args.seed)
    scenarios: list[Scenario] = []
    # Oversample 3x — many candidates will have sparse verification coverage;
    # this lets us keep going until we have n_scenarios actually-renderable ones.
    pool = rng.sample(imminent, len(imminent))
    for ev in pool:
        if len(scenarios) >= args.n_scenarios:
            break
        result = _gather_scenario_frames(ev, args.archive)
        if result is None:
            _LOGGER.warning("no verification frames for %s — skipping", ev)
            continue
        frames_in, frames_truth = result
        scenarios.append(Scenario(
            label=f"hist_{ev.strftime('%Y%m%d_%H%M')}",
            event_time=ev,
            frames_in=frames_in,
            frames_truth=frames_truth,
        ))
        _LOGGER.info("  kept %s (verification leads: %s)",
                     ev.isoformat(timespec="minutes"),
                     [l for l, _ in frames_truth])
    # Sort the kept set chronologically for stable dashboard order.
    scenarios.sort(key=lambda s: s.event_time)

    # Add "latest" — pick the two most-recent archive frames as input,
    # but no truth verification (we don't have the future yet).
    files_by_ts = sorted(
        [(p, _parse_archive_filename(p)) for p in args.archive.glob("dk.com.*.h5")],
        key=lambda x: x[1] or datetime.min.replace(tzinfo=timezone.utc),
    )
    if len(files_by_ts) >= 2:
        latest_curr_p, latest_curr_t = files_by_ts[-1]
        # Need a 10-min-earlier frame
        target_prev = latest_curr_t - timedelta(minutes=10)
        latest_prev_p = None
        for p, t in reversed(files_by_ts[:-1]):
            if abs(t - target_prev) < timedelta(minutes=3):
                latest_prev_p = p
                break
        if latest_prev_p is not None:
            # For "latest" we approximate truth using whatever frames we
            # have after curr — should usually be empty (it's the latest)
            scenarios.insert(0, Scenario(
                label="latest_live",
                event_time=latest_curr_t,
                frames_in=[latest_prev_p, latest_curr_p],
                frames_truth=[],  # no future data yet
            ))

    _LOGGER.info("Step 3: render %d scenarios", len(scenarios))
    scenario_outputs = []
    for sc in scenarios:
        _LOGGER.info("  rendering %s", sc.label)
        try:
            scenario_outputs.append(_render_scenario(sc, args.output))
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning("failed to render %s: %s", sc.label, exc)
            import traceback; traceback.print_exc()

    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_scenarios": len(scenario_outputs),
        "scenarios": scenario_outputs,
    }, indent=2))
    _LOGGER.info("manifest → %s", manifest_path)

    if args.no_server:
        return 0
    _Handler.output_dir = args.output
    _Handler.manifest_path = manifest_path
    _LOGGER.info("Dashboard at http://localhost:%d/", args.port)
    try:
        ThreadingHTTPServer(("127.0.0.1", args.port), _Handler).serve_forever()
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
