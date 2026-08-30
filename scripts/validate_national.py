#!/usr/bin/env python3
"""Live validation: national products must agree with the point forecast.

Post-deploy companion to ``sidecar/tests/test_validation_archive.py``:
hits a RUNNING sidecar and cross-checks its three surfaces against each
other —

- ``/state.json``          (the configured reference point's forecast)
- ``/forecast?lat=&lon=``  (point lookup into the in-memory national grids)
- ``/nowcast/manifest.json`` + one quantised ``p_rain`` PNG artifact

Checks (one row each in the pass/fail table):

- per-lead ``p_ensemble`` (state) vs ``p_rain`` (/forecast) — exact match
  expected; the max abs diff is reported,
- ``eta_p50_window_min`` (state) vs ``eta_min`` (/forecast) — the ETA must
  lie within the window extended by one timestep on each side; the
  minority-member caveat (window present, ETA null, exceedance < 0.5) and
  the both-null dry case are accepted per the documented semantics,
- ``n_members`` / ``calibrated`` consistency across surfaces,
- radar timestamps consistent across state / forecast / manifest,
- PNG round-trip: the ``p_rain`` artifact for one lead, dequantised with
  the manifest's scale/offset and sampled at the reference pixel through
  the manifest's grid-geometry block (proj4 + UL corner + pixel scale —
  needs ``pyproj``; the check is SKIPped with a note when it isn't
  importable), must match /forecast within half a quantisation step.

It also prints the diagnostics block (``ensemble_ms``, ``national_ms``,
``artifact_bytes``, ``cycle_ms``) — the deployment budget gates read from
these.

Usage::

    # --base-url (or DMI_NOWCAST_BASE_URL) points at the running service
    python scripts/validate_national.py --base-url http://localhost:8081
    python scripts/validate_national.py --token SECRET        # bearer auth
    python scripts/validate_national.py --lat 55.4 --lon 10.4 # other point

By default the reference coordinates are read from ``/state.json``'s home
block, so the script validates exactly the pixel the cross-surface
guarantee is about. Exit code 0 when every check passes, 1 otherwise
(including the "sidecar reachable but no cycle completed yet" 503 case,
which gets its own message).

Dependencies: stdlib + numpy + Pillow (both already in the sidecar venv);
pyproj is optional and only gates the PNG reference-pixel check.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime

try:
    import numpy as np
except ImportError:  # pragma: no cover - environment guard
    sys.exit(
        "validate_national.py needs numpy (pip install numpy, or run it "
        "with the sidecar venv: sidecar/.venv/bin/python scripts/validate_national.py)"
    )
try:
    from PIL import Image
except ImportError:  # pragma: no cover - environment guard
    sys.exit(
        "validate_national.py needs Pillow (pip install Pillow, or run it "
        "with the sidecar venv: sidecar/.venv/bin/python scripts/validate_national.py)"
    )

# No baked-in host: point the script at your own deployment with
# --base-url, or export DMI_NOWCAST_BASE_URL (see .env.example).
DEFAULT_BASE_URL = os.environ.get("DMI_NOWCAST_BASE_URL", "http://localhost:8081")
NODATA_LEVEL = 255  # quantised-PNG nodata level, per national_artifacts.py
HTTP_TIMEOUT_S = 15.0


class Report:
    """Collects (status, check, detail) rows; renders the plain-text table."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, ok: bool, check: str, detail: str) -> None:
        self.rows.append(("PASS" if ok else "FAIL", check, detail))

    def skip(self, check: str, detail: str) -> None:
        self.rows.append(("SKIP", check, detail))

    @property
    def failed(self) -> bool:
        return any(status == "FAIL" for status, _, _ in self.rows)

    def render(self) -> str:
        width = max((len(check) for _, check, _ in self.rows), default=0)
        lines = ["", "live validation — national grids vs point forecast", "=" * 72]
        for status, check, detail in self.rows:
            lines.append(f"  {status:<4}  {check:<{width}}  {detail}")
        lines.append("=" * 72)
        lines.append("RESULT: " + ("FAIL" if self.failed else "PASS (all checks)"))
        return "\n".join(lines)


def _fetch(base_url: str, path: str, token: str | None) -> bytes:
    req = urllib.request.Request(base_url.rstrip("/") + path)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
        return resp.read()


def _fetch_json(base_url: str, path: str, token: str | None) -> dict:
    return json.loads(_fetch(base_url, path, token))


def _parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _check_per_lead(report: Report, state: dict, forecast: dict) -> None:
    """state per_lead p_ensemble vs /forecast per_lead p_rain, shared leads."""
    state_p = {
        e["lead_min"]: e.get("p_ensemble")
        for e in state.get("forecast", {}).get("per_lead", [])
    }
    fc_p = {e["lead_min"]: e.get("p_rain") for e in forecast.get("per_lead", [])}
    shared = sorted(set(state_p) & set(fc_p))
    if not shared:
        report.add(False, "per-lead p_ensemble vs p_rain",
                   "no shared leads between state.json and /forecast")
        return
    if any(state_p[lead] is None for lead in shared):
        report.add(
            False, "per-lead p_ensemble vs p_rain",
            "p_ensemble is null — the STEPS ensemble is not running "
            "(steps.enabled? check the sidecar log for ensemble_failed)",
        )
        return
    diffs = []
    for lead in shared:
        sp, fp = state_p[lead], fc_p[lead]
        diffs.append(abs(sp - fp) if fp is not None else math.inf)
    max_diff = max(diffs)
    exact = sum(1 for d in diffs if d == 0.0)
    # Tolerance 1e-6, not exact: the national grid is float32 while the
    # home aggregate is float64. With a power-of-two member count every
    # fraction k/N is dyadic and both sides are bit-equal, but e.g. 24
    # members yields fractions like 5/24 that differ at ~1e-8 between the
    # two precisions. 1e-6 stays far below half a quantisation step.
    report.add(
        max_diff <= 1e-6,
        "per-lead p_ensemble vs p_rain",
        f"leads {shared}: {exact}/{len(shared)} exact, max |diff| = {_fmt(max_diff)}",
    )


def _check_eta(report: Report, state: dict, forecast: dict,
               timestep_min: float) -> None:
    """eta_p50_window_min (state) vs eta_min (national pixel)."""
    window = state.get("forecast", {}).get("eta_p50_window_min")
    eta = forecast.get("eta_min")
    check = "eta_min vs eta_p50_window_min"
    if window is None and eta is None:
        report.add(True, check, "both null — no rain expected on either side")
    elif window is None:
        report.add(
            False, check,
            f"national ETA {eta} min but home window is null "
            "(no member crossed at home — should imply exceedance < 0.5)",
        )
    elif eta is None:
        # Minority-member caveat: national ETA is conservatively NaN when
        # fewer than half the members ever cross, while the home window may
        # still summarise the crossing minority. Verify via the largest lead.
        p_last = None
        per_lead = forecast.get("per_lead") or []
        if per_lead:
            p_last = per_lead[-1].get("p_rain")
        ok = p_last is not None and p_last < 0.5
        report.add(
            ok, check,
            f"window {window} but national ETA null — minority-member case; "
            f"exceedance at max lead = {_fmt(p_last)} (must stay < 0.5)",
        )
    else:
        lo, hi = float(window[0]), float(window[1])
        ok = (lo - timestep_min) <= float(eta) <= (hi + timestep_min)
        report.add(
            ok, check,
            f"ETA {eta} min vs window ({lo}, {hi}) ± {timestep_min} min",
        )


def _grid_index(grid: dict, lat: float, lon: float) -> tuple[int, int] | None:
    """Home pixel via the manifest's grid geometry; None if pyproj is missing.

    The two-line projection: lon/lat → projection metres with the grid's
    proj4 string, then ``col = (x - x_ul) / scale_x``,
    ``row = (y_ul - y) / scale_y`` (rows grow southward).
    """
    try:
        from pyproj import CRS, Transformer
    except ImportError:
        return None
    transformer = Transformer.from_crs(
        "EPSG:4326", CRS.from_proj4(grid["proj4"]), always_xy=True,
    )
    x, y = transformer.transform(lon, lat)
    col = int(round((x - grid["x_ul_m"]) / grid["pixel_scale_x_m"]))
    row = int(round((grid["y_ul_m"] - y) / grid["pixel_scale_y_m"]))
    return row, col


def _check_png(report: Report, base_url: str, token: str | None,
               manifest: dict, forecast: dict, lat: float, lon: float) -> None:
    """Dequantise one p_rain PNG, sample the home pixel, compare to /forecast."""
    check = "p_rain PNG round-trip at home pixel"
    fc_p = {e["lead_min"]: e.get("p_rain") for e in forecast.get("per_lead", [])}
    candidates = [
        a for a in manifest.get("artifacts", [])
        if a.get("product") == "p_rain" and a.get("lead_min") in fc_p
    ]
    if not candidates:
        report.add(False, check, "no p_rain artifact for a /forecast lead in manifest")
        return
    # Prefer the wettest lead so the round-trip exercises non-zero levels.
    entry = max(
        candidates,
        key=lambda a: (fc_p[a["lead_min"]] is not None, fc_p[a["lead_min"]] or 0.0),
    )
    index = _grid_index(manifest["grid"], lat, lon)
    if index is None:
        report.skip(
            check,
            "pyproj not importable — cannot project lat/lon onto the grid; "
            "run with the sidecar venv (it ships pyproj) to enable this check",
        )
        return
    row, col = index
    h, w = manifest["grid"]["shape"]
    if not (0 <= row < h and 0 <= col < w):
        report.add(False, check, f"home pixel ({row}, {col}) outside grid {h}x{w}")
        return
    png = _fetch(base_url, f"/nowcast/{entry['filename']}", token)
    levels = np.asarray(Image.open(io.BytesIO(png)))
    if levels.shape != (h, w):
        report.add(False, check,
                   f"{entry['filename']}: shape {levels.shape} != manifest {h}x{w}")
        return
    level = int(levels[row, col])
    png_val = (
        math.nan if level == entry.get("nodata", NODATA_LEVEL)
        else level * entry["scale"] + entry["offset"]
    )
    fc_val = fc_p[entry["lead_min"]]
    half_step = entry["scale"] / 2.0
    if fc_val is None:
        ok = math.isnan(png_val)
        detail = f"{entry['filename']} pixel ({row}, {col}): forecast null, PNG {png_val}"
    elif math.isnan(png_val):
        ok = False
        detail = f"{entry['filename']} pixel ({row}, {col}): PNG nodata, forecast {fc_val}"
    else:
        err = abs(png_val - fc_val)
        ok = err <= half_step + 1e-6
        detail = (
            f"{entry['filename']} pixel ({row}, {col}): PNG {_fmt(png_val)} vs "
            f"forecast {_fmt(fc_val)}, |diff| {_fmt(err)} (half step {_fmt(half_step)})"
        )
    report.add(ok, check, detail)


def _check_calibration(report: Report, state: dict, forecast: dict,
                       manifest: dict) -> None:
    """Calibration story consistent across state/forecast/manifest (§B5).

    The three surfaces may each be calibrated or raw (home and national
    lead sets differ, so per-lead curve coverage can too), but their
    stories must agree: identical ``fitted_at`` wherever calibration is
    claimed, no flag claiming calibration while the manifest serves raw
    grids, and — before the first national fit — everything uniformly
    uncalibrated.
    """
    check = "calibration consistency"
    prob = state.get("probabilistic") or {}
    man_cal = manifest.get("calibration")
    st_fit = prob.get("calibration_fitted_at")
    fc_fit = forecast.get("calibration_fitted_at")
    st_flag = prob.get("calibrated")
    fc_flag = forecast.get("calibrated")

    if man_cal is None and st_fit is None and fc_fit is None:
        ok = st_flag is False and fc_flag is False
        report.add(
            ok, check,
            "uncalibrated everywhere (national curves not fitted yet)"
            if ok else
            f"nothing served calibrated, but flags claim state={st_flag} "
            f"forecast={fc_flag}",
        )
        return

    fits = {
        "state": st_fit,
        "forecast": fc_fit,
        "manifest": (man_cal or {}).get("fitted_at"),
    }
    # Compare as instants, not strings — the three surfaces serialise the
    # same timestamp differently ("...Z" vs "...+00:00").
    present = {k: _parse_ts(v) for k, v in fits.items() if v is not None}
    ok = bool(present) and len(set(present.values())) == 1
    if fc_flag and man_cal is None:
        ok = False  # /forecast claims calibration but the grids are raw
    detail = ", ".join(f"{k}={v or 'null'}" for k, v in fits.items())
    if man_cal is not None:
        detail += f"; calibrated_leads={man_cal.get('calibrated_leads')}"
    report.add(ok, check, detail)

    # Calibrated probabilities must stay probabilities.
    ps = [e.get("p_rain") for e in forecast.get("per_lead", [])]
    ps += [
        e.get("p_ensemble")
        for e in state.get("forecast", {}).get("per_lead", [])
    ]
    finite = [p for p in ps if p is not None]
    in_range = all(0.0 <= p <= 1.0 for p in finite)
    report.add(
        in_range, "calibrated probabilities in [0, 1]",
        f"{len(finite)} values checked"
        + ("" if in_range else f"; offenders: {[p for p in finite if not 0.0 <= p <= 1.0]}"),
    )


def _print_diagnostics(state: dict) -> None:
    diag = state.get("diagnostics") or {}
    print("\nDiagnostics (§A5 budget gates):")
    for key in ("cycle_ms", "fetch_ms", "compute_ms", "render_ms",
                "ensemble_ms", "national_ms", "artifact_bytes"):
        print(f"  {key:<16} {_fmt(diag.get(key, 'n/a'))}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a running sidecar's national products "
                    "against its point forecast.",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help="sidecar base URL (default: $DMI_NOWCAST_BASE_URL "
                             f"or {DEFAULT_BASE_URL})")
    parser.add_argument("--token", default=None,
                        help="bearer token for /forecast and /nowcast/* "
                             "(only needed when server.api_key is set)")
    parser.add_argument("--lat", type=float, default=None,
                        help="latitude to validate (default: /state.json home)")
    parser.add_argument("--lon", type=float, default=None,
                        help="longitude to validate (default: /state.json home)")
    args = parser.parse_args(argv)

    report = Report()
    try:
        state = _fetch_json(args.base_url, "/state.json", args.token)
        home = state.get("home", {})
        lat = args.lat if args.lat is not None else home.get("lat")
        lon = args.lon if args.lon is not None else home.get("lon")
        if lat is None or lon is None:
            print("state.json has no home block and no --lat/--lon given")
            return 1
        forecast = _fetch_json(
            args.base_url, f"/forecast?lat={lat}&lon={lon}", args.token,
        )
        manifest = _fetch_json(args.base_url, "/nowcast/manifest.json", args.token)
    except urllib.error.HTTPError as exc:
        if exc.code == 503:
            print(
                f"sidecar at {args.base_url} is up but has no completed "
                f"national cycle yet (503 on {exc.url}) — wait one poll "
                "interval (~5 min) and retry; check steps.enabled / "
                "national.enabled if it persists"
            )
        else:
            print(f"HTTP {exc.code} from {exc.url}: {exc.reason}")
        return 1
    except urllib.error.URLError as exc:
        print(
            f"cannot reach sidecar at {args.base_url}: {exc.reason}\n"
            "(check --base-url / $DMI_NOWCAST_BASE_URL, and that anything "
            "between you and the host — VPN, firewall — allows it)"
        )
        return 1

    print(f"sidecar:  {args.base_url}")
    print(f"point:    lat={lat} lon={lon}"
          + ("  (from /state.json home block)" if args.lat is None else ""))
    print(f"radar ts: {state.get('radar', {}).get('latest_ts')}"
          f"  (age {state.get('radar', {}).get('data_age_minutes')} min)")

    # --- consistency checks -------------------------------------------------
    prob = state.get("probabilistic")
    if prob is None:
        report.add(False, "probabilistic block present",
                   "state.json has no probabilistic block — ensemble not running")
    else:
        report.add(True, "probabilistic block present",
                   f"n_members={prob.get('n_members')}")
        report.add(
            prob.get("n_members") == forecast.get("n_members"),
            "n_members consistent",
            f"state {prob.get('n_members')} vs forecast {forecast.get('n_members')}",
        )
        _check_calibration(report, state, forecast, manifest)

    state_ts = _parse_ts(state["radar"]["latest_ts"])
    fc_ts = _parse_ts(forecast["radar_ts_utc"])
    man_ts = _parse_ts(manifest["radar_ts_utc"])
    report.add(
        state_ts == fc_ts == man_ts,
        "radar timestamps consistent",
        f"state {state_ts.isoformat()} / forecast {fc_ts.isoformat()} / "
        f"manifest {man_ts.isoformat()}"
        + ("" if state_ts == fc_ts == man_ts
           else " — a cycle may have rolled between requests; rerun to confirm"),
    )

    _check_per_lead(report, state, forecast)
    # The ETA-window slack is ± one STEPS timestep, and since C0 the
    # timestep is derived from the frame spacing (~10 min fullRange
    # cadence) — so it MUST come from the served manifest, never a
    # hardcoded default. A manifest without it (pre-C0 sidecar, corrupt
    # write) is a failure, not something to paper over with 5.0.
    ts_min = manifest.get("timestep_min")
    if ts_min is None:
        report.add(False, "eta_min vs eta_p50_window_min",
                   "manifest has no timestep_min — cannot size the ETA "
                   "window slack (pre-C0 sidecar?)")
    else:
        _check_eta(report, state, forecast, timestep_min=float(ts_min))
    _check_png(report, args.base_url, args.token, manifest, forecast, lat, lon)

    print(report.render())
    _print_diagnostics(state)
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
