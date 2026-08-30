"""Live dashboard for the calibration corpus builder.

Runs a tiny HTTP server (stdlib only — no Flask/FastAPI dep) that reads
the progress JSON written by ``build_calibration_corpus.py`` and renders
an auto-refreshing HTML page::

    python scripts/calibration_dashboard.py --port 8765 \\
        --progress /tmp/calib_progress.json \\
        --parquet reports/calibration_corpus.parquet

Then open http://localhost:8765 in a browser. The page reloads itself
every 5 s. No HTML framework — just an inline template so the script
stays a single file.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from collections import Counter
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    import pyarrow.parquet as pq  # type: ignore
except ImportError:
    pq = None  # type: ignore


PROGRESS_PATH = Path("/tmp/calib_progress.json")
PARQUET_PATH = Path("reports/calibration_corpus.parquet")


def _load_progress() -> dict:
    if not PROGRESS_PATH.exists():
        return {"_status": "waiting for corpus builder to start"}
    try:
        return json.loads(PROGRESS_PATH.read_text())
    except Exception as exc:  # noqa: BLE001
        return {"_status": f"progress file unreadable: {exc}"}


def _parquet_summary() -> dict:
    if pq is None:
        return {"rows": "(pyarrow not installed)"}
    if not PARQUET_PATH.exists():
        return {"rows": 0}
    try:
        tbl = pq.read_table(PARQUET_PATH)
        return {
            "rows": tbl.num_rows,
            "unique_events": len(set(tbl.column("event_time").to_pylist())),
            "leads": sorted(set(tbl.column("lead_min").to_pylist())),
        }
    except Exception as exc:  # noqa: BLE001
        return {"rows": f"(read error: {exc})"}


def _histogram(values: list[float], bins: int = 10) -> list[tuple[float, float, int]]:
    """Return [(lo, hi, count), ...] for an even-spaced histogram on [0, 1]."""
    if not values:
        return []
    out: list[tuple[float, float, int]] = []
    width = 1.0 / bins
    for i in range(bins):
        lo = i * width
        hi = lo + width
        count = sum(1 for v in values if lo <= v < hi or (i == bins - 1 and v == 1.0))
        out.append((lo, hi, count))
    return out


def _per_lead_histograms() -> dict[int, list[tuple[float, float, int]]]:
    """Read the Parquet and compute per-lead histograms of raw_prob."""
    if pq is None or not PARQUET_PATH.exists():
        return {}
    try:
        tbl = pq.read_table(PARQUET_PATH, columns=["lead_min", "raw_prob", "outcome"])
        leads = tbl.column("lead_min").to_pylist()
        raws = tbl.column("raw_prob").to_pylist()
        outs = tbl.column("outcome").to_pylist()
    except Exception:  # noqa: BLE001
        return {}
    by_lead: dict[int, list[float]] = {}
    base_rate: dict[int, list[int]] = {}
    for lead, raw, out in zip(leads, raws, outs):
        if raw is None or raw != raw:  # NaN
            continue
        by_lead.setdefault(lead, []).append(float(raw))
        if out in (0, 1):
            base_rate.setdefault(lead, []).append(int(out))
    hists = {lead: _histogram(vals) for lead, vals in by_lead.items()}
    return hists, base_rate  # type: ignore


def _render_html(progress: dict, parquet_info: dict) -> str:
    """Inline HTML template. Auto-refreshes every 5 s via <meta>."""
    now = datetime.now().isoformat(timespec="seconds")
    completed = progress.get("completed", 0)
    total = progress.get("todo", progress.get("total", 0))
    pct = (100.0 * completed / total) if total else 0.0
    errored = progress.get("errored", 0)
    eta = progress.get("eta_min")
    eta_str = f"{eta:.1f} min" if eta else "—"
    rate = progress.get("events_per_min", 0)
    elapsed = progress.get("elapsed_s", 0)
    lead_summary = progress.get("lead_summary", {})
    recent = progress.get("recent", []) or []

    try:
        hists, base_rates = _per_lead_histograms()
    except Exception:
        hists, base_rates = {}, {}

    rows_html = []
    for ev in reversed(recent[-15:]):
        raws = ev.get("raw", [])
        outs = ev.get("outcomes", [])
        err = ev.get("error")
        cells = ""
        if err:
            cells = f'<td colspan="3" style="color:#c33;">{html.escape(str(err))}</td>'
        else:
            cells = "".join(
                f'<td>{r*100:.0f}% → {"yes" if o == 1 else ("no" if o == 0 else "—")}</td>'
                for r, o in zip(raws, outs)
            )
        rows_html.append(f'<tr><td><code>{html.escape(ev["event_time"])}</code></td>{cells}</tr>')
    rows_html_str = "\n".join(rows_html) or '<tr><td colspan="4"><em>(no events yet)</em></td></tr>'

    # Per-lead histograms as ASCII bars.
    hist_blocks = []
    for lead in sorted(hists.keys()):
        rows = hists[lead]
        max_count = max((c for _, _, c in rows), default=1)
        bars = []
        for lo, hi, c in rows:
            bar = "█" * int(round(40 * c / max_count)) if c else ""
            bars.append(f"  {lo*100:3.0f}-{hi*100:3.0f}%  {bar} {c}")
        outs_for_lead = base_rates.get(lead, []) if isinstance(base_rates, dict) else []
        br = (sum(outs_for_lead) / len(outs_for_lead)) if outs_for_lead else 0.0
        hist_blocks.append(
            f'<div class="hist"><h3>Lead +{lead} min  ·  base rate {br*100:.1f}%</h3>'
            f'<pre>{"\n".join(bars)}</pre></div>'
        )
    hist_html = "\n".join(hist_blocks) or "<em>(no data yet)</em>"

    # Lead summary cards from the live progress JSON.
    cards = []
    for lead_s, info in (lead_summary or {}).items():
        n = info.get("n_finite", 0)
        mean = info.get("mean_raw")
        br = info.get("base_rate")
        cards.append(
            f'<div class="card">'
            f'<h4>+{lead_s} min</h4>'
            f'<div>n: {n}</div>'
            f'<div>mean raw: {mean*100:.1f}%</div>' if mean is not None else '<div>mean raw: —</div>'
        )
        cards.append(
            f'<div>obs rate: {br*100:.1f}%</div>' if br is not None else '<div>obs rate: —</div>'
        )
        cards.append('</div>')
    cards_html = "".join(cards) or "<em>(stats collecting…)</em>"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="5">
  <title>Calibration corpus progress</title>
  <style>
    body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 1100px; margin: 2em auto; padding: 0 1em; color: #222; }}
    h1 {{ font-size: 1.4em; margin-bottom: 0.2em; }}
    .muted {{ color: #888; font-size: 0.9em; }}
    .bar {{ background: #eee; border-radius: 4px; overflow: hidden; height: 24px; margin: 0.5em 0; }}
    .bar-fill {{ background: linear-gradient(90deg, #4a90e2, #2c5fbb); height: 100%; transition: width 0.3s; }}
    .stats {{ display: flex; gap: 1em; margin: 1em 0; }}
    .card {{ background: #fafafa; border: 1px solid #ddd; border-radius: 6px; padding: 0.5em 1em; min-width: 120px; }}
    .card h4 {{ margin: 0 0 0.3em 0; font-size: 1em; }}
    .hist pre {{ background: #f7f7f7; padding: 0.8em; border-radius: 6px; font-size: 0.85em; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.85em; }}
    td, th {{ border-bottom: 1px solid #eee; padding: 0.3em 0.6em; text-align: left; }}
    code {{ font-size: 0.85em; color: #555; }}
    section {{ margin: 1.5em 0; }}
  </style>
</head>
<body>
<h1>Calibration corpus builder</h1>
<div class="muted">page auto-refreshes every 5 s · last rendered {now}</div>

<section>
  <strong>{completed} / {total}</strong> events processed
  &nbsp;·&nbsp; {pct:.1f}%
  &nbsp;·&nbsp; {errored} errored
  &nbsp;·&nbsp; rate {rate} ev/min
  &nbsp;·&nbsp; ETA {eta_str}
  &nbsp;·&nbsp; elapsed {elapsed:.0f} s
  <div class="bar"><div class="bar-fill" style="width: {pct:.1f}%"></div></div>
</section>

<section>
  <h2>Per-lead progress</h2>
  <div class="stats">{cards_html}</div>
</section>

<section>
  <h2>Raw probability histograms (from Parquet)</h2>
  {hist_html}
</section>

<section>
  <h2>Recent events</h2>
  <table>
    <thead><tr><th>Event time</th><th>+10 min raw → outcome</th><th>+30 min</th><th>+60 min</th></tr></thead>
    <tbody>
      {rows_html_str}
    </tbody>
  </table>
</section>

<section class="muted">
  <h3>Parquet</h3>
  <pre>{html.escape(json.dumps(parquet_info, indent=2))}</pre>
</section>
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path not in ("/", "/index.html"):
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"not found")
            return
        body = _render_html(_load_progress(), _parquet_summary()).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # noqa: A002
        # Silence the default access log so the dashboard doesn't drown
        # the corpus builder's terminal output.
        return


def main() -> int:
    global PROGRESS_PATH, PARQUET_PATH
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--progress", type=Path, default=PROGRESS_PATH)
    ap.add_argument("--parquet", type=Path, default=PARQUET_PATH)
    args = ap.parse_args()
    PROGRESS_PATH = args.progress
    PARQUET_PATH = args.parquet
    print(f"Dashboard: http://localhost:{args.port}/")
    print(f"  progress: {PROGRESS_PATH}")
    print(f"  parquet:  {PARQUET_PATH}")
    try:
        ThreadingHTTPServer(("127.0.0.1", args.port), _Handler).serve_forever()
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
