"""Render the lightning-collection monitoring dashboard as one self-contained
HTML page.

Stat cards + a CSS per-day bar chart + a Leaflet heat-map of every archived
strike with the anchor circles. Leaflet + leaflet.heat come from a CDN (loaded
by the browser); the strike points + per-day counts are injected server-side, so
the produced file is fully functional when mirrored to HA's ``/local/`` and
embedded in an iframe. No Python plotting dependency.
"""
from __future__ import annotations

import json

# Fixed anchors to draw (name, lat, lon, radius_km) — the five Danish
# collection entries, whose 100 km circles tile the country. Moving targets
# are dynamic and aren't drawn here. Alpine anchors were dropped on
# 2026-08-14 when continuous Alps collection stopped; re-add them here if
# collection ever resumes. The archived Alpine strikes stay either way —
# this list is what to draw, not what has been collected.
ANCHORS: list[tuple[str, float, float, float]] = [
    ("Fyn", 55.33, 10.32, 100.0),
    ("Aalborg", 57.05, 9.92, 100.0),
    ("Silkeborg", 56.17, 9.55, 100.0),
    ("Esbjerg", 55.47, 8.46, 100.0),
    ("Zealand (Ringsted)", 55.45, 11.80, 100.0),
]

_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lightning collection</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>
<style>
  :root { color-scheme: light dark; }
  body { font-family: system-ui, sans-serif; margin: 0; padding: 12px;
         background: #f6f7f9; color: #1b1b1f; }
  @media (prefers-color-scheme: dark){ body{ background:#15171c; color:#e6e6e6; } .card,.panel{ background:#1f232b !important; } }
  h1 { font-size: 1.05rem; margin: 0 0 2px; }
  .sub { font-size: .8rem; opacity: .65; margin-bottom: 12px; }
  .cards { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 14px; }
  .card { background: #fff; border-radius: 10px; padding: 10px 14px; min-width: 92px;
          box-shadow: 0 1px 3px rgba(0,0,0,.08); }
  .card .v { font-size: 1.5rem; font-weight: 650; }
  .card .l { font-size: .72rem; opacity: .65; text-transform: uppercase; letter-spacing: .03em; }
  .panel { background:#fff; border-radius:10px; padding:12px 14px; margin-bottom:14px;
           box-shadow:0 1px 3px rgba(0,0,0,.08); }
  .panel h2 { font-size:.82rem; margin:0 0 10px; opacity:.7; text-transform:uppercase; letter-spacing:.03em; }
  .bars { display:flex; align-items:flex-end; gap:4px; height:120px; }
  .bar { flex:1 1 0; background:#5b6cff; border-radius:3px 3px 0 0; min-width:6px;
         position:relative; }
  .bar span { position:absolute; bottom:-16px; left:50%; transform:translateX(-50%) rotate(0deg);
              font-size:.6rem; opacity:.6; white-space:nowrap; }
  #mapwrap { position: relative; }
  #map { height: 380px; border-radius: 10px; }
  #maplock { position:absolute; inset:0; z-index:500; display:flex; align-items:center;
             justify-content:center; background:rgba(0,0,0,.04); cursor:pointer; border-radius:10px; }
  #maplock span { background:rgba(20,20,20,.72); color:#fff; padding:7px 14px;
                  border-radius:16px; font-size:.85rem; }
  #maprelock { position:absolute; top:8px; right:8px; z-index:600; display:none;
               background:rgba(20,20,20,.72); color:#fff; border:0; padding:6px 12px;
               border-radius:14px; font-size:.8rem; cursor:pointer; }
  .empty { opacity:.6; font-size:.85rem; }
</style></head>
<body>
  <h1>⚡ Lightning data collection</h1>
  <div class="sub">__SUB__</div>
  <div class="cards">__CARDS__</div>
  <div class="panel"><h2>Strikes per day</h2><div class="bars" id="bars">__BARS__</div>
    <div style="height:18px"></div></div>
  <div class="panel"><h2>Coverage</h2>
    <div id="mapwrap">
      <div id="map"></div>
      <div id="maplock"><span>Tap to interact</span></div>
      <button id="maprelock" type="button">🔒 Lock</button>
    </div>
    <div style="font-size:.7rem;opacity:.55;margin-top:6px;">Tap the map to pan/zoom · Lock to scroll the page</div>
  </div>
<script>
  const POINTS = __POINTS__;        // [[lat,lon,weight],...]
  const ANCHORS = __ANCHORS__;      // [[name,lat,lon,radius_km],...]
  const BBOX = __BBOX__;            // [latmin,lonmin,latmax,lonmax] or null
  // Locked by default so page/iframe scroll isn't trapped (esp. on mobile).
  const map = L.map('map', { dragging:false, scrollWheelZoom:false, touchZoom:false,
    doubleClickZoom:false, boxZoom:false, keyboard:false, tap:false });
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    { maxZoom: 19, attribution: '© OpenStreetMap' }).addTo(map);
  const _lock = document.getElementById('maplock'), _relock = document.getElementById('maprelock');
  function _enable(){ map.dragging.enable(); map.touchZoom.enable(); map.scrollWheelZoom.enable();
    map.doubleClickZoom.enable(); _lock.style.display='none'; _relock.style.display='block'; }
  function _disable(){ map.dragging.disable(); map.touchZoom.disable(); map.scrollWheelZoom.disable();
    map.doubleClickZoom.disable(); _lock.style.display='flex'; _relock.style.display='none'; }
  _lock.addEventListener('click', _enable);
  _relock.addEventListener('click', _disable);
  let maxW = 1; for (const p of POINTS) maxW = Math.max(maxW, p[2]);
  if (POINTS.length) L.heatLayer(POINTS, { radius: 18, blur: 14, max: maxW, minOpacity: .4 }).addTo(map);
  const group = [];
  for (const a of ANCHORS) {
    L.circle([a[1], a[2]], { radius: a[3]*1000, color:'#888', weight:1, fill:false, dashArray:'4 4' }).addTo(map);
    L.circleMarker([a[1], a[2]], { radius:4, color:'#e23', fill:true, fillOpacity:1 })
      .bindTooltip(a[0]).addTo(map);
    group.push([a[1], a[2]]);
  }
  if (BBOX) { group.push([BBOX[0], BBOX[1]]); group.push([BBOX[2], BBOX[3]]); }
  if (group.length) map.fitBounds(group, { padding:[30,30] }); else map.setView([52,9], 5);
  // Self-refresh with a cache-buster so HA's /local cache can't pin a stale copy.
  setTimeout(function(){
    var u = new URL(window.location.href);
    u.searchParams.set('t', Date.now());
    window.location.replace(u.toString());
  }, 60000);
</script>
</body></html>
"""


def _card(value, label: str) -> str:
    return f'<div class="card"><div class="v">{value}</div><div class="l">{label}</div></div>'


def render_dashboard_html(summary: dict, points: list, anchors=ANCHORS) -> str:
    per_region = summary.get("per_region", {})
    cards = "".join([
        _card(summary.get("total", 0), "total"),
        _card(summary.get("today", 0), "today"),
        _card(summary.get("last_7d", 0), "last 7d"),
        _card(summary.get("days", 0), "days"),
        _card(per_region.get("Denmark", 0), "Denmark"),
        _card(per_region.get("Alps", 0), "Alps"),
    ])
    if per_region.get("Other"):
        cards += _card(per_region["Other"], "other")

    per_day = summary.get("per_day", {})
    items = list(per_day.items())[-30:]  # last 30 days
    if items:
        mx = max(n for _, n in items) or 1
        bars = "".join(
            f'<div class="bar" title="{d}: {n}" style="height:{max(4, round(100*n/mx))}%">'
            f'<span>{d[5:]}</span></div>'
            for d, n in items
        )
    else:
        bars = '<div class="empty">No strikes archived yet.</div>'

    last = summary.get("last_strike_utc") or "—"
    sub = (f'{summary.get("total", 0)} strikes · {summary.get("days", 0)} day(s) · '
           f'last strike {last} · generated {summary.get("generated_at", "")[:19]}Z')

    return (
        _TEMPLATE
        .replace("__SUB__", sub)
        .replace("__CARDS__", cards)
        .replace("__BARS__", bars)
        .replace("__POINTS__", json.dumps(points))
        .replace("__ANCHORS__", json.dumps([list(a) for a in anchors]))
        .replace("__BBOX__", json.dumps(summary.get("bbox")))
    )


__all__ = ["render_dashboard_html", "ANCHORS"]
