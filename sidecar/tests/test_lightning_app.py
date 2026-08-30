"""Lightning POST/GET endpoints — HTTP surface + auth + buffer behaviour."""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from dmi_nowcast_sidecar.app import create_app
from dmi_nowcast_sidecar.config import Config

HOME_LAT, HOME_LON = 55.33, 10.32


def _km_to_dlon(km: float, lat0: float = HOME_LAT) -> float:
    return km / (111.320 * math.cos(math.radians(lat0)))


def _approaching_payload(n: int = 9) -> dict:
    """Cell due west of home, marching east at 1 km/min; nearest (now) 18 km."""
    now = datetime.now(timezone.utc)
    strikes = []
    for m in range(n):
        dist = 18.0 + m  # older strikes are farther west
        strikes.append({
            "lat": HOME_LAT,
            "lon": HOME_LON - _km_to_dlon(dist),
            "t": (now - timedelta(minutes=m)).isoformat(),
        })
    return {"strikes": strikes}


@pytest.fixture
def client(minimal_config: Config) -> TestClient:
    minimal_config.server.api_key = None  # LAN trust
    app = create_app(minimal_config, auto_start_scheduler=False)
    with TestClient(app) as c:
        yield c


def test_ingest_then_eta_approaching(client: TestClient):
    payload = _approaching_payload(9)
    r = client.post("/lightning/strikes", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["accepted"] == 9
    assert body["buffer"] == 9

    r = client.get("/lightning/eta", params={"lat": HOME_LAT, "lon": HOME_LON, "rings": "3,10"})
    assert r.status_code == 200, r.text
    eta = r.json()
    assert eta["state"] == "approaching"
    assert eta["n_cells"] == 1
    assert eta["leading_edge_km"] == pytest.approx(18.0, abs=2.0)
    by_ring = {r_["ring_km"]: r_ for r_ in eta["rings"]}
    assert by_ring[10.0]["eta_min"] is not None
    assert by_ring[3.0]["eta_min"] > by_ring[10.0]["eta_min"]
    assert 0.0 < eta["confidence"] <= 1.0


def test_eta_empty_buffer_insufficient(client: TestClient):
    r = client.get("/lightning/eta", params={"lat": HOME_LAT, "lon": HOME_LON})
    assert r.status_code == 200
    eta = r.json()
    assert eta["state"] == "insufficient_data"
    assert all(ring["eta_min"] is None for ring in eta["rings"])


def test_probability_endpoint_shape_and_elevated(client: TestClient):
    client.post("/lightning/strikes", json=_approaching_payload(9))
    r = client.get("/lightning/probability", params={"lat": HOME_LAT, "lon": HOME_LON})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["region"] == "Denmark"
    assert body["n_strikes_buffer"] == 9
    rings = {rr["ring_km"]: rr for rr in body["rings"]}
    assert set(rings) == {3.0, 10.0}
    for rr in body["rings"]:
        assert {"ring_km", "lead_min", "p_raw", "p"} <= set(rr)
        assert 0.0 <= rr["p"] <= 1.0 and 0.0 <= rr["p_raw"] <= 1.0
    # An approaching cell (18 km, ~1 km/min) reaches 10 km in ~8 min (< 30 lead)
    # → a non-zero ensemble probability.
    assert rings[10.0]["p_raw"] > 0.0


def test_probability_empty_buffer_zero(client: TestClient):
    r = client.get("/lightning/probability", params={"lat": HOME_LAT, "lon": HOME_LON})
    assert r.status_code == 200
    # No strikes → raw ensemble probability 0 (the calibrated p may be a small
    # climatological floor, so assert on p_raw).
    assert all(rr["p_raw"] == 0.0 for rr in r.json()["rings"])


def test_probability_503_when_disabled(minimal_config: Config):
    minimal_config.server.api_key = None
    minimal_config.lightning.enabled = False
    app = create_app(minimal_config, auto_start_scheduler=False)
    with TestClient(app) as c:
        r = c.get("/lightning/probability", params={"lat": HOME_LAT, "lon": HOME_LON})
        assert r.status_code == 503


def test_dedup_on_repeat_post(client: TestClient):
    payload = _approaching_payload(9)
    assert client.post("/lightning/strikes", json=payload).json()["accepted"] == 9
    # Re-POSTing the same strikes adds nothing.
    second = client.post("/lightning/strikes", json=payload).json()
    assert second["accepted"] == 0
    assert second["buffer"] == 9


def test_invalid_rings_400(client: TestClient):
    r = client.get("/lightning/eta", params={"lat": HOME_LAT, "lon": HOME_LON, "rings": "abc"})
    assert r.status_code == 400


def test_auth_required_when_key_set(minimal_config: Config):
    minimal_config.server.api_key = "secret-key"
    app = create_app(minimal_config, auto_start_scheduler=False)
    with TestClient(app) as c:
        payload = _approaching_payload(6)
        assert c.post("/lightning/strikes", json=payload).status_code == 401
        ok = c.post(
            "/lightning/strikes", json=payload,
            headers={"Authorization": "Bearer secret-key"},
        )
        assert ok.status_code == 200
        # GET (read) stays open even with a key set.
        assert c.get("/lightning/eta", params={"lat": HOME_LAT, "lon": HOME_LON}).status_code == 200


def test_clusters_endpoint(client: TestClient):
    client.post("/lightning/strikes", json=_approaching_payload(9))
    r = client.get("/lightning/clusters", params={"lat": HOME_LAT, "lon": HOME_LON})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_strikes_buffer"] == 9
    assert body["n_clusters"] >= 1
    c = body["clusters"][0]
    assert {"centroid_lat", "centroid_lon", "speed_kmh", "bearing_deg",
            "threatening", "leading_edge_km"} <= set(c)


def test_map_503_without_radar_geometry(client: TestClient):
    # No cycle has run in the test app, so engine.geo is None.
    r = client.get("/lightning/map.png")
    assert r.status_code == 503


def test_eta_smoother_prior_alpha_and_reset(minimal_config: Config):
    from datetime import datetime, timedelta, timezone

    from dmi_nowcast_sidecar.eta_smoother import EtaSmoother

    s = EtaSmoother(minimal_config.lightning)  # tau=3, max_gap=10 (defaults)
    now = datetime(2026, 6, 7, 12, 0, tzinfo=timezone.utc)
    k = EtaSmoother.key_for(55.35, 10.35)
    assert s.prior(k, now) == (None, None, 1.0)  # no prior yet
    s.store(k, 60.0, 20.0, now)
    pc, pe, alpha = s.prior(k, now + timedelta(minutes=3))
    assert (pc, pe) == (60.0, 20.0)
    assert alpha == pytest.approx(1.0 - math.e ** -1, abs=0.01)  # 1-exp(-3/3)
    # Gap beyond smoothing_max_gap_min → reset (use raw).
    assert s.prior(k, now + timedelta(minutes=20)) == (None, None, 1.0)


def test_eta_smoothing_path_runs(client: TestClient):
    client.post("/lightning/strikes", json=_approaching_payload(9))
    a = client.get("/lightning/eta", params={"lat": HOME_LAT, "lon": HOME_LON})
    b = client.get("/lightning/eta", params={"lat": HOME_LAT, "lon": HOME_LON})
    assert a.status_code == 200 and b.status_code == 200
    assert b.json()["state"] == "approaching"


def test_strike_archive_writes_ndjson(tmp_path):
    import json
    from datetime import datetime, timezone

    from dmi_nowcast_core.lightning import LightningStrike
    from dmi_nowcast_sidecar.strike_archive import StrikeArchive

    arch = StrikeArchive(tmp_path / "strikes")
    t = datetime(2026, 6, 7, 12, 0, tzinfo=timezone.utc)
    n = arch.append([LightningStrike(55.0, 10.0, t), LightningStrike(56.0, 11.0, t)])
    assert n == 2
    f = tmp_path / "strikes" / "strikes_2026-06-07.ndjson"
    lines = f.read_text().strip().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert set(rec) == {"lat", "lon", "t"}


def test_strike_archive_dedups_on_read(tmp_path):
    # A restart makes the HA push re-append already-archived strikes verbatim;
    # iter_strikes / snapshot must drop the duplicate lines (M2).
    import json
    from datetime import datetime, timezone

    from dmi_nowcast_sidecar.strike_archive import StrikeArchive

    d = tmp_path / "strikes"
    d.mkdir()
    recs = [
        {"lat": 55.1, "lon": 10.1, "t": "2026-06-07T12:00:00+00:00"},
        {"lat": 55.1, "lon": 10.1, "t": "2026-06-07T12:00:00+00:00"},  # dup (re-post)
        {"lat": 55.2, "lon": 10.2, "t": "2026-06-07T12:01:00+00:00"},
        {"lat": 55.1, "lon": 10.1, "t": "2026-06-07T12:00:00+00:00"},  # dup again
    ]
    (d / "strikes_2026-06-07.ndjson").write_text(
        "\n".join(json.dumps(r) for r in recs) + "\n")
    arch = StrikeArchive(d)
    assert sum(1 for _ in arch.iter_strikes()) == 2
    summary, _ = arch.snapshot(now=datetime(2026, 6, 7, 13, 0, tzinfo=timezone.utc), ttl_s=0)
    assert summary["total"] == 2


def test_tracker_archives_new_strikes_only(tmp_path, minimal_config: Config):
    import glob
    from datetime import datetime, timezone

    from dmi_nowcast_core.lightning import LightningStrike
    from dmi_nowcast_sidecar.lightning_tracker import LightningTracker
    from dmi_nowcast_sidecar.strike_archive import StrikeArchive

    arch = StrikeArchive(tmp_path / "s")
    tr = LightningTracker(minimal_config.lightning, archive=arch)
    s = [LightningStrike(55.0, 10.0, datetime.now(timezone.utc))]
    assert tr.add(s) == 1
    assert tr.add(s) == 0  # dedup → not re-archived
    files = glob.glob(str(tmp_path / "s" / "*.ndjson"))
    assert len(files) == 1
    assert len(open(files[0]).read().strip().splitlines()) == 1


def test_strike_archive_snapshot(tmp_path):
    from datetime import datetime, timezone

    from dmi_nowcast_core.lightning import LightningStrike
    from dmi_nowcast_sidecar.strike_archive import StrikeArchive

    arch = StrikeArchive(tmp_path / "s")
    d1 = datetime(2026, 6, 7, 12, 0, tzinfo=timezone.utc)
    d2 = datetime(2026, 6, 8, 12, 0, tzinfo=timezone.utc)
    arch.append([LightningStrike(55.8, 9.6, d1), LightningStrike(55.0, 10.0, d1)])  # DK
    arch.append([LightningStrike(46.0, 7.0, d2)])  # Alps
    summary, points = arch.snapshot(now=d2, ttl_s=0)
    assert summary["total"] == 3
    assert summary["per_region"]["Denmark"] == 2
    assert summary["per_region"]["Alps"] == 1
    assert summary["per_day"] == {"2026-06-07": 2, "2026-06-08": 1}
    assert summary["days"] == 2
    assert summary["bbox"][0] == 46.0  # latmin
    assert len(points) == 3


def test_archive_endpoints(minimal_config: Config, tmp_path):
    minimal_config.lightning.archive_dir = tmp_path / "strikes"
    minimal_config.server.api_key = None
    app = create_app(minimal_config, auto_start_scheduler=False)
    with TestClient(app) as c:
        c.post("/lightning/strikes", json=_approaching_payload(9))
        r = c.get("/lightning/archive/summary")
        assert r.status_code == 200, r.text
        assert r.json()["total"] == 9
        h = c.get("/lightning/archive/dashboard.html")
        assert h.status_code == 200
        assert "text/html" in h.headers["content-type"]
        assert "Lightning data collection" in h.text


def test_disabled_returns_503(minimal_config: Config):
    minimal_config.lightning.enabled = False
    app = create_app(minimal_config, auto_start_scheduler=False)
    with TestClient(app) as c:
        assert c.get("/lightning/eta", params={"lat": HOME_LAT, "lon": HOME_LON}).status_code == 503
        assert c.post("/lightning/strikes", json=_approaching_payload(6)).status_code == 503
