"""/frames/* endpoints + safe-name guard."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dmi_nowcast_sidecar.app import _safe_frame_name, create_app
from dmi_nowcast_sidecar.config import Config


def test_safe_frame_name_accepts_valid() -> None:
    assert _safe_frame_name("frame_00.png")
    assert _safe_frame_name("frame_12.png")
    assert _safe_frame_name("frame_999.png")  # ≥ 3 digits also fine
    assert _safe_frame_name("frame_1234.png")
    assert _safe_frame_name("loop.png")  # the animated APNG for image entity


def test_safe_frame_name_rejects_traversal() -> None:
    assert not _safe_frame_name("../etc/passwd")
    assert not _safe_frame_name("frame_aa.png")
    assert not _safe_frame_name("frame_1.png")        # 1 digit too few
    assert not _safe_frame_name("frame_12.PNG")        # uppercase ext
    assert not _safe_frame_name("frame_12.png.json")
    assert not _safe_frame_name("../frame_00.png")
    assert not _safe_frame_name("frame_00.png/../")


def test_manifest_endpoint_503_before_any_render(minimal_config: Config) -> None:
    app = create_app(minimal_config, auto_start_scheduler=False)
    with TestClient(app) as c:
        r = c.get("/frames/manifest.json")
        assert r.status_code == 503


def test_manifest_and_frames_served_after_writes(
    minimal_config: Config, tmp_path: Path,
) -> None:
    """Pre-seed a frames.json + a frame PNG and verify the endpoints serve them."""
    frames_dir = minimal_config.storage.data_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    # Tiny valid 1x1 PNG (transparent).
    png_bytes = bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
        "0000000a49444154789c63000000000200013e9a3a200000000049454e44ae426082"
    )
    (frames_dir / "frame_00.png").write_bytes(png_bytes)
    manifest = {
        "version": 1,
        "generated_at": "2026-05-20T07:00:00Z",
        "frame_count": 1,
        "frames": [
            {"index": 0, "filename": "frame_00.png",
             "timestamp_utc": "2026-05-20T07:00:00Z", "kind": "observed",
             "label": "now", "duration_ms": 1500},
        ],
        "now_stats_subline": None,
    }
    (frames_dir / "frames.json").write_text(json.dumps(manifest))

    app = create_app(minimal_config, auto_start_scheduler=False)
    with TestClient(app) as c:
        m = c.get("/frames/manifest.json")
        assert m.status_code == 200
        assert m.json()["frame_count"] == 1
        f = c.get("/frames/frame_00.png")
        assert f.status_code == 200
        assert f.headers["content-type"] == "image/png"
        assert f.content == png_bytes
        # Bad name → 400, missing → 404.
        assert c.get("/frames/etc..").status_code == 400
        assert c.get("/frames/frame_99.png").status_code == 404
