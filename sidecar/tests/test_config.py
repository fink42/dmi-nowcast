"""Config loading + validation."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from dmi_nowcast_sidecar.config import Config, load_config


def test_load_minimal_yaml(config_yaml: Path) -> None:
    cfg = load_config(config_yaml)
    assert cfg.home.lat == 55.33
    assert cfg.home.lon == 10.32
    # Defaults applied
    assert cfg.poll.interval_min == 5
    assert cfg.server.port == 8081
    assert cfg.forecast.method == "farneback"


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="sidecar config not found"):
        load_config(tmp_path / "nope.yaml")


def test_top_level_must_be_mapping(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("- not\n- a\n- mapping\n")
    with pytest.raises(ValueError, match="top-level must be a mapping"):
        load_config(p)


def test_extra_fields_forbidden(tmp_path: Path) -> None:
    p = tmp_path / "extra.yaml"
    p.write_text(
        "home:\n  lat: 55.33\n  lon: 10.32\n"
        "bogus_section: hello\n"
    )
    with pytest.raises(Exception):  # pydantic ValidationError
        load_config(p)


def test_invalid_lat_rejected(tmp_path: Path) -> None:
    p = tmp_path / "bad_lat.yaml"
    p.write_text("home:\n  lat: 999\n  lon: 10\n")
    with pytest.raises(Exception):
        load_config(p)


def test_leads_must_be_sorted(tmp_path: Path) -> None:
    p = tmp_path / "leads.yaml"
    p.write_text(
        "home:\n  lat: 55\n  lon: 10\n"
        "forecast:\n  leads_min: [10, 5, 20]\n"
    )
    with pytest.raises(Exception, match="sorted"):
        load_config(p)


def test_env_override_nested(config_yaml: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DMI_NOWCAST_SERVER__PORT", "9999")
    cfg = load_config(config_yaml)
    assert cfg.server.port == 9999


def test_env_overrides_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Env vars must beat explicit YAML values (docker-compose pattern)."""
    p = tmp_path / "config.yaml"
    p.write_text(
        "home:\n  lat: 55.33\n  lon: 10.32\n"
        "server:\n  port: 8081\n"
    )
    monkeypatch.setenv("DMI_NOWCAST_SERVER__PORT", "8082")
    cfg = load_config(p)
    assert cfg.server.port == 8082


def test_env_override_api_key(config_yaml: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DMI_NOWCAST_SERVER__API_KEY", "secret-token-123")
    cfg = load_config(config_yaml)
    assert cfg.server.api_key == "secret-token-123"


def test_default_config_path_via_env(
    config_yaml: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DMI_NOWCAST_CONFIG", str(config_yaml))
    # Pass no path → falls back to env var.
    cfg = load_config()
    assert cfg.home.lat == 55.33
