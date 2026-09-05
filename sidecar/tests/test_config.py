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


# ---------------------------------------------------------------------------
# STEPS horizon (measured from radar-frame time)
# ---------------------------------------------------------------------------

def test_steps_horizon_default_and_env_override(
    config_yaml: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """90 by default: 60 min of served lead plus the 14-18 min (up to ~30)
    frame age at compute, so the longest served lead is still a forecast
    for that lead and not the ensemble's clamped final timestep."""
    cfg = load_config(config_yaml)
    assert cfg.forecast.steps.horizon_min == 90

    monkeypatch.setenv("DMI_NOWCAST_FORECAST__STEPS__HORIZON_MIN", "120")
    assert load_config(config_yaml).forecast.steps.horizon_min == 120


def test_steps_horizon_yaml_and_bounds(tmp_path: Path) -> None:
    p = tmp_path / "horizon.yaml"
    p.write_text(
        "home:\n  lat: 55\n  lon: 10\n"
        "forecast:\n  steps:\n    horizon_min: 60\n"
    )
    assert load_config(p).forecast.steps.horizon_min == 60

    # Below the floor: a horizon shorter than 30 min cannot outlive the
    # frame age at compute, so it is a config error, not a tuning choice.
    too_short = tmp_path / "short.yaml"
    too_short.write_text(
        "home:\n  lat: 55\n  lon: 10\n"
        "forecast:\n  steps:\n    horizon_min: 29\n"
    )
    with pytest.raises(Exception):
        load_config(too_short)

    too_long = tmp_path / "long.yaml"
    too_long.write_text(
        "home:\n  lat: 55\n  lon: 10\n"
        "forecast:\n  steps:\n    horizon_min: 181\n"
    )
    with pytest.raises(Exception):
        load_config(too_long)


# ---------------------------------------------------------------------------
# Web Push section (website Phase D)
# ---------------------------------------------------------------------------

def test_push_defaults_are_off(config_yaml: Path) -> None:
    """Additive config: an existing deployment keeps its behaviour."""
    cfg = load_config(config_yaml)
    assert cfg.push.enabled is False
    assert cfg.push.vapid_subject is None
    assert cfg.push.vapid_private_key_file is None
    assert cfg.push.db_path is None
    assert cfg.push.min_lead_min == 20
    # Phase G: unset means "derive as before" and the fitted table lands
    # beside the other synced artifacts in the data volume.
    assert cfg.push.lead_options is None
    assert cfg.push.thresholds_path is None
    assert cfg.push.threshold_options_pct == [40, 60, 80]
    assert cfg.push.default_threshold_pct == 60
    assert cfg.push.default_lead_min == 30
    assert cfg.push.max_subscriptions == 200
    assert cfg.push.persistence_obs == 2
    assert cfg.push.rearm_after_min == 60
    assert "fcm.googleapis.com" in cfg.push.allowed_endpoint_host_suffixes


def test_push_enabled_without_subject_rejected(tmp_path: Path) -> None:
    """Push services may throttle a sender with no usable contact, so an
    enabled feature without one is a config error, not a warning."""
    p = tmp_path / "push.yaml"
    p.write_text("home:\n  lat: 55\n  lon: 10\npush:\n  enabled: true\n")
    with pytest.raises(Exception, match="vapid_subject"):
        load_config(p)


def test_push_subject_scheme_enforced(tmp_path: Path) -> None:
    p = tmp_path / "push_scheme.yaml"
    p.write_text(
        "home:\n  lat: 55\n  lon: 10\n"
        "push:\n  enabled: true\n  vapid_subject: ops@example.com\n"
    )
    with pytest.raises(Exception, match="mailto"):
        load_config(p)


def test_push_defaults_must_be_offered_options(tmp_path: Path) -> None:
    p = tmp_path / "push_opts.yaml"
    p.write_text(
        "home:\n  lat: 55\n  lon: 10\n"
        "push:\n  default_threshold_pct: 55\n"
    )
    with pytest.raises(Exception, match="threshold_options_pct"):
        load_config(p)

    p2 = tmp_path / "push_lead.yaml"
    p2.write_text(
        "home:\n  lat: 55\n  lon: 10\n"
        "push:\n  min_lead_min: 30\n  default_lead_min: 20\n"
    )
    with pytest.raises(Exception, match="min_lead_min"):
        load_config(p2)


def test_push_lead_options_must_be_leads_the_grids_carry(tmp_path: Path) -> None:
    """A horizon outside forecast.national.leads_min could be chosen and
    then never evaluated: the subscription would sample None for ever."""
    good = tmp_path / "push_leads_ok.yaml"
    good.write_text(
        "home:\n  lat: 55\n  lon: 10\n"
        "push:\n  lead_options: [20, 30, 45, 60]\n"
    )
    assert load_config(good).push.lead_options == [20, 30, 45, 60]

    bad = tmp_path / "push_leads_bad.yaml"
    bad.write_text(
        "home:\n  lat: 55\n  lon: 10\n"
        "push:\n  lead_options: [20, 44]\n"
    )
    with pytest.raises(Exception, match="lead_options"):
        load_config(bad)

    unsorted = tmp_path / "push_leads_unsorted.yaml"
    unsorted.write_text(
        "home:\n  lat: 55\n  lon: 10\n"
        "push:\n  lead_options: [30, 20]\n"
    )
    with pytest.raises(Exception, match="ascending"):
        load_config(unsorted)


def test_push_default_lead_must_be_offered(tmp_path: Path) -> None:
    p = tmp_path / "push_default_lead.yaml"
    p.write_text(
        "home:\n  lat: 55\n  lon: 10\n"
        "push:\n  lead_options: [20, 45]\n  default_lead_min: 30\n"
    )
    with pytest.raises(Exception, match="lead_options"):
        load_config(p)


def test_quality_report_threshold_fit_is_off_by_default(
    config_yaml: Path,
) -> None:
    fit = load_config(config_yaml).quality_report.fit_thresholds
    assert fit.enabled is False
    assert fit.decisions_dirs == []
    assert fit.thresholds == "20:80:5"
    assert (fit.min_warnings, fit.min_delta_pct) == (30, 5)
    assert fit.min_useful_lead_min == 5.0
    assert fit.thresholds_out is None


def test_push_quiet_hours_must_be_hhmm(tmp_path: Path) -> None:
    p = tmp_path / "push_quiet.yaml"
    p.write_text(
        "home:\n  lat: 55\n  lon: 10\n"
        'push:\n  default_quiet_start: "25:00"\n'
    )
    with pytest.raises(Exception, match="HH:MM"):
        load_config(p)


def test_push_env_overrides(
    config_yaml: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The compose stacks turn push on through the environment."""
    monkeypatch.setenv("DMI_NOWCAST_PUSH__ENABLED", "true")
    monkeypatch.setenv("DMI_NOWCAST_PUSH__VAPID_SUBJECT", "mailto:ops@example.com")
    monkeypatch.setenv("DMI_NOWCAST_PUSH__MAX_SUBSCRIPTIONS", "5")
    cfg = load_config(config_yaml)
    assert cfg.push.enabled is True
    assert cfg.push.vapid_subject == "mailto:ops@example.com"
    assert cfg.push.max_subscriptions == 5


def test_push_paths_resolve_against_the_data_dir(
    config_yaml: Path, tmp_path: Path,
) -> None:
    from dmi_nowcast_sidecar.push.paths import resolved_db_path, resolved_key_path

    cfg = load_config(config_yaml)
    assert resolved_db_path(cfg) == cfg.storage.data_dir / "push" / "subscriptions.sqlite"
    assert resolved_key_path(cfg) == cfg.storage.data_dir / "push" / "vapid_private.pem"

    cfg.push.db_path = tmp_path / "elsewhere.sqlite"
    cfg.push.vapid_private_key_file = tmp_path / "elsewhere.pem"
    assert resolved_db_path(cfg) == tmp_path / "elsewhere.sqlite"
    assert resolved_key_path(cfg) == tmp_path / "elsewhere.pem"
