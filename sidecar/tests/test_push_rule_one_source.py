"""One rule, one source: the push engine's timing, everywhere it is read.

DECIDE-14 (2026-09-13). The push fan-out required TWO over-threshold
observations before notifying, and the nightly threshold fit — which
chooses the percent each horizon warns at — replayed the rule the same
way. Everything that *measured* that rule required ONE: the live station
scoreboard, the quality page's served-rule scoreboard, the manual fit
script, the historical replay and the benchmark. So the table in service
was fitted for a rule the page never scored, and a manual fit and a
nightly fit produced different tables from the same rows.

The fix is structural, and this module is the pin for it: the number lives
in :mod:`dmi_nowcast_core.push_rules`, ``push.persistence_obs`` /
``push.rearm_after_min`` are the only place it can be overridden, and
every consumer reads THAT rather than carrying a default of its own. A
test that only checked the shipped values would pass again the moment
someone re-introduced a second default, so each assertion here is made
twice: once on the shipped config, and once on a config whose push rule
has been moved somewhere neither number could be reached by accident.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from dmi_nowcast_core.push_rules import (
    DEFAULT_PERSISTENCE_OBS,
    DEFAULT_REARM_AFTER_MIN,
)

from dmi_nowcast_sidecar.config import Config
from dmi_nowcast_sidecar.push.engine import Rules
from dmi_nowcast_sidecar.push.service import PushService
from dmi_nowcast_sidecar.quality_report import QualityReportTask
from dmi_nowcast_sidecar.served_rule import ServedRuleOptions
from dmi_nowcast_sidecar.station_eval import StationEvalService
from dmi_nowcast_sidecar.threshold_sweep import SweepOptions


def _config(tmp_path: Path) -> Config:
    """The private instance's shape: scoreboard on, nightly fit on."""
    return Config(  # type: ignore[arg-type]
        home={"lat": 55.33, "lon": 10.32},
        calibration={
            "curves_path": tmp_path / "curves.json",
            "national_curves_path": tmp_path / "national_curves.json",
        },
        storage={"data_dir": tmp_path / "data", "corpus_dir": tmp_path / "corpus"},
        lightning={"archive_dir": tmp_path / "strikes"},
        station_eval={
            "enabled": True,
            "points_file": tmp_path / "station_points.json",
        },
        quality_report={
            "enabled": True,
            "fit_thresholds": {
                "enabled": True,
                "decisions_dirs": [tmp_path / "corpus" / "stations" / "eval"],
                "thresholds": "40:60:10",
            },
        },
    )


def _timing(config: Config) -> dict[str, tuple[int, int]]:
    """``(persistence_obs, rearm_after_min)`` as each consumer computes it.

    ``_rules`` on both services is a pure function of ``self.config``, so a
    namespace carrying the config is enough — and keeps this test away from
    a real store, a real engine and a VAPID key it has no use for.
    """
    task = QualityReportTask(config)
    engine_rules: Rules = PushService._rules(SimpleNamespace(config=config))
    station_rules: Rules = StationEvalService._rules(SimpleNamespace(config=config))
    served = task._served_rule_options()
    fit = task._fit_options()
    return {
        "push fan-out": (
            engine_rules.persistence_obs, engine_rules.rearm_after_min,
        ),
        "station scoreboard": (
            station_rules.persistence_obs, station_rules.rearm_after_min,
        ),
        "served-rule hook": (
            int(served.persistence_obs), int(served.rearm_after_min),
        ),
        "nightly threshold fit": (
            int(fit.persistence_obs), int(fit.rearm_after_min),
        ),
    }


def test_the_shipped_rule_is_one_observation_and_a_sixty_minute_rearm(
    tmp_path: Path,
) -> None:
    """The default every consumer gets is the core constant, not a literal."""
    assert (DEFAULT_PERSISTENCE_OBS, DEFAULT_REARM_AFTER_MIN) == (1, 60)
    shipped = (DEFAULT_PERSISTENCE_OBS, DEFAULT_REARM_AFTER_MIN)

    config = _config(tmp_path)
    assert (config.push.persistence_obs, config.push.rearm_after_min) == shipped
    for consumer, timing in _timing(config).items():
        assert timing == shipped, consumer

    # The dataclasses a batch job constructs directly (the manual fit, the
    # replay-driven scoreboard) default to the same numbers.
    assert (
        SweepOptions(
            decisions_dirs=[tmp_path], corpus_dir=tmp_path,
        ).persistence_obs,
        SweepOptions(
            decisions_dirs=[tmp_path], corpus_dir=tmp_path,
        ).rearm_after_min,
    ) == shipped
    assert (
        ServedRuleOptions().persistence_obs,
        ServedRuleOptions().rearm_after_min,
    ) == shipped


@pytest.mark.parametrize("persistence,rearm", [(1, 60), (2, 45), (3, 90)])
def test_every_consumer_follows_push_when_it_is_overridden(
    tmp_path: Path, persistence: int, rearm: int,
) -> None:
    """``push.*`` is the one knob; nothing may keep its own answer.

    This is the assertion the old code failed: ``station_eval.rules`` and
    the served-rule hook carried their own pair, so moving the service's
    rule moved the service alone and the measurement stayed where it was.
    """
    config = _config(tmp_path)
    config.push.persistence_obs = persistence
    config.push.rearm_after_min = rearm

    for consumer, timing in _timing(config).items():
        assert timing == (persistence, rearm), consumer


@pytest.mark.parametrize("enabled,readings", [(True, 2), (False, 2), (True, 3)])
def test_every_consumer_follows_push_for_the_all_clear(
    tmp_path: Path, enabled: bool, readings: int,
) -> None:
    """The all-clear has one knob too: ``push.allclear_*``."""
    from dmi_nowcast_core.push_rules import (
        DEFAULT_ALLCLEAR_ENABLED,
        DEFAULT_ALLCLEAR_READINGS,
    )

    config = _config(tmp_path)
    assert (config.push.allclear_enabled, config.push.allclear_readings) == (
        DEFAULT_ALLCLEAR_ENABLED, DEFAULT_ALLCLEAR_READINGS,
    )
    config.push.allclear_enabled = enabled
    config.push.allclear_readings = readings
    engine_rules: Rules = PushService._rules(SimpleNamespace(config=config))
    station_rules: Rules = StationEvalService._rules(SimpleNamespace(config=config))
    served = QualityReportTask(config)._served_rule_options()
    for consumer, pair in {
        "push fan-out": (engine_rules.allclear_enabled, engine_rules.allclear_readings),
        "station scoreboard": (
            station_rules.allclear_enabled, station_rules.allclear_readings,
        ),
        "served-rule hook": (served.allclear_enabled, served.allclear_readings),
    }.items():
        assert pair == (enabled, readings), consumer
