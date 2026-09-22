"""The nightly refit stands down for a model it did not fit (S3).

A post-processing model fitted on the workstation — a tree ensemble, a v2
design, the random-point protocol — is installed into the same file the
nightly refit publishes into. Without a guard the refit overwrites it a
few hours later with tonight's pooled logistic, the threshold sweep is
then fitted on the replacement, and the only trace is a
``postprocess.prev.json`` nobody is watching.

So the job asks one question before it reads a row: *could this
configuration have produced the document in service?* The comparison is on
what the document says about itself — its kind, its design version, and
the protocol it was fitted under — and a "no" means no ``.prev`` copy, no
write, and one log line.

The documents here are hand-written headers, deliberately: the guard reads
the header and nothing else, and the full document — loaded, evaluated,
scored without LightGBM — is ``test_postprocess_v2.py``'s subject.

The last section is the other half of the same operation: the installer
that puts such a document there (``sidecar/deploy/install_artifact.sh``),
exercised through ``--dry-run`` with ssh and scp stubbed to fail. What is
under test there is the validation and the exact shape of the swap — one
generation kept, chowned to the container's uid, renamed into place — and
never a connection.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from dmi_nowcast_core import postprocess as pp
from dmi_nowcast_sidecar import quality_job
from dmi_nowcast_sidecar.config import FitPostprocessConfig
from dmi_nowcast_sidecar.postprocess_fit import (
    NIGHTLY_PROTOCOL,
    PostprocessFitOptions,
    describe_served_model,
    refit_skip_reason,
)

INSTALLED = json.dumps({
    "schema_version": pp.SCHEMA_VERSION,
    "fitted_at_utc": "2026-09-19T14:08:48+00:00",
    "kind": pp.KIND_TREES,
    "design": {"version": pp.DESIGN_V2, "extra_columns": []},
    "leads": [20, 30, 45, 60],
    "training": {"rows": 4018648, "protocol": pp.PROTOCOL_RANDOM_POINT},
    "models": {},
}, indent=1) + "\n"

NIGHTLY = json.dumps({
    "schema_version": pp.SCHEMA_VERSION,
    "fitted_at_utc": "2026-09-21T03:40:00+00:00",
    "kind": pp.KIND_LOGISTIC,
    "design": {"version": pp.DESIGN_V1, "extra_columns": []},
    "leads": [20, 30, 45, 60],
    "training": {"rows": 120000},
    "models": {},
}, indent=1) + "\n"


def _options(out: Path, **over) -> PostprocessFitOptions:
    kwargs: dict = {
        "decisions_dirs": [out.parent],
        "corpus_dir": out.parent,
        "out": out,
    }
    kwargs.update(over)
    return PostprocessFitOptions(**kwargs)  # type: ignore[arg-type]


def _job(out: Path, **over) -> dict:
    fit: dict = {
        "out": str(out),
        "options": {
            "decisions_dirs": [str(out.parent)],
            "corpus_dir": str(out.parent),
            "out": str(out),
        },
    }
    fit.update(over)
    return fit


# ---------------------------------------------------------------------------
# Reading the header
# ---------------------------------------------------------------------------


class TestWhatTheServedFileSaysItIs:
    def test_the_installed_artefact_is_read_as_what_it_is(
        self, tmp_path: Path,
    ) -> None:
        path = tmp_path / "postprocess.json"
        path.write_text(INSTALLED)
        served = describe_served_model(path)
        assert served is not None
        assert served.kind == pp.KIND_TREES
        assert served.design == pp.DESIGN_V2
        assert served.protocol == pp.PROTOCOL_RANDOM_POINT

    def test_a_top_level_protocol_key_wins_over_the_training_block(
        self, tmp_path: Path,
    ) -> None:
        """The schema is growing one; until then ``training`` carries it."""
        doc = json.loads(INSTALLED)
        doc["protocol"] = pp.PROTOCOL_RANDOM_POINT
        doc["training"]["protocol"] = pp.PROTOCOL_AT_GAUGE
        path = tmp_path / "postprocess.json"
        path.write_text(json.dumps(doc))
        served = describe_served_model(path)
        assert served is not None
        assert served.protocol == pp.PROTOCOL_RANDOM_POINT

    def test_a_document_that_names_no_protocol_is_an_at_gauge_fit(
        self, tmp_path: Path,
    ) -> None:
        """Every model fitted before the distinction existed was one."""
        path = tmp_path / "postprocess.json"
        path.write_text(NIGHTLY)
        served = describe_served_model(path)
        assert served is not None
        assert served.protocol == NIGHTLY_PROTOCOL == pp.PROTOCOL_AT_GAUGE
        assert served.kind == pp.KIND_LOGISTIC
        assert served.design == pp.DESIGN_V1

    @pytest.mark.parametrize(
        "text", ["", "not json at all", "[1, 2, 3]", '"a string"'],
    )
    def test_nothing_readable_is_nothing_to_protect(
        self, tmp_path: Path, text: str,
    ) -> None:
        """A junk file is not a model; the refit overwrites it as it always did."""
        path = tmp_path / "postprocess.json"
        path.write_text(text)
        assert describe_served_model(path) is None

    def test_no_file_is_no_model(self, tmp_path: Path) -> None:
        assert describe_served_model(tmp_path / "nothing.json") is None


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------


class TestTheGuard:
    def test_the_nightly_model_is_refitted_as_before(self, tmp_path: Path) -> None:
        out = tmp_path / "postprocess.json"
        out.write_text(NIGHTLY)
        assert refit_skip_reason(out, _options(out)) is None

    def test_a_first_fit_is_not_blocked(self, tmp_path: Path) -> None:
        out = tmp_path / "postprocess.json"
        assert refit_skip_reason(out, _options(out)) is None

    def test_an_installed_tree_model_stops_the_refit(self, tmp_path: Path) -> None:
        out = tmp_path / "postprocess.json"
        out.write_text(INSTALLED)
        reason = refit_skip_reason(out, _options(out))
        assert reason == (
            "served model kind=trees design=v2 protocol=random-point is not "
            "reproducible by the nightly fit (model=logistic design=v1 "
            "protocol=at-gauge); left in place"
        )

    def test_the_protocol_alone_is_enough(self, tmp_path: Path) -> None:
        """A logistic v1 fitted at random points is still not this fit."""
        doc = json.loads(NIGHTLY)
        doc["training"]["protocol"] = pp.PROTOCOL_RANDOM_POINT
        out = tmp_path / "postprocess.json"
        out.write_text(json.dumps(doc))
        reason = refit_skip_reason(out, _options(out))
        assert reason is not None
        assert "protocol=random-point" in reason

    def test_the_design_alone_is_enough(self, tmp_path: Path) -> None:
        doc = json.loads(NIGHTLY)
        doc["design"]["version"] = pp.DESIGN_V2
        out = tmp_path / "postprocess.json"
        out.write_text(json.dumps(doc))
        assert refit_skip_reason(out, _options(out)) is not None
        # ...and a config that ASKS for v2 refits its own kind of document.
        assert refit_skip_reason(
            out, _options(out, design=pp.DESIGN_V2),
        ) is None

    def test_the_kind_alone_is_enough(self, tmp_path: Path) -> None:
        doc = json.loads(NIGHTLY)
        doc["kind"] = pp.KIND_LOGISTIC_SHARED
        out = tmp_path / "postprocess.json"
        out.write_text(json.dumps(doc))
        assert refit_skip_reason(out, _options(out)) is not None
        assert refit_skip_reason(
            out, _options(out, model=pp.KIND_LOGISTIC_SHARED),
        ) is None

    def test_a_tree_kind_is_refused_even_when_the_config_asks_for_one(
        self, tmp_path: Path,
    ) -> None:
        """The image has no LightGBM, so such a fit could only fail anyway."""
        out = tmp_path / "postprocess.json"
        out.write_text(INSTALLED)
        reason = refit_skip_reason(
            out, _options(out, model=pp.KIND_TREES, design=pp.DESIGN_V2),
        )
        assert reason is not None

    def test_hold_stops_a_refit_of_the_jobs_own_model(self, tmp_path: Path) -> None:
        out = tmp_path / "postprocess.json"
        out.write_text(NIGHTLY)
        reason = refit_skip_reason(out, _options(out), hold=True)
        assert reason is not None
        assert "hold" in reason
        assert "kind=logistic design=v1 protocol=at-gauge" in reason

    def test_hold_with_nothing_in_service_still_writes_nothing(
        self, tmp_path: Path,
    ) -> None:
        out = tmp_path / "postprocess.json"
        reason = refit_skip_reason(out, _options(out), hold=True)
        assert reason is not None and "hold" in reason


def test_the_config_field_defaults_to_off() -> None:
    assert FitPostprocessConfig().hold is False
    assert FitPostprocessConfig(hold=True).hold is True


def test_the_public_stack_pulls_the_model_it_can_never_fit() -> None:
    """The install lands on the private stack; ``sync`` carries it onward.

    The public instance runs its own push fan-out and has no gauge store,
    so ``calibration/postprocess.json`` has to be in its ``sync.files`` or
    it warns at thresholds fitted on a probability it does not have. And
    the body ceiling has to clear a tree document, which is megabytes where
    the other three artefacts are kilobytes.
    """
    from dmi_nowcast_sidecar.config import load_config
    from dmi_nowcast_sidecar.push.paths import resolved_postprocess_path
    from dmi_nowcast_sidecar.sync import POSTPROCESS_FILE, target_path

    path = (
        Path(__file__).resolve().parents[1]
        / "deploy" / "public" / "config.public.example.yaml"
    )
    cfg = load_config(path)
    assert POSTPROCESS_FILE in cfg.sync.files
    assert cfg.sync.max_bytes >= 8 * 1024 * 1024
    # And it lands where the running cycle reads it, not under data_dir by
    # its URL path.
    assert target_path(cfg, POSTPROCESS_FILE) == resolved_postprocess_path(cfg)


# ---------------------------------------------------------------------------
# The step
# ---------------------------------------------------------------------------


class TestTheStepLeavesItAlone:
    def test_it_neither_fits_nor_copies_nor_writes(self, tmp_path: Path) -> None:
        out = tmp_path / "postprocess.json"
        out.write_text(INSTALLED)
        calls: list[object] = []
        lines: list[str] = []
        summary = quality_job.run_postprocess_fit(
            _job(out),
            fitter=lambda options: calls.append(options) or {"model": None},
            log=lines.append,
        )
        assert calls == []
        assert out.read_text() == INSTALLED
        assert not (tmp_path / "postprocess.prev.json").exists()
        assert summary == {
            "postprocess_skipped": (
                "served model kind=trees design=v2 protocol=random-point is "
                "not reproducible by the nightly fit (model=logistic "
                "design=v1 protocol=at-gauge); left in place"
            ),
        }
        assert "postprocess_error" not in summary
        assert len(lines) == 1
        assert lines[0].startswith("postprocess_refit_skipped_external_model:")

    def test_the_hold_flag_is_honoured_on_the_step(self, tmp_path: Path) -> None:
        out = tmp_path / "postprocess.json"
        out.write_text(NIGHTLY)
        lines: list[str] = []
        summary = quality_job.run_postprocess_fit(
            _job(out, hold=True),
            fitter=lambda options: pytest.fail("the fit must not run"),
            log=lines.append,
        )
        assert "hold" in summary["postprocess_skipped"]
        assert out.read_text() == NIGHTLY
        assert lines == [
            f"postprocess_refit_skipped_hold: {summary['postprocess_skipped']}"
        ]

    def test_the_hold_flag_also_travels_inside_the_options(
        self, tmp_path: Path,
    ) -> None:
        """Whichever half of the job config carries it, the answer is the same."""
        out = tmp_path / "postprocess.json"
        out.write_text(NIGHTLY)
        job = _job(out)
        job["options"]["hold"] = True
        summary = quality_job.run_postprocess_fit(
            job, fitter=lambda options: pytest.fail("the fit must not run"),
        )
        assert "hold" in summary["postprocess_skipped"]

    def test_the_threshold_fit_still_runs_afterwards(self, tmp_path: Path) -> None:
        """The whole point of skipping rather than failing.

        The thresholds are refitted every night on the probabilities of
        whatever model is in service, so an installed artefact gets its own
        table without anyone doing anything.
        """
        out = tmp_path / "postprocess.json"
        out.write_text(INSTALLED)
        thresholds_out = tmp_path / "push_thresholds.json"
        config = {
            "quality": {
                "out_json": str(tmp_path / "quality.json"),
                "markdown_dir": None,
                "inputs": {},
            },
            "fit_postprocess": {"enabled": True, **_job(out)},
            "fit": {
                "enabled": True,
                "thresholds_out": str(thresholds_out),
                "options": {
                    "decisions_dirs": [str(tmp_path)],
                    "corpus_dir": str(tmp_path),
                },
            },
        }
        summary = quality_job.run_job(
            config,
            builder=lambda inputs, **kw: {"built_at_utc": "2026-09-22"},
            renderer=lambda report: "# quality\n",
            postprocess_fitter=lambda options: pytest.fail(
                "the refit must not run",
            ),
            fitter=lambda options: {"thresholds": {
                "schema_version": 1,
                "fitted_at_utc": "2026-09-22T03:40:00+00:00",
                "objective": {"probability_column": "p_post_{lead}"},
                "window": {},
                "fallback_threshold_pct": 40,
                "leads": {},
            }},
        )
        assert summary["ok"] is True
        assert "not reproducible" in summary["postprocess_skipped"]
        assert summary["postprocess_path"] is None
        assert summary["postprocess_error"] is None
        assert out.read_text() == INSTALLED
        # And the table was refitted and published beside it.
        assert summary["thresholds_path"] == str(thresholds_out)
        table = json.loads(thresholds_out.read_text())
        assert table["objective"]["probability_column"] == "p_post_{lead}"

    def test_a_refit_that_can_run_still_runs(self, tmp_path: Path) -> None:
        """The guard is narrow: it must not stop the normal night."""
        out = tmp_path / "postprocess.json"
        out.write_text(NIGHTLY)

        class _Model:
            fitted_at_utc = "2026-09-22T03:40:00+00:00"

            def dumps(self) -> str:
                return '{"tonight": true}\n'

        summary = quality_job.run_postprocess_fit(
            _job(out),
            fitter=lambda options: {"model": _Model(), "summary": {"rows": 12}},
        )
        assert summary["postprocess_path"] == str(out)
        assert out.read_text() == '{"tonight": true}\n'
        assert (tmp_path / "postprocess.prev.json").read_text() == NIGHTLY


# ---------------------------------------------------------------------------
# The other half: getting the artefact in there
# ---------------------------------------------------------------------------

INSTALLER = (
    Path(__file__).resolve().parents[1] / "deploy" / "install_artifact.sh"
)

#: The same header as ``INSTALLED``, with the per-lead models the installer
#: insists on: it validates the whole document, where the nightly guard
#: reads only what the document claims to be.
INSTALLED_FULL = json.dumps({
    **json.loads(INSTALLED),
    "models": {"20": {}, "30": {}, "45": {}, "60": {}},
}, indent=1) + "\n"

#: ssh and scp that fail loudly, so a --dry-run that tried to connect would
#: be a failing test rather than a slow one.
REFUSING_SSH = """#!/usr/bin/env bash
echo "$(basename "$0") must not be called" >&2
exit 99
"""


def _installer(tmp_path: Path, *args: str):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name in ("ssh", "scp"):
        stub = bin_dir / name
        stub.write_text(REFUSING_SSH)
        stub.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '/usr/bin:/bin')}"
    return subprocess.run(
        ["bash", str(INSTALLER), *args],
        capture_output=True, text=True, env=env, cwd=str(tmp_path),
    )


class TestTheInstaller:
    def test_a_dry_run_validates_and_prints_the_atomic_swap(
        self, tmp_path: Path,
    ) -> None:
        artefact = tmp_path / "postprocess.json"
        artefact.write_text(INSTALLED_FULL)
        proc = _installer(
            tmp_path, "--stack", "private", "--dry-run", str(artefact),
        )
        assert proc.returncode == 0, proc.stderr
        out = proc.stdout
        # What it is, read off the document rather than assumed.
        assert "kind=trees design=v2 protocol=random-point" in out
        assert "will skip itself" in out
        # The swap: staged under .tmp, one generation kept, renamed into
        # place, readable by the container's uid.
        assert "docker cp" in out
        assert "postprocess.json.tmp" in out
        assert "cp -p postprocess.json postprocess.prev.json" in out
        assert "chown 10001:10001 postprocess.json.tmp" in out
        assert "mv postprocess.json.tmp postprocess.json" in out
        # The right container for the stack, and the served path.
        assert "dmi-nowcast-sidecar" in out
        assert "/var/lib/dmi-nowcast/postprocess.json" in out
        assert "nothing was copied" in out

    def test_the_public_stack_names_the_public_container(
        self, tmp_path: Path,
    ) -> None:
        artefact = tmp_path / "postprocess.json"
        artefact.write_text(INSTALLED_FULL)
        proc = _installer(
            tmp_path, "--stack", "public", "--dry-run", str(artefact),
        )
        assert proc.returncode == 0, proc.stderr
        assert "dmi-nowcast-public" in proc.stdout
        assert "dmi-nowcast-sidecar:" not in proc.stdout

    def test_an_unknown_basename_is_refused(self, tmp_path: Path) -> None:
        """Only the two files something actually reads by path."""
        stray = tmp_path / "quality.json"
        stray.write_text("{}\n")
        proc = _installer(
            tmp_path, "--stack", "private", "--dry-run", str(stray),
        )
        assert proc.returncode == 2
        assert "refusing quality.json" in proc.stderr

    def test_a_document_that_is_not_a_model_is_refused(
        self, tmp_path: Path,
    ) -> None:
        artefact = tmp_path / "postprocess.json"
        artefact.write_text('{"schema_version": 1, "kind": "randomforest"}\n')
        proc = _installer(
            tmp_path, "--stack", "private", "--dry-run", str(artefact),
        )
        assert proc.returncode != 0
        assert "randomforest" in proc.stderr

    def test_a_threshold_table_reports_the_column_it_was_fitted_on(
        self, tmp_path: Path,
    ) -> None:
        """The classic mistake: a table fitted on one scale, served on another."""
        table = tmp_path / "push_thresholds.json"
        table.write_text(json.dumps({
            "schema_version": 1,
            "fitted_at_utc": "2026-09-22T03:40:00+00:00",
            "objective": {"probability_column": "p_rain_{lead}"},
            "window": {},
            "fallback_threshold_pct": 40,
            "leads": {"20": {"threshold_pct": 50, "guard": "first_fit"}},
        }))
        proc = _installer(
            tmp_path, "--stack", "private", "--dry-run", str(table),
        )
        assert proc.returncode == 0, proc.stderr
        assert "fitted_on=p_rain_{lead}" in proc.stdout
        assert "push.probability_source matches" in proc.stdout
        assert "mv push_thresholds.json.tmp push_thresholds.json" in proc.stdout
