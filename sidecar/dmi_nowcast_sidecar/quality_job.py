"""The nightly quality build, as a process that exits.

``python -m dmi_nowcast_sidecar.quality_job --config-json '{...}'``

Everything the nightly job does — the push-threshold fit, the report
build, the two atomic writes — lives here, behind a ``main()`` that takes
its resolved inputs and outputs as one JSON document and prints a
one-line JSON summary on stdout. Nothing else is written to stdout, so a
caller can parse the last line and be sure of what it got.

Why a separate process at all
-----------------------------

The build reads the whole evidence base into Arrow and numpy: the radar
corpus (~1.7 M rows), the station corpus (~1.5 M), every decision row of
the replay plus the live scoreboard, and the gauge store behind them.
Peak is gigabytes, and neither Arrow's memory pool nor CPython's
allocator hands that back to the kernel afterwards — a service that
builds the report in-process keeps the high-water mark for the rest of
its life. On the 12 GB VM the live sidecar went from ~0.9 GB to 5.5 GB
anon RSS after one nightly build and was then picked by the kernel's OOM
killer while a batch replay ran beside it.

A child process gives the memory back by exiting. It also means a crash,
a hang or an OOM kill inside the build is contained: the parent sees a
non-zero exit or a timeout, logs it, and the service keeps serving
yesterday's report.

Both callers come through here
------------------------------

The nightly task (``quality_report.QualityReportTask``) spawns this
module; ``sidecar/deploy/quality_report.sh`` runs the same command line
in a throwaway container for the first build. One implementation, so the
manual path and the scheduled one cannot drift.

The one thing that stays in the parent is the in-process hook: after a
successful fit the running service is told to re-read its threshold
table (``push.thresholds.ThresholdTable.note_changed``). A child process
cannot reach into the parent's objects, and it does not need to — the
file is on disk by then.

Failure policy, unchanged from the in-process version: a missing input
nulls its section, a failed fit leaves the table already in service
exactly where it was, and a failed build leaves the previous report
exactly where it is. Both writes are tmp + rename in the target
directory, so a reader never sees half a document.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

#: What the parent spawns: ``sys.executable -m <this>``. Named here so the
#: task, the tests and the deploy script all quote the same string.
JOB_MODULE = "dmi_nowcast_sidecar.quality_job"

#: ``QualityInputs`` fields that carry a filesystem path.
_INPUT_PATH_FIELDS = frozenset({
    "radar_corpus",
    "station_corpus",
    "replay_dir",
    "corpus_dir",
    "persistence_json",
    "national_curves",
    "thresholds_path",
})

#: ``SweepOptions`` fields that carry a path, or a list of them.
_SWEEP_PATH_FIELDS = frozenset({"corpus_dir"})
_SWEEP_PATH_LIST_FIELDS = frozenset({"decisions_dirs", "radar_decisions_dirs"})
_SWEEP_TUPLE_FIELDS = frozenset({"leads", "thresholds", "strata"})

#: The report sections named in the summary line, in document order.
REPORT_SECTIONS = (
    "windows", "headline", "reliability", "raining_now",
    "stations", "events", "methods",
)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def write_atomic(path: Path, text: str) -> None:
    """tmp + rename in the target directory; a reader never sees a half file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


# ---------------------------------------------------------------------------
# The config document: dataclasses across a process boundary
# ---------------------------------------------------------------------------


def _as_path(value: Any) -> Path | None:
    return None if value in (None, "") else Path(str(value))


def inputs_to_json(inputs: Any) -> dict:
    """A :class:`QualityInputs` as plain JSON types."""
    out: dict[str, Any] = {}
    for field in dataclasses.fields(inputs):
        value = getattr(inputs, field.name)
        if value is None:
            out[field.name] = None
        elif field.name in _INPUT_PATH_FIELDS:
            out[field.name] = str(value)
        elif isinstance(value, datetime):
            out[field.name] = value.isoformat()
        elif isinstance(value, (list, tuple)):
            out[field.name] = list(value)
        else:
            out[field.name] = value
    return out


def inputs_from_json(payload: dict) -> Any:
    """The :class:`QualityInputs` ``inputs_to_json`` encoded."""
    from dmi_nowcast_core.quality_report import QualityInputs

    known = {field.name for field in dataclasses.fields(QualityInputs)}
    kwargs: dict[str, Any] = {}
    for name, value in (payload or {}).items():
        if name not in known:
            continue  # a newer writer, an older reader: ignore, don't crash
        if name in _INPUT_PATH_FIELDS:
            kwargs[name] = _as_path(value)
        elif name == "now":
            kwargs[name] = None if value is None else datetime.fromisoformat(value)
        elif name == "served_leads":
            kwargs[name] = None if value is None else tuple(int(v) for v in value)
        else:
            kwargs[name] = value
    return QualityInputs(**kwargs)


def sweep_options_to_json(options: Any) -> dict:
    """A :class:`SweepOptions` as plain JSON types."""
    out: dict[str, Any] = {}
    for field in dataclasses.fields(options):
        value = getattr(options, field.name)
        if field.name in _SWEEP_PATH_LIST_FIELDS:
            out[field.name] = None if value is None else [str(p) for p in value]
        elif field.name in _SWEEP_PATH_FIELDS:
            out[field.name] = None if value is None else str(value)
        elif isinstance(value, (list, tuple)):
            out[field.name] = list(value)
        else:
            out[field.name] = value
    return out


def sweep_options_from_json(payload: dict) -> Any:
    """The :class:`SweepOptions` ``sweep_options_to_json`` encoded."""
    from .threshold_sweep import SweepOptions

    known = {field.name for field in dataclasses.fields(SweepOptions)}
    kwargs: dict[str, Any] = {}
    for name, value in (payload or {}).items():
        if name not in known:
            continue
        if name in _SWEEP_PATH_LIST_FIELDS:
            kwargs[name] = None if value is None else [Path(p) for p in value]
        elif name in _SWEEP_PATH_FIELDS:
            kwargs[name] = None if value is None else Path(value)
        elif name in _SWEEP_TUPLE_FIELDS:
            kwargs[name] = () if value is None else tuple(value)
        else:
            kwargs[name] = value
    return SweepOptions(**kwargs)


# ---------------------------------------------------------------------------
# The two steps
# ---------------------------------------------------------------------------


def run_threshold_fit(
    fit: dict,
    *,
    fitter: Callable[[Any], dict] | None = None,
    log: Callable[[str], None] | None = None,
) -> dict:
    """Refit the push thresholds and publish the guarded table.

    Returns the summary fields this step owns: ``thresholds_path`` and
    ``thresholds_guard`` on success, ``thresholds_error`` otherwise.

    Never raises. Every failure — no corpus, no rows, a sweep that blew
    up — leaves the table already in service exactly where it was, which
    is the same failure policy as the report itself.
    """
    from dmi_nowcast_core.push_thresholds import (
        apply_stability_guard,
        load_thresholds,
    )

    from .threshold_sweep import SweepError, run_fit

    out = Path(fit["thresholds_out"])
    try:
        run = fitter or run_fit
        payload = run(sweep_options_from_json(fit.get("options") or {}))
        new_doc = payload["thresholds"]
        # The full sweep record — every cell, the picks, the radar
        # cross-check — kept beside the table when a caller asks for it.
        # The nightly job does not; the manual run does, because that one
        # is read by a human deciding whether to trust the picks.
        sweep_json = fit.get("sweep_json")
        if sweep_json:
            write_atomic(Path(sweep_json), json.dumps(payload, indent=1) + "\n")
        # Guard against the table in service, which is the file we are
        # about to overwrite — read it BEFORE the write, obviously, and
        # treat an unusable one as a first fit.
        previous = load_thresholds(out)
        doc = apply_stability_guard(
            new_doc, previous,
            min_delta_pct=int(fit.get("min_delta_pct", 5)),
            min_warnings=int(fit.get("min_warnings", 30)),
        )
        write_atomic(out, json.dumps(doc, indent=1) + "\n")
    except SweepError as exc:
        # Nothing to fit on: not a bug, and not worth a warning every
        # night while the corpus is still filling up.
        if log:
            log(f"threshold fit skipped: {exc}")
        return {"thresholds_error": str(exc)}
    except Exception as exc:  # noqa: BLE001 — the report still builds
        if log:
            log(f"threshold fit failed: {type(exc).__name__}: {exc}")
        return {"thresholds_error": f"{type(exc).__name__}: {exc}"}
    guard = {
        key: str(entry.get("guard"))
        for key, entry in (doc.get("leads") or {}).items()
        if isinstance(entry, dict)
    }
    if log:
        log(f"threshold fit done: {out} {guard}")
    return {"thresholds_path": str(out), "thresholds_guard": guard}


def run_job(
    config: dict,
    *,
    builder: Callable[[Any], dict] | None = None,
    renderer: Callable[[dict], str] | None = None,
    fitter: Callable[[Any], dict] | None = None,
    log: Callable[[str], None] | None = None,
) -> dict:
    """The whole nightly job: fit, build, write, summarise.

    The injectables exist for the tests, which exercise the wiring — the
    guard, the atomic writes, the summary — without a corpus on disk.
    Production passes none of them and gets the real builder.

    Raises whatever the builder raises: a build that fails must not
    overwrite the report that is already on disk, and the caller (the
    parent task, or :func:`main`) is what turns that into a log line.
    """
    from dmi_nowcast_core.quality_report import (
        build_quality_report,
        render_markdown,
        validate_report,
    )

    build = builder or build_quality_report
    render = renderer or render_markdown
    quality = config["quality"]

    # The fit first, so the builder is handed tonight's table and
    # ``quality.json``'s thresholds section describes the rule the service
    # is on as of now.
    fit_summary: dict = {}
    fit = config.get("fit") or {}
    if fit.get("enabled"):
        fit_summary = run_threshold_fit(fit, fitter=fitter, log=log)

    inputs = inputs_from_json(quality.get("inputs") or {})
    if log:
        log("building the report")
    report = build(inputs)
    problems = validate_report(report)
    payload = json.dumps(report, indent=1, sort_keys=False)
    path = Path(quality["out_json"])
    write_atomic(path, payload)

    markdown_dir = quality.get("markdown_dir")
    if markdown_dir is not None:
        stamp = str(report.get("generated_at_utc") or "")[:10] or (
            datetime.now(timezone.utc).date().isoformat()
        )
        try:
            write_atomic(Path(markdown_dir) / f"{stamp}.md", render(report))
        except Exception as exc:  # noqa: BLE001 — the archive twin is a nicety
            if log:
                log(f"markdown twin failed: {type(exc).__name__}: {exc}")

    summary = {
        "ok": True,
        "path": str(path),
        "bytes": len(payload.encode("utf-8")),
        "sections": [
            key for key in REPORT_SECTIONS
            if isinstance(report, dict) and report.get(key) is not None
        ],
        "schema_problems": problems,
        "thresholds_path": None,
        "thresholds_guard": {},
        "thresholds_error": None,
        "error": None,
    }
    summary.update(fit_summary)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _read_config(value: str) -> dict:
    """The job config: inline JSON, ``@path`` to a file, or ``-`` for stdin."""
    if value == "-":
        text = sys.stdin.read()
    elif value.startswith("@"):
        text = Path(value[1:]).read_text(encoding="utf-8")
    else:
        text = value
    config = json.loads(text)
    if not isinstance(config, dict) or "quality" not in config:
        raise ValueError("job config must be an object carrying 'quality'")
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"python -m {JOB_MODULE}",
        description="Build quality.json (and optionally refit the push "
                    "thresholds) in a process that exits afterwards.",
    )
    parser.add_argument(
        "--config-json", required=True,
        help="the resolved job config as JSON: inline, @file, or - for stdin",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the job described by ``--config-json``; print one JSON line.

    Exit 0 when the report was written, 1 otherwise. Progress goes to
    stderr so stdout carries exactly one thing: the summary the parent
    parses.
    """
    args = build_parser().parse_args(argv)

    def log(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    try:
        config = _read_config(args.config_json)
        summary = run_job(config, log=log)
    except Exception as exc:  # noqa: BLE001 — the exit code is the contract
        traceback.print_exc(file=sys.stderr)
        summary = {
            "ok": False,
            "path": None,
            "bytes": 0,
            "sections": [],
            "schema_problems": [],
            "thresholds_path": None,
            "thresholds_guard": {},
            "thresholds_error": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if summary.get("ok") else 1


__all__ = [
    "JOB_MODULE",
    "REPORT_SECTIONS",
    "build_parser",
    "inputs_from_json",
    "inputs_to_json",
    "main",
    "redirect_structlog_to_stderr",
    "run_job",
    "run_threshold_fit",
    "sweep_options_from_json",
    "sweep_options_to_json",
    "write_atomic",
]


def redirect_structlog_to_stderr() -> None:
    """Keep stdout clean for the summary line.

    Anything in the tree that logs through structlog would otherwise
    print to stdout under the default configuration and corrupt the one
    line the parent parses. Called only when this module IS the process —
    it mutates global logging state, which a caller inside another
    program has every right not to want.
    """
    try:
        import structlog

        structlog.configure(
            logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        )
    except Exception:  # noqa: BLE001 — logging setup must never fail the job
        pass


if __name__ == "__main__":  # pragma: no cover — exercised as a subprocess
    redirect_structlog_to_stderr()
    raise SystemExit(main())
