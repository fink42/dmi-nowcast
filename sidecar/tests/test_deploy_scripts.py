"""Tests for the deploy-dir batch scripts (``sidecar/deploy/*.sh``).

Two things are worth a test here, and only two:

1. Every script parses. ``bash -n`` catches the quoting mistake that would
   otherwise surface three hours into a monthly run on the VM.
2. ``lib/batch.sh``'s two safety behaviours actually happen: the memory cap
   is issued with the configured size, and the "one batch at a time" guard
   refuses (or allows) as specified. Both exist because an uncapped batch
   container beside the live sidecar got the service OOM-killed 18 times.

No docker, no network: a stub ``docker`` on PATH records its argv.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_DIR = REPO_ROOT / "sidecar" / "deploy"
LIB = DEPLOY_DIR / "lib" / "batch.sh"

# The scripts that source the library and therefore must honour its rules.
BATCH_SCRIPTS = ["calibrate.sh", "station_corpus.sh", "replay.sh",
                 "radar_replay.sh"]

FAKE_DOCKER = """#!/usr/bin/env bash
# Stub docker: log the argv, answer ``ps`` from the environment.
printf '%s\\n' "$*" >> "$FAKE_DOCKER_LOG"
if [[ "${1:-}" == "ps" && -n "${FAKE_DOCKER_PS:-}" ]]; then
    printf '%s\\n' "$FAKE_DOCKER_PS"
fi
exit "${FAKE_DOCKER_RC:-0}"
"""


def _shell(snippet: str, tmp_path: Path, **env_extra: str):
    """Source lib/batch.sh with a stub docker on PATH and run ``snippet``."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    stub = bin_dir / "docker"
    stub.write_text(FAKE_DOCKER)
    stub.chmod(0o755)
    log = tmp_path / "docker.log"
    log.touch()

    env = {
        "PATH": f"{bin_dir}:{os.environ.get('PATH', '/usr/bin:/bin')}",
        "HOME": str(tmp_path),
        "FAKE_DOCKER_LOG": str(log),
        # Don't wait 25 s for the cap in a unit test.
        "BATCH_CAP_DELAY_S": "0",
    }
    env.update(env_extra)

    proc = subprocess.run(
        ["bash", "-c", f'set -uo pipefail\nsource "{LIB}"\n{snippet}\n'],
        capture_output=True, text=True, env=env, cwd=str(tmp_path),
    )
    return proc, log.read_text()


# --- 1. every script parses -------------------------------------------

@pytest.mark.parametrize(
    "script",
    sorted(p.name for p in DEPLOY_DIR.glob("*.sh")) + ["lib/batch.sh"],
)
def test_script_parses(script: str) -> None:
    proc = subprocess.run(["bash", "-n", str(DEPLOY_DIR / script)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, f"{script}: {proc.stderr}"


@pytest.mark.parametrize("script", BATCH_SCRIPTS)
def test_batch_scripts_use_the_library(script: str) -> None:
    """Each batch script sources the lib and takes the one-at-a-time guard.

    A script that grew its own ``run_in_repo`` again would run uncapped.
    """
    text = (DEPLOY_DIR / script).read_text()
    assert 'source "$DEPLOY_DIR/lib/batch.sh"' in text
    assert "require_no_batch_running" in text
    # No script may define its own run_in_repo — that is how the cap gets lost.
    assert "\nrun_in_repo() {" not in text


# --- 2. the memory cap ------------------------------------------------

def test_mem_cap_uses_the_configured_size(tmp_path: Path) -> None:
    proc, log = _shell(
        "batch_apply_mem_cap; batch_wait_mem_cap", tmp_path,
        BATCH_MEM_CAP="1234m", FAKE_DOCKER_PS="deploy-sidecar-run-abc123",
    )
    assert proc.returncode == 0, proc.stderr
    assert "update --memory 1234m --memory-swap 1234m deploy-sidecar-run-abc123" in log
    assert "memory cap 1234m applied to deploy-sidecar-run-abc123" in proc.stdout


def test_mem_cap_default_is_5000m(tmp_path: Path) -> None:
    """The default is the size measured to survive on the 12 GB VM."""
    proc, log = _shell(
        "batch_apply_mem_cap; batch_wait_mem_cap", tmp_path,
        FAKE_DOCKER_PS="deploy-sidecar-run-abc123",
    )
    assert proc.returncode == 0, proc.stderr
    assert "update --memory 5000m --memory-swap 5000m deploy-sidecar-run-abc123" in log


def test_mem_cap_targets_the_newest_container(tmp_path: Path) -> None:
    """``docker ps`` lists newest first; the cap goes on that one only."""
    proc, log = _shell(
        "batch_apply_mem_cap; batch_wait_mem_cap", tmp_path,
        FAKE_DOCKER_PS="deploy-sidecar-run-new\ndeploy-sidecar-run-old",
    )
    assert proc.returncode == 0, proc.stderr
    assert "deploy-sidecar-run-new" in log
    assert "update --memory 5000m --memory-swap 5000m deploy-sidecar-run-old" not in log


def test_mem_cap_complains_when_no_container_appears(tmp_path: Path) -> None:
    proc, log = _shell(
        "batch_apply_mem_cap; batch_wait_mem_cap", tmp_path, FAKE_DOCKER_PS="",
    )
    assert "update --memory" not in log
    assert "memory cap NOT applied" in proc.stderr


def test_capped_run_reaps_a_watcher_it_did_not_need(tmp_path: Path) -> None:
    """A step shorter than the delay leaves no stray background job."""
    proc, log = _shell(
        'run_in_repo_capped python -c pass; echo "pid=[${BATCH_CAP_PID}]"',
        tmp_path, BATCH_CAP_DELAY_S="30", FAKE_DOCKER_PS="deploy-sidecar-run-abc",
    )
    assert proc.returncode == 0, proc.stderr
    assert "pid=[]" in proc.stdout
    assert "update --memory" not in log


# --- 3. the one-batch-at-a-time guard ---------------------------------

def test_guard_refuses_when_a_batch_is_running(tmp_path: Path) -> None:
    proc, _ = _shell(
        "require_no_batch_running || exit 1", tmp_path,
        FAKE_DOCKER_PS="deploy-sidecar-run-abc123",
    )
    assert proc.returncode == 1
    assert "REFUSING TO START" in proc.stderr
    assert "deploy-sidecar-run-abc123" in proc.stderr


def test_guard_allows_when_idle(tmp_path: Path) -> None:
    proc, _ = _shell(
        "require_no_batch_running || exit 1; echo allowed", tmp_path,
        FAKE_DOCKER_PS="",
    )
    assert proc.returncode == 0, proc.stderr
    assert "allowed" in proc.stdout


def test_guard_can_be_forced(tmp_path: Path) -> None:
    proc, _ = _shell(
        "require_no_batch_running || exit 1; echo allowed", tmp_path,
        FAKE_DOCKER_PS="deploy-sidecar-run-abc123", BATCH_FORCE="1",
    )
    assert proc.returncode == 0, proc.stderr
    assert "allowed" in proc.stdout
    assert "BATCH_FORCE=1" in proc.stderr


# --- 4. run_in_repo contract ------------------------------------------

def test_run_in_repo_mounts_the_tree_read_only(tmp_path: Path) -> None:
    proc, log = _shell("run_in_repo python -c pass", tmp_path, FAKE_DOCKER_PS="")
    assert proc.returncode == 0, proc.stderr
    assert f"compose run --rm -T -v {REPO_ROOT}:/repo:ro" in log
    assert "--workdir /repo" in log
    assert "-e PYTHONPATH=/repo/src" in log


def test_run_in_repo_takes_extra_mounts(tmp_path: Path) -> None:
    """The replay scripts bind their days file in this way."""
    proc, log = _shell(
        'BATCH_RUN_ARGS=(-v "/tmp/days.txt:/tmp/replay_days.txt:ro")\n'
        "run_in_repo python -c pass",
        tmp_path, FAKE_DOCKER_PS="",
    )
    assert proc.returncode == 0, proc.stderr
    assert "-v /tmp/days.txt:/tmp/replay_days.txt:ro" in log


def test_helpers_ignore_the_callers_cwd(tmp_path: Path) -> None:
    """A deploy can delete the cwd under a long run; helpers cd themselves."""
    proc, log = _shell(
        'mkdir -p "$HOME/gone" && cd "$HOME/gone" && rmdir "$HOME/gone"\n'
        "run_in_repo python -c pass",
        tmp_path, FAKE_DOCKER_PS="",
    )
    assert proc.returncode == 0, proc.stderr
    assert f"compose run --rm -T -v {REPO_ROOT}:/repo:ro" in log


def test_default_worker_count_is_two(tmp_path: Path) -> None:
    """Two STEPS workers is the number this VM survives. See lib/batch.sh."""
    proc, _ = _shell('echo "workers=$BATCH_WORKERS"', tmp_path, FAKE_DOCKER_PS="")
    assert "workers=2" in proc.stdout


# --- 5. the stable copies are published atomically --------------------

def test_publish_atomic_replaces_the_target(tmp_path: Path) -> None:
    """The nightly report may read the stable name mid-run: no torn file.

    ``run_in_repo`` is shimmed to run the embedded python locally instead
    of in a container, so this exercises the real copy-then-rename.
    """
    src = tmp_path / "national_corpus_20260901.parquet"
    src.write_bytes(b"new corpus bytes")
    dst = tmp_path / "sub" / "latest.parquet"
    dst.parent.mkdir()
    dst.write_bytes(b"last month")

    proc, _ = _shell(
        'run_in_repo() { python3 "${@:2}"; }\n'
        f'batch_publish_atomic "{src}" "{dst}"',
        tmp_path, FAKE_DOCKER_PS="",
    )
    assert proc.returncode == 0, proc.stderr
    assert dst.read_bytes() == b"new corpus bytes"
    # The temp sibling is renamed, never left behind.
    assert list(dst.parent.glob("*.tmp.*")) == []


def test_publish_atomic_uses_rename_not_a_plain_copy() -> None:
    """A plain ``cp`` onto the live path is the bug this replaced."""
    body = LIB.read_text().split("batch_publish_atomic()")[1]
    assert "os.replace(tmp, dst)" in body
    assert "shutil.copyfile(src, tmp)" in body


@pytest.mark.parametrize(
    "script,src_hint",
    [("calibrate.sh", "latest_path"), ("station_corpus.sh", "stable")],
)
def test_stable_names_are_published_atomically(script: str, src_hint: str) -> None:
    text = (DEPLOY_DIR / script).read_text()
    assert 'batch_publish_atomic "$' in text
    assert src_hint in text
    # No un-atomic copy onto a stable name survived the change.
    assert 'run_in_repo cp -f "$joined"' not in text
    assert 'run_in_repo cp -f "$corpus_path"' not in text


# --- 6. calibrate.sh carries the station step -------------------------

def test_calibrate_runs_the_gauge_half() -> None:
    """The systemd timer covers the whole routine without a unit change."""
    text = (DEPLOY_DIR / "calibrate.sh").read_text()
    assert "scripts/join_gauge_truth.py" in text
    assert 'CALIBRATION_STATIONS:-1' in text
    # The CALIBRATION_UNION=0 fallback still delegates to the old path.
    assert '"$DEPLOY_DIR/station_corpus.sh"' in text


def test_calibrate_never_fails_over_the_gauge_half() -> None:
    """A missing points file or a failed join must not fail the curves."""
    text = (DEPLOY_DIR / "calibrate.sh").read_text()
    tail = text.split("--- the gauge half")[1]
    assert "non-fatal" in tail
    # The only exit condition left is whether the curves are being served.
    assert '[[ "$fit_served" == 1 ]] || exit 1' in tail
    # A missing points file downgrades the run, it does not abort it.
    head = text.split("--- the gauge half")[0]
    assert "batch_container_file_exists \"$station_points\"" in head
    assert "want_stations=0" in head


# --- 7. the union build -----------------------------------------------

def test_calibrate_builds_one_corpus_for_both_point_sets() -> None:
    text = (DEPLOY_DIR / "calibrate.sh").read_text()
    # Repeatable --points, assembled before the build.
    assert 'points_args=(--points "$points_path")' in text
    assert 'points_args+=(--points "$station_points")' in text
    assert '"${points_args[@]}"' in text
    # One builder invocation, not two.
    assert text.count("scripts/build_calibration_corpus.py") == 1


def test_calibrate_names_the_point_set_on_every_consumer() -> None:
    """fit and report take the radar set; the join takes the station set."""
    text = (DEPLOY_DIR / "calibrate.sh").read_text()
    for script, arg in [
        ("scripts/fit_national_calibration.py", '--point-set "$radar_set"'),
        ("scripts/national_calibration_report.py", '--point-set "$radar_set"'),
        ("scripts/join_gauge_truth.py", '--point-set "$station_set"'),
    ]:
        block = text.split(script)[1][:400]
        assert arg in block, f"{script} is missing {arg}"


def test_point_set_labels_are_derived_from_the_file_stems() -> None:
    """points_set_name() in the builder is the file's stem; match it."""
    text = (DEPLOY_DIR / "calibrate.sh").read_text()
    assert 'radar_set=$(basename "$points_path" .json)' in text
    assert 'station_set=$(basename "$station_points" .json)' in text
    station = (DEPLOY_DIR / "station_corpus.sh").read_text()
    assert 'point_set=$(basename "$points" .json)' in station


def test_station_corpus_prefers_joining_the_union_corpus() -> None:
    """Rebuilding rows that are already on disk costs 3.5 h for nothing."""
    text = (DEPLOY_DIR / "station_corpus.sh").read_text()
    assert "STATION_REUSE_CORPUS" in text
    assert "STATION_UNION_CORPUS" in text
    assert 'calibration/latest.parquet' in text
    # The build is behind the reuse check, and the join is not.
    reuse_at = text.index("reuse=$(run_in_repo python -")
    build_at = text.index("scripts/build_calibration_corpus.py")
    join_at = text.index("scripts/join_gauge_truth.py")
    assert reuse_at < build_at < join_at
    assert '--point-set "$point_set"' in text


def test_calibration_stations_zero_drops_both_halves() -> None:
    """CALIBRATION_STATIONS=0 means no station points AND no join."""
    text = (DEPLOY_DIR / "calibrate.sh").read_text()
    head = text.split("--- the gauge half")[0]
    # The union is only assembled when stations are wanted.
    gate = head.split("want_stations=1")[1]
    assert 'CALIBRATION_STATIONS:-1' in gate
    assert "points_args+=" in gate
    tail = text.split("--- the gauge half")[1]
    assert '"$want_stations" == 0' in tail
