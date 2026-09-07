#!/usr/bin/env bash
# Shared helpers for the deploy dir's BATCH jobs (calibrate, station
# corpus, replays). Source it; do not execute it:
#
#   source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/batch.sh"
#
# ---------------------------------------------------------------------
# Why this file exists: the memory rules
# ---------------------------------------------------------------------
# The VM has 12 GB shared between the LIVE sidecar, the public stack, and
# whatever batch job is running. A STEPS worker (16 members, 432x496)
# holds 1.3-2.0 GB of anonymous RSS. Uncapped ``docker compose run``
# containers with 3-5 workers got the LIVE sidecar chosen by the kernel's
# global OOM killer 18 times on 2026-09-05/06 — ``oom_score_adj: -500``
# on the service biases the choice but does not make it impossible when
# the machine is genuinely out of memory.
#
# What works, and what every batch job here does:
#   1. TWO workers (BATCH_WORKERS), not three or five.
#      Three is now plausible and is the next thing to try: the September
#      2026 STEPS work brought peak RSS per worker to ~1.65 GB, so
#      3 x 1.65 = 5.0 GB sits right at the cap and a squeeze still costs
#      a batch worker rather than the service. The shipped default stays
#      2 until a real monthly run confirms it — the measurement that
#      matters is a 3.5 h build under the live sidecar, not a benchmark.
#   2. A HARD memory cap on the batch container, applied ~25 s after it
#      starts (BATCH_MEM_CAP, default 5000m). Under the cap the cgroup's
#      own OOM killer fires first and kills a BATCH worker — the pool
#      restarts it — instead of the global one killing the service.
#      The cap cannot go in docker-compose.yml: it must land on the
#      throwaway ``deploy-sidecar-run-*`` container, not on the service,
#      and compose gives both the same service definition. Hence
#      ``docker update`` on the container once it exists.
#   3. ONE batch job at a time (require_no_batch_running).
#
# Every helper ``cd``s into the deploy dir itself. Nothing here may rely
# on an inherited cwd: a deploy that replaces ~/dmi-nowcast mid-run
# leaves a long-running chain's shell sitting on a deleted directory.
#
# Configuration (all overridable by the caller's environment):
#   BATCH_MEM_CAP       hard cap for the batch container (default 5000m)
#   BATCH_WORKERS       STEPS/pool workers (default 2; try 3, see above)
#   BATCH_CAP_DELAY_S   seconds to wait for the container (default 25)
#   BATCH_FORCE         1 to bypass require_no_batch_running
#   BATCH_PYTHONPATH    PYTHONPATH inside the container (default /repo/src)
#   BATCH_RUN_ARGS      bash array of extra ``docker compose run`` flags,
#                       e.g. BATCH_RUN_ARGS=(-v "$HOME/days.txt:/tmp/d:ro")
#   DOCKER              docker binary (default ``docker``; the tests
#                       point this at a stub via PATH)

# Idempotent: the chain scripts source each other's libs.
[[ -n "${_DMI_BATCH_LIB_LOADED:-}" ]] && return 0
_DMI_BATCH_LIB_LOADED=1

# Absolute, resolved once. Every helper cd's here by absolute path, so a
# deleted-and-recreated deploy dir re-resolves instead of erroring.
_BATCH_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="${DEPLOY_DIR:-$(cd "$_BATCH_LIB_DIR/.." && pwd)}"
REPO_ROOT="${REPO_ROOT:-$(cd "$DEPLOY_DIR/../.." && pwd)}"

DOCKER=${DOCKER:-docker}
BATCH_MEM_CAP=${BATCH_MEM_CAP:-5000m}
BATCH_WORKERS=${BATCH_WORKERS:-2}
BATCH_CAP_DELAY_S=${BATCH_CAP_DELAY_S:-25}
BATCH_CONTAINER_FILTER=${BATCH_CONTAINER_FILTER:-deploy-sidecar-run}
BATCH_PYTHONPATH=${BATCH_PYTHONPATH:-/repo/src}
BATCH_FORCE=${BATCH_FORCE:-0}
# Extra flags for ``docker compose run`` (additional -v mounts, -e vars).
# Guarded so a caller can set it before sourcing.
if [[ -z "${BATCH_RUN_ARGS+x}" ]]; then
    declare -a BATCH_RUN_ARGS=()
fi
BATCH_CAP_PID=""

# --- docker plumbing --------------------------------------------------

# ``docker compose ...`` from the deploy dir, whatever the caller's cwd.
compose() {
    ( cd "$DEPLOY_DIR" && "$DOCKER" compose "$@" )
}

# Run a command in a throwaway container with the working tree mounted
# read-only at /repo. Scripts are not baked into the runtime image (it
# stays lean), so they are mounted on demand — same pattern calibrate.sh
# has always used, now in one place.
#
# ``-T`` because every caller is non-interactive (cron, systemd, a
# detached ssh chain): without it compose may try to allocate a TTY it
# cannot have.
run_in_repo() {
    ( cd "$DEPLOY_DIR" && "$DOCKER" compose run --rm -T \
        -v "$REPO_ROOT:/repo:ro" \
        ${BATCH_RUN_ARGS[@]+"${BATCH_RUN_ARGS[@]}"} \
        --workdir /repo \
        -e PYTHONPATH="$BATCH_PYTHONPATH" \
        sidecar \
        "$@" )
}

# Newest ``deploy-sidecar-run-*`` container, or "" if none is running.
# ``docker ps`` lists newest first.
batch_running_container() {
    "$DOCKER" ps --filter "name=${BATCH_CONTAINER_FILTER}" --format '{{.Names}}' 2>/dev/null | head -1
}

# --- the guard --------------------------------------------------------

# Refuse to start a second batch job beside a running one. Two of these
# next to the live service is exactly the memory shape that OOMs the VM.
# Returns 1 (the caller should exit) unless BATCH_FORCE=1.
require_no_batch_running() {
    local existing
    existing=$(batch_running_container)
    if [[ -z "$existing" ]]; then
        return 0
    fi
    if [[ "${BATCH_FORCE:-0}" == "1" ]]; then
        echo "!! a batch container is already running (${existing}) — BATCH_FORCE=1, continuing" >&2
        return 0
    fi
    echo "REFUSING TO START: a batch container is already running (${existing})." >&2
    echo "  Two batch jobs beside the live sidecar is how this VM runs out of" >&2
    echo "  memory. Wait for it to finish (docker ps), or set BATCH_FORCE=1 if" >&2
    echo "  you have checked the memory budget yourself." >&2
    return 1
}

# --- the memory cap ---------------------------------------------------

# Start a background watcher that caps the batch container once it
# exists. Call it immediately BEFORE the long-running run_in_repo, so the
# container the watcher finds is that one and not a short-lived helper.
# Sets BATCH_CAP_PID.
batch_apply_mem_cap() {
    local cap=${1:-$BATCH_MEM_CAP}
    local delay=${2:-$BATCH_CAP_DELAY_S}
    (
        sleep "$delay"
        name=$(batch_running_container)
        if [[ -z "$name" ]]; then
            echo "!! no ${BATCH_CONTAINER_FILTER}-* container after ${delay}s — memory cap NOT applied" >&2
            exit 1
        fi
        if "$DOCKER" update --memory "$cap" --memory-swap "$cap" "$name" >/dev/null; then
            echo "    memory cap ${cap} applied to ${name}"
        else
            echo "!! docker update failed on ${name} — the batch is running UNCAPPED" >&2
            exit 1
        fi
    ) &
    BATCH_CAP_PID=$!
}

# Wait for the watcher to finish (it is a sleep plus one docker call).
batch_wait_mem_cap() {
    [[ -n "${BATCH_CAP_PID:-}" ]] || return 0
    wait "$BATCH_CAP_PID" 2>/dev/null || true
    BATCH_CAP_PID=""
}

# Stop a watcher that is no longer needed (the step finished before the
# delay elapsed, so there is nothing left to cap).
#
# The ``pkill -P`` matters: killing the subshell alone orphans its
# ``sleep``, which holds the script's stdout pipe open until the delay
# runs out — anything reading that pipe (a log collector, the test suite)
# then waits on a job that is already finished.
batch_reap_mem_cap() {
    [[ -n "${BATCH_CAP_PID:-}" ]] || return 0
    pkill -P "$BATCH_CAP_PID" 2>/dev/null || true
    kill "$BATCH_CAP_PID" 2>/dev/null || true
    wait "$BATCH_CAP_PID" 2>/dev/null || true
    BATCH_CAP_PID=""
}

# run_in_repo with the cap watcher around it. This is what every heavy
# step should use; run_in_repo bare is for the seconds-long helpers
# (mkdir, cp, file tests) that never grow.
run_in_repo_capped() {
    local rc=0
    batch_apply_mem_cap
    run_in_repo "$@" || rc=$?
    batch_reap_mem_cap
    return "$rc"
}

# --- misc -------------------------------------------------------------

# Read the live settings out of the RUNNING config, so a corpus can never
# drift from what the service serves. Prints one space-separated line:
#   radius_m ensemble_size cascades downsample threshold stat leads_csv
batch_live_settings() {
    compose exec -T sidecar python - <<'PY'
from dmi_nowcast_sidecar.config import load_config
c = load_config()
print(c.home.radius_km * 1000.0,
      c.forecast.steps.ensemble_size, c.forecast.steps.n_cascade_levels,
      c.forecast.steps.downsample_factor, c.forecast.rain_threshold_mm_h,
      c.forecast.detection_stat, ",".join(str(l) for l in c.forecast.leads_min))
PY
}

# Publish <src> at <dst> atomically, from inside the container.
#
# The stable names (calibration/latest.parquet,
# stations/station_corpus_gauge.parquet) are read by the nightly quality
# report's child process at 03:30 UTC while this job may still be running.
# A plain ``cp`` onto a live path lets that reader open a half-written
# parquet and fail on a truncated footer. So: copy to a temp name in the
# SAME directory, then os.replace — atomic on POSIX within one filesystem,
# and a sibling in the same directory always is one.
batch_publish_atomic() {
    run_in_repo python - "$1" "$2" <<'ATOMIC'
import os
import shutil
import sys

src, dst = sys.argv[1], sys.argv[2]
tmp = f"{dst}.tmp.{os.getpid()}"
os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
try:
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
except BaseException:
    try:
        os.unlink(tmp)
    except OSError:
        pass
    raise
print(f"    published {dst}")
ATOMIC
}

# True when a container-side path exists. ``test`` runs in the container
# because these are container paths on a bind mount the host may not see
# at the same location.
batch_container_file_exists() {
    run_in_repo python -c 'import os,sys; sys.exit(0 if os.path.isfile(sys.argv[1]) else 1)' "$1"
}
