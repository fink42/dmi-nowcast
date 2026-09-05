#!/usr/bin/env bash
# One-off build of the quality report ("How good are we?", Phase F, F4).
#
# The running sidecar builds this nightly once ``quality_report.enabled``
# is on. This script is for the FIRST one — the page 503s until a document
# exists, and waiting until 03:30 to find out the paths were wrong is a
# poor way to learn it.
#
# Same shape as calibrate.sh: run on the *host*, mount the repo into a
# throwaway container so scripts/ is available (the runtime image stays
# lean), write into the volumes the service already has.
#
# Every input is OPTIONAL. A missing one nulls its section of the report
# instead of faking it, so this is safe to run before the whole corpus
# exists — the first report can be a window and a persistence margin, and
# it grows as the evidence does. The summary printed at the end names the
# sections that came out non-null.
#
# Configuration (env or CLI):
#   QUALITY_RADAR_CORPUS      national calibration corpus parquet
#                             (default: the newest under
#                             $CORPUS_DIR/calibration/)
#   QUALITY_STATION_CORPUS    station corpus widened by join_gauge_truth.py
#                             (default: $CORPUS_DIR/stations/station_corpus_gauge.parquet)
#   QUALITY_REPLAY_DIR        replay_warnings.py output directory
#                             (default: $CORPUS_DIR/stations/replay)
#   QUALITY_PERSISTENCE_JSON  persistence_vs_advection.py results.json
#                             (default: $CORPUS_DIR/pva/results.json)
#   QUALITY_CURVES            served national isotonic curves
#                             (default: /var/lib/dmi-nowcast/national_curves.json)
#   QUALITY_CORPUS_DIR        corpus root (default /var/lib/dmi-nowcast-corpus)
#   QUALITY_OUT               served document
#                             (default /var/lib/dmi-nowcast/nowcast/quality.json)
#   QUALITY_MD_DIR            markdown archive directory
#                             (default $CORPUS_DIR/quality_reports)
#   QUALITY_LIVE_DAYS         live stations/eval lookback (default 90)
#
# The push-threshold fit (Phase G, G4) is an OPTIONAL first step, off
# unless QUALITY_FIT_THRESHOLDS=1. It replays the push rule over the
# decision rows at every threshold on the grid and writes the horizon →
# threshold table the notifications warn by. The running service re-reads
# it at its next fan-out — no restart — and the report below then embeds
# it. This is how the FIRST table is made; after that, turn on
# quality_report.fit_thresholds.enabled and it happens nightly.
#
#   QUALITY_FIT_THRESHOLDS    1 to fit the push thresholds first (default off)
#   QUALITY_THRESHOLDS_OUT    the table the service reads
#                             (default /var/lib/dmi-nowcast/push_thresholds.json)
#   QUALITY_DECISIONS_DIRS    space-separated decision trees, later wins
#                             (default $CORPUS_DIR/stations/replay/decisions
#                                      $CORPUS_DIR/stations/eval)
#   QUALITY_RADAR_DECISIONS   optional radar cross-check decision tree
#   QUALITY_FIT_LEADS         horizons to fit (default 20,30,45,60)
#   QUALITY_FIT_GRID          threshold grid (default 20:80:5)
#   QUALITY_FIT_WORKERS       processes over cells (default 4)
#   QUALITY_FIT_MIN_WARNINGS  evidence floor per lead (default 30)
#   QUALITY_SWEEP_JSON        also keep the full sweep record here
#                             (default $CORPUS_DIR/thresholds/sweep.json)
#
# Usage:
#   sidecar/deploy/quality_report.sh
#   QUALITY_FIT_THRESHOLDS=1 sidecar/deploy/quality_report.sh
#   QUALITY_RADAR_CORPUS=/var/lib/dmi-nowcast-corpus/calibration/national_corpus_20260901_020000.parquet \
#       sidecar/deploy/quality_report.sh
#
# Verify afterwards, from the host:
#   curl -fs http://localhost:8081/nowcast/quality.json | head -c 400
#   curl -fs http://localhost:8081/calibration/push_thresholds.json | head -c 400
#   curl -fs http://localhost:8081/api/push/options
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DEPLOY_DIR"

corpus_dir=${QUALITY_CORPUS_DIR:-/var/lib/dmi-nowcast-corpus}
station_corpus=${QUALITY_STATION_CORPUS:-$corpus_dir/stations/station_corpus_gauge.parquet}
replay_dir=${QUALITY_REPLAY_DIR:-$corpus_dir/stations/replay}
persistence_json=${QUALITY_PERSISTENCE_JSON:-$corpus_dir/pva/results.json}
curves=${QUALITY_CURVES:-/var/lib/dmi-nowcast/national_curves.json}
out=${QUALITY_OUT:-/var/lib/dmi-nowcast/nowcast/quality.json}
md_dir=${QUALITY_MD_DIR:-$corpus_dir/quality_reports}
live_days=${QUALITY_LIVE_DAYS:-90}

fit_thresholds=${QUALITY_FIT_THRESHOLDS:-0}
thresholds_out=${QUALITY_THRESHOLDS_OUT:-/var/lib/dmi-nowcast/push_thresholds.json}
decisions_dirs=${QUALITY_DECISIONS_DIRS:-"$corpus_dir/stations/replay/decisions $corpus_dir/stations/eval"}
radar_decisions=${QUALITY_RADAR_DECISIONS:-}
fit_leads=${QUALITY_FIT_LEADS:-20,30,45,60}
fit_grid=${QUALITY_FIT_GRID:-20:80:5}
fit_workers=${QUALITY_FIT_WORKERS:-4}
fit_min_warnings=${QUALITY_FIT_MIN_WARNINGS:-30}
sweep_json=${QUALITY_SWEEP_JSON:-$corpus_dir/thresholds/sweep.json}

# Bring scripts/ into the container on demand — the runtime image does not
# carry them. Repo mounted read-only; every output goes to a volume.
run_in_repo() {
    docker compose run --rm \
        -v "$DEPLOY_DIR/../..:/repo:ro" \
        --workdir /repo \
        -e PYTHONPATH=/repo/src \
        sidecar \
        "$@"
}

# The calibration corpus is stamped per run; default to the newest one so
# the common case needs no argument at all.
radar_corpus=${QUALITY_RADAR_CORPUS:-}
if [[ -z "$radar_corpus" ]]; then
    radar_corpus=$(run_in_repo python - "$corpus_dir" <<'PY' | tr -d '\r'
import sys
from pathlib import Path

candidates = sorted(Path(sys.argv[1], "calibration").glob("*.parquet"))
print(candidates[-1] if candidates else "")
PY
)
fi

# Only pass the flags whose inputs actually exist: an absent path and an
# omitted flag mean the same thing to the builder (that section is null),
# and omitting it keeps the log free of "file not found" noise.
args=(--out-json "$out" --out-md "$md_dir/$(date -u +%Y-%m-%d).md"
      --corpus-dir "$corpus_dir" --live-days "$live_days")
add_if_exists() {   # add_if_exists <flag> <path> <test-flag>
    if run_in_repo python -c "import sys,os; sys.exit(0 if os.path.$3(sys.argv[1]) else 1)" "$2"; then
        args+=("$1" "$2")
        echo "    $1 $2"
    else
        echo "    (skipping $1 — $2 not present; that section will be null)"
    fi
}

# --- optional: fit the push thresholds first --------------------------
# Before the report, so quality.json embeds the table this run produced.
# --previous is the file already in service: the stability guard keeps its
# value for any lead the new fit does not clearly improve on. On the very
# first run that file does not exist, which the guard reads as a first fit.
if [[ "$fit_thresholds" == "1" ]]; then
    echo "==> Fitting the push thresholds (Phase G)"
    echo "    decisions → $decisions_dirs"
    echo "    leads $fit_leads over grid $fit_grid, $fit_workers worker(s)"
    echo "    out → $thresholds_out"
    run_in_repo python -c "from pathlib import Path; \
        Path('$(dirname "$thresholds_out")').mkdir(parents=True, exist_ok=True); \
        Path('$(dirname "$sweep_json")').mkdir(parents=True, exist_ok=True)"
    fit_args=(--corpus-dir "$corpus_dir"
              --leads "$fit_leads" --thresholds "$fit_grid"
              --workers "$fit_workers" --min-warnings "$fit_min_warnings"
              --out-thresholds "$thresholds_out"
              --out-json "$sweep_json")
    for d in $decisions_dirs; do
        fit_args+=(--decisions-dir "$d")
    done
    # An `if`, not `[[ ... ]] && ...`: under `set -e` a false test as a
    # bare compound command would end the script.
    if [[ -n "$radar_decisions" ]]; then
        fit_args+=(--radar-decisions-dir "$radar_decisions")
    fi
    # Guard against whatever is in service right now; absent on run one.
    if run_in_repo python -c "import sys,os; sys.exit(0 if os.path.isfile(sys.argv[1]) else 1)" "$thresholds_out"; then
        fit_args+=(--previous "$thresholds_out")
        echo "    guarding against the table in service"
    else
        echo "    no table in service yet — this is the first fit"
    fi
    run_in_repo python scripts/sweep_thresholds.py "${fit_args[@]}"
    args+=(--thresholds "$thresholds_out")
    echo "    the running service re-reads it at its next fan-out"
else
    echo "==> Skipping the push-threshold fit (QUALITY_FIT_THRESHOLDS=1 to run it)"
    if run_in_repo python -c "import sys,os; sys.exit(0 if os.path.isfile(sys.argv[1]) else 1)" "$thresholds_out"; then
        args+=(--thresholds "$thresholds_out")
    fi
fi

echo "==> Building the quality report"
echo "    corpus dir → $corpus_dir"
[[ -n "$radar_corpus" ]] && add_if_exists --radar-corpus "$radar_corpus" isfile
add_if_exists --station-corpus "$station_corpus" isfile
add_if_exists --replay-dir "$replay_dir" isdir
add_if_exists --persistence-json "$persistence_json" isfile
add_if_exists --national-curves "$curves" isfile
echo "    out → $out"
echo "    markdown → $md_dir"

run_in_repo python -c "from pathlib import Path; \
    Path('$(dirname "$out")').mkdir(parents=True, exist_ok=True); \
    Path('$md_dir').mkdir(parents=True, exist_ok=True)"

run_in_repo python scripts/quality_report.py "${args[@]}"

echo
echo "==> Done. The running sidecar serves it immediately — no restart needed:"
echo "    curl -fs http://localhost:8081/nowcast/quality.json | head -c 400"
echo "    (the public instance picks it up on its next sync interval)"
