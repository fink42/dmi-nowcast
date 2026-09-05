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
# Usage:
#   sidecar/deploy/quality_report.sh
#   QUALITY_RADAR_CORPUS=/var/lib/dmi-nowcast-corpus/calibration/national_corpus_20260901_020000.parquet \
#       sidecar/deploy/quality_report.sh
#
# Verify afterwards, from the host:
#   curl -fs http://localhost:8081/nowcast/quality.json | head -c 400
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
