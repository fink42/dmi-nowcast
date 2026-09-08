#!/usr/bin/env bash
# PRODUCT STUDY (Phase H, H-L): how the two DMI composites compare at the
# rain gauges (L1), and the doppler → fullRange harmonisation map (L2).
#
# WHAT THIS IS FOR
#
# The cycle anchors on `fullRange` (:x0, 240 km) and never reads `doppler`
# (:x5, 120 km), which publishes five minutes fresher. Before that
# freshness can be spent, two things have to be measured:
#
#   L1  scripts/gauge_agreement_study.py
#       Every 10-minute gauge slot on the chosen days, at every station:
#       the 1 km disc p90 from fullRange at the start of the slot, doppler
#       in the middle, fullRange at the end — scored against the gauge wet
#       rule. Which product is closer to the ground, and where.
#
#   L2  scripts/fit_doppler_harmonisation.py
#       The quantile map doppler dBZ → fullRange dBZ, per distance band
#       and season, fitted on half the days and evaluated on the other
#       half. Produces doppler_harmonisation.json.
#
# Both are READ-ONLY over the archive and the gauge store. Neither writes
# into anything the monthly routine owns: every output lands in
# <corpus>/stations/product_study/, which nothing else reads.
#
# Runs on the *host*, like calibrate.sh and replay.sh: a throwaway
# container with the working tree mounted read-only at /repo, two workers
# under BATCH_MEM_CAP. See lib/batch.sh for why that number and that cap.
#
# Configuration (env):
#   STUDY_DAYS_FILE    one YYYY-MM-DD per line (default ~/study_days.txt;
#                      the 30 wettest-of-month days is what this is for)
#   STUDY_CORPUS_DIR   corpus root (default /var/lib/dmi-nowcast-corpus)
#   STUDY_OUT_DIR      container path for the outputs
#                      (default <corpus>/stations/product_study)
#   STUDY_POINTS       station points JSON
#                      (default <corpus>/stations/station_points.json)
#   STUDY_RADIUS_M     disc radius in metres (default 1000 — production)
#   STUDY_THRESHOLDS   radar-wet thresholds, mm/h (default 0.1,0.5,1.0)
#   STUDY_STRIDE       L2: use every Nth triple in a day (default 1)
#   STUDY_SKIP_L1      1 to run the harmonisation fit only
#   STUDY_SKIP_L2      1 to run the gauge-agreement study only
#   BATCH_WORKERS      parallel workers (default 2)
#   BATCH_MEM_CAP      hard cap on the batch container (default 5000m)
#   BATCH_FORCE        1 to run beside another batch job (don't)
#
# Usage:
#   sidecar/deploy/gauge_agreement.sh
#   STUDY_DAYS_FILE=~/wettest30.txt sidecar/deploy/gauge_agreement.sh
#   STUDY_SKIP_L2=1 sidecar/deploy/gauge_agreement.sh
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The helpers cd into $DEPLOY_DIR themselves — never rely on an inherited
# cwd here, a deploy can replace the checkout underneath a long run.
source "$DEPLOY_DIR/lib/batch.sh"

require_no_batch_running || exit 1

corpus_dir=${STUDY_CORPUS_DIR:-/var/lib/dmi-nowcast-corpus}
days_file=${STUDY_DAYS_FILE:-$HOME/study_days.txt}
out_dir=${STUDY_OUT_DIR:-$corpus_dir/stations/product_study}
points=${STUDY_POINTS:-$corpus_dir/stations/station_points.json}
radius_m=${STUDY_RADIUS_M:-1000}
thresholds=${STUDY_THRESHOLDS:-0.1,0.5,1.0}
stride=${STUDY_STRIDE:-1}

stamp=$(date -u +%Y%m%d_%H%M%S)

# The days file lives on the HOST (it is an operator's list, not corpus
# data) and is bind-mounted in read-only — same contract as replay.sh.
if [[ ! -f "$days_file" ]]; then
    echo "FATAL: days file not found: ${days_file}" >&2
    echo "  One YYYY-MM-DD per line. Set STUDY_DAYS_FILE to point elsewhere." >&2
    exit 2
fi

echo "==> Radar product study (L1 + L2)"
echo "    days: $(wc -l < "$days_file" | tr -d ' ') from ${days_file}"
echo "    corpus → ${corpus_dir}"
echo "    points → ${points}"
echo "    out → ${out_dir} (stamp ${stamp})"
echo "    workers: ${BATCH_WORKERS}   memory cap: ${BATCH_MEM_CAP}"

BATCH_RUN_ARGS=(-v "$days_file:/tmp/study_days.txt:ro")

# --- L1: gauge agreement per product ----------------------------------
# The station points file is corpus data, not code — the deploy does not
# ship it. Fail here rather than an hour in.
if [[ "${STUDY_SKIP_L1:-0}" != "1" ]]; then
    if ! batch_container_file_exists "$points"; then
        echo >&2
        echo "FATAL: station points not found at ${points}" >&2
        echo "  Build it once with scripts/build_station_points.py (see" >&2
        echo "  station_corpus.sh for the exact invocation), or set" >&2
        echo "  STUDY_POINTS to another location." >&2
        exit 2
    fi

    echo "==> L1 gauge agreement → ${out_dir}/gauge_agreement_${stamp}.md"
    run_in_repo_capped python scripts/gauge_agreement_study.py \
            --corpus-dir "$corpus_dir" \
            --points "$points" \
            --days-file /tmp/study_days.txt \
            --workers "$BATCH_WORKERS" \
            --radius-m "$radius_m" \
            --thresholds "$thresholds" \
            --out-parquet "$out_dir/gauge_agreement_${stamp}.parquet" \
            --out-json "$out_dir/gauge_agreement_${stamp}.json" \
            --out-md "$out_dir/gauge_agreement_${stamp}.md"
fi

# --- L2: the harmonisation map ----------------------------------------
if [[ "${STUDY_SKIP_L2:-0}" != "1" ]]; then
    echo "==> L2 harmonisation map → ${out_dir}/doppler_harmonisation_${stamp}.json"
    run_in_repo_capped python scripts/fit_doppler_harmonisation.py \
            --corpus-dir "$corpus_dir" \
            --days-file /tmp/study_days.txt \
            --workers "$BATCH_WORKERS" \
            --stride "$stride" \
            --out-json "$out_dir/doppler_harmonisation_${stamp}.json" \
            --out-md "$out_dir/doppler_harmonisation_${stamp}.md"

    # Stable name for whatever reads the map next (L3's replay variant).
    # Published atomically for the same reason every other stable name in
    # this directory is: a reader may open it while this job runs again.
    batch_publish_atomic \
        "$out_dir/doppler_harmonisation_${stamp}.json" \
        "$out_dir/doppler_harmonisation.json"
fi

echo "==> Done → ${out_dir}"
echo "    L1: gauge_agreement_${stamp}.{md,json,parquet}"
echo "    L2: doppler_harmonisation_${stamp}.{md,json}"
echo "    stable map: ${out_dir}/doppler_harmonisation.json"
