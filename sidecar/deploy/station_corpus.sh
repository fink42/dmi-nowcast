#!/usr/bin/env bash
# STATION-POINT corpus + gauge-truth join.
#
# Runs automatically as step 2 of calibrate.sh (and therefore of the
# monthly dmi-calibrate timer) whenever the station points file exists.
# Run it directly to redo just this half — after a gauge backfill, say,
# or when step 2 failed and step 1 did not.
#
# The sibling of calibrate.sh. Same builder, same live settings read from
# the same running config — only the point set differs: DMI's rain-gauge
# stations instead of the 120 fixed radar calibration points. The join
# afterwards widens the corpus with metObs gauge amounts, which is what
# makes the gauge column of the quality report a measurement against the
# ground rather than against the radar the forecast was made from.
#
# Its output, `stations/station_corpus_gauge.parquet`, is the stable name
# `quality_report.station_corpus` reads. The stamped file beside it is
# kept so a past month can be re-scored.
#
# Runs on the *host*, like calibrate.sh: a throwaway container with the
# working tree mounted read-only at /repo. Two workers under
# BATCH_MEM_CAP — see lib/batch.sh for why that number and that cap.
#
# Configuration (env):
#   STATION_N_EVENTS       events sampled (default 4000)
#   STATION_WET_BIAS       oversample wet hours (default 0.15)
#   STATION_SEED           random seed (default 246 — the Phase F run's)
#   STATION_DAYS_BACK      0 = the whole corpus archive (default, and what
#                          the gauge history is short enough to want)
#   STATION_POINTS         points JSON, container path (default
#                          <corpus>/stations/station_points.json, built by
#                          scripts/build_station_points.py)
#   STATION_CORPUS_DIR     corpus root (default /var/lib/dmi-nowcast-corpus)
#   STATION_FRAME_AGE_RANGE simulated live frame age (default 12,18 — must
#                          match calibrate.sh or the two corpora are not
#                          comparable)
#   BATCH_WORKERS          STEPS workers (default 2)
#   BATCH_MEM_CAP          hard cap on the batch container (default 5000m)
#   BATCH_FORCE            1 to run beside another batch job (don't)
#
# Usage:
#   sidecar/deploy/station_corpus.sh
#   STATION_N_EVENTS=1000 sidecar/deploy/station_corpus.sh   # quick pass
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The helpers cd into $DEPLOY_DIR themselves — never rely on an inherited
# cwd here, a deploy can replace the checkout underneath a long run.
source "$DEPLOY_DIR/lib/batch.sh"

require_no_batch_running || exit 1

corpus_dir=${STATION_CORPUS_DIR:-/var/lib/dmi-nowcast-corpus}
points=${STATION_POINTS:-$corpus_dir/stations/station_points.json}
n_events=${STATION_N_EVENTS:-4000}
wet_bias=${STATION_WET_BIAS:-0.15}
seed=${STATION_SEED:-246}
days_back=${STATION_DAYS_BACK:-0}
frame_age_range=${STATION_FRAME_AGE_RANGE:-12,18}

stamp=$(date -u +%Y%m%d_%H%M%S)
out=$corpus_dir/stations/station_corpus_${stamp}.parquet
joined=${out%.parquet}_gauge.parquet
stable=$corpus_dir/stations/station_corpus_gauge.parquet

# Corpus/runtime parity: the same one-shot read calibrate.sh does.
settings=$(batch_live_settings)
read -r radius_m ensemble_size cascades downsample threshold stat leads_csv <<< "$settings"

echo "==> Station corpus"
echo "    settings from live config: ${ensemble_size} members, thr ${threshold} mm/h,"
echo "      ds ${downsample}, ${stat}, leads [${leads_csv}], disc ${radius_m} m"
echo "    points → ${points}"
echo "    n_events: ${n_events}   wet_bias: ${wet_bias}   seed: ${seed}"
echo "    window: --days-back ${days_back} (0 = whole corpus archive)"
echo "    simulated frame age: ${frame_age_range} min"
echo "    workers: ${BATCH_WORKERS}   memory cap: ${BATCH_MEM_CAP}"
echo "    corpus → ${out}"
echo "    joined → ${joined}"
echo "    stable copy → ${stable}"

# Fail here, loudly, rather than three hours into a STEPS run: the points
# file is built once by scripts/build_station_points.py and does not come
# with the deploy (it is corpus data, not code).
if ! batch_container_file_exists "$points"; then
    echo >&2
    echo "FATAL: station points not found at ${points}" >&2
    echo "  This file is corpus data, not code — the deploy does not ship it." >&2
    echo "  Build it once, after the station observations have been backfilled:" >&2
    echo "    docker compose run --rm -T -v \"\$PWD/../..:/repo:ro\" --workdir /repo \\" >&2
    echo "      -e PYTHONPATH=/repo/src sidecar python scripts/build_station_points.py \\" >&2
    echo "      --corpus-dir ${corpus_dir} --from <YYYY-MM-DD> --to <YYYY-MM-DD> \\" >&2
    echo "      --min-coverage 0.8 --out ${points}" >&2
    echo "  Set STATION_POINTS to override the location." >&2
    exit 2
fi

run_in_repo_capped python scripts/build_calibration_corpus.py \
        --points "$points" \
        --days-back "$days_back" \
        --n-events "$n_events" \
        --wet-bias "$wet_bias" \
        --seed "$seed" \
        --workers "$BATCH_WORKERS" \
        --cache-dir "$corpus_dir/calib_cache" \
        --corpus-dir "$corpus_dir" \
        --ensemble-size "$ensemble_size" \
        --n-cascade-levels "$cascades" \
        --downsample-factor "$downsample" \
        --threshold-mm-h "$threshold" \
        --detection-stat "$stat" \
        --disc-radius-m "$radius_m" \
        --leads "$leads_csv" \
        --frame-age-range "$frame_age_range" \
        --output "$out" \
        --progress "$corpus_dir/stations/station_corpus_progress.json"

# Gauge truth. Reads the mirrored metObs store under <corpus>/stations/obs.
run_in_repo_capped python scripts/join_gauge_truth.py \
        --corpus "$out" \
        --corpus-dir "$corpus_dir" \
        --out "$joined"

# Stable name for quality_report.station_corpus. A copy for the same
# reason calibrate.sh copies latest.parquet: a bind-mounted symlink into a
# stamped sibling fails silently when the target moves or is pruned. And
# published atomically — the nightly report's child process may be reading
# this exact path at 03:30 UTC while this job is still going.
batch_publish_atomic "$joined" "$stable"

echo "==> Done: ${joined}"
echo "    stable copy: ${stable} (this is what the nightly report reads)"
