#!/usr/bin/env bash
# STATION-POINT corpus + gauge-truth join.
#
# THE MONTHLY ROUTINE NO LONGER NEEDS THIS. calibrate.sh builds one union
# corpus over both point sets and joins the gauge rows out of it, which is
# half the wall time. This script is what you run to redo just the gauge
# half — after a gauge backfill, or when the join failed and the curves
# did not — and it is the fallback calibrate.sh calls under
# CALIBRATION_UNION=0.
#
# It therefore does the cheapest thing that produces the file:
#
#   1. If a union corpus is on disk (calibration/latest.parquet by
#      default) and it holds rows for the station point set, JOIN THAT —
#      seconds, not hours. STATION_REUSE_CORPUS=0 forces a rebuild.
#   2. Otherwise build a station-only corpus first (~3.5 h), then join.
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
#   STATION_UNION_CORPUS   union corpus to join instead of building
#                          (default <corpus>/calibration/latest.parquet)
#   STATION_REUSE_CORPUS   0 to always build, never reuse (default 1)
#   BATCH_WORKERS          STEPS workers (default 2)
#   BATCH_MEM_CAP          hard cap on the batch container (default 5000m)
#   BATCH_FORCE            1 to run beside another batch job (don't)
#
# Usage:
#   sidecar/deploy/station_corpus.sh                         # join, or build
#   STATION_REUSE_CORPUS=0 sidecar/deploy/station_corpus.sh  # force a build
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
joined=$corpus_dir/stations/station_corpus_${stamp}_gauge.parquet
stable=$corpus_dir/stations/station_corpus_gauge.parquet
union_corpus=${STATION_UNION_CORPUS:-$corpus_dir/calibration/latest.parquet}
# The point_set label is the points file's STEM — the rule
# build_calibration_corpus.py's points_set_name() applies.
point_set=$(basename "$points" .json)

echo "==> Station corpus (point_set ${point_set})"
echo "    joined → ${joined}"
echo "    stable copy → ${stable}"

# --- can we skip the build? -------------------------------------------
# A union corpus already contains the station rows: one STEPS run per
# event served every point in both --points files. Re-running the builder
# to reproduce rows that are on disk costs 3.5 h and, because the corpus
# is a deterministic function of its settings hash, produces the same
# numbers. So look first.
reuse=""
if [[ "${STATION_REUSE_CORPUS:-1}" != "0" ]]; then
    reuse=$(run_in_repo python - "$union_corpus" "$point_set" <<'REUSE' | tr -d '\r' | tail -n 1
import sys

# Prints the corpus path when it holds rows for this point set, else an
# empty line. Never raises: a missing, truncated or pre-union file just
# means "build instead". pc.unique keeps this a dictionary scan rather
# than materialising millions of python strings.
path, point_set = sys.argv[1], sys.argv[2]
answer = ""
try:
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    if "point_set" in pq.ParquetFile(path).schema_arrow.names:
        col = pq.read_table(path, columns=["point_set"]).column("point_set")
        if point_set in pc.unique(col).to_pylist():
            answer = path
except Exception:
    answer = ""
print(answer)
REUSE
) || reuse=""
fi

if [[ -n "$reuse" ]]; then
    echo "    reusing the union corpus → ${reuse}"
    echo "    (STATION_REUSE_CORPUS=0 to rebuild instead)"
    out="$reuse"
else
    echo "    no union corpus with ${point_set} rows at ${union_corpus} — building"

    # Corpus/runtime parity: the same one-shot read calibrate.sh does.
    settings=$(batch_live_settings)
    read -r radius_m ensemble_size cascades downsample threshold stat leads_csv <<< "$settings"

    echo "    settings from live config: ${ensemble_size} members, thr ${threshold} mm/h,"
    echo "      ds ${downsample}, ${stat}, leads [${leads_csv}], disc ${radius_m} m"
    echo "    points → ${points}"
    echo "    n_events: ${n_events}   wet_bias: ${wet_bias}   seed: ${seed}"
    echo "    window: --days-back ${days_back} (0 = whole corpus archive)"
    echo "    simulated frame age: ${frame_age_range} min"
    echo "    workers: ${BATCH_WORKERS}   memory cap: ${BATCH_MEM_CAP}"
    echo "    corpus → ${out}"

    # Only the BUILD needs the points file. Fail here, loudly, rather than
    # three hours in: it is built once by scripts/build_station_points.py
    # and does not come with the deploy (it is corpus data, not code).
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
fi

# Gauge truth. Reads the mirrored metObs store under <corpus>/stations/obs.
# --point-set selects the station rows: mandatory on a union corpus (the
# radar points are grid coordinates, not station ids, so they can never
# join a gauge observation) and correct on a station-only one, where every
# row carries the same label anyway.
run_in_repo_capped python scripts/join_gauge_truth.py \
        --corpus "$out" \
        --corpus-dir "$corpus_dir" \
        --point-set "$point_set" \
        --out "$joined"

# Stable name for quality_report.station_corpus. A copy for the same
# reason calibrate.sh copies latest.parquet: a bind-mounted symlink into a
# stamped sibling fails silently when the target moves or is pruned. And
# published atomically — the nightly report's child process may be reading
# this exact path at 03:30 UTC while this job is still going.
batch_publish_atomic "$joined" "$stable"

echo "==> Done: ${joined} (from ${out})"
echo "    stable copy: ${stable} (this is what the nightly report reads)"
