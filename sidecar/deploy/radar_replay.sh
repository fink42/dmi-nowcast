#!/usr/bin/env bash
# The same warning replay as replay.sh, over the 120 fixed RADAR
# calibration points instead of the gauge stations — truth is the radar
# composite itself, so it produces decision rows for every point in the
# country rather than only where a gauge happens to sit.
#
# WHEN YOU NEED THIS
#
# Rarely, and always alongside replay.sh. Its purpose is a cross-check:
# the gauge replay picks a push threshold from ~110 stations, and this run
# says whether the same threshold sits on a plateau when the country is
# sampled evenly. If the two disagree, the gauge answer wins (it is the
# independent instrument) but the disagreement is worth understanding.
#
# `--no-score` because there is no gauge truth to score against here; the
# decision rows are the product, consumed as
# QUALITY_RADAR_DECISIONS by quality_report.sh's threshold fit.
#
# Same trigger list as replay.sh: run it after a change to the decision
# rule, the ensemble settings, or the served curves — not on a schedule.
#
# Configuration (env):
#   RADAR_REPLAY_DAYS_FILE  one YYYY-MM-DD per line (default ~/replay_days.txt)
#   RADAR_REPLAY_OUT_DIR    container path for the output tree
#                           (default <corpus>/points/replay)
#   RADAR_REPLAY_POINTS     points JSON (default the repo's v2 set)
#   RADAR_REPLAY_CORPUS_DIR corpus root (default /var/lib/dmi-nowcast-corpus)
#   RADAR_REPLAY_FRAME_AGE  simulated frame age in minutes (default 14)
#   RADAR_REPLAY_HORIZON_MIN forecast horizon (default 90)
#   RADAR_REPLAY_CURVES     served national curves (default the live ones)
#   BATCH_WORKERS           parallel workers (default 2)
#   BATCH_MEM_CAP           hard cap on the batch container (default 5000m)
#   BATCH_FORCE             1 to run beside another batch job (don't)
#
# Usage:
#   sidecar/deploy/radar_replay.sh
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DEPLOY_DIR/lib/batch.sh"

require_no_batch_running || exit 1

corpus_dir=${RADAR_REPLAY_CORPUS_DIR:-/var/lib/dmi-nowcast-corpus}
days_file=${RADAR_REPLAY_DAYS_FILE:-$HOME/replay_days.txt}
out_dir=${RADAR_REPLAY_OUT_DIR:-$corpus_dir/points/replay}
points=${RADAR_REPLAY_POINTS:-/repo/src/dmi_nowcast_core/calibration_points_v2.json}
frame_age=${RADAR_REPLAY_FRAME_AGE:-14}
horizon=${RADAR_REPLAY_HORIZON_MIN:-90}
curves=${RADAR_REPLAY_CURVES:-/var/lib/dmi-nowcast/national_curves.json}

if [[ ! -f "$days_file" ]]; then
    echo "FATAL: days file not found: ${days_file}" >&2
    echo "  One YYYY-MM-DD per line. Set RADAR_REPLAY_DAYS_FILE to point elsewhere." >&2
    exit 2
fi

echo "==> Radar-point warning replay"
echo "    days: $(wc -l < "$days_file" | tr -d ' ') from ${days_file}"
echo "    points → ${points}"
echo "    out → ${out_dir}"
echo "    frame age ${frame_age} min, horizon ${horizon} min, --no-score"
echo "    workers: ${BATCH_WORKERS}   memory cap: ${BATCH_MEM_CAP}"

settings=$(batch_live_settings)
read -r _radius_m ensemble_size cascades downsample _threshold _stat _leads <<< "$settings"
echo "    ensemble ${ensemble_size} members, ${cascades} cascade levels, ds ${downsample}"

BATCH_RUN_ARGS=(-v "$days_file:/tmp/replay_days.txt:ro")

run_in_repo_capped python scripts/replay_warnings.py \
        --archive-dir "$corpus_dir/composites" \
        --corpus-dir "$corpus_dir" \
        --points "$points" \
        --days-file /tmp/replay_days.txt \
        --workers "$BATCH_WORKERS" \
        --frame-age-min "$frame_age" \
        --ensemble-size "$ensemble_size" \
        --cascade-levels "$cascades" \
        --downsample-factor "$downsample" \
        --horizon-min "$horizon" \
        --no-score \
        --national-curves "$curves" \
        --out-dir "$out_dir" \
        --progress "$out_dir/progress.json"

echo "==> Radar replay done → ${out_dir}"
echo "    Feed it to the threshold fit as the cross-check:"
echo "    QUALITY_FIT_THRESHOLDS=1 QUALITY_RADAR_DECISIONS=${out_dir}/decisions \\"
echo "        sidecar/deploy/quality_report.sh"
