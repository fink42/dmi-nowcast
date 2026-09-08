#!/usr/bin/env bash
# Warning replay over the STATION points — a virtual subscriber at every
# usable DMI rain gauge, driven through the same push decision engine the
# real subscriptions go through, over a list of past days.
#
# WHEN YOU NEED THIS
#
# Not monthly. The live gauge scoreboard (station_eval, one row per gauge
# per cycle) accumulates the same decision rows continuously and is what
# the nightly quality report normally scores. Reach for a replay only when
# the pipeline itself changed and the history has to be re-derived under
# the new code:
#
#   - the push decision rule, its thresholds, or the onset definition moved
#   - the ensemble settings changed (members, cascade levels, downsample)
#   - the national curves were refit in a way that shifts served probability
#   - you need a season of evidence NOW and the live scoreboard is young
#
# It re-derives history; it does not observe it. Between changes the live
# rows are strictly better evidence, because they are what the service
# actually served.
#
# Output goes to <corpus>/stations/replay, which is
# `quality_report.replay_dir`; its decision rows and the live station_eval
# rows are scored as one table, deduplicated on (radar_ts, station_id)
# with the live row winning.
#
# Configuration (env):
#   REPLAY_DAYS_FILE   one YYYY-MM-DD per line (default ~/replay_days.txt,
#                      historically the 30 wettest days from the
#                      persistence study)
#   REPLAY_OUT_DIR     container path for the output tree
#                      (default <corpus>/stations/replay)
#   REPLAY_POINTS      points JSON (default <corpus>/stations/station_points.json)
#   REPLAY_CORPUS_DIR  corpus root (default /var/lib/dmi-nowcast-corpus)
#   REPLAY_FRAME_AGE   simulated frame age in minutes (default 14)
#   REPLAY_HORIZON_MIN forecast horizon (default 90)
#   REPLAY_CURVES      served national curves (default the live ones)
#   BATCH_WORKERS      parallel workers (default 2)
#   BATCH_MEM_CAP      hard cap on the batch container (default 5000m)
#   BATCH_FORCE        1 to run beside another batch job (don't)
#
# Usage:
#   sidecar/deploy/replay.sh
#   REPLAY_DAYS_FILE=~/august.txt sidecar/deploy/replay.sh
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DEPLOY_DIR/lib/batch.sh"

require_no_batch_running || exit 1

corpus_dir=${REPLAY_CORPUS_DIR:-/var/lib/dmi-nowcast-corpus}
days_file=${REPLAY_DAYS_FILE:-$HOME/replay_days.txt}
out_dir=${REPLAY_OUT_DIR:-$corpus_dir/stations/replay}
points=${REPLAY_POINTS:-$corpus_dir/stations/station_points.json}
frame_age=${REPLAY_FRAME_AGE:-14}
horizon=${REPLAY_HORIZON_MIN:-90}
curves=${REPLAY_CURVES:-/var/lib/dmi-nowcast/national_curves.json}

# The days file lives on the HOST (it is an operator's list, not corpus
# data) and is bind-mounted in read-only.
if [[ ! -f "$days_file" ]]; then
    echo "FATAL: days file not found: ${days_file}" >&2
    echo "  One YYYY-MM-DD per line. Set REPLAY_DAYS_FILE to point elsewhere." >&2
    exit 2
fi

echo "==> Station warning replay"
echo "    days: $(wc -l < "$days_file" | tr -d ' ') from ${days_file}"
echo "    points → ${points}"
echo "    out → ${out_dir}"
echo "    frame age ${frame_age} min, horizon ${horizon} min"
echo "    workers: ${BATCH_WORKERS}   memory cap: ${BATCH_MEM_CAP}"

# Live ensemble settings, read from the running config so a replay cannot
# drift from what the service serves.
settings=$(batch_live_settings)
read -r _radius_m ensemble_size cascades downsample _threshold _stat _leads <<< "$settings"
echo "    ensemble ${ensemble_size} members, ${cascades} cascade levels, ds ${downsample}"
# Motion completion too (H-F, 2026-09-08): it decides the velocity STEPS
# runs on, so a replay under the other policy is scoring a different
# forecast. The choice is echoed into the run summary's "flow" block.
flow_settings=$(batch_live_flow_settings)
read -r flow_completion _flow_window _flow_conf_pct _flow_texture_pct <<< "$flow_settings"
echo "    flow completion ${flow_completion}"

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
        --flow-completion "$flow_completion" \
        --national-curves "$curves" \
        --out-dir "$out_dir" \
        --progress "$out_dir/progress.json"

echo "==> Replay done → ${out_dir}"
echo "    The nightly quality report picks it up as quality_report.replay_dir."
