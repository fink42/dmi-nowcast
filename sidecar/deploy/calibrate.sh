#!/usr/bin/env bash
# Monthly NATIONAL recalibration job. Runs on the sidecar host via the
# dmi-calibrate systemd timer. Builds a multi-point
# corpus over the last CALIBRATION_INPUT_MONTHS months (default 6 ≈
# DMI's full archive depth) — one STEPS run per event feeding all ~120
# calibration points — then fits the pooled, weight-corrected national
# isotonic curves and writes them to the data volume so the running
# sidecar serves calibrated probabilities after restart.
#
# The LEGACY single-point curves (calibration_curves.json, feeding the
# binary p_calibrated field) are deliberately NOT refitted — that path
# is frozen; national_curves.json is the maintained fit. STEPS/sampling
# settings are read from the RUNNING config so corpus and runtime can
# never drift.
#
# The default window is 6 months because that is the *entire* depth DMI's
# open-data archive exposes (~180 days). Frames are resolved from the
# persistent corpus archive first (--corpus-dir), so after the one-time
# backfill this job downloads only frames newer than the last run.
#
# This script is run on the *host*, not inside the container — it shells
# into the existing sidecar container via ``docker compose exec`` to
# reuse the venv and the data volume.
#
# Configuration (env or CLI):
#   CALIBRATION_INPUT_MONTHS    history depth (months; default 6 ≈ DMI's max)
#   CALIBRATION_N_EVENTS        events sampled (default 4000)
#   CALIBRATION_WET_BIAS        oversample wet hours (default 0.15)
#   CALIBRATION_SEED            random seed (default $(date +%j))
#   CALIBRATION_WORKERS         parallel STEPS workers (default 3). Each is
#                               ~0.9 GB RSS — this is the memory knob.
#   CALIBRATION_CACHE_DIR       wet/dry index + gap-download cache (container
#                               path; default /var/lib/dmi-nowcast-corpus/calib_cache)
#   CALIBRATION_CORPUS_DIR      persistent corpus archive to resolve frames
#                               from (container path; default
#                               /var/lib/dmi-nowcast-corpus)
#   CALIBRATION_WET_REFS        wet/dry reference points as
#                               "lat,lon;lat,lon;..." (default: the
#                               builder's five spread national references).
#                               The set re-keys the wet/dry index cache.
#   CALIBRATION_FRAME_AGE_RANGE simulated live frame age, "LO,HI" minutes
#                               (default 12,18). The live cycle finishes
#                               12-18 min after its newest frame's radar
#                               timestamp and shifts every lead by that age
#                               before reading an ensemble timestep, so the
#                               corpus draws an age per event and verifies
#                               at the same instant the service serves.
#                               "0,0" restores the old zero-age convention.
#                               Joins the settings hash: a corpus built
#                               under another range cannot be resumed.
#
# Usage:
#   sidecar/deploy/calibrate.sh                                    # use defaults
#   CALIBRATION_INPUT_MONTHS=3 sidecar/deploy/calibrate.sh         # 3-month window
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DEPLOY_DIR"

months=${CALIBRATION_INPUT_MONTHS:-6}
days=$((months * 30))
n_events=${CALIBRATION_N_EVENTS:-4000}
wet_bias=${CALIBRATION_WET_BIAS:-0.15}
seed=${CALIBRATION_SEED:-$(date +%j)}
# Simulated live frame age. Must match the latency the runtime actually
# has (fetch + STEPS + render after the radar timestamp): the fitted
# curve corrects the lead the service SERVES only if the corpus verified
# at the same instant.
frame_age_range=${CALIBRATION_FRAME_AGE_RANGE:-12,18}
# Persistent corpus archive — frames are resolved from here first; only
# gaps are downloaded (straight into the corpus). This is the 51k-frame
# backfilled archive, so after backfill the monthly run fetches almost
# nothing from DMI.
corpus_dir=${CALIBRATION_CORPUS_DIR:-/var/lib/dmi-nowcast-corpus}
# Small-metadata + gap-download cache. Must be a *writable* container path —
# /repo is mounted read-only, so the script's default (``radar_archive``
# relative to cwd) fails with EROFS. Holds wet_dry_index.json; lives on the
# corpus bind-mount so it's durable and survives ``docker compose down -v``.
cache_dir=${CALIBRATION_CACHE_DIR:-/var/lib/dmi-nowcast-corpus/calib_cache}
# Parallelism for the STEPS event loop. Each worker is a *spawned* process
# carrying its own numpy/pysteps state and holds ~0.9 GB RSS, so this is the
# knob that sets the job's memory footprint: 6 workers ≈ 5.5 GB, which on a
# ~10 GB host left barely 1.3 GB free for 6.5 h. 3 workers ≈ 2.8 GB and
# roughly doubles the wall time — fine for a job that runs once a month.
workers=${CALIBRATION_WORKERS:-3}

# Read ALL corpus-relevant settings from the running config in one shot
# (corpus/runtime parity — the builder must see exactly what the live
# cycle serves). The wet/dry reference set is deliberately NOT taken
# from the configured reference point: this is a national fit, so it
# uses the builder's spread national references unless
# CALIBRATION_WET_REFS overrides them.
settings=$(docker compose exec -T sidecar python - <<'PY'
from dmi_nowcast_sidecar.config import load_config
c = load_config()
print(c.home.radius_km * 1000.0,
      c.forecast.steps.ensemble_size, c.forecast.steps.n_cascade_levels,
      c.forecast.steps.downsample_factor, c.forecast.rain_threshold_mm_h,
      c.forecast.detection_stat, ",".join(str(l) for l in c.forecast.leads_min))
PY
)
read -r radius_m ensemble_size cascades downsample threshold stat leads_csv <<< "$settings"

# Wet/dry references. Empty means "let the builder use its national
# default set" (five spread points; see build_calibration_corpus.py).
wet_ref_args=()
if [[ -n "${CALIBRATION_WET_REFS:-}" ]]; then
    wet_ref_args=(--wet-ref "$CALIBRATION_WET_REFS")
fi

stamp=$(date -u +%Y%m%d_%H%M%S)
# Durable outputs live on the corpus bind-mount (survive ``down -v``);
# the curves go on the data volume where the sidecar reads them.
corpus_path=/var/lib/dmi-nowcast-corpus/calibration/national_corpus_${stamp}.parquet
curves_path=/var/lib/dmi-nowcast/national_curves.json
report_dir=/var/lib/dmi-nowcast-corpus/calibration_reports/${stamp}
points_path=/repo/src/dmi_nowcast_core/calibration_points_v2.json

echo "==> National recalibration window: ${months} months (~${days} days)"
echo "    settings from live config: ${ensemble_size} members, thr ${threshold} mm/h,"
echo "      ds ${downsample}, ${stat}, leads [${leads_csv}], disc ${radius_m} m"
echo "    wet-bias refs: ${CALIBRATION_WET_REFS:-<builder default: 5 spread national points>}"
echo "    n_events: ${n_events}   wet_bias: ${wet_bias}   seed: ${seed}"
echo "    simulated frame age: ${frame_age_range} min (per-event uniform draw)"
echo "    workers: ${workers}"
echo "    points → ${points_path}"
echo "    corpus archive (frame source) → ${corpus_dir}"
echo "    cache → ${cache_dir}"
echo "    corpus parquet → ${corpus_path}"
echo "    curves → ${curves_path}"
echo "    report → ${report_dir}"

# Bring scripts/ into the container on demand — they're not baked into
# the runtime image, so we mount the repo from the host using `docker
# compose run --rm -v`. Alternatively bake them into the image; this
# script keeps the runtime image lean by mounting on demand.
run_in_repo() {
    docker compose run --rm \
        -v "$DEPLOY_DIR/../..:/repo:ro" \
        --workdir /repo \
        -e PYTHONPATH=/repo/src \
        sidecar \
        "$@"
}

# Durable output dirs live on the corpus bind-mount — create them first.
run_in_repo python -c "from pathlib import Path; \
    Path('$(dirname "$corpus_path")').mkdir(parents=True, exist_ok=True); \
    Path('$report_dir').mkdir(parents=True, exist_ok=True)"

run_in_repo python scripts/build_calibration_corpus.py \
        --points "$points_path" \
        --days-back "$days" \
        --n-events "$n_events" \
        --wet-bias "$wet_bias" \
        ${wet_ref_args[@]+"${wet_ref_args[@]}"} \
        --seed "$seed" \
        --workers "$workers" \
        --cache-dir "$cache_dir" \
        --corpus-dir "$corpus_dir" \
        --ensemble-size "$ensemble_size" \
        --n-cascade-levels "$cascades" \
        --downsample-factor "$downsample" \
        --threshold-mm-h "$threshold" \
        --detection-stat "$stat" \
        --disc-radius-m "$radius_m" \
        --leads "$leads_csv" \
        --frame-age-range "$frame_age_range" \
        --output "$corpus_path"

run_in_repo python scripts/fit_national_calibration.py \
        --corpus "$corpus_path" \
        --output "$curves_path"

# Reliability report (plan §B3) — non-fatal: duckdb ships in the image
# from the Phase B Dockerfile on, but an older image must not fail the
# whole calibration over a missing report.
if ! run_in_repo python scripts/national_calibration_report.py \
        --corpus "$corpus_path" \
        --out-dir "$report_dir"; then
    echo "!! reliability report failed (older image without duckdb?)." >&2
    echo "   Curves are still fitted. Generate the report on the dev box:" >&2
    echo "   python scripts/national_calibration_report.py --corpus <synced parquet>" >&2
fi

# Record the currently-served fit timestamp BEFORE restart so we can prove
# the new curves were actually picked up. ``|| true`` so a missing/garbage
# state.json (e.g. very first calibration) doesn't abort under ``set -e``.
read_fitted_at() {
    # National calibration surfaces on the probabilistic block (§B4);
    # null until the first fit, hence the ``or ''`` guards.
    docker compose exec -T sidecar curl -fs http://localhost:8081/state.json 2>/dev/null \
        | python3 -c "import json,sys; d=json.load(sys.stdin); print(((d.get('probabilistic') or {}).get('calibration_fitted_at')) or '')" 2>/dev/null \
        || true
}
old_fitted_at=$(read_fitted_at)

echo "==> New curves at $curves_path"
echo "==> Restarting sidecar to pick them up (was fitted_at=${old_fitted_at:-none})"
docker compose restart sidecar

# Guard: poll until the served fitted_at advances past the old value. If it
# never does, the fit silently failed to take effect — exit non-zero so the
# systemd unit is marked failed and ``systemctl --failed`` / journald surface
# it, instead of quietly serving stale curves for another month.
echo "==> Verifying the new fit is being served"
new_fitted_at=""
for i in $(seq 1 20); do
    sleep 3
    new_fitted_at=$(read_fitted_at)
    if [[ -n "$new_fitted_at" && "$new_fitted_at" != "$old_fitted_at" ]]; then
        echo "    ✓ sidecar now serving fitted_at=${new_fitted_at}"
        exit 0
    fi
done

echo "    ✗ fitted_at did not advance (still '${new_fitted_at:-unreadable}') after restart." >&2
echo "      The new curves were written to ${curves_path} but the sidecar is not" >&2
echo "      serving them. Check 'docker compose logs sidecar' for a load error." >&2
exit 1
