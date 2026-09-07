#!/usr/bin/env bash
# The MONTHLY ROUTINE, end to end. Runs on the sidecar host via the
# dmi-calibrate systemd timer (1st of the month, 03:00 UTC + up to 15 min
# of randomised delay), and by hand for a re-run.
#
ONE corpus build serves BOTH point sets. ``--points`` is repeatable
# and one STEPS run per event already feeds every point in the union, so
# the ~120 radar calibration points and the DMI gauge stations come out of
# a single ~3.5 h pass instead of two:
#
#   build   --points calibration_points_v2.json --points station_points.json
#   fit     --point-set calibration_points_v2      (the served curves)
#   report  --point-set calibration_points_v2      (reliability)
#   join    --point-set station_points             (gauge truth)
#
# Every row carries a ``point_set`` column naming the file its point came
# from, which is what lets one corpus be split back into the two it
# replaces. ``calibration/latest.parquet`` is the whole union — the
# quality report's radar section filters it by point_set itself
# (RADAR_POINT_SET in core/quality_report.py).
#
# This lives here rather than in a second scheduled unit deliberately:
# the timer already exists and changing it needs host root, and two
# schedulers on one 12 GB VM is how batch jobs end up running on top of
# each other. CALIBRATION_STATIONS=0 drops the station points AND the
# join; a missing points file logs one line and is not an error — the
# calibration must never fail over the station half.
#
# CALIBRATION_UNION=0 falls back to the old two-build path (a radar-only
# corpus, then station_corpus.sh) — the escape hatch if a union run ever
# misbehaves.
#
# Builds a multi-point
# corpus over the last CALIBRATION_INPUT_MONTHS months (default "all" —
# the entire persistent archive) — one STEPS run per event feeding all
# ~120 calibration points — then fits the pooled, weight-corrected
# national isotonic curves and writes them to the data volume so the
# running sidecar serves calibrated probabilities after restart.
#
# The LEGACY single-point curves (calibration_curves.json, feeding the
# binary p_calibrated field) are deliberately NOT refitted — that path
# is frozen; national_curves.json is the maintained fit. STEPS/sampling
# settings are read from the RUNNING config so corpus and runtime can
# never drift.
#
# The default window is the WHOLE archive (CALIBRATION_INPUT_MONTHS=all →
# --days-back 0). The builder lists frames from the persistent corpus
# archive first (--corpus-dir) and only falls back to DMI's items API for
# windows the archive does not hold, so the calibration window is bounded
# by the archive's own depth rather than by DMI's 180-day listing —
# which is the reason the archive is kept in the first place. It grows by
# one month every month; a fixed number of months would throw that away.
# Set CALIBRATION_INPUT_MONTHS=<n> for a shorter, fixed window (e.g. to
# recalibrate on a recent season only).
#
# This script is run on the *host*, not inside the container — it shells
# into the existing sidecar container via ``docker compose exec`` to
# reuse the venv and the data volume.
#
# Configuration (env or CLI):
#   CALIBRATION_INPUT_MONTHS    history depth in months, or "all" for the
#                               whole corpus archive (default "all")
#   CALIBRATION_N_EVENTS        events sampled (default 4000)
#   CALIBRATION_WET_BIAS        oversample wet hours (default 0.15)
#   CALIBRATION_SEED            random seed (default $(date +%j))
#   CALIBRATION_STATIONS        0 to skip the station points and the gauge
#                               join entirely (default on, when the points
#                               file exists)
#   CALIBRATION_UNION           0 to build the two corpora separately
#                               instead of one union run (default 1)
#   STATION_POINTS              gauge points JSON (container path; default
#                               <corpus>/stations/station_points.json)
#   CALIBRATION_WORKERS         parallel STEPS workers (default 2, from
#                               BATCH_WORKERS). Each worker is a spawned
#                               process holding 1.3-2.0 GB of anon RSS at
#                               16 members / 432x496 — this is THE memory
#                               knob, and 2 is the number this VM survives.
#                               3 is now plausible — peak RSS per worker is
#                               ~1.65 GB after the Sept 2026 STEPS work, so
#                               3 x 1.65 GB fits under the 5 GB cap — but 2
#                               ships until a real monthly run confirms it.
#                               See lib/batch.sh for the full reasoning.
#   BATCH_MEM_CAP               hard cap put on the batch container ~25 s
#                               after it starts (default 5000m). Under the
#                               cap the cgroup OOM killer takes a batch
#                               worker; without it the global one took the
#                               LIVE sidecar, 18 times in two days.
#   BATCH_FORCE                 1 to start even though another batch
#                               container is already running (don't)
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
# Outputs (all on the corpus bind-mount, so they survive ``down -v``):
#   calibration/national_corpus_<stamp>.parquet   the corpus this run built
#   calibration/latest.parquet                    a COPY of it (see below)
#   calibration/latest.md                         which run, which report dir
#   calibration_reports/<stamp>/                  the reliability report
#   stations/station_corpus_<stamp>_gauge.parquet the gauge-joined rows
#   stations/station_corpus_gauge.parquet         a COPY of it
#   /var/lib/dmi-nowcast/national_curves.json     the served curves
#
# The two stable names are COPIES, not symlinks, on purpose: they are read
# through a docker bind-mount by a *different* container than the one that
# writes them, and by the nightly report's child process, which resolves
# the configured path itself. A symlink into a stamped sibling survives
# neither a mount whose root differs nor a prune of the stamped file, and
# it fails by reading nothing rather than loudly. A parquet is tens of MB;
# a copy once a month is the cheap, boring option. Each copy is published
# atomically (temp name in the same directory, then rename), because the
# nightly report at 03:30 UTC reads them while this job is still running.
# ``quality_report.radar_corpus`` in config.example.yaml points here.
#
# Usage:
#   sidecar/deploy/calibrate.sh                                    # whole archive
#   CALIBRATION_INPUT_MONTHS=3 sidecar/deploy/calibrate.sh         # 3-month window
#   CALIBRATION_INPUT_MONTHS=all sidecar/deploy/calibrate.sh       # explicit default
#   CALIBRATION_STATIONS=0 sidecar/deploy/calibrate.sh             # curves only
#   CALIBRATION_UNION=0 sidecar/deploy/calibrate.sh                # two builds
#
# By hand, keep a log — the timer's runs go to journald:
#   sidecar/deploy/calibrate.sh 2>&1 | tee ~/dmi-nowcast-logs/calibrate-$(date -u +%Y%m%d_%H%M%S).log
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Every helper below cd's into $DEPLOY_DIR itself — nothing here relies on
# an inherited cwd, because a deploy that replaces the checkout mid-run
# leaves a long chain's shell sitting on a deleted directory.
source "$DEPLOY_DIR/lib/batch.sh"

# One batch job at a time on this VM.
require_no_batch_running || exit 1

# "all" (the default) → --days-back 0, which the builder reads as "start
# at the oldest archived frame". Any number is still a fixed month window.
months=${CALIBRATION_INPUT_MONTHS:-all}
if [[ "$months" == "all" ]]; then
    days=0
    window_desc="entire corpus archive (--days-back 0)"
else
    if ! [[ "$months" =~ ^[0-9]+$ ]] || [[ "$months" -lt 1 ]]; then
        echo "CALIBRATION_INPUT_MONTHS must be a positive integer or 'all', got '${months}'" >&2
        exit 2
    fi
    days=$((months * 30))
    window_desc="${months} months (~${days} days)"
fi
n_events=${CALIBRATION_N_EVENTS:-4000}
wet_bias=${CALIBRATION_WET_BIAS:-0.15}
seed=${CALIBRATION_SEED:-$(date +%j)}
# Simulated live frame age. Must match the latency the runtime actually
# has (fetch + STEPS + render after the radar timestamp): the fitted
# curve corrects the lead the service SERVES only if the corpus verified
# at the same instant.
frame_age_range=${CALIBRATION_FRAME_AGE_RANGE:-12,18}
# Persistent corpus archive. Frames are both LISTED and resolved from
# here first; DMI is only consulted for windows the archive does not
# hold, and only gaps are downloaded (straight into the corpus). This is
# also what --days-back 0 measures its window against, so the archive —
# not DMI's 180-day listing — is what bounds the calibration window.
# Required for the default "all" window.
corpus_dir=${CALIBRATION_CORPUS_DIR:-/var/lib/dmi-nowcast-corpus}
# Small-metadata + gap-download cache. Must be a *writable* container path —
# /repo is mounted read-only, so the script's default (``radar_archive``
# relative to cwd) fails with EROFS. Holds wet_dry_index.json; lives on the
# corpus bind-mount so it's durable and survives ``docker compose down -v``.
cache_dir=${CALIBRATION_CACHE_DIR:-/var/lib/dmi-nowcast-corpus/calib_cache}
# Parallelism for the STEPS event loop. Each worker is a *spawned* process
# carrying its own numpy/pysteps state and holds 1.3-2.0 GB of anon RSS at
# the live 16-member / 432x496 settings, so this is the knob that sets the
# job's memory footprint. On this 12 GB VM, shared with two live sidecars,
# 2 workers under BATCH_MEM_CAP is the combination that has never taken the
# service down; 3-5 uncapped workers took it down 18 times in two days. It
# costs wall time (~3.5 h) — fine for a job that runs once a month.
workers=${CALIBRATION_WORKERS:-$BATCH_WORKERS}

# Read ALL corpus-relevant settings from the running config in one shot
# (corpus/runtime parity — the builder must see exactly what the live
# cycle serves). The wet/dry reference set is deliberately NOT taken
# from the configured reference point: this is a national fit, so it
# uses the builder's spread national references unless
# CALIBRATION_WET_REFS overrides them.
settings=$(batch_live_settings)
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
points_path=${CALIBRATION_POINTS:-/repo/src/dmi_nowcast_core/calibration_points_v2.json}
station_points=${STATION_POINTS:-/var/lib/dmi-nowcast-corpus/stations/station_points.json}
latest_path=/var/lib/dmi-nowcast-corpus/calibration/latest.parquet
latest_md=/var/lib/dmi-nowcast-corpus/calibration/latest.md
gauge_path=/var/lib/dmi-nowcast-corpus/stations/station_corpus_${stamp}_gauge.parquet
gauge_stable=/var/lib/dmi-nowcast-corpus/stations/station_corpus_gauge.parquet
# point_set labels are the points files' STEMS — the same rule
# build_calibration_corpus.py's points_set_name() applies. Derived rather
# than hardcoded so CALIBRATION_POINTS / STATION_POINTS stay honest.
radar_set=$(basename "$points_path" .json)
station_set=$(basename "$station_points" .json)

# Decide the shape of the run BEFORE the build: the union needs both
# points files on the one builder invocation.
# NB an ``x && y=0`` one-liner here would abort the whole script under
# ``set -e`` whenever the test is false.
want_stations=1
if [[ "${CALIBRATION_STATIONS:-1}" == "0" ]]; then
    want_stations=0
fi
points_args=(--points "$points_path")
union=0
if [[ "$want_stations" == 1 ]]; then
    if batch_container_file_exists "$station_points"; then
        if [[ "${CALIBRATION_UNION:-1}" != "0" ]]; then
            union=1
            points_args+=(--points "$station_points")
        fi
    else
        echo "==> No station points at ${station_points} — radar points only" \
             "(build them with scripts/build_station_points.py)"
        want_stations=0
    fi
fi

echo "==> National recalibration window: ${window_desc}"
echo "    settings from live config: ${ensemble_size} members, thr ${threshold} mm/h,"
echo "      ds ${downsample}, ${stat}, leads [${leads_csv}], disc ${radius_m} m"
echo "    wet-bias refs: ${CALIBRATION_WET_REFS:-<builder default: 5 spread national points>}"
echo "    n_events: ${n_events}   wet_bias: ${wet_bias}   seed: ${seed}"
echo "    simulated frame age: ${frame_age_range} min (per-event uniform draw)"
echo "    workers: ${workers}"
if [[ "$union" == 1 ]]; then
    echo "    points → ${points_path} + ${station_points} (ONE union build)"
    echo "      fit/report point_set: ${radar_set}   gauge join: ${station_set}"
elif [[ "$want_stations" == 1 ]]; then
    echo "    points → ${points_path} (CALIBRATION_UNION=0: station corpus built separately)"
else
    echo "    points → ${points_path} (no station corpus this run)"
fi
echo "    corpus archive (frame source) → ${corpus_dir}"
echo "    cache → ${cache_dir}"
echo "    corpus parquet → ${corpus_path}"
echo "    curves → ${curves_path}"
echo "    report → ${report_dir}"
echo "    stable copy → ${latest_path} (+ ${latest_md})"
echo "    memory cap → ${BATCH_MEM_CAP} on the batch container"

# ``run_in_repo`` / ``run_in_repo_capped`` come from lib/batch.sh: they
# mount the working tree read-only at /repo (scripts/ is not baked into the
# runtime image), cd into the deploy dir themselves, and the *_capped form
# additionally puts BATCH_MEM_CAP on the throwaway container once it exists.

# Durable output dirs live on the corpus bind-mount — create them first.
run_in_repo python -c "from pathlib import Path; \
    Path('$(dirname "$corpus_path")').mkdir(parents=True, exist_ok=True); \
    Path('$report_dir').mkdir(parents=True, exist_ok=True)"

run_in_repo_capped python scripts/build_calibration_corpus.py \
        "${points_args[@]}" \
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

# --point-set even on a single-set corpus: every row the current builder
# writes carries the column, and naming the set makes the fit's scope
# explicit in the log rather than implied by what happened to be built.
run_in_repo_capped python scripts/fit_national_calibration.py \
        --corpus "$corpus_path" \
        --point-set "$radar_set" \
        --output "$curves_path"

# Reliability report (plan §B3) — non-fatal: duckdb ships in the image
# from the Phase B Dockerfile on, but an older image must not fail the
# whole calibration over a missing report.
if ! run_in_repo_capped python scripts/national_calibration_report.py \
        --corpus "$corpus_path" \
        --point-set "$radar_set" \
        --out-dir "$report_dir"; then
    echo "!! reliability report failed (older image without duckdb?)." >&2
    echo "   Curves are still fitted. Generate the report on the dev box:" >&2
    echo "   python scripts/national_calibration_report.py --corpus <synced parquet>" >&2
fi

# --- stable pointers for the nightly report ---------------------------
# ``quality_report.radar_corpus`` in config names latest.parquet; until now
# nothing maintained it, so the host config had to be hand-edited after
# every run (and wasn't — it named the 2026-09-03 corpus for a fortnight).
# A COPY rather than a symlink, published atomically: see the header.
echo "==> Updating ${latest_path}"
if batch_publish_atomic "$corpus_path" "$latest_path"; then
    echo "    ✓ latest.parquet is now a copy of $(basename "$corpus_path")"
else
    echo "!! could not update ${latest_path} — the nightly report will keep" >&2
    echo "   reading the previous corpus. Copy it by hand from the deploy dir:" >&2
    echo "   docker compose run --rm -T sidecar cp -f $corpus_path $latest_path" >&2
fi

# A one-screen note beside it: which run, and which report to read. Written
# through python so the quoting survives the container boundary.
run_in_repo python - "$latest_md" "$stamp" "$corpus_path" "$curves_path" \
    "$report_dir" "$window_desc" "$radar_set" "$([[ "$union" == 1 ]] && echo "$station_set" || echo "")" \
    <<'MD' || echo "!! could not write ${latest_md} (non-fatal)" >&2
import os
import sys
from pathlib import Path

out, stamp, corpus, curves, report_dir, window, radar_set, station_set = sys.argv[1:9]
sets = f"{radar_set} + {station_set}" if station_set else radar_set
tmp = Path(f"{out}.tmp.{os.getpid()}")
tmp.write_text(
    f"# Latest national calibration\n\n"
    f"- run stamp: `{stamp}` (UTC)\n"
    f"- window: {window}\n"
    f"- point sets: {sets}\n"
    f"- corpus: `{corpus}`\n"
    f"- `latest.parquet` is a copy of that file\n"
    f"- curves: fitted on `{radar_set}` → `{curves}`\n"
    f"- reliability report: `{report_dir}`\n\n"
    "Written by sidecar/deploy/calibrate.sh. The nightly quality report\n"
    "reads `latest.parquet` as `quality_report.radar_corpus` and filters\n"
    "it to the radar point set itself.\n"
)
os.replace(tmp, out)
MD

# Record the currently-served fit timestamp BEFORE restart so we can prove
# the new curves were actually picked up. ``|| true`` so a missing/garbage
# state.json (e.g. very first calibration) doesn't abort under ``set -e``.
read_fitted_at() {
    # National calibration surfaces on the probabilistic block (§B4);
    # null until the first fit, hence the ``or ''`` guards.
    compose exec -T sidecar curl -fs http://localhost:8081/state.json 2>/dev/null \
        | python3 -c "import json,sys; d=json.load(sys.stdin); print(((d.get('probabilistic') or {}).get('calibration_fitted_at')) or '')" 2>/dev/null \
        || true
}
old_fitted_at=$(read_fitted_at)

echo "==> New curves at $curves_path"
echo "==> Restarting sidecar to pick them up (was fitted_at=${old_fitted_at:-none})"
compose restart sidecar

# Guard: poll until the served fitted_at advances past the old value. If it
# never does, the fit silently failed to take effect — exit non-zero so the
# systemd unit is marked failed and ``systemctl --failed`` / journald surface
# it, instead of quietly serving stale curves for another month.
echo "==> Verifying the new fit is being served"
fit_served=0
new_fitted_at=""
for i in $(seq 1 20); do
    sleep 3
    new_fitted_at=$(read_fitted_at)
    if [[ -n "$new_fitted_at" && "$new_fitted_at" != "$old_fitted_at" ]]; then
        echo "    ✓ sidecar now serving fitted_at=${new_fitted_at}"
        fit_served=1
        break
    fi
done

if [[ "$fit_served" != 1 ]]; then
    echo "    ✗ fitted_at did not advance (still '${new_fitted_at:-unreadable}') after restart." >&2
    echo "      The new curves were written to ${curves_path} but the sidecar is not" >&2
    echo "      serving them. Check 'docker compose logs sidecar' for a load error." >&2
fi

# --- the gauge half ---------------------------------------------------
# AFTER the restart, so the new curves are in service within minutes
# rather than after the join. Nothing here depends on them, so it runs
# even when the verification above failed; the exit code at the bottom
# still reports that failure to systemd.
#
# Never fatal. The gauge corpus is a *report input*; the curves are the
# product. A gauge store that has not been backfilled, a join that errors
# — each costs the gauge column of the quality page for a month and
# nothing else.
#
# With a union corpus this is just the join: the station rows are already
# built. Without one (CALIBRATION_UNION=0) it is the old separate build,
# which station_corpus.sh does end to end.
station_rc="skipped"
if [[ "$want_stations" == 0 ]]; then
    echo "==> No gauge corpus this run (CALIBRATION_STATIONS=0 or no points file)"
elif [[ "$union" == 1 ]]; then
    echo
    echo "==> Gauge truth join over the union corpus  $(date -u +%FT%TZ)"
    echo "    --point-set ${station_set} → ${gauge_path}"
    if run_in_repo_capped python scripts/join_gauge_truth.py \
            --corpus "$corpus_path" \
            --corpus-dir "$corpus_dir" \
            --point-set "$station_set" \
            --out "$gauge_path" \
        && batch_publish_atomic "$gauge_path" "$gauge_stable"; then
        station_rc=0
        echo "    ✓ ${gauge_stable}"
    else
        station_rc=$?
        echo "!! gauge join failed (exit ${station_rc}) — non-fatal." >&2
        echo "   The curves above are unaffected; the quality page's gauge column" >&2
        echo "   keeps last month's corpus. The union corpus is already built, so" >&2
        echo "   a re-run only redoes the join: sidecar/deploy/station_corpus.sh" >&2
    fi
else
    echo
    echo "==> Station corpus, separate build (CALIBRATION_UNION=0)  $(date -u +%FT%TZ)"
    if "$DEPLOY_DIR/station_corpus.sh"; then
        station_rc=0
    else
        station_rc=$?
        echo "!! station corpus failed (exit ${station_rc}) — non-fatal." >&2
        echo "   The curves above are unaffected; the quality page's gauge column" >&2
        echo "   keeps last month's corpus. Re-run: sidecar/deploy/station_corpus.sh" >&2
    fi
fi

echo
echo "==> Monthly routine done $(date -u +%FT%TZ)"
echo "    curves served:  $([[ "$fit_served" == 1 ]] && echo yes || echo NO)"
echo "    gauge corpus:   ${station_rc}"
echo "    The nightly quality report (03:30 UTC) picks both up by their"
echo "    stable names — nothing further to run."

# Non-zero only when the curves are not in service: that is the failure
# systemd should surface. ``set -e`` is on, so be explicit.
[[ "$fit_served" == 1 ]] || exit 1
exit 0
