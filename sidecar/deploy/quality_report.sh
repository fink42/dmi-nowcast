#!/usr/bin/env bash
# One-off build of the quality report ("How good are we?", Phase F, F4).
#
# The running sidecar builds this nightly once ``quality_report.enabled``
# is on. This script is for the FIRST one — the page 503s until a document
# exists, and waiting until 03:30 to find out the paths were wrong is a
# poor way to learn it.
#
# Same shape as calibrate.sh: run on the *host*, mount the repo into a
# throwaway container so the working tree's code is what runs (the
# runtime image stays lean), write into the volumes the service already
# has.
#
# It runs the SAME module the nightly task spawns —
# ``python -m dmi_nowcast_sidecar.quality_job`` — so the manual first
# build and the 03:30 one cannot drift. This script's job is only to
# resolve the paths and hand them over as one JSON document. A throwaway
# container is also why the memory never lands on the live service: this
# path has always been out of process, and now the nightly one is too.
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
#   QUALITY_FIT_MIN_DELTA     points a pick must move to be published (5)
#   QUALITY_SWEEP_JSON        also keep the full sweep record here
#                             (default $CORPUS_DIR/thresholds/sweep.json)
#
# The gauge-trained post-processing refit (Phase H, H-P) is a second
# OPTIONAL step, off unless QUALITY_FIT_POSTPROCESS=1, and it runs BEFORE
# the threshold fit — a threshold is a percent ON a probability, so the
# model has to exist before the sweep can be fitted on what the engine
# reads. This is how the FIRST model is made (or seeded: an artefact from
# scripts/fit_postprocess.py is the same document, and copying it into
# QUALITY_POSTPROCESS_OUT is a valid seed); after that, turn on
# quality_report.fit_postprocess.enabled and it happens nightly.
#
#   QUALITY_FIT_POSTPROCESS   1 to refit the model first (default off)
#   QUALITY_POSTPROCESS_OUT   the model the cycle reads
#                             (default /var/lib/dmi-nowcast/postprocess.json)
#   QUALITY_FIT_L2            ridge on the slopes (default 1.0)
#   QUALITY_FIT_DESIGN_LEADS  leads whose raw fraction enters the design
#                             (default 10,20,30,45,60 — the served leads)
#   QUALITY_FIT_PROBABILITY   which probability the THRESHOLD fit is fitted
#                             on: postprocess (default, matching
#                             push.probability_source) or curve
#
# Usage:
#   sidecar/deploy/quality_report.sh
#   QUALITY_GAUGE_RELIABILITY=0 sidecar/deploy/quality_report.sh
#   QUALITY_FIT_THRESHOLDS=1 sidecar/deploy/quality_report.sh
#   QUALITY_FIT_POSTPROCESS=1 QUALITY_FIT_THRESHOLDS=1 \
#       sidecar/deploy/quality_report.sh
#   QUALITY_RADAR_CORPUS=/var/lib/dmi-nowcast-corpus/calibration/national_corpus_20260901_020000.parquet \
#       sidecar/deploy/quality_report.sh
#
# Verify afterwards, from the host:
#   curl -fs http://localhost:8081/nowcast/quality.json | head -c 400
#   curl -fs http://localhost:8081/calibration/push_thresholds.json | head -c 400
#   curl -fs http://localhost:8081/calibration/postprocess.json | head -c 400
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
fit_min_delta=${QUALITY_FIT_MIN_DELTA:-5}
sweep_json=${QUALITY_SWEEP_JSON:-$corpus_dir/thresholds/sweep.json}

fit_postprocess=${QUALITY_FIT_POSTPROCESS:-0}
postprocess_out=${QUALITY_POSTPROCESS_OUT:-/var/lib/dmi-nowcast/postprocess.json}
fit_l2=${QUALITY_FIT_L2:-1.0}
fit_design_leads=${QUALITY_FIT_DESIGN_LEADS:-10,20,30,45,60}
fit_probability=${QUALITY_FIT_PROBABILITY:-postprocess}

# The page's gauge reliability curve, scored on the decision rows rather
# than on the station corpus (Phase H). On by default: the site serves the
# post-processed probability, so a diagram of the isotonic curve would be
# of a number nobody is shown. QUALITY_GAUGE_RELIABILITY=0 falls back to
# the corpus fit, which is what the page showed before the model existed.
gauge_reliability=${QUALITY_GAUGE_RELIABILITY:-1}

# Bring scripts/ into the container on demand — the runtime image does not
# carry them. Repo mounted read-only; every output goes to a volume.
run_in_repo() {
    docker compose run --rm \
        -v "$DEPLOY_DIR/../..:/repo:ro" \
        --workdir /repo \
        -e PYTHONPATH=/repo/src:/repo/sidecar \
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

# Only name the inputs that actually exist: an absent path and an omitted
# key mean the same thing to the builder (that section is null), and
# leaving it out keeps the log free of "file not found" noise.
inputs=()
add_if_exists() {   # add_if_exists <json-key> <path> <test-flag>
    if run_in_repo python -c "import sys,os; sys.exit(0 if os.path.$3(sys.argv[1]) else 1)" "$2"; then
        inputs+=("$1=$2")
        echo "    $1 $2"
    else
        echo "    (skipping $1 — $2 not present; that section will be null)"
    fi
}

# --- optional: fit the push thresholds first --------------------------
# Before the report, so quality.json embeds the table this run produced.
# The stability guard is against the file already in service: the job
# reads it before it overwrites it, and an absent one reads as a first
# fit. That is the same code the nightly task runs.
# --- optional: refit the post-processing model first -------------------
# Before the threshold fit, and for the reason in the header: a percent is
# a threshold ON a probability.
post_on=false
if [[ "$fit_postprocess" == "1" ]]; then
    echo "==> Refitting the post-processing model (Phase H)"
    echo "    decisions → $decisions_dirs"
    echo "    leads $fit_leads, design leads $fit_design_leads, L2 $fit_l2"
    echo "    out → $postprocess_out"
    echo "    rows without feature columns are skipped and counted"
    post_on=true
    echo "    the running cycle re-reads it at its next full cycle"
else
    echo "==> Skipping the post-processing refit (QUALITY_FIT_POSTPROCESS=1 to run it)"
fi

fit_on=false
if [[ "$fit_thresholds" == "1" ]]; then
    echo "==> Fitting the push thresholds (Phase G)"
    echo "    decisions → $decisions_dirs"
    echo "    leads $fit_leads over grid $fit_grid, $fit_workers worker(s)"
    echo "    out → $thresholds_out"
    echo "    full sweep record → $sweep_json"
    fit_on=true
    if run_in_repo python -c "import sys,os; sys.exit(0 if os.path.isfile(sys.argv[1]) else 1)" "$thresholds_out"; then
        echo "    guarding against the table in service"
    else
        echo "    no table in service yet — this is the first fit"
    fi
    echo "    the running service re-reads it at its next fan-out"
else
    echo "==> Skipping the push-threshold fit (QUALITY_FIT_THRESHOLDS=1 to run it)"
fi

echo "==> Building the quality report"
echo "    corpus dir → $corpus_dir"
[[ -n "$radar_corpus" ]] && add_if_exists radar_corpus "$radar_corpus" isfile
add_if_exists station_corpus "$station_corpus" isfile
add_if_exists replay_dir "$replay_dir" isdir
add_if_exists persistence_json "$persistence_json" isfile
add_if_exists national_curves "$curves" isfile
# The fitted table: tonight's once the step above has run, otherwise
# whatever is in service. A missing file nulls that section, it is not an
# error, so this is named whenever there is any chance of one.
if [[ "$fit_on" == "true" ]] || run_in_repo python -c "import sys,os; sys.exit(0 if os.path.isfile(sys.argv[1]) else 1)" "$thresholds_out"; then
    inputs+=("thresholds_path=$thresholds_out")
fi
echo "    out → $out"
echo "    markdown → $md_dir"

# The job config, built by the same code that parses the threshold grid
# for the nightly task, printed as one line of JSON. What this script
# resolved is exactly what the job runs — no second set of defaults.
config_json=$(run_in_repo python - \
    "$corpus_dir" "$out" "$md_dir" "$live_days" \
    "$fit_on" "$thresholds_out" "$sweep_json" "$fit_leads" "$fit_grid" \
    "$fit_workers" "$fit_min_warnings" "$fit_min_delta" \
    "$radar_decisions" "$decisions_dirs" \
    "$post_on" "$postprocess_out" "$fit_l2" "$fit_design_leads" \
    "$fit_probability" "$gauge_reliability" ${inputs[@]+"${inputs[@]}"} <<'CFG' | tr -d '\r' | tail -n 1
import json
import sys

from dmi_nowcast_core.postprocess import POST_COLUMN_TEMPLATE
from dmi_nowcast_sidecar.threshold_sweep import parse_thresholds

(corpus_dir, out, md_dir, live_days, fit_on, thresholds_out, sweep_json,
 leads, grid, workers, min_warnings, min_delta, radar_decisions,
 decisions_dirs, post_on, postprocess_out, l2, design_leads,
 probability, gauge_reliability, *pairs) = sys.argv[1:]

inputs = {"corpus_dir": corpus_dir, "live_days": int(live_days)}
for pair in pairs:
    key, _, value = pair.partition("=")
    inputs[key] = value

config = {
    "quality": {
        "out_json": out,
        "markdown_dir": md_dir or None,
        "inputs": inputs,
    },
    "fit": {"enabled": False},
    "fit_postprocess": {"enabled": False},
    "gauge_reliability": {"enabled": False},
}
lead_list = [int(v) for v in leads.split(",") if v.strip()]
design_list = [int(v) for v in design_leads.split(",") if v.strip()]
if gauge_reliability == "1":
    # The same rows, the same gauge rule and the same probability column
    # the threshold fit above uses — the page, the fitted model and the
    # served thresholds have to stand on one sample.
    config["gauge_reliability"] = {
        "enabled": True,
        "options": {
            "decisions_dirs": decisions_dirs.split(),
            "corpus_dir": corpus_dir,
            "leads": lead_list,
            "probability_column": (
                POST_COLUMN_TEMPLATE if probability == "postprocess" else None
            ),
            # The served model, so the ten months of replay rows that
            # carry features and no stored probability are scored rather
            # than excluded. Without it the diagram would cover only the
            # hours since the serving path shipped.
            "postprocess_model": (
                postprocess_out if probability == "postprocess" else None
            ),
            "design_leads": design_list,
        },
    }
if post_on == "true":
    config["fit_postprocess"] = {
        "enabled": True,
        "out": postprocess_out,
        "options": {
            "decisions_dirs": decisions_dirs.split(),
            "corpus_dir": corpus_dir,
            "out": postprocess_out,
            "leads": lead_list,
            "design_leads": design_list,
            "l2": float(l2),
        },
    }
if fit_on == "true":
    config["fit"] = {
        "enabled": True,
        "thresholds_out": thresholds_out,
        "sweep_json": sweep_json or None,
        "min_delta_pct": int(min_delta),
        "min_warnings": int(min_warnings),
        "options": {
            "decisions_dirs": decisions_dirs.split(),
            "corpus_dir": corpus_dir,
            "radar_decisions_dirs": [radar_decisions] if radar_decisions else None,
            "leads": lead_list,
            "thresholds": list(parse_thresholds(grid)),
            "workers": int(workers),
            "min_warnings": int(min_warnings),
            # Fitted on the probability the engine decides with, or the
            # served one — never a third answer.
            "probability_column": (
                POST_COLUMN_TEMPLATE if probability == "postprocess" else None
            ),
            "postprocess_model": (
                postprocess_out if probability == "postprocess" else None
            ),
            "design_leads": design_list,
        },
    }
print(json.dumps(config, sort_keys=True))
CFG
)

# One process, both steps, exactly as the nightly task runs them. Its
# stdout is one line of JSON: the summary.
run_in_repo python -m dmi_nowcast_sidecar.quality_job --config-json "$config_json"

echo
echo "==> Done. The running sidecar serves it immediately — no restart needed:"
echo "    curl -fs http://localhost:8081/nowcast/quality.json | head -c 400"
echo "    (the public instance picks it up on its next sync interval)"
