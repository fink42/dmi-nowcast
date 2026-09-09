#!/usr/bin/env bash
# LAYER A SCREEN (Phase H, H4): score one or more candidate motion fields
# against the radar's own future, on identical cases.
#
# WHAT THIS IS FOR
#
# `scripts/persistence_vs_advection.py` is the plan's Layer A harness
# (`forecast_skill_plan.md` §2): per horizon and threshold it scores the
# advected field and the persistence field against the truth frame, over
# a list of days, and reports CSI / POD / FAR / bias / ETS, FSS at 2-32 km
# and the H-F flow-stall diagnostic, plus the per-day contingency blocks
# `scripts/compare_layer_a.py` bootstraps. `--variant NAME` picks which
# motion field, from the registry in `dmi_nowcast_core.variants`.
#
# The last comparison (bulk vs confidence, 2026-09-08) was launched by
# hand as two `docker run` commands, which is why this file exists: H4
# queues 24 Farnebäck cells plus `lucaskanade`, `median3` and `oracle`,
# and 27 hand-typed invocations is 27 chances to mistype a stride and
# quietly compare two runs that scored different case lists.
#
# ONE CONTAINER AT A TIME. `LAYER_A_VARIANTS` runs a list SEQUENTIALLY,
# each in its own throwaway container under its own memory cap — not in
# parallel. Two of these beside the live sidecar is the memory shape that
# OOM-killed the service eighteen times in September 2026; see
# lib/batch.sh. The hand-run comparison did use two concurrent containers
# at 3 GB each, and that is exactly the thing not to repeat now that the
# list is 27 long.
#
# READ-ONLY over the archive. Everything lands in <corpus>/layer_a, which
# nothing the monthly routine owns ever reads.
#
# COST. A stride-2 run over 30 days is ~4.5 h per variant (measured,
# 2026-09-08, 2 workers). Screen the grid at `LAYER_A_STRIDE=4` on the 15
# wettest days first and give only the survivors a full run — the plan's
# §5 H4 note says so, and 27 x 4.5 h is a week of VM.
#
# Configuration (env):
#   LAYER_A_VARIANT     the flow variant to score. REQUIRED unless
#                       LAYER_A_VARIANTS is set.
#   LAYER_A_VARIANTS    space- or comma-separated list of variants, run
#                       one after another in one invocation (each its own
#                       container). Overrides LAYER_A_VARIANT.
#   LAYER_A_DAYS_FILE   one YYYY-MM-DD per line, a HOST path, bind-mounted
#                       read-only (default ~/layer_a_days.txt)
#   LAYER_A_STRIDE      use every Nth candidate frame (default 2)
#   LAYER_A_HORIZONS    forecast horizons in minutes (default 10,20,30,45)
#   LAYER_A_CORPUS_DIR  corpus root (default /var/lib/dmi-nowcast-corpus)
#   LAYER_A_OUT_DIR     container path for the outputs
#                       (default <corpus>/layer_a)
#   BATCH_WORKERS       parallel workers per run (default 2)
#   BATCH_MEM_CAP       hard cap on the batch container (default 5000m)
#   BATCH_FORCE         1 to run beside another batch job (don't)
#
# Outputs, per variant:
#   <out>/<variant>_stride<N>_<stamp>.json   day blocks for compare_layer_a.py
#   <out>/<variant>_stride<N>_<stamp>.md     the readable report
# One <stamp> for the whole invocation, so a list run groups on disk.
#
# Usage:
#   LAYER_A_VARIANT=lucaskanade sidecar/deploy/layer_a.sh
#   LAYER_A_VARIANTS="confidence,median3,oracle" \
#       LAYER_A_STRIDE=4 LAYER_A_DAYS_FILE=~/wettest15.txt \
#       sidecar/deploy/layer_a.sh
#
# Then compare a candidate against the baseline:
#   docker compose -f sidecar/deploy/docker-compose.yml run --rm -T \
#       -v "$PWD:/repo:ro" --workdir /repo -e PYTHONPATH=/repo/src sidecar \
#       python scripts/compare_layer_a.py \
#         --baseline <out>/confidence_stride4_<stamp>.json \
#         --candidate <out>/median3_stride4_<stamp>.json \
#         --out-md <out>/compare_confidence_vs_median3.md
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The helpers cd into $DEPLOY_DIR themselves — never rely on an inherited
# cwd here, a deploy can replace the checkout underneath a long run.
source "$DEPLOY_DIR/lib/batch.sh"

require_no_batch_running || exit 1

corpus_dir=${LAYER_A_CORPUS_DIR:-/var/lib/dmi-nowcast-corpus}
days_file=${LAYER_A_DAYS_FILE:-$HOME/layer_a_days.txt}
out_dir=${LAYER_A_OUT_DIR:-$corpus_dir/layer_a}
stride=${LAYER_A_STRIDE:-2}
horizons=${LAYER_A_HORIZONS:-10,20,30,45}

stamp=$(date -u +%Y%m%d_%H%M%S)

# --- what to run ------------------------------------------------------
# The list form wins when both are set: a caller that passes a list means
# the list, and silently scoring only the first name would produce a
# report whose header looks right.
if [[ -n "${LAYER_A_VARIANTS:-}" ]]; then
    # Commas or spaces, either way.
    read -r -a variants <<< "${LAYER_A_VARIANTS//,/ }"
elif [[ -n "${LAYER_A_VARIANT:-}" ]]; then
    variants=("$LAYER_A_VARIANT")
else
    echo "FATAL: set LAYER_A_VARIANT=<name> (or LAYER_A_VARIANTS='a,b,c')." >&2
    echo "  The registry's names:" >&2
    run_in_repo python scripts/persistence_vs_advection.py --variant-list \
        | sed 's/^/    /' >&2 || true
    exit 2
fi

if [[ ${#variants[@]} -eq 0 ]]; then
    echo "FATAL: LAYER_A_VARIANTS is set but empty." >&2
    exit 2
fi

# --- preflight --------------------------------------------------------
# The days file lives on the HOST (it is an operator's list, not corpus
# data) and is bind-mounted in read-only — same contract as replay.sh.
if [[ ! -f "$days_file" ]]; then
    echo "FATAL: days file not found: ${days_file}" >&2
    echo "  One YYYY-MM-DD per line. Set LAYER_A_DAYS_FILE to point elsewhere." >&2
    exit 2
fi

if [[ ! "$stride" =~ ^[0-9]+$ ]] || [[ "$stride" -lt 1 ]]; then
    echo "FATAL: LAYER_A_STRIDE must be a positive integer, got '${stride}'." >&2
    exit 2
fi

# Check every name against the registry BEFORE the first container: a
# typo found four hours in has cost four hours, and with a list it would
# also have produced a set of reports with a hole in it.
echo "==> Checking variant names against the registry"
known=$(run_in_repo python scripts/persistence_vs_advection.py --variant-list)
bad=()
for variant in "${variants[@]}"; do
    if ! grep -qxF "$variant" <<< "$known"; then
        bad+=("$variant")
    fi
done
if [[ ${#bad[@]} -gt 0 ]]; then
    echo "FATAL: unknown flow variant(s): ${bad[*]}" >&2
    echo "  Registered:" >&2
    sed 's/^/    /' <<< "$known" >&2
    exit 2
fi

echo "==> Layer A screen"
echo "    variants: ${variants[*]} (${#variants[@]} run(s), sequential)"
echo "    days: $(wc -l < "$days_file" | tr -d ' ') from ${days_file}"
echo "    archive → ${corpus_dir}/composites"
echo "    stride ${stride}, horizons ${horizons}"
echo "    out → ${out_dir} (stamp ${stamp})"
echo "    workers: ${BATCH_WORKERS}   memory cap: ${BATCH_MEM_CAP}"

# The harness writes with ``Path.write_text``, which does not create
# parents. Cheap helper, so bare run_in_repo (see lib/batch.sh).
run_in_repo mkdir -p "$out_dir"

BATCH_RUN_ARGS=(-v "$days_file:/tmp/layer_a_days.txt:ro")

# --- the runs ---------------------------------------------------------
# A failure does not abort the list. On a 27-variant screen the useful
# thing after one variant OOMs at hour three is the other twenty-six
# reports, not an empty directory; the names that failed are repeated at
# the end and the exit status carries the failure.
failed=()
for variant in "${variants[@]}"; do
    base="${out_dir}/${variant}_stride${stride}_${stamp}"
    echo
    echo "==> ${variant} → ${base}.{json,md}"
    if run_in_repo_capped python scripts/persistence_vs_advection.py \
            --archive-dir "$corpus_dir/composites" \
            --days-file /tmp/layer_a_days.txt \
            --workers "$BATCH_WORKERS" \
            --stride "$stride" \
            --horizons "$horizons" \
            --variant "$variant" \
            --out-json "${base}.json" \
            --out-md "${base}.md"; then
        echo "    ${variant} done"
    else
        rc=$?
        echo "!! ${variant} FAILED (exit ${rc}) — continuing with the rest" >&2
        failed+=("$variant")
    fi
done

echo
echo "==> Done → ${out_dir}"
echo "    stamp ${stamp}, stride ${stride}"
for variant in "${variants[@]}"; do
    echo "    ${variant}_stride${stride}_${stamp}.{json,md}"
done
echo "    Compare with scripts/compare_layer_a.py; check each report's"
echo "    'cases (frames)' and 'skipped by reason' agree before reading a"
echo "    difference — a paired bootstrap needs the same case list."

if [[ ${#failed[@]} -gt 0 ]]; then
    echo "!! failed: ${failed[*]}" >&2
    exit 1
fi
