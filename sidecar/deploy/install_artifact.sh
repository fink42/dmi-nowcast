#!/usr/bin/env bash
# Install a fitted artefact — postprocess.json, postprocess_push.json (the
# push-only onset model, S11) or push_thresholds.json —
# into a running stack's data volume, atomically, keeping one generation.
#
# WHY THIS EXISTS
# ---------------
# Some models cannot be fitted on the VM. A tree model needs LightGBM,
# which the sidecar image deliberately does not carry, and the
# random-point protocol fit costs hours over the full archive. Those are
# fitted on the workstation (``scripts/fit_postprocess.py`` in
# ``.venv-fit``) and the resulting document — the same schema the nightly
# refit writes, scored by the numpy evaluator that IS in the image — is
# copied into the file the service reads. This script is that copy, done
# the way the service expects it: validated first, staged beside the
# target, one ``.prev`` generation kept, and swapped in with a rename.
#
# ROLLOUT PROCEDURE (private stack, a post-processing model)
# ---------------------------------------------------------
#   1. Fit and report on the workstation. Ship nothing without the
#      out-of-fold numbers behind it.
#   2. ``install_artifact.sh --stack private --dry-run <postprocess.json>``
#      and read what it says it will do. Then run it for real.
#   3. Nothing else is needed for the model to be served: the cycle
#      re-reads postprocess.json when its (mtime, size) moves
#      (``push/postprocess.py`` PostprocessTable.maybe_reload).
#   4. The nightly refit then LEAVES IT ALONE. It compares the served
#      document's kind, design version and protocol against its own
#      configuration and skips the refit when it could not have produced
#      what is there (``postprocess_fit.refit_skip_reason``); the nightly
#      summary says ``postprocess_skipped``. A model it cannot tell apart
#      from its own — same kind, same design, at-gauge — needs the
#      explicit ``quality_report.fit_postprocess.hold: true``.
#   5. The threshold fit still runs, the same night, and is fitted on the
#      probabilities of the model now in service, because
#      ``push.probability_source: postprocess`` makes the sweep read
#      ``p_post_<lead>`` and fill it from the SERVED postprocess.json
#      (``quality_report._fit_options``). The thresholds therefore catch
#      up by themselves. To not wait a night, fit them by hand:
#
#        python scripts/sweep_thresholds.py ... \
#            --probability-column 'p_post_{lead}' \
#            --postprocess-model /var/lib/dmi-nowcast/postprocess.json \
#            --previous /var/lib/dmi-nowcast/push_thresholds.json \
#            --out-thresholds push_thresholds.json
#
#      and install the result with this script.
#   6. The public stack needs nothing: its ``sync`` task pulls
#      ``calibration/postprocess.json`` from the private instance hourly
#      and nudges its own table (``sync.py``). Installing on --stack
#      public only makes sense in a hurry or with sync off — the next sync
#      overwrites it with whatever the private instance serves.
#
# ROLLBACK
# --------
#   * Put the previous generation back — one command, no refit:
#       docker exec -u 0 dmi-nowcast-sidecar sh -c \
#         'cd /var/lib/dmi-nowcast && cp -p postprocess.prev.json postprocess.json'
#     Then refit the thresholds on it (or let the nightly job do it).
#   * To go back to the nightly logistic altogether: make sure
#     ``fit_postprocess.hold`` is false, move the installed model out of
#     the way (``mv postprocess.json postprocess.installed.json``) and the
#     next nightly refit writes a fresh logistic into the empty slot and
#     refits the thresholds on it.
#
# Configuration via env vars, loaded from the repo-root .env exactly as
# deploy.sh does (see .env.example): DEPLOY_SSH_HOST, DEPLOY_SSH_PORT,
# DEPLOY_SSH_USER, DEPLOY_SSH_KEY, REMOTE_DIR. DATA_DIR overrides the
# in-container data directory (default /var/lib/dmi-nowcast).
#
# Usage:
#   sidecar/deploy/install_artifact.sh --stack private [--dry-run] FILE...
#   sidecar/deploy/install_artifact.sh --stack public  [--dry-run] FILE...
#
# --dry-run validates the files locally and prints every remote command it
# would run, without opening a connection.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "$REPO_ROOT"

# The only two basenames this installs. Both are read by path from the
# data volume and hot-reloaded on (mtime, size); anything else would land
# somewhere nothing looks at.
#   postprocess.json      push/paths.py resolved_postprocess_path
#   postprocess_push.json push/paths.py resolved_onset_model_path (S11)
#   push_thresholds.json  push/paths.py resolved_thresholds_path
KNOWN_NAMES=(postprocess.json postprocess_push.json push_thresholds.json)

#: uid:gid of the unprivileged ``dmi`` user the container runs as — the
#: same 10001 deploy.sh chowns the corpus bind-mount to. docker cp brings
#: the host file's ownership in with it, so the swapped-in file is chowned
#: before the rename or the service cannot read it.
CONTAINER_UID=10001
CONTAINER_GID=10001

stack=""
dry_run=0
files=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stack) stack="${2:?--stack needs a value}"; shift;;
        --stack=*) stack="${1#*=}";;
        --dry-run) dry_run=1;;
        -h|--help) sed -n '2,75p' "${BASH_SOURCE[0]}"; exit 0;;
        -*) echo "unknown arg: $1" >&2; exit 2;;
        *) files+=("$1");;
    esac
    shift
done

if [[ ${#files[@]} -eq 0 ]]; then
    echo "usage: install_artifact.sh --stack private|public [--dry-run] FILE..." >&2
    exit 2
fi

# Load .env if present (sets DEPLOY_SSH_* / REMOTE_DIR). Same file, same
# variables, same precedence as deploy.sh.
if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

if [[ "$dry_run" == 1 ]]; then
    # A dry run must work on a workstation with no deploy credentials at
    # all — it prints commands, it does not connect.
    : "${DEPLOY_SSH_HOST:=<DEPLOY_SSH_HOST>}"
    : "${DEPLOY_SSH_USER:=<DEPLOY_SSH_USER>}"
    : "${DEPLOY_SSH_KEY:=<DEPLOY_SSH_KEY>}"
fi
: "${DEPLOY_SSH_HOST:?DEPLOY_SSH_HOST not set (configure in .env)}"
: "${DEPLOY_SSH_PORT:=22}"
: "${DEPLOY_SSH_USER:?DEPLOY_SSH_USER not set}"
: "${DEPLOY_SSH_KEY:?DEPLOY_SSH_KEY not set}"

# ssh uses -p for the port; scp uses -P. Same key, same host-key policy as
# deploy.sh, so the two scripts cannot disagree about which host this is.
SSH=(ssh -p "$DEPLOY_SSH_PORT" -i "$DEPLOY_SSH_KEY"
     -o StrictHostKeyChecking=accept-new
     "${DEPLOY_SSH_USER}@${DEPLOY_SSH_HOST}")
SCP=(scp -P "$DEPLOY_SSH_PORT" -i "$DEPLOY_SSH_KEY"
     -o StrictHostKeyChecking=accept-new)

# Stack-dependent layout, mirroring deploy.sh: separate remote dir,
# separate container, separate data volume.
if [[ "$stack" == "public" ]]; then
    REMOTE_DIR="${REMOTE_DIR:-/home/${DEPLOY_SSH_USER}/dmi-nowcast-public}"
    CONTAINER="dmi-nowcast-public"
elif [[ "$stack" == "private" ]]; then
    REMOTE_DIR="${REMOTE_DIR:-/home/${DEPLOY_SSH_USER}/dmi-nowcast}"
    CONTAINER="dmi-nowcast-sidecar"
else
    echo "unknown --stack: ${stack:-<missing>} (private|public)" >&2
    exit 2
fi
DATA_DIR="${DATA_DIR:-/var/lib/dmi-nowcast}"

PYTHON="${PYTHON:-python3}"

# --- 1. validate locally ---------------------------------------------------
# Before anything is copied anywhere. A document that is not what it says
# it is must be caught on the workstation, where the cost is a message —
# not in the data volume, where the cost is a fan-out that falls back to
# the curve for every subscriber until someone reads the logs.

# The schema versions the running code accepts, read out of the code rather
# than typed here, so a bump cannot leave this script quietly wrong.
model_schema=$(sed -n 's/^SCHEMA_VERSION *= *\([0-9][0-9]*\).*/\1/p' \
    src/dmi_nowcast_core/postprocess.py | head -n 1)
table_schema=$(sed -n 's/^THRESHOLDS_SCHEMA_VERSION *= *\([0-9][0-9]*\).*/\1/p' \
    src/dmi_nowcast_core/push_thresholds.py | head -n 1)

validate() {
    local path="$1" name="$2"
    "$PYTHON" - "$path" "$name" "${model_schema:-}" "${table_schema:-}" <<'PY'
import json
import os
import sys

path, name, model_schema, table_schema = sys.argv[1:5]

# The ceiling the public instance's sync will pull this through
# (``SyncConfig.max_bytes``, 16 MB by default). A bigger document reaches
# the private stack and then silently never reaches the public one.
SYNC_MAX_BYTES = 16 * 1024 * 1024
KINDS = ("logistic", "logistic-shared", "trees", "trees-shared")
DESIGNS = ("v1", "v2")


def die(message: str) -> None:
    print(f"    ERROR {name}: {message}", file=sys.stderr)
    raise SystemExit(1)


size = os.path.getsize(path)
try:
    with open(path, "rb") as fh:
        doc = json.load(fh)
except ValueError as exc:
    die(f"not valid JSON ({exc})")
if not isinstance(doc, dict):
    die("not a JSON object")

print(f"    {name}: {size} bytes ({size / 1024 / 1024:.2f} MiB)")
if size > SYNC_MAX_BYTES:
    die(
        f"{size} bytes is over the {SYNC_MAX_BYTES}-byte sync ceiling "
        "(sync.max_bytes); the public instance could never pull it"
    )

version = doc.get("schema_version")
if not isinstance(version, int):
    die("no integer schema_version")

if name in ("postprocess.json", "postprocess_push.json"):
    # The two model slots serve different targets and each refuses the
    # other's document at load (push/postprocess.py PostprocessTable):
    # postprocess.json is what the SITE shows (target wet), and
    # postprocess_push.json feeds only the push rule's onset AND half.
    target = str(
        doc.get("target")
        or (doc.get("training") or {}).get("target")
        or "wet"
    )
    wanted = "onset" if name == "postprocess_push.json" else "wet"
    if target != wanted:
        die(f"target {target!r}, but {name} must be a {wanted!r} model")
    if model_schema and version != int(model_schema):
        die(
            f"schema_version {version}, but this checkout serves "
            f"{model_schema} (postprocess.SCHEMA_VERSION)"
        )
    kind = str(doc.get("kind") or "logistic")
    if kind not in KINDS:
        die(f"kind {kind!r} is not one of {KINDS}")
    design = doc.get("design")
    design_version = (
        design.get("version") if isinstance(design, dict) else None
    ) or "v1"
    if design_version not in DESIGNS:
        die(f"design.version {design_version!r} is not one of {DESIGNS}")
    training = doc.get("training") if isinstance(doc.get("training"), dict) else {}
    # Top-level first, then the training block, then the protocol every
    # document written before the distinction existed was fitted under.
    protocol = doc.get("protocol") or training.get("protocol") or "at-gauge"
    leads = doc.get("leads")
    if not isinstance(leads, list) or not leads:
        die("no leads")
    models = doc.get("models")
    if not isinstance(models, dict):
        die("no models block")
    missing = [lead for lead in leads if str(lead) not in models]
    if missing:
        die(f"leads {missing} have no model")
    print(
        f"    schema_version={version} kind={kind} design={design_version} "
        f"protocol={protocol} target={target}"
    )
    print(
        f"    leads={list(leads)} fitted_at={doc.get('fitted_at_utc') or '?'} "
        f"rows={training.get('rows', '?')} stations={training.get('stations', '?')}"
    )
    if name == "postprocess.json" and (
        kind.startswith("trees") or design_version != "v1" or protocol != "at-gauge"
    ):
        print(
            "    note: the nightly refit cannot reproduce this document and "
            "will skip itself while it is in service"
        )
elif name == "push_thresholds.json":
    if table_schema and version != int(table_schema):
        die(
            f"schema_version {version}, but this checkout serves "
            f"{table_schema} (push_thresholds.THRESHOLDS_SCHEMA_VERSION)"
        )
    leads = doc.get("leads")
    if not isinstance(leads, dict) or not leads:
        die("no leads block")
    column = str((doc.get("objective") or {}).get("probability_column") or "?")
    picked = 0
    for lead in sorted(leads, key=lambda value: int(value)):
        entry = leads[lead] if isinstance(leads[lead], dict) else {}
        pct = entry.get("threshold_pct")
        # S11: a lead on the onset AND rule may carry threshold_pct 0
        # (onset alone decides) beside a whole-percent onset_threshold_pct.
        onset = entry.get("onset_threshold_pct")
        if onset is not None and not (isinstance(onset, int) and 0 < onset < 100):
            die(f"lead {lead}: onset_threshold_pct {onset!r} is not a percent")
        floor = 0 if onset is not None else 1
        if isinstance(pct, int) and floor <= pct < 100:
            picked += 1
        elif pct is not None:
            die(f"lead {lead}: threshold_pct {pct!r} is not a percent")
        rule = "" if onset is None else (
            f" onset_threshold_pct={onset} "
            f"single_threshold_pct={entry.get('single_threshold_pct', 'fallback')}"
        )
        print(
            f"    lead {lead}: threshold_pct={pct if pct is not None else 'none'}{rule} "
            f"guard={entry.get('guard', '?')} warnings={entry.get('warnings', '?')} "
            f"f1={entry.get('f1', '?')}"
        )
    if not picked:
        die("no lead has a usable threshold_pct")
    print(
        f"    schema_version={version} fitted_on={column} "
        f"fallback={doc.get('fallback_threshold_pct', '?')}"
    )
    if column != "p_post_{lead}":
        print(
            "    note: this table was fitted on "
            f"{column} — install it only where push.probability_source "
            "matches, or the rule warns at the wrong percent"
        )
else:
    die("unknown basename")
PY
}

run() {
    # Every remote command goes through here, so --dry-run cannot miss one.
    # Printed to be read and pasted, not re-escaped: the remote snippets
    # below deliberately use single quotes and no double quotes, so
    # wrapping a whitespace-carrying argument in double quotes here is
    # faithful to what ssh is handed.
    if [[ "$dry_run" == 1 ]]; then
        local line="" arg
        for arg in "$@"; do
            if [[ "$arg" == *[[:space:]]* ]]; then
                line+=" \"$arg\""
            else
                line+=" $arg"
            fi
        done
        printf '    +%s\n' "$line"
        return 0
    fi
    "$@"
}

echo "==> Validating ${#files[@]} file(s) locally"
for file in "${files[@]}"; do
    name="$(basename "$file")"
    known=0
    for candidate in "${KNOWN_NAMES[@]}"; do
        [[ "$name" == "$candidate" ]] && known=1
    done
    if [[ "$known" != 1 ]]; then
        echo "    ERROR: refusing $name — install only: ${KNOWN_NAMES[*]}" >&2
        exit 2
    fi
    if [[ ! -f "$file" ]]; then
        echo "    ERROR: $file is not a file" >&2
        exit 2
    fi
    validate "$file" "$name"
done

echo "==> Installing into $CONTAINER:$DATA_DIR ($stack stack)"
if [[ "$dry_run" == 1 ]]; then
    echo "    DRY RUN — no connection is opened; the commands follow"
fi

for file in "${files[@]}"; do
    name="$(basename "$file")"
    # ``keep_previous`` in quality_job.py names the rollback copy
    # <stem>.prev<suffix>; matching it means one place to look whichever
    # writer last published, and one command to roll back.
    prev="${name%.json}.prev.json"
    staged="$REMOTE_DIR/$name.incoming"
    target="$DATA_DIR/$name"

    echo "    $file → $CONTAINER:$target"
    run "${SSH[@]}" "mkdir -p '$REMOTE_DIR'"
    run "${SCP[@]}" "$file" \
        "${DEPLOY_SSH_USER}@${DEPLOY_SSH_HOST}:${staged}"
    # Into the volume under a .tmp name, so nothing reading the target
    # ever sees a partial file. docker cp writes it with the HOST file's
    # ownership, which is why the swap below chowns it first.
    run "${SSH[@]}" "docker cp '$staged' '$CONTAINER:$target.tmp'"
    # The swap, as one remote shell: keep one generation, make the file
    # readable by the service's uid, then rename. ``mv`` within a
    # directory is a rename, which is atomic — a reader gets the old file
    # or the new one and never half of either.
    run "${SSH[@]}" "docker exec -u 0 $CONTAINER sh -c 'set -e; cd $DATA_DIR; if [ -f $name ]; then cp -p $name $prev; fi; chown $CONTAINER_UID:$CONTAINER_GID $name.tmp; chmod 0644 $name.tmp; mv $name.tmp $name'"
    run "${SSH[@]}" "rm -f '$staged'"
    # What is actually there now. The stamp matters: both tables reload on
    # a change of (mtime, size), so this line is the evidence that the
    # running process will pick the file up. Comma-separated because the
    # format string must not carry a space — see ``run``.
    run "${SSH[@]}" "docker exec $CONTAINER sh -c 'stat -c %n,%s-bytes,mtime=%y,%U:%G,%a $target; stat -c %n,%s-bytes,mtime=%y $DATA_DIR/$prev 2>/dev/null || echo no-previous-generation'"
done

if [[ "$dry_run" == 1 ]]; then
    echo "==> Dry run complete — nothing was copied"
    exit 0
fi

echo "==> Done"
echo "    postprocess.json / postprocess_push.json are re-read at the start"
echo "    of the next radar cycle;"
echo "    push_thresholds.json at the start of the next fan-out. No restart."
echo "    The nightly refit will skip a model it could not have produced —"
echo "    check tonight's summary for postprocess_skipped."
