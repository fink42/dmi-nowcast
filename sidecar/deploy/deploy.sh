#!/usr/bin/env bash
# Deploy the sidecar to a remote docker host over SSH.
#
# What it does:
#   1. Tar up the repo (sidecar/ + src/ + scripts/ + pyproject.toml) —
#      excluding the dev .venv, the radar archive, and other dev junk.
#      scripts/ MUST be included: calibrate.sh / backfill_corpus.sh /
#      build_corpus_manifest.sh mount the repo into a one-off container and
#      run ``python scripts/...``. Omitting it makes the monthly
#      dmi-calibrate.timer fail with "can't open file scripts/...".
#   2. SCP the tarball to $REMOTE_DIR on the host
#   3. Extract, bring up `docker compose` with rebuild
#   4. Poll /healthz until it returns 200
#
# Configuration via env vars (loaded from repo-root .env if present —
# see .env.example):
#   DEPLOY_SSH_HOST   — docker host hostname / IP
#   DEPLOY_SSH_PORT   — SSH port
#   DEPLOY_SSH_USER   — SSH username
#   DEPLOY_SSH_KEY    — path to the private key
#   REMOTE_DIR        — checkout dir on the host
#                       (default ~$DEPLOY_SSH_USER/dmi-nowcast)
#   CORPUS_HOST_DIR   — persistent corpus archive on the host
#                       (default ~$DEPLOY_SSH_USER/dmi-nowcast-corpus)
#
# Usage:
#   sidecar/deploy/deploy.sh
#   sidecar/deploy/deploy.sh --no-build       # restart only, skip build
#   sidecar/deploy/deploy.sh --logs           # tail logs after deploy

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "$REPO_ROOT"

# Load .env if present (sets DEPLOY_SSH_* vars). See .env.example.
if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

: "${DEPLOY_SSH_HOST:?DEPLOY_SSH_HOST not set (configure in .env)}"
: "${DEPLOY_SSH_PORT:=22}"
: "${DEPLOY_SSH_USER:?DEPLOY_SSH_USER not set}"
: "${DEPLOY_SSH_KEY:?DEPLOY_SSH_KEY not set}"

REMOTE_DIR="${REMOTE_DIR:-/home/${DEPLOY_SSH_USER}/dmi-nowcast}"
# ssh uses -p for the port; scp uses -P. Same key, same StrictHostKeyChecking.
SSH=(ssh -p "$DEPLOY_SSH_PORT" -i "$DEPLOY_SSH_KEY"
     -o StrictHostKeyChecking=accept-new
     "${DEPLOY_SSH_USER}@${DEPLOY_SSH_HOST}")
SCP=(scp -P "$DEPLOY_SSH_PORT" -i "$DEPLOY_SSH_KEY"
     -o StrictHostKeyChecking=accept-new)

do_build=1
do_tail=0
for arg in "$@"; do
    case "$arg" in
        --no-build) do_build=0;;
        --logs) do_tail=1;;
        *) echo "unknown arg: $arg"; exit 2;;
    esac
done

echo "==> Packing sidecar bundle"
tar_excludes=(
    --exclude='./.venv'
    --exclude='./sidecar/.venv'
    --exclude='./radar_archive'
    --exclude='./reports'
    --exclude='./.recon_cache'
    --exclude='./.pytest_cache'
    --exclude='./.git'
    --exclude='./__pycache__'
    --exclude='*.pyc'
    --exclude='./tests'
)
# COPYFILE_DISABLE prevents macOS resource forks (._foo) from contaminating
# the bundle when packed on a Mac.
# README.md is referenced by the root pyproject.toml (``readme = "README.md"``)
# so hatchling reads it during the editable install of dmi-nowcast-core.
# Forgetting it makes `uv sync --frozen` blow up with an OSError.
COPYFILE_DISABLE=1 tar czf /tmp/dmi-sidecar-bundle.tar.gz \
    "${tar_excludes[@]}" \
    --no-mac-metadata \
    ./sidecar ./src ./scripts ./sql ./pyproject.toml ./uv.lock ./README.md ./.python-version
echo "    bundle: $(du -h /tmp/dmi-sidecar-bundle.tar.gz | awk '{print $1}')"

echo "==> Ensuring remote dirs (incl. persistent corpus archive)"
"${SSH[@]}" "mkdir -p '$REMOTE_DIR'"
# Persistent corpus archive — bind-mounted into the container (host side
# $CORPUS_HOST_DIR → container /var/lib/dmi-nowcast-corpus). It defaults to
# the deploy user's home so mkdir needs no host sudo. The container runs as
# uid 10001 (the unprivileged ``dmi`` user), so the host dir must be owned
# by 10001; we chown via a throwaway root busybox container — the docker
# daemon is root, so this needs no host sudo and keeps the whole deploy
# unattended (important for re-runs from the monthly timer's host).
CORPUS_HOST="${CORPUS_HOST_DIR:-/home/${DEPLOY_SSH_USER}/dmi-nowcast-corpus}"
"${SSH[@]}" "mkdir -p '$CORPUS_HOST' && docker run --rm -v '$CORPUS_HOST:/mnt' busybox chown -R 10001:10001 /mnt"

echo "==> Copying bundle"
"${SCP[@]}" /tmp/dmi-sidecar-bundle.tar.gz \
    "${DEPLOY_SSH_USER}@${DEPLOY_SSH_HOST}:${REMOTE_DIR}/"

echo "==> Unpacking on the host (preserving the host-edited config.yaml)"
# ``rm -rf sidecar`` would delete the host's edited config.yaml and the
# bootstrap below would silently reset it to the example (this bit us:
# a deploy reverted ensemble_size 16 → 24). Stash it across the wipe.
"${SSH[@]}" "cd '$REMOTE_DIR' \
    && { test -f sidecar/deploy/config.yaml && cp sidecar/deploy/config.yaml /tmp/dmi-config-keep.yaml || true; } \
    && rm -rf sidecar src scripts && tar xzf dmi-sidecar-bundle.tar.gz \
    && { test -f /tmp/dmi-config-keep.yaml && mv /tmp/dmi-config-keep.yaml sidecar/deploy/config.yaml && echo '    (preserved existing config.yaml)' || true; }"

# First-time bootstrap only: stage the example config if none survived.
"${SSH[@]}" "test -f '$REMOTE_DIR/sidecar/deploy/config.yaml' || { cp '$REMOTE_DIR/sidecar/config.example.yaml' '$REMOTE_DIR/sidecar/deploy/config.yaml' && echo '    (default config.yaml staged — edit it on the host)'; }"

if [[ "$do_build" == 1 ]]; then
    echo "==> Building image + restarting service"
    "${SSH[@]}" "cd '$REMOTE_DIR/sidecar/deploy' && docker compose up -d --build"
else
    echo "==> Restarting service (no rebuild)"
    "${SSH[@]}" "cd '$REMOTE_DIR/sidecar/deploy' && docker compose restart"
fi

echo "==> Waiting for /healthz to come up"
for i in $(seq 1 60); do
    if "${SSH[@]}" "curl -fs http://localhost:8081/healthz >/dev/null 2>&1"; then
        echo "    ✓ sidecar healthy after ${i}s"
        "${SSH[@]}" "curl -s http://localhost:8081/healthz | python3 -m json.tool"
        break
    fi
    sleep 1
    if [[ "$i" == 60 ]]; then
        echo "    ✗ healthz never returned 200; recent logs:"
        "${SSH[@]}" "cd '$REMOTE_DIR/sidecar/deploy' && docker compose logs --tail 40 sidecar"
        exit 1
    fi
done

if [[ "$do_tail" == 1 ]]; then
    "${SSH[@]}" "cd '$REMOTE_DIR/sidecar/deploy' && docker compose logs -f sidecar"
fi
