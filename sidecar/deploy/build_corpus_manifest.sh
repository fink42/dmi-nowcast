#!/usr/bin/env bash
# Build (or update) manifest.parquet over the persistent corpus archive.
# Same mount-on-demand pattern as backfill_corpus.sh.
#
# Configuration (env or CLI):
#   CORPUS_DIR       host bind-mount path (default ~/dmi-nowcast-corpus)
#
# Usage:
#   sidecar/deploy/build_corpus_manifest.sh              # incremental
#   sidecar/deploy/build_corpus_manifest.sh --rebuild    # re-parse every frame
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DEPLOY_DIR"

corpus_host=${CORPUS_DIR:-$HOME/dmi-nowcast-corpus}
corpus_mount=/var/lib/dmi-nowcast-corpus

extra_args=()
for arg in "$@"; do
    extra_args+=("$arg")
done

echo "==> Building corpus manifest"
echo "    corpus_host:  ${corpus_host}"
echo "    corpus_mount: ${corpus_mount}"

docker compose run --rm \
    -v "$DEPLOY_DIR/../..:/repo:ro" \
    -v "${corpus_host}:${corpus_mount}" \
    --workdir /repo \
    -e PYTHONPATH=/repo/src \
    sidecar \
    python scripts/build_corpus_manifest.py \
        --corpus-dir "${corpus_mount}" \
        "${extra_args[@]}"

echo "==> Manifest written to ${corpus_host}/manifest.parquet (host) = ${corpus_mount}/manifest.parquet (container)"
