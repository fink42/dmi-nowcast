#!/usr/bin/env bash
# One-time (or occasional) backfill of the persistent corpus archive from
# DMI's 180-day open-data window. Mirrors calibrate.sh's mount-on-demand
# pattern: scripts aren't baked into the runtime image, so we mount the
# repo into a one-off container that reuses the sidecar's venv.
#
# Configuration (env or CLI):
#   DAYS_BACK        history depth (default 180 = DMI's archive ceiling)
#   CONCURRENCY      parallel downloads (default 4; honors DMI rate limit)
#   CORPUS_DIR       host bind-mount path (default ~/dmi-nowcast-corpus)
#
# Usage:
#   sidecar/deploy/backfill_corpus.sh              # default 180 days
#   DAYS_BACK=30 sidecar/deploy/backfill_corpus.sh # last 30 days only
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DEPLOY_DIR"

days_back=${DAYS_BACK:-180}
concurrency=${CONCURRENCY:-4}
# Host side of the corpus bind-mount (under the user's home — no sudo).
corpus_host=${CORPUS_DIR:-$HOME/dmi-nowcast-corpus}
# Container-internal path — kept identical to the live sidecar's mount and
# ``storage.corpus_dir`` so backfilled frames land exactly where the running
# sidecar reads them.
corpus_mount=/var/lib/dmi-nowcast-corpus

echo "==> Backfilling corpus from DMI"
echo "    days_back:    ${days_back}"
echo "    concurrency:  ${concurrency}"
echo "    corpus_host:  ${corpus_host}"
echo "    corpus_mount: ${corpus_mount}"

# Mount the repo so /repo/scripts is visible, plus the corpus bind-mount
# so the script can write directly into the persistent archive.
docker compose run --rm \
    -v "$DEPLOY_DIR/../..:/repo:ro" \
    -v "${corpus_host}:${corpus_mount}" \
    --workdir /repo \
    -e PYTHONPATH=/repo/src \
    sidecar \
    python scripts/backfill_corpus.py \
        --corpus-dir "${corpus_mount}" \
        --days-back "${days_back}" \
        --concurrency "${concurrency}"

echo "==> Done. To build the manifest:"
echo "    sidecar/deploy/build_corpus_manifest.sh"
