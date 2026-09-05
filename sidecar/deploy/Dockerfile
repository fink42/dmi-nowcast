# Two-stage build keeps the runtime image lean — wheels with C extensions
# (numpy, pyproj, h5py, opencv-python-headless, Pillow) all need build deps
# during install but not at runtime. Stage 2 copies only the resolved
# site-packages.
FROM python:3.12-slim AS builder

# Compiler toolchain only — every wheel we need ships as a manylinux
# binary on python:3.12-slim/x86_64 (numpy, h5py, pyproj, Pillow, opencv-
# python-headless, scipy). If a future dep slips off the wheel index uv
# will still resolve it; reintroduce libhdf5-dev / libgeos-dev / libproj-
# dev here if a source build complains.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv for fast resolver + lockfile-aware sync.
RUN pip install --no-cache-dir uv==0.5.4

# Build at /app/ — same path the runtime image uses — so uv's editable
# install records the right absolute paths in the .pth files. If we built
# at /build/ and then copied to /app/ the imports would fail at runtime
# ("No module named dmi_nowcast_sidecar").
COPY pyproject.toml README.md /app/
COPY src /app/src
COPY sidecar /app/sidecar

# Resolve sidecar deps into /app/sidecar/.venv. ``--frozen`` requires the
# lockfile to be up to date; the deploy script regenerates it on cut over.
WORKDIR /app/sidecar
RUN uv sync --frozen --no-dev

# duckdb is needed at *deploy* time, not by the live server: the corpus
# reports and the recalibration (national_calibration_report.py) query
# Parquet with it, inside this image via ``docker compose run``. It is in
# core's ``dev`` group, so the ``--no-dev`` sync above skips it; install it
# explicitly rather than dropping ``--no-dev`` (which would also pull
# pytest/matplotlib into the image). pyarrow is no longer listed here: since
# Phase F the sidecar itself writes Parquet (gauge store, station eval), so
# it is a runtime dependency resolved by the lockfile sync.
RUN uv pip install --python /app/sidecar/.venv/bin/python "duckdb==1.5.5"

# ─────────────────────────────────────────────────────────────────────────
# Optional built frontend (website Phase C §P2). The SvelteKit package in
# ``frontend/`` is built by ITS OWN toolchain, not here — node in this image
# would double its size for a site that releases on its own cadence.
#
# Build order:
#   1. cd frontend && npm ci && npm run build     → frontend/build/
#   2. docker compose -f sidecar/deploy/public/docker-compose.yml build
#
# The copy must not fail when step 1 hasn't run. ``fronten[d]/buil[d]`` is
# a glob that matches nothing when either directory is absent — BOTH path
# segments must be patterns, because buildkit lstat()s a literal parent
# (``frontend/buil[d]`` dies with "lstat /frontend: no such file or
# directory" against the private-stack bundle, which ships no frontend/
# at all). The paired ``pyproject.toml`` (always present) keeps the source
# set non-empty so COPY can't error out; the RUN below deletes that
# stand-in again. Result: an image with the site when it was built, and an
# API-only image when it wasn't — the service logs ``frontend_dir_missing``
# and boots either way.
FROM python:3.12-slim AS frontend
COPY pyproject.toml fronten[d]/buil[d] /frontend/build/
RUN rm -f /frontend/build/pyproject.toml

# ─────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Runtime libs only — no compilers in the final image. The manylinux wheels
# for h5py / pyproj / Pillow / opencv-python-headless on x86_64 ship their
# native deps bundled inside the wheel, so we only need:
#   - libgomp1   — OpenMP runtime that opencv links against (not bundled)
#   - libglib2.0-0 — used by some opencv pathways even in headless build
#   - tini       — PID 1 / signal forwarding for the entrypoint
#   - curl + ca-certificates — for the HEALTHCHECK probe and TLS chains
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libglib2.0-0 tini ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Unprivileged service user. Pre-create the data dir owned by the service
# user so that on first volume mount docker propagates the right
# permissions (named-volume initialisation inherits the underlying
# directory's owner + mode).
RUN useradd --create-home --shell /bin/bash --uid 10001 dmi \
    && mkdir -p /var/lib/dmi-nowcast /var/lib/dmi-nowcast-corpus \
    && chown -R dmi:dmi /var/lib/dmi-nowcast /var/lib/dmi-nowcast-corpus

WORKDIR /app

# Copy the entire built /app tree, preserving the build paths so the
# editable .pth references inside the venv still resolve.
COPY --from=builder --chown=dmi:dmi /app /app

# The static site, when one was built (see the ``frontend`` stage). Served
# only by instances whose config sets ``server.frontend_dir`` to this path —
# the public stack does, the LAN stack doesn't.
COPY --from=frontend --chown=dmi:dmi /frontend/build /app/frontend/build

USER dmi

# Data directory mounted as a volume in compose. The default below matches
# the compose mount-point so a bare `docker run` also works.
ENV DMI_NOWCAST_CONFIG=/app/sidecar/config.yaml \
    DMI_NOWCAST_STORAGE__DATA_DIR=/var/lib/dmi-nowcast \
    PATH=/app/sidecar/.venv/bin:$PATH

EXPOSE 8081

# Seed-and-exec shim runs as the unprivileged user inside the volume
# (which the Dockerfile chown'd to dmi). tini wraps it to forward
# SIGTERM cleanly through to the python process.
COPY --chown=dmi:dmi sidecar/deploy/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/usr/bin/tini", "--", "/app/entrypoint.sh"]
CMD ["python", "-m", "dmi_nowcast_sidecar"]

# Healthcheck the orchestrator can wire to.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fs http://localhost:8081/healthz || exit 1
