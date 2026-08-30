#!/usr/bin/env bash
# Wrapper that seeds the persisted data volume with the baked-in
# calibration curves on first boot, then execs the sidecar.
#
# Why a separate script: the curves config-path is in the data volume so
# the monthly recalibration job (calibrate.sh) can write fresh curves
# that survive image rebuilds. But on first boot the volume is empty —
# without seeding, calibration would be unavailable until the first
# recalibration ran (potentially weeks).
set -euo pipefail

BAKED_CURVES=/app/src/dmi_nowcast_core/calibration_curves.json
VOLUME_CURVES=/var/lib/dmi-nowcast/calibration_curves.json

if [[ -f "$BAKED_CURVES" && ! -f "$VOLUME_CURVES" ]]; then
    echo "[entrypoint] seeding calibration curves from baked-in copy" >&2
    cp "$BAKED_CURVES" "$VOLUME_CURVES"
fi

exec "$@"
