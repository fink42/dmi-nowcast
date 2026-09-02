"""VAPID key generator.

    python -m dmi_nowcast_sidecar.push.keygen /var/lib/dmi-nowcast/push/vapid_private.pem

Writes a new P-256 private key (mode 0600) and prints the public key in
the form a browser wants. The service generates its own key on first start
when one is missing, so this CLI is for the case where you want the key
*before* the container boots — provisioning a volume, or pinning the same
identity across two instances.

Rotating an existing key invalidates every stored subscription, so the
command refuses to overwrite without ``--force``.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .vapid import generate_private_key_pem, public_key_b64url


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dmi_nowcast_sidecar.push.keygen",
        description="Generate a VAPID private key for Web Push.",
    )
    parser.add_argument("path", type=Path, help="where to write the PEM")
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "overwrite an existing key — this invalidates every stored "
            "subscription, which then has to re-subscribe"
        ),
    )
    args = parser.parse_args(argv)

    path: Path = args.path
    if path.exists() and not args.force:
        print(
            f"refusing to overwrite {path} (pass --force to rotate the key; "
            "every existing subscription stops working)",
            file=sys.stderr,
        )
        return 1

    pem = generate_private_key_pem()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Write through a 0600 descriptor so the key is never briefly world-readable.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(pem)
    os.chmod(path, 0o600)

    print(f"private key: {path}")
    print(f"public key:  {public_key_b64url(pem)}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via main()
    raise SystemExit(main())
