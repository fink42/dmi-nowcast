"""VAPID key handling.

Web Push identifies the sender with a P-256 key pair (RFC 8292). The
browser is handed the **public** key as the raw uncompressed X9.62 point,
base64url without padding — that exact 65-byte encoding is what
``PushManager.subscribe({applicationServerKey})`` accepts, and anything
else fails in the browser with an opaque error.

The private key is a PKCS#8 PEM in the data volume, generated on first
start with mode 0600. It is the service's identity: rotate it and every
existing subscription's ``applicationServerKey`` stops matching, so the
browsers have to re-subscribe. Back it up with the subscription DB.

``cryptography`` is used directly (it arrives as a ``pywebpush``
dependency) rather than ``py_vapid``'s file helpers: the PEM this writes
loads cleanly through ``py_vapid.Vapid02.from_pem`` for the actual send.
"""
from __future__ import annotations

import base64
import os
from pathlib import Path

import structlog
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

_log = structlog.get_logger(__name__)

_PEM_HEADER = b"-----BEGIN PRIVATE KEY-----"


def generate_private_key_pem() -> bytes:
    """A fresh P-256 private key as an unencrypted PKCS#8 PEM."""
    key = ec.generate_private_key(ec.SECP256R1())
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def ensure_private_key(path: Path) -> bytes:
    """Read the PEM at ``path``, creating it (mode 0600) when missing.

    Created with ``O_EXCL`` so two processes racing on the same data volume
    cannot both think they wrote the key — the loser re-reads the winner's.
    """
    path = Path(path)
    if path.is_file():
        return path.read_bytes()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    pem = generate_private_key_pem()
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return path.read_bytes()
    with os.fdopen(fd, "wb") as fh:
        fh.write(pem)
    _log.info("vapid_key_generated", path=str(path))
    return pem


def public_key_b64url(pem: bytes) -> str:
    """The ``applicationServerKey`` a browser needs, from a private PEM.

    Raw uncompressed point (65 bytes, leading ``0x04``), base64url, no
    padding.
    """
    key = serialization.load_pem_private_key(pem, password=None)
    if not isinstance(key, ec.EllipticCurvePrivateKey):  # pragma: no cover
        raise ValueError("VAPID key must be an EC (P-256) private key")
    raw = key.public_key().public_bytes(
        encoding=serialization.Encoding.X962,
        format=serialization.PublicFormat.UncompressedPoint,
    )
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


__all__ = ["ensure_private_key", "generate_private_key_pem", "public_key_b64url"]
