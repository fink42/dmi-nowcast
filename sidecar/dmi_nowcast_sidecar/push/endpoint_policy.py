"""SSRF guard over browser-supplied push endpoints.

A subscribe request hands the service a URL and the service later POSTs to
it — from a VM that can see the LAN, the Docker network and the metadata
range. Without a policy, ``POST /api/push/subscribe`` is an open
request-forgery primitive for anyone who can reach the public site.

The policy is an allow-list of the real push services (see
``push.allowed_endpoint_host_suffixes``), plus the structural rules that
make an allow-list meaningful:

- ``https`` only — no ``http``, no ``file``, no ``gopher``;
- a DNS name, never an IP literal (``https://192.168.1.10/...`` and
  ``https://[::1]/...`` are the LAN pivot the allow-list exists to stop);
- no embedded credentials, and port 443 only;
- the host must **equal** an allowed suffix or end with ``.`` + it, so
  ``fcm.googleapis.com.evil.example`` is rejected — a plain
  ``str.endswith`` would accept it.

Returns a reason string rather than raising: the caller turns it into a
400 with that text, which is the only feedback a subscriber gets.
"""
from __future__ import annotations

import ipaddress
from typing import Sequence
from urllib.parse import urlsplit


def validate_endpoint(url: str, allowed_suffixes: Sequence[str]) -> str | None:
    """``None`` when ``url`` is an acceptable push endpoint, else why not."""
    if not url or not url.strip():
        return "endpoint must not be empty"
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return "endpoint is not a valid URL"

    if parts.scheme.lower() != "https":
        return "endpoint must use https"
    if parts.username or parts.password:
        return "endpoint must not embed credentials"

    try:
        host = parts.hostname
    except ValueError:
        return "endpoint host is not valid"
    if not host:
        return "endpoint must have a host"
    host = host.strip().rstrip(".").lower()
    if not host:
        return "endpoint must have a host"

    try:
        parts.port  # noqa: B018 - raises on a malformed port
    except ValueError:
        return "endpoint port is not valid"
    if parts.port is not None and parts.port != 443:
        return "endpoint must use the default https port"

    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        return "endpoint host must be a DNS name, not an IP address"

    for suffix in allowed_suffixes:
        candidate = suffix.strip().rstrip(".").lower()
        if not candidate:
            continue
        if host == candidate or host.endswith("." + candidate):
            return None
    return "endpoint host is not a known push service"


__all__ = ["validate_endpoint"]
