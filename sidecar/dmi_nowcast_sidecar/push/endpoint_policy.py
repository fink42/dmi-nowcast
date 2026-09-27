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

**Parser differentials.** The URL is checked here with ``urllib.parse``
but connected to by ``requests``/``urllib3``, and the two do not always
agree on what the host is: ``https://127.0.0.1\\.fcm.googleapis.com/x`` is
host ``127.0.0.1\\.fcm.googleapis.com`` (an allowed suffix!) to the first
and ``127.0.0.1`` to the second. So the URL must be plain ASCII with no
whitespace or control character anywhere, no backslash anywhere, no
``%``/``@`` in the authority, the host must be ``[a-z0-9.-]`` only (no
trailing dot, no empty label), and ``urllib3.util.parse_url`` must read
the very same scheme, host and port. Anything the two parsers could read
differently is refused rather than reasoned about. The sender also never
follows redirects (``push.fanout``), so an allowed host cannot bounce the
POST somewhere else.

Returns a reason string rather than raising: the caller turns it into a
400 with that text, which is the only feedback a subscriber gets.
"""
from __future__ import annotations

import ipaddress
import re
from typing import Sequence
from urllib.parse import urlsplit

from urllib3.util import parse_url

_HOST_RE = re.compile(r"^[a-z0-9.-]+$")


def _has_unsafe_char(text: str) -> bool:
    return any(ch.isspace() or ord(ch) < 0x20 or ord(ch) == 0x7F for ch in text)


def validate_endpoint(url: str, allowed_suffixes: Sequence[str]) -> str | None:
    """``None`` when ``url`` is an acceptable push endpoint, else why not."""
    if not url or not url.strip():
        return "endpoint must not be empty"
    if not url.isascii():
        return "endpoint must be plain ASCII"
    if _has_unsafe_char(url):
        return "endpoint must not contain whitespace or control characters"
    if "\\" in url:
        return "endpoint must not contain a backslash"
    try:
        parts = urlsplit(url)
    except ValueError:
        return "endpoint is not a valid URL"

    if parts.scheme.lower() != "https":
        return "endpoint must use https"
    authority = parts.netloc
    if "@" in authority or parts.username or parts.password:
        return "endpoint must not embed credentials"
    if "%" in authority:
        return "endpoint host must not be percent-encoded"

    try:
        host = parts.hostname
    except ValueError:
        return "endpoint host is not valid"
    if not host:
        return "endpoint must have a host"
    host = host.lower()

    try:
        port = parts.port
    except ValueError:
        return "endpoint port is not valid"
    if port is not None and port != 443:
        return "endpoint must use the default https port"

    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        return "endpoint host must be a DNS name, not an IP address"

    if not _HOST_RE.match(host):
        return "endpoint host is not valid"
    if host.endswith(".") or host.startswith(".") or ".." in host:
        return "endpoint host is not valid"

    # The connecting library must see exactly what we just checked.
    try:
        other = parse_url(url)
    except Exception:  # noqa: BLE001 - urllib3 raises LocationParseError & co
        return "endpoint is not a valid URL"
    if (
        (other.scheme or "").lower() != "https"
        or (other.host or "").lower() != host
        or other.port != port
        or other.auth is not None
    ):
        return "endpoint is not a valid URL"

    for suffix in allowed_suffixes:
        candidate = suffix.strip().rstrip(".").lower()
        if not candidate:
            continue
        if host == candidate or host.endswith("." + candidate):
            return None
    return "endpoint host is not a known push service"


__all__ = ["validate_endpoint"]
