"""Notification payloads, localised server-side.

The push payload is a contract with the service worker: it renders what
it is handed, so every string, number format and rounding decision is
made here from the subscription's stored ``lang``.

Danish is the default and the fallback — an unknown language tag is not
an error, it is Danish. Numbers follow the language: decimal comma and
``mm/t`` in Danish, decimal point and ``mm/h`` in English.

The intensity bands are the frontend's (``frontend/src/lib/format.ts``),
deliberately duplicated rather than derived, so a notification and the
panel never disagree about the word for 3 mm/h:

    <= 0.05 none · < 2.5 light · < 10 moderate · < 50 heavy · else violent

Titles stay short — iOS truncates around 40 characters.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

__all__ = ["rain_incoming_payload", "test_payload"]

#: Both notification types share a tag, so a newer alert replaces an
#: unread one on the device instead of stacking.
TAG = "rain-incoming"

_STRINGS: dict[str, dict[str, str]] = {
    "da": {
        "title_eta": "Regn på vej — ca. {eta} min",
        "title_window": "Regn på vej inden for {lead} min",
        "intensity_clause": "{word} regn (~{mm} mm/t). ",
        "prob_eta": "Sandsynlighed inden for {lead} min: {pct} %.",
        "prob_window": "Sandsynlighed for regn ved dit punkt: {pct} %.",
        "test_title": "Testbesked fra Regnradar",
        "test_body": "Notifikationer virker på denne enhed.",
        "intensity_light": "Let",
        "intensity_moderate": "Moderat",
        "intensity_heavy": "Kraftig",
        "intensity_violent": "Voldsom",
    },
    "en": {
        "title_eta": "Rain incoming — ~{eta} min",
        "title_window": "Rain incoming within {lead} min",
        "intensity_clause": "{word} rain (~{mm} mm/h). ",
        "prob_eta": "Probability within {lead} min: {pct}%.",
        "prob_window": "Probability of rain at your point: {pct}%.",
        "test_title": "Test message from Rain radar",
        "test_body": "Notifications work on this device.",
        "intensity_light": "Light",
        "intensity_moderate": "Moderate",
        "intensity_heavy": "Heavy",
        "intensity_violent": "Violent",
    },
}


def _lang(lang: str | None) -> str:
    """Normalise a language tag to ``"da"`` or ``"en"``; anything else is Danish."""
    tag = str(lang or "").strip().lower().replace("_", "-").split("-")[0]
    return "en" if tag == "en" else "da"


def _round_half_up(value: float) -> int:
    """Round like the frontend's ``Math.round`` (ties away from zero, upward)."""
    return int(math.floor(value + 0.5))


def _decimal(value: float, lang: str) -> str:
    """One decimal place, with a comma in Danish."""
    text = f"{value:.1f}"
    return text.replace(".", ",") if lang == "da" else text


def _intensity_word(mm_h: float | None, lang: str) -> str | None:
    """The band word, or ``None`` for "no meaningful rain rate"."""
    if mm_h is None or mm_h <= 0.05:
        return None
    strings = _STRINGS[lang]
    if mm_h < 2.5:
        return strings["intensity_light"]
    if mm_h < 10:
        return strings["intensity_moderate"]
    if mm_h < 50:
        return strings["intensity_heavy"]
    return strings["intensity_violent"]


def _iso_z(moment: datetime) -> str:
    """ISO 8601 UTC with a ``Z`` suffix, second precision.

    A naive datetime is read as UTC — the house rule is that naive means
    UTC internally — and an aware one is converted.
    """
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _base(
    *, kind: str, lang: str, lat: float, lon: float, sent_utc: datetime
) -> dict:
    return {
        "type": kind,
        "lang": lang,
        "lat": round(lat, 4),
        "lon": round(lon, 4),
        "url": f"/?lat={lat:.4f}&lon={lon:.4f}",
        "tag": TAG,
        "sent_utc": _iso_z(sent_utc),
    }


def rain_incoming_payload(
    *,
    lang: str,
    lat: float,
    lon: float,
    eta_min: float | None,
    p_rain: float,
    lead_min: int,
    intensity_mm_h: float | None,
    sent_utc: datetime,
) -> dict:
    """The "rain incoming at your saved point" payload.

    With an ETA the headline names the minutes; without one it names the
    lead window. The intensity clause is dropped whenever the rate is
    missing or below the detection band — a "(~0,0 mm/t)" tells nobody
    anything.
    """
    lang = _lang(lang)
    strings = _STRINGS[lang]
    pct = _round_half_up(p_rain * 100)

    if eta_min is None:
        title = strings["title_window"].format(lead=lead_min)
        body_tail = strings["prob_window"].format(pct=pct)
        eta_out: int | None = None
    else:
        eta_out = _round_half_up(eta_min)
        title = strings["title_eta"].format(eta=eta_out)
        body_tail = strings["prob_eta"].format(lead=lead_min, pct=pct)

    word = _intensity_word(intensity_mm_h, lang)
    if word is None:
        body = body_tail
    else:
        body = (
            strings["intensity_clause"].format(
                word=word, mm=_decimal(float(intensity_mm_h), lang)
            )
            + body_tail
        )

    payload = _base(
        kind="rain_incoming", lang=lang, lat=lat, lon=lon, sent_utc=sent_utc
    )
    payload.update(
        {
            "title": title,
            "body": body,
            "eta_min": eta_out,
            "p_pct": pct,
            "lead_min": lead_min,
            "intensity_mm_h": (
                None
                if intensity_mm_h is None
                else round(float(intensity_mm_h), 1)
            ),
        }
    )
    return payload


def test_payload(
    *, lang: str, lat: float, lon: float, sent_utc: datetime
) -> dict:
    """The bearer-gated "does this device receive pushes at all" payload."""
    lang = _lang(lang)
    strings = _STRINGS[lang]
    payload = _base(kind="test", lang=lang, lat=lat, lon=lon, sent_utc=sent_utc)
    payload.update(
        {"title": strings["test_title"], "body": strings["test_body"]}
    )
    return payload
