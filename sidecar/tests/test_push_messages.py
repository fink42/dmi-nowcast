"""Notification wording, number formats and the payload contract.

The service worker renders these strings verbatim, so the assertions are
on exact text — a change here is a change the phone shows.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from dmi_nowcast_sidecar.push.messages import TAG, rain_incoming_payload
# Aliased on import: pytest would otherwise collect the payload builder
# itself as a test case, because of its name.
from dmi_nowcast_sidecar.push.messages import test_payload as build_test_payload

SENT = datetime(2026, 9, 2, 14, 5, 30, tzinfo=timezone.utc)
LAT, LON = 55.33, 10.32

RAIN_KEYS = {
    "type",
    "title",
    "body",
    "lang",
    "lat",
    "lon",
    "url",
    "tag",
    "sent_utc",
    "eta_min",
    "p_pct",
    "lead_min",
    "intensity_mm_h",
}
TEST_KEYS = {
    "type",
    "title",
    "body",
    "lang",
    "lat",
    "lon",
    "url",
    "tag",
    "sent_utc",
}


def _rain(**overrides) -> dict:
    kwargs = dict(
        lang="da",
        lat=LAT,
        lon=LON,
        eta_min=18.0,
        p_rain=0.78,
        lead_min=30,
        intensity_mm_h=4.0,
        sent_utc=SENT,
    )
    kwargs.update(overrides)
    return rain_incoming_payload(**kwargs)


# --------------------------------------------------------------------------
# Wording, both languages, both headline branches
# --------------------------------------------------------------------------


def test_danish_with_eta() -> None:
    payload = _rain(lang="da")
    assert payload["title"] == "Regn på vej — ca. 18 min"
    assert payload["body"] == (
        "Moderat regn (~4,0 mm/t). Sandsynlighed inden for 30 min: 78 %."
    )


def test_english_with_eta() -> None:
    payload = _rain(lang="en")
    assert payload["title"] == "Rain incoming — ~18 min"
    assert payload["body"] == (
        "Moderate rain (~4.0 mm/h). Probability within 30 min: 78%."
    )


def test_danish_without_eta() -> None:
    payload = _rain(lang="da", eta_min=None, p_rain=0.45, intensity_mm_h=None)
    assert payload["title"] == "Regn på vej inden for 30 min"
    assert payload["body"] == "Sandsynlighed for regn ved dit punkt: 45 %."


def test_english_without_eta() -> None:
    payload = _rain(lang="en", eta_min=None, p_rain=0.45, intensity_mm_h=None)
    assert payload["title"] == "Rain incoming within 30 min"
    assert payload["body"] == "Probability of rain at your point: 45%."


def test_without_eta_keeps_a_meaningful_intensity_clause() -> None:
    payload = _rain(lang="da", eta_min=None, p_rain=0.45, intensity_mm_h=12.0)
    assert payload["body"] == (
        "Kraftig regn (~12,0 mm/t). Sandsynlighed for regn ved dit punkt: 45 %."
    )


@pytest.mark.parametrize("lang", ["da", "en"])
def test_test_message(lang: str) -> None:
    payload = build_test_payload(lang=lang, lat=LAT, lon=LON, sent_utc=SENT)
    assert payload["type"] == "test"
    if lang == "da":
        assert payload["title"] == "Testbesked fra Regnradar"
        assert payload["body"] == "Notifikationer virker på denne enhed."
    else:
        assert payload["title"] == "Test message from Rain radar"
        assert payload["body"] == "Notifications work on this device."


# --------------------------------------------------------------------------
# Intensity bands — the frontend's, exactly
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mm_h", "da_word", "en_word"),
    [
        (0.06, "Let", "Light"),
        (2.4, "Let", "Light"),
        (2.49, "Let", "Light"),
        (2.5, "Moderat", "Moderate"),
        (9.99, "Moderat", "Moderate"),
        (10.0, "Kraftig", "Heavy"),
        (49.9, "Kraftig", "Heavy"),
        (50.0, "Voldsom", "Violent"),
        (120.0, "Voldsom", "Violent"),
    ],
)
def test_intensity_bands(mm_h: float, da_word: str, en_word: str) -> None:
    assert _rain(lang="da", intensity_mm_h=mm_h)["body"].startswith(
        f"{da_word} regn ("
    )
    assert _rain(lang="en", intensity_mm_h=mm_h)["body"].startswith(
        f"{en_word} rain ("
    )


@pytest.mark.parametrize("mm_h", [None, 0.0, 0.05])
def test_no_intensity_clause_below_the_detection_band(mm_h: float | None) -> None:
    da = _rain(lang="da", intensity_mm_h=mm_h)
    en = _rain(lang="en", intensity_mm_h=mm_h)
    assert da["body"] == "Sandsynlighed inden for 30 min: 78 %."
    assert en["body"] == "Probability within 30 min: 78%."
    # The machine-readable field stays faithful to the reading (1 decimal)
    # even when the sentence drops the clause.
    assert da["intensity_mm_h"] == (None if mm_h is None else round(mm_h, 1))


def test_danish_uses_a_decimal_comma_and_english_a_point() -> None:
    assert "(~2,4 mm/t)" in _rain(lang="da", intensity_mm_h=2.4)["body"]
    assert "(~2.4 mm/h)" in _rain(lang="en", intensity_mm_h=2.4)["body"]


# --------------------------------------------------------------------------
# Rounding and machine-readable fields
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("eta_in", "eta_out"), [(17.4, 17), (17.5, 18), (17.6, 18), (0.4, 0), (59.5, 60)]
)
def test_eta_rounds_half_up(eta_in: float, eta_out: int) -> None:
    payload = _rain(eta_min=eta_in)
    assert payload["eta_min"] == eta_out
    assert f"{eta_out} min" in payload["title"]


@pytest.mark.parametrize(
    ("p_in", "pct"), [(0.0, 0), (0.444, 44), (0.446, 45), (0.78, 78), (1.0, 100)]
)
def test_probability_percent(p_in: float, pct: int) -> None:
    payload = _rain(p_rain=p_in)
    assert payload["p_pct"] == pct
    assert isinstance(payload["p_pct"], int)
    assert f"{pct} %" in payload["body"]


def test_machine_readable_fields() -> None:
    payload = _rain(lead_min=45, intensity_mm_h=4.04)
    assert payload["type"] == "rain_incoming"
    assert payload["lead_min"] == 45
    assert payload["intensity_mm_h"] == 4.0
    assert payload["eta_min"] == 18


def test_eta_and_intensity_are_null_when_absent() -> None:
    payload = _rain(eta_min=None, intensity_mm_h=None)
    assert payload["eta_min"] is None
    assert payload["intensity_mm_h"] is None


# --------------------------------------------------------------------------
# Language fallback
# --------------------------------------------------------------------------


@pytest.mark.parametrize("lang", ["xx", "", "de", "fr-FR", None, "danish"])
def test_unknown_language_falls_back_to_danish(lang) -> None:
    payload = _rain(lang=lang)
    assert payload["lang"] == "da"
    assert payload["title"].startswith("Regn på vej")
    assert build_test_payload(lang=lang, lat=LAT, lon=LON, sent_utc=SENT)["lang"] == "da"


@pytest.mark.parametrize("lang", ["en", "EN", "en-GB", "en_US", " en "])
def test_english_tags_are_normalised(lang: str) -> None:
    payload = _rain(lang=lang)
    assert payload["lang"] == "en"
    assert payload["title"].startswith("Rain incoming")


# --------------------------------------------------------------------------
# The rest of the payload contract
# --------------------------------------------------------------------------


def test_url_and_coordinates() -> None:
    payload = _rain(lat=55.3, lon=10.3212345)
    assert payload["url"] == "/?lat=55.3000&lon=10.3212"
    assert payload["lat"] == 55.3
    assert payload["lon"] == 10.3212


def test_negative_coordinates_survive_the_url_format() -> None:
    payload = build_test_payload(lang="en", lat=-33.8688, lon=-70.1234, sent_utc=SENT)
    assert payload["url"] == "/?lat=-33.8688&lon=-70.1234"


@pytest.mark.parametrize("kind", ["rain", "test"])
def test_tag_is_shared_so_a_newer_push_replaces_an_unread_one(kind: str) -> None:
    payload = (
        _rain()
        if kind == "rain"
        else build_test_payload(lang="da", lat=LAT, lon=LON, sent_utc=SENT)
    )
    assert payload["tag"] == TAG == "rain-incoming"


def test_sent_utc_is_iso_z_at_second_precision() -> None:
    assert _rain()["sent_utc"] == "2026-09-02T14:05:30Z"
    assert _rain(sent_utc=SENT.replace(microsecond=123456))["sent_utc"] == (
        "2026-09-02T14:05:30Z"
    )


def test_sent_utc_converts_from_another_zone() -> None:
    local = SENT.astimezone(timezone(timedelta(hours=2)))
    assert local.hour == 16
    assert _rain(sent_utc=local)["sent_utc"] == "2026-09-02T14:05:30Z"


def test_naive_sent_utc_is_read_as_utc() -> None:
    assert _rain(sent_utc=SENT.replace(tzinfo=None))["sent_utc"] == (
        "2026-09-02T14:05:30Z"
    )


def test_keys_are_exactly_the_contract() -> None:
    assert set(_rain().keys()) == RAIN_KEYS
    assert set(
        build_test_payload(lang="da", lat=LAT, lon=LON, sent_utc=SENT).keys()
    ) == TEST_KEYS


@pytest.mark.parametrize("lang", ["da", "en"])
@pytest.mark.parametrize("eta", [18.0, None])
def test_payloads_are_json_serialisable(lang: str, eta: float | None) -> None:
    payload = _rain(lang=lang, eta_min=eta)
    restored = json.loads(json.dumps(payload))
    assert restored == payload
    restored_test = json.loads(
        json.dumps(build_test_payload(lang=lang, lat=LAT, lon=LON, sent_utc=SENT))
    )
    assert restored_test["type"] == "test"


@pytest.mark.parametrize("lang", ["da", "en"])
@pytest.mark.parametrize("lead", [20, 30, 45, 60])
@pytest.mark.parametrize("eta", [None, 5.0, 18.0, 59.0])
def test_titles_stay_short_enough_for_ios(
    lang: str, lead: int, eta: float | None
) -> None:
    payload = _rain(lang=lang, lead_min=lead, eta_min=eta)
    assert len(payload["title"]) <= 40
    assert len(build_test_payload(lang=lang, lat=LAT, lon=LON, sent_utc=SENT)["title"]) <= 40
