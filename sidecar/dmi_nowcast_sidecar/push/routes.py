"""The ``/api/push/*`` HTTP surface.

Five routes, in two groups:

**Public** (allow-listed in ``app.py``'s public-mode gate, reachable by an
anonymous browser):

- ``GET  /api/push/config``      — is push on, the VAPID public key, and
  the option lists the UI renders. ``{"enabled": false}`` when off.
- ``POST /api/push/subscribe``   — store or update one browser
  subscription plus its alert preferences.
- ``POST /api/push/unsubscribe`` — forget one endpoint. Idempotent.

**Private** (bearer-gated *and* deliberately absent from the public
allow-list, so in public mode they 404 without the key):

- ``POST /api/push/test``  — send the canned test notification.
- ``GET  /api/push/stats`` — counts and the last fan-out summary.

Every route answers ``503`` while push is disabled (or failed to
initialise), never ``404`` — the routes exist, the feature does not.

Validation is layered: pydantic handles shapes and bounds (422), the
handler handles anything that depends on configuration or on the radar
grid (400 with a readable ``detail``). Nothing is written unless every
check passed.
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Annotated, Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from ..config import Config
from ..national_sample import sample_point
from .endpoint_policy import validate_endpoint
from .store import NewSubscription, PushStore, sub_id

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .service import PushService

_log = structlog.get_logger(__name__)

_NO_STORE = {"Cache-Control": "no-store"}
_HHMM = r"^(?:[01]\d|2[0-3]):[0-5]\d$"
# Endpoint and key bounds. A browser endpoint is a URL well under 1 KB and
# the two keys are fixed-size base64; the caps exist so a hostile body
# cannot fill the SQLite file.
_MAX_ENDPOINT = 2048
_MAX_KEY = 512


def _require_api_key(request: Request) -> None:
    """Bearer dependency for the private routes.

    Imported lazily: ``app`` imports this module to register the router, so
    a module-level import of ``app.require_api_key`` would be circular.
    """
    from ..app import require_api_key

    require_api_key(request)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class PushKeys(BaseModel):
    """The two keys a ``PushSubscription`` carries (ECDH + auth secret)."""

    p256dh: Annotated[str, Field(min_length=1, max_length=_MAX_KEY)]
    auth: Annotated[str, Field(min_length=1, max_length=_MAX_KEY)]


class BrowserSubscription(BaseModel):
    """``PushSubscription.toJSON()`` as the browser produces it.

    ``expirationTime`` is accepted (browsers send it, usually ``null``) and
    ignored — the store keeps only what a send needs.
    """

    model_config = ConfigDict(extra="ignore")

    endpoint: Annotated[str, Field(min_length=1, max_length=_MAX_ENDPOINT)]
    keys: PushKeys
    expirationTime: Any | None = None  # noqa: N815 - the browser's field name


class QuietHoursIn(BaseModel):
    enabled: bool
    start: Annotated[str, Field(pattern=_HHMM)] = "22:00"
    end: Annotated[str, Field(pattern=_HHMM)] = "07:00"


class SubscribeRequest(BaseModel):
    subscription: BrowserSubscription
    lat: Annotated[float, Field(ge=-90.0, le=90.0)]
    lon: Annotated[float, Field(ge=-180.0, le=180.0)]
    threshold_pct: Annotated[int, Field(ge=1, le=99)]
    lead_min: Annotated[int, Field(ge=1, le=180)]
    quiet_hours: QuietHoursIn | None = None
    # IANA zone name; the length cap keeps a junk value out of the DB
    # before ``ZoneInfo`` even sees it.
    tz: Annotated[str, Field(min_length=1, max_length=64)]
    lang: Literal["da", "en"]


class UnsubscribeRequest(BaseModel):
    endpoint: Annotated[str, Field(min_length=1, max_length=_MAX_ENDPOINT)]


class TestRequest(BaseModel):
    endpoint: Annotated[str | None, Field(max_length=_MAX_ENDPOINT)] = None


class QuietHoursOut(BaseModel):
    enabled: bool
    start: str
    end: str


class PushDefaults(BaseModel):
    threshold_pct: int
    lead_min: int
    quiet_hours: QuietHoursOut


class PushConfigResponse(BaseModel):
    """``{"enabled": false}`` when off; the full shape when on.

    Served with ``response_model_exclude_none=True`` so the disabled answer
    really is that one key — a frontend switches on it.
    """

    enabled: bool
    vapid_public_key: str | None = None
    threshold_options_pct: list[int] | None = None
    lead_options_min: list[int] | None = None
    defaults: PushDefaults | None = None
    capacity_reached: bool | None = None


class SubscribeResponse(BaseModel):
    ok: bool
    created: bool


class UnsubscribeResponse(BaseModel):
    ok: bool
    deleted: bool


class SendResponse(BaseModel):
    sent: int
    failed: int
    removed: int


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def lead_options(config: Config) -> list[int]:
    """Offered lead times: national probability leads at/after ``min_lead_min``.

    Derived, never configured separately — a lead the grids do not carry
    could be selected and then never evaluated.
    """
    return sorted(
        int(lead)
        for lead in config.forecast.national.leads_min
        if int(lead) >= config.push.min_lead_min
    )


def build_router(
    config: Config,
    *,
    store: PushStore | None = None,
    public_key: str | None = None,
    service: "PushService | None" = None,
) -> APIRouter:
    """Build the ``/api/push`` router.

    ``store``/``public_key``/``service`` are ``None`` when push is disabled
    **or** when its initialisation failed; either way the routes answer as
    "disabled" rather than pretending to work.
    """
    router = APIRouter(prefix="/api/push", tags=["push"])
    enabled = bool(config.push.enabled and store is not None and public_key)

    def _require_enabled() -> PushStore:
        if not enabled or store is None:
            raise HTTPException(
                status_code=503, detail="push notifications are disabled",
            )
        return store

    @router.get(
        "/config",
        response_model=PushConfigResponse,
        response_model_exclude_none=True,
    )
    async def push_config(response: Response) -> PushConfigResponse:
        """What the browser needs to render the subscribe form."""
        response.headers.update(_NO_STORE)
        if not enabled or store is None:
            return PushConfigResponse(enabled=False)
        count = await asyncio.to_thread(store.count)
        pc = config.push
        return PushConfigResponse(
            enabled=True,
            vapid_public_key=public_key,
            threshold_options_pct=list(pc.threshold_options_pct),
            lead_options_min=lead_options(config),
            defaults=PushDefaults(
                threshold_pct=pc.default_threshold_pct,
                lead_min=pc.default_lead_min,
                quiet_hours=QuietHoursOut(
                    enabled=False,
                    start=pc.default_quiet_start,
                    end=pc.default_quiet_end,
                ),
            ),
            capacity_reached=count >= pc.max_subscriptions,
        )

    @router.post("/subscribe", response_model=SubscribeResponse)
    async def subscribe(
        body: SubscribeRequest, request: Request, response: Response,
    ) -> SubscribeResponse:
        """Store (or replace) one subscription and its preferences."""
        response.headers.update(_NO_STORE)
        active = _require_enabled()
        pc = config.push

        if body.threshold_pct not in pc.threshold_options_pct:
            raise HTTPException(
                status_code=400,
                detail=(
                    "threshold_pct must be one of "
                    f"{list(pc.threshold_options_pct)}"
                ),
            )
        leads = lead_options(config)
        if body.lead_min not in leads:
            raise HTTPException(
                status_code=400, detail=f"lead_min must be one of {leads}",
            )
        try:
            ZoneInfo(body.tz)
        except (ZoneInfoNotFoundError, ValueError):
            raise HTTPException(
                status_code=400, detail="tz must be an IANA time-zone name",
            ) from None

        reason = validate_endpoint(
            body.subscription.endpoint, pc.allowed_endpoint_host_suffixes,
        )
        if reason is not None:
            raise HTTPException(status_code=400, detail=reason)

        # Off-coverage check, using the same sampler /forecast uses, so a
        # point that can never produce a notification is refused at
        # subscribe time rather than silently never firing. Before the
        # first national cycle there is nothing to check against: accept.
        engine = getattr(request.app.state, "engine", None)
        latest = getattr(engine, "national_latest", None) if engine else None
        geo = getattr(engine, "geo", None) if engine else None
        if latest is not None and geo is not None:
            products, _radar_ts = latest
            if sample_point(products, geo, body.lat, body.lon) is None:
                raise HTTPException(
                    status_code=400,
                    detail="coordinates outside the radar composite grid",
                )

        endpoint = body.subscription.endpoint
        existing = await asyncio.to_thread(active.get, endpoint)
        if existing is None:
            count = await asyncio.to_thread(active.count)
            if count >= pc.max_subscriptions:
                raise HTTPException(
                    status_code=503,
                    detail="subscription capacity reached; try again later",
                )

        quiet = body.quiet_hours
        created = await asyncio.to_thread(
            active.upsert,
            NewSubscription(
                endpoint=endpoint,
                p256dh=body.subscription.keys.p256dh,
                auth=body.subscription.keys.auth,
                lat=body.lat,
                lon=body.lon,
                threshold_pct=body.threshold_pct,
                lead_min=body.lead_min,
                quiet_enabled=bool(quiet.enabled) if quiet else False,
                quiet_start=quiet.start if quiet else pc.default_quiet_start,
                quiet_end=quiet.end if quiet else pc.default_quiet_end,
                tz=body.tz,
                lang=body.lang,
            ),
        )
        _log.info(
            "push_subscribed",
            sub=sub_id(endpoint),
            created=created,
            threshold_pct=body.threshold_pct,
            lead_min=body.lead_min,
            lang=body.lang,
        )
        return SubscribeResponse(ok=True, created=created)

    @router.post("/unsubscribe", response_model=UnsubscribeResponse)
    async def unsubscribe(
        body: UnsubscribeRequest, response: Response,
    ) -> UnsubscribeResponse:
        """Forget one endpoint. 200 whether or not it was there — a browser
        unsubscribing twice is not an error, and the difference would leak
        whether an endpoint is registered."""
        response.headers.update(_NO_STORE)
        active = _require_enabled()
        deleted = await asyncio.to_thread(active.delete, body.endpoint)
        _log.info("push_unsubscribed", sub=sub_id(body.endpoint), deleted=deleted)
        return UnsubscribeResponse(ok=True, deleted=deleted)

    @router.post(
        "/test",
        response_model=SendResponse,
        dependencies=[Depends(_require_api_key)],
    )
    async def send_test(
        response: Response, body: TestRequest | None = None,
    ) -> SendResponse:
        """Send the canned test payload to one endpoint, or to all of them."""
        response.headers.update(_NO_STORE)
        _require_enabled()
        if service is None:
            raise HTTPException(
                status_code=503, detail="push notifications are disabled",
            )
        counts = await service.send_test(body.endpoint if body else None)
        return SendResponse(**counts)

    @router.get(
        "/stats",
        response_model=None,
        dependencies=[Depends(_require_api_key)],
    )
    async def stats(response: Response) -> dict:
        """Operator view: how many subscriptions, and what the last cycle did.

        Counts only — no endpoints, no coordinates.
        """
        response.headers.update(_NO_STORE)
        active = _require_enabled()
        base = await asyncio.to_thread(active.stats)
        last_ts = getattr(service, "last_evaluated_radar_ts", None)
        return {
            **base,
            "last_evaluated_radar_ts": last_ts.isoformat() if last_ts else None,
            "last_fanout": getattr(service, "last_fanout", None),
        }

    return router


__all__ = [
    "BrowserSubscription",
    "PushConfigResponse",
    "SubscribeRequest",
    "build_router",
    "lead_options",
]
