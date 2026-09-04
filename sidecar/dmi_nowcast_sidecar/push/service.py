"""Per-cycle evaluation and fan-out.

The scheduler awaits :meth:`PushService.after_cycle` once per cycle, after
the state has been written. From there:

1. **Skip anything that is not a new radar observation.** The cycle fires
   every 5 min but fullRange composites land every ~10, so about half the
   cycles re-emit the previous state with the same ``radar.latest_ts``.
   The decision state machine counts *observations*, not poll firings —
   evaluating a repeated frame would double-count a streak and fire a
   notification one observation early. The last evaluated radar timestamp
   is only advanced by an evaluation that actually happened, so a skipped
   cycle never swallows a frame.

2. **Evaluate every subscription off the loop.** ``sample_point`` is the
   same sampler ``/forecast`` uses, fed the same held grids *including the
   observed-rain grid* — a notification and the panel can never disagree
   about which pixel a point reads, nor about whether it is already
   raining there.

3. **Persist first, send second.** The new state is written before any
   network call, so a crash mid-fan-out can only cost a notification, not
   cause a repeat. The failure the user forgives is a missed alert; the
   one they uninstall over is the same alert five times.

4. **Fan out sequentially inside a wall-clock budget.** The push services
   are the slow part and the cycle must not be held hostage to them:
   whatever is still queued when the budget expires is dropped and
   counted. ``404``/``410`` from a push service means the browser is gone —
   the row is deleted, which is the only garbage collection the store has.

Logging is aggregate: one ``push_fanout`` event with counts. Endpoints are
capabilities to notify a browser and never appear in a log line; the short
``sub_id`` hash is used where an individual row must be identified.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any

import structlog

from ..compute import CycleEngine, CycleResult
from ..config import Config
from ..national_sample import sample_point
from . import engine as decision_engine
from . import fanout
from .engine import Observation, QuietHours, Rules, SubState
from .messages import rain_incoming_payload, test_payload
from .store import PushStore, Subscription, sub_id

_log = structlog.get_logger(__name__)


class PushService:
    """Owns the push side-effects of a cycle. One instance per process."""

    def __init__(
        self,
        config: Config,
        engine: CycleEngine,
        store: PushStore,
        vapid_private_pem: bytes,
        public_key: str,
    ) -> None:
        self.config = config
        self.engine = engine
        self.store = store
        self.vapid_private_pem = vapid_private_pem
        self.public_key = public_key
        self._last_evaluated_radar_ts: datetime | None = None
        self._last_fanout: dict | None = None

    # -- introspection (served by /api/push/stats) --------------------------

    @property
    def last_evaluated_radar_ts(self) -> datetime | None:
        return self._last_evaluated_radar_ts

    @property
    def last_fanout(self) -> dict | None:
        return self._last_fanout

    # -- the cycle hook -----------------------------------------------------

    async def after_cycle(self, result: CycleResult) -> None:
        """Evaluate + notify for one completed cycle. Never raises."""
        if result.state is None:
            return
        radar_ts = getattr(result.state.radar, "latest_ts", None)
        if radar_ts is None:
            return
        if radar_ts.tzinfo is None:
            radar_ts = radar_ts.replace(tzinfo=timezone.utc)
        if (
            self._last_evaluated_radar_ts is not None
            and radar_ts == self._last_evaluated_radar_ts
        ):
            # The no-new-frame fast path, or a re-emitted state.
            return

        latest = self.engine.national_latest
        geo = self.engine.geo
        if latest is None or geo is None:
            _log.info("push_eval_skipped", reason="no_national_products")
            return
        products, products_ts = latest
        if products_ts is not None and products_ts.tzinfo is None:
            products_ts = products_ts.replace(tzinfo=timezone.utc)
        if products_ts != radar_ts:
            # The held grids belong to a different frame than the state we
            # were just handed. Evaluating them would attribute one frame's
            # probabilities to another frame's timestamp — and advancing
            # the last-evaluated marker would then hide the real frame.
            _log.info(
                "push_eval_skipped",
                reason="products_radar_ts_mismatch",
                products_ts=products_ts.isoformat() if products_ts else None,
                radar_ts=radar_ts.isoformat(),
            )
            return

        now_utc = datetime.now(timezone.utc)
        summary = await asyncio.to_thread(
            self._evaluate_and_send,
            products,
            geo,
            radar_ts,
            now_utc,
            # Additive on the snapshot: a cycle whose observed reduction
            # failed still evaluates, on the ETA test alone.
            getattr(latest, "observed_mm_h", None),
        )
        self._last_evaluated_radar_ts = radar_ts
        self._last_fanout = summary

    # -- the work (runs in a worker thread) ---------------------------------

    def _rules(self) -> Rules:
        return Rules(
            persistence_obs=self.config.push.persistence_obs,
            rearm_after_min=self.config.push.rearm_after_min,
            # One detection threshold for the whole pipeline: what counts
            # as rain falling at the point here is what counts as rain
            # everywhere else (Home Assistant's ``raining_now``, the
            # ensemble exceedance, the motion support mask).
            raining_now_mm_h=self.config.forecast.rain_threshold_mm_h,
        )

    def _evaluate_and_send(
        self,
        products: Any,
        geo: Any,
        radar_ts: datetime,
        now_utc: datetime,
        observed_mm_h: Any = None,
    ) -> dict:
        """Decide for every subscription, persist, then fan out. Blocking."""
        subs = self.store.list()
        rules = self._rules()
        pending: list[tuple[Subscription, dict]] = []
        errors = 0
        actions: dict[str, int] = {}

        for sub in subs:
            sample = sample_point(
                products, geo, sub.lat, sub.lon, observed_mm_h=observed_mm_h,
            )
            obs = Observation(
                radar_ts_utc=radar_ts,
                p_rain=sample.p_rain.get(sub.lead_min) if sample else None,
                eta_min=sample.eta_min if sample else None,
                intensity_mm_h=sample.intensity_mm_h if sample else None,
                observed_mm_h=sample.observed_mm_h if sample else None,
            )
            state = SubState(
                armed=sub.armed,
                streak=sub.streak,
                below_since_utc=sub.below_since_utc,
                last_eval_radar_ts=sub.last_eval_radar_ts,
            )
            quiet = (
                QuietHours(start=sub.quiet_start, end=sub.quiet_end)
                if sub.quiet_enabled
                else None
            )
            try:
                decision = decision_engine.evaluate(
                    state,
                    obs,
                    threshold_pct=sub.threshold_pct,
                    quiet=quiet,
                    tz=sub.tz,
                    now_utc=now_utc,
                    rules=rules,
                )
            except Exception as exc:  # noqa: BLE001 - one bad row must not
                # stop the others; the state is simply left as it was.
                errors += 1
                _log.warning(
                    "push_eval_error", sub=sub_id(sub.endpoint), error=str(exc),
                )
                continue

            actions[decision.action] = actions.get(decision.action, 0) + 1
            notify = decision.action == "notify" and obs.p_rain is not None
            new_state = decision.state
            # Persist BEFORE sending: a crash may cost a notification, it
            # must never cause a duplicate one.
            self.store.update_state(
                sub.endpoint,
                armed=new_state.armed,
                streak=new_state.streak,
                below_since_utc=new_state.below_since_utc,
                last_eval_radar_ts=new_state.last_eval_radar_ts,
                **({"last_notified_utc": now_utc} if notify else {}),
            )
            if notify:
                pending.append((
                    sub,
                    rain_incoming_payload(
                        lang=sub.lang,
                        lat=sub.lat,
                        lon=sub.lon,
                        eta_min=obs.eta_min,
                        p_rain=float(obs.p_rain),  # type: ignore[arg-type]
                        lead_min=sub.lead_min,
                        intensity_mm_h=obs.intensity_mm_h,
                        sent_utc=now_utc,
                    ),
                ))

        counts = self._fanout(pending)
        summary = {
            "radar_ts": radar_ts.isoformat(),
            "subscriptions": len(subs),
            "notified": len(pending),
            "eval_errors": errors,
            "actions": actions,
            **counts,
        }
        _log.info("push_fanout", **summary)
        return summary

    def _fanout(self, pending: list[tuple[Subscription, dict]]) -> dict:
        """Send each queued payload within the wall-clock budget."""
        sent = failed = removed = skipped = 0
        deadline = time.monotonic() + self.config.push.fanout_budget_s
        for index, (sub, payload) in enumerate(pending):
            if time.monotonic() >= deadline:
                skipped = len(pending) - index
                _log.warning("push_fanout_budget_exhausted", skipped=skipped)
                break
            ok, gone = self._send_one(sub, payload)
            if ok:
                sent += 1
            else:
                failed += 1
            if gone:
                self.store.delete(sub.endpoint)
                removed += 1
        return {
            "sent": sent, "failed": failed,
            "removed": removed, "skipped": skipped,
        }

    def _send_one(self, sub: Subscription, payload: dict) -> tuple[bool, bool]:
        """One delivery → ``(ok, gone)``. Any exception counts as a failure."""
        try:
            result = fanout.send(
                endpoint=sub.endpoint,
                p256dh=sub.p256dh,
                auth=sub.auth,
                payload=payload,
                vapid_private_pem=self.vapid_private_pem,
                vapid_subject=str(self.config.push.vapid_subject),
                ttl_s=self.config.push.ttl_s,
            )
        except Exception as exc:  # noqa: BLE001 - one dead push service must
            # not cost every other subscriber their notification.
            _log.warning(
                "push_send_error", sub=sub_id(sub.endpoint), error=str(exc),
            )
            return False, False
        if not result.ok:
            _log.info(
                "push_send_failed",
                sub=sub_id(sub.endpoint),
                status=result.status,
                gone=result.gone,
                error=result.error,
            )
        return bool(result.ok), bool(result.gone)

    # -- the test route -----------------------------------------------------

    async def send_test(self, endpoint: str | None = None) -> dict:
        """Send the canned test payload to one subscription, or to all."""
        return await asyncio.to_thread(self._send_test_sync, endpoint)

    def _send_test_sync(self, endpoint: str | None) -> dict:
        if endpoint is None:
            targets = self.store.list()
        else:
            one = self.store.get(endpoint)
            targets = [one] if one is not None else []
        now_utc = datetime.now(timezone.utc)
        sent = failed = removed = 0
        for sub in targets:
            ok, gone = self._send_one(
                sub,
                test_payload(
                    lang=sub.lang, lat=sub.lat, lon=sub.lon, sent_utc=now_utc,
                ),
            )
            if ok:
                sent += 1
            else:
                failed += 1
            if gone:
                self.store.delete(sub.endpoint)
                removed += 1
        _log.info(
            "push_test_sent",
            targets=len(targets),
            sent=sent,
            failed=failed,
            removed=removed,
        )
        return {"sent": sent, "failed": failed, "removed": removed}


__all__ = ["PushService"]
