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

3. **One rule per horizon, re-read between cycles.** The threshold a
   subscription is judged at comes from the fitted table
   (``push.thresholds``) unless the row carries an explicit override. The
   table is re-read at the start of the fan-out and never during one, so
   every ``push_eval`` line of a cycle was decided under the same rule and
   says which rule that was.

3a. **And one probability, chosen once per cycle.** Under
   ``push.probability_source: postprocess`` (the default since Phase H)
   the rule reads the gauge-trained post-processed probability the cycle
   computed for this subscription's point — ΔBSS +0.14…+0.19 against the
   curve at the gauges, ΔF1 +0.03…+0.06 on this very rule. The fallback is
   per observation, not per cycle: a point the model could not score is
   judged on the curve and the log line says so, because one point off the
   model's coverage must not silence it, and one model outage must not
   silence everyone. ``push.probability_source: curve`` restores the
   pre-Phase-H behaviour exactly, and is the rollback.

4. **Persist first, send second.** The new state is written before any
   network call, so a crash mid-fan-out can only cost a notification, not
   cause a repeat. The failure the user forgives is a missed alert; the
   one they uninstall over is the same alert five times.

4a. **The all-clear retracts only what arrived.** An ``"all_clear"``
   decision (two below-threshold observations after a push, before the
   re-arm — ``push.engine``) queues one silent replacement through the
   same fan-out as a warning. It is queued only when the store says the
   warning it retracts was DELIVERED (``Subscription.
   last_warning_delivered``: the fan-out records the warning's
   ``last_notified_utc`` stamp in ``last_delivered_utc`` after a
   successful send). A warning whose send failed, was skipped by the
   budget, or never happened because the process died mid-fan-out has
   nothing on the device to retract; the all-clear is then recorded as
   spent and not sent, and ``push_all_clear_suppressed`` says why.

5. **Fan out concurrently inside a wall-clock budget.** The push services
   are the slow part and the cycle must not be held hostage to them:
   ``push.fanout_workers`` threads share one keep-alive session per push
   host, and whatever has not started when the budget expires is not sent
   — its subscription's state is written back to what it was before the
   observation, so the next observation retries it rather than the row
   reading "notified" for a warning nobody attempted. ``404``/``410`` from
   a push service means the browser is gone and the row is deleted; rows
   that keep failing are collected too (``PushStore.record_send``).

Logging is aggregate: one ``push_fanout`` event with counts. Endpoints are
capabilities to notify a browser and never appear in a log line; the short
``sub_id`` hash is used where an individual row must be identified.
"""
from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import requests
import structlog
from urllib3.util import parse_url

from ..compute import CycleEngine, CycleResult
from ..config import Config
from ..national_sample import sample_point
from . import engine as decision_engine
from . import fanout
from .endpoint_policy import validate_endpoint
from .engine import Observation, QuietHours, Rules, SubState
from .messages import all_clear_payload, rain_incoming_payload, test_payload
from .paths import resolved_thresholds_path
from .store import PushStore, Subscription, sub_id
from .thresholds import ThresholdTable

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
        thresholds: ThresholdTable | None = None,
    ) -> None:
        self.config = config
        self.engine = engine
        self.store = store
        self.vapid_private_pem = vapid_private_pem
        self.public_key = public_key
        #: The fitted horizon → threshold table (Phase G). Shared with the
        #: routes so the subscribe response and the fan-out can never
        #: disagree about which rule a subscription is on; built here when
        #: nobody handed one over, so a service constructed in a test or a
        #: script still reads the same file the deployment does.
        self.thresholds = thresholds if thresholds is not None else ThresholdTable(
            resolved_thresholds_path(config),
        )
        self._last_evaluated_radar_ts: datetime | None = None
        self._last_fanout: dict | None = None
        #: One keep-alive, no-redirect session per push host (``fanout``).
        self._sessions: dict[str, requests.Session] = {}
        self._sessions_lock = threading.Lock()

    # -- introspection (served by /api/push/stats) --------------------------

    @property
    def last_evaluated_radar_ts(self) -> datetime | None:
        return self._last_evaluated_radar_ts

    @property
    def last_fanout(self) -> dict | None:
        return self._last_fanout

    # -- what the cycle needs from us ---------------------------------------

    def decision_points(self) -> list[tuple[float, float]]:
        """Every subscription's point, for the cycle's feature table (H-P).

        Registered with the engine by ``app.create_app`` and called once
        per full cycle, inside the cycle worker. Coordinates only: the
        cycle scores *places*, and the endpoint — a bearer capability —
        never leaves this store.
        """
        return [(sub.lat, sub.lon) for sub in self.store.list()]

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
            # Deterministic forecast series — carried for the evaluation
            # log only; it changes no decision.
            getattr(latest, "forecast_mm_h", None),
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
            allclear_enabled=self.config.push.allclear_enabled,
            allclear_readings=self.config.push.allclear_readings,
        )

    def _postprocess_for(self, radar_ts: datetime) -> Any:
        """This cycle's post-processing answer, or None to use the curve.

        Three ways to get None, all of them meaning "decide on the served
        probability": the operator asked for ``curve``; the cycle produced
        nothing (no model, a feature failure, a points source that
        raised); or the object belongs to a different frame — the same
        trap the products check guards, and for the same reason.
        """
        if self.config.push.probability_source != "postprocess":
            return None
        latest = getattr(self.engine, "postprocess_latest", None)
        if latest is None or not getattr(latest, "active", False):
            return None
        stamp = getattr(latest, "radar_ts_utc", None)
        if stamp is not None and stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        if stamp != radar_ts:
            _log.info(
                "push_postprocess_stale",
                postprocess_ts=stamp.isoformat() if stamp else None,
                radar_ts=radar_ts.isoformat(),
            )
            return None
        return latest

    def _evaluate_and_send(
        self,
        products: Any,
        geo: Any,
        radar_ts: datetime,
        now_utc: datetime,
        observed_mm_h: Any = None,
        forecast_mm_h: Any = None,
    ) -> dict:
        """Decide for every subscription, persist, then fan out. Blocking.

        Each subscription is decided inside its own ``try``: one row that
        cannot be sampled, evaluated or written costs that row this
        observation and nothing else. Whatever was queued is sent in a
        ``finally``, so no failure after the first persisted decision can
        leave a warning persisted-as-sent but never attempted.
        """
        pending: list[_Pending] = []
        subs: list[Subscription] = []
        post: Any = None
        source: str = self.config.push.probability_source
        notified_count = 0
        all_clear_count = 0
        all_clear_suppressed = 0
        errors = 0
        stale_writes = 0
        curve_fallbacks = 0
        #: Observations on a lead with the onset AND rule (S11) that had no
        #: ``p_onset`` and were judged on the single threshold instead.
        onset_fallbacks = 0
        actions: dict[str, int] = {}
        counts: dict[str, int] = {}

        try:
            # One stat per cycle, a JSON parse only when the fitted table
            # moved. The start of a fan-out is the safe moment to swap it:
            # the whole cycle then evaluates every subscription against one
            # version of the rule, and the ``push_eval`` lines of a cycle
            # are comparable with each other.
            self.thresholds.maybe_reload()
            # Resolved once, for the same reason: every subscription of one
            # cycle is judged on one probability source.
            post = self._postprocess_for(radar_ts)
            subs = self.store.list()
            rules = self._rules()

            for sub in subs:
                try:
                    sample = sample_point(
                        products, geo, sub.lat, sub.lon,
                        observed_mm_h=observed_mm_h,
                        forecast_mm_h=forecast_mm_h,
                    )
                    series = sample.forecast_mm_h if sample else None
                    # S11: the onset AND rule, for a lead whose fitted row
                    # carries it. A row-level override is a single-threshold
                    # rule, as it always was. ``p_onset`` is read only for
                    # such a lead, and only off a frame-matched cycle object
                    # (``post``), so the curve rollback also rolls this back.
                    onset = (
                        None if sub.threshold_pct is not None
                        else self.thresholds.onset_rule(sub.lead_min)
                    )
                    p_onset = (
                        None if onset is None or post is None
                        else post.onset_probability(sub.lat, sub.lon, sub.lead_min)
                    )
                    if onset is not None and p_onset is None:
                        onset_fallbacks += 1
                    obs = Observation(
                        radar_ts_utc=radar_ts,
                        p_rain=sample.p_rain.get(sub.lead_min) if sample else None,
                        eta_min=sample.eta_min if sample else None,
                        intensity_mm_h=sample.intensity_mm_h if sample else None,
                        observed_mm_h=sample.observed_mm_h if sample else None,
                        forecast_now_mm_h=series.get(0) if series else None,
                        p_post=(
                            None if post is None
                            else post.probability(sub.lat, sub.lon, sub.lead_min)
                        ),
                        p_source="postprocess" if post is not None else "curve",
                        p_onset=p_onset,
                    )
                    if post is not None and obs.p_post is None:
                        curve_fallbacks += 1
                    state = SubState(
                        armed=sub.armed,
                        streak=sub.streak,
                        below_since_utc=sub.below_since_utc,
                        last_eval_radar_ts=sub.last_eval_radar_ts,
                        notified=sub.notified,
                        below_streak=sub.below_streak,
                        all_clear_sent=sub.all_clear_sent,
                    )
                    quiet = (
                        QuietHours(start=sub.quiet_start, end=sub.quiet_end)
                        if sub.quiet_enabled
                        else None
                    )
                    # The one knob: the subscription chose a horizon, the
                    # fitted table chose the percent that horizon warns at.
                    # A non-null ``threshold_pct`` on the row is a deliberate
                    # override and wins over both. ``evaluate`` stays pure —
                    # it is handed a number, and never learns where it came
                    # from.
                    if sub.threshold_pct is not None:
                        threshold_pct, threshold_source = (
                            int(sub.threshold_pct), "override",
                        )
                    else:
                        threshold_pct, threshold_source = self.thresholds.effective(
                            sub.lead_min,
                        )
                    decision = decision_engine.evaluate(
                        state,
                        obs,
                        threshold_pct=threshold_pct,
                        quiet=quiet,
                        tz=sub.tz,
                        now_utc=now_utc,
                        rules=rules,
                        onset_threshold_pct=None if onset is None else onset[0],
                        single_threshold_pct=None if onset is None else onset[1],
                    )

                    actions[decision.action] = actions.get(decision.action, 0) + 1
                    # One line per subscription per cycle: everything needed
                    # to replay why this row did or did not get a push.
                    # Identified by the row's hashed handle only — never the
                    # endpoint (a bearer capability) and never the
                    # coordinates.
                    _log.info(
                        "push_eval",
                        sub=sub_id(sub.endpoint),
                        radar_ts=radar_ts.isoformat(),
                        action=decision.action,
                        lead_min=sub.lead_min,
                        threshold_pct=threshold_pct,
                        threshold_source=threshold_source,
                        # Both probabilities, and which one the decision
                        # used: every ``push_eval`` line has to be replayable
                        # on its own, and "would the curve have fired here?"
                        # is the first question anyone asks of a
                        # post-processed warning.
                        p_source=obs.p_decision_source,
                        p_rain=obs.p_rain,
                        p_post=obs.p_post,
                        p_onset=obs.p_onset,
                        onset_threshold_pct=None if onset is None else onset[0],
                        eta_min=obs.eta_min,
                        intensity_mm_h=obs.intensity_mm_h,
                        observed_mm_h=obs.observed_mm_h,
                        forecast_now_mm_h=obs.forecast_now_mm_h,
                    )
                    notify = (
                        decision.action == "notify" and obs.p_decision is not None
                    )
                    all_clear = decision.action == "all_clear"
                    new_state = decision.state
                    # Persist BEFORE sending: a crash may cost a
                    # notification, it must never cause a duplicate one. An
                    # all-clear is marked spent here too. The write is
                    # conditional on the row's ``version``: a subscribe that
                    # landed while this cycle was deciding wins, and a
                    # decision that could not be persisted is not acted on.
                    written = self.store.update_state(
                        sub.endpoint,
                        armed=new_state.armed,
                        streak=new_state.streak,
                        below_since_utc=new_state.below_since_utc,
                        last_eval_radar_ts=new_state.last_eval_radar_ts,
                        notified=new_state.notified,
                        below_streak=new_state.below_streak,
                        all_clear_sent=new_state.all_clear_sent,
                        expected_version=sub.version,
                        **({"last_notified_utc": now_utc} if notify else {}),
                    )
                    if not written:
                        stale_writes += 1
                        _log.info(
                            "push_state_write_skipped",
                            sub=sub_id(sub.endpoint),
                            reason="row_changed_mid_cycle",
                            action=decision.action,
                        )
                        continue
                    if all_clear:
                        minutes_since_push = (
                            None if sub.last_notified_utc is None
                            else round(
                                (now_utc - sub.last_notified_utc).total_seconds()
                                / 60.0,
                                1,
                            )
                        )
                        if not sub.last_warning_delivered:
                            # Nothing on the device to retract: the
                            # warning's send failed, was skipped, or was
                            # never attempted.
                            all_clear_suppressed += 1
                            _log.info(
                                "push_all_clear_suppressed",
                                sub=sub_id(sub.endpoint),
                                reason="warning_not_delivered",
                                minutes_since_push=minutes_since_push,
                            )
                            continue
                        all_clear_count += 1
                        _log.info(
                            "push_all_clear",
                            sub=sub_id(sub.endpoint),
                            radar_ts=radar_ts.isoformat(),
                            lead_min=sub.lead_min,
                            threshold_pct=threshold_pct,
                            p_decision=obs.p_decision,
                            minutes_since_push=minutes_since_push,
                        )
                        pending.append(_Pending(
                            sub=sub,
                            payload=all_clear_payload(
                                lang=sub.lang,
                                lat=sub.lat,
                                lon=sub.lon,
                                lead_min=sub.lead_min,
                                sent_utc=now_utc,
                                tz=sub.tz,
                            ),
                            notified_utc=None,
                            previous=state,
                        ))
                    if notify:
                        notified_count += 1
                        pending.append(_Pending(
                            sub=sub,
                            payload=rain_incoming_payload(
                                lang=sub.lang,
                                lat=sub.lat,
                                lon=sub.lon,
                                eta_min=obs.eta_min,
                                # The number the decision was taken on, so
                                # the text a subscriber reads and the rule
                                # that woke them are the same probability.
                                p_rain=float(obs.p_decision),  # type: ignore[arg-type]
                                lead_min=sub.lead_min,
                                intensity_mm_h=obs.intensity_mm_h,
                                sent_utc=now_utc,
                            ),
                            notified_utc=now_utc,
                            previous=state,
                        ))
                except Exception as exc:  # noqa: BLE001 - one bad row must
                    # not stop the others; its state is left as it was
                    # unless the write above already landed.
                    errors += 1
                    _log.warning(
                        "push_eval_error",
                        sub=sub_id(sub.endpoint),
                        error=type(exc).__name__,
                    )
        finally:
            # Whatever was decided and persisted is attempted, whatever
            # happened after it.
            counts = self._fanout(pending, now_utc)

        summary = {
            "radar_ts": radar_ts.isoformat(),
            "thresholds_fitted_at": self.thresholds.fitted_at_utc,
            # Additive (Phase H): which probability this cycle decided on,
            # the model behind it, and how many observations fell back to
            # the curve because the model could not speak for them.
            "probability_source": source,
            "postprocess_active": post is not None,
            "postprocess_fitted_at": (
                None if post is None else post.fitted_at_utc
            ),
            "postprocess_curve_fallbacks": curve_fallbacks,
            # S11: onset-rule observations judged on the single threshold
            # because the onset model had nothing to say for them.
            "onset_rule_fallbacks": onset_fallbacks,
            "subscriptions": len(subs),
            "notified": notified_count,
            # Silent retractions queued this cycle, beside the warnings;
            # and the ones not sent because their warning never arrived.
            "all_clear": all_clear_count,
            "all_clear_suppressed": all_clear_suppressed,
            "eval_errors": errors,
            # Decisions dropped because the row was edited mid-cycle.
            "stale_writes": stale_writes,
            "actions": actions,
            **counts,
        }
        _log.info("push_fanout", **summary)
        return summary

    # -- delivery -------------------------------------------------------------

    def _session_for(self, endpoint: str) -> requests.Session:
        """The shared keep-alive session for this endpoint's push host."""
        try:
            host = (parse_url(endpoint).host or "").lower()
        except Exception:  # noqa: BLE001 - the policy check refuses it anyway
            host = ""
        with self._sessions_lock:
            session = self._sessions.get(host)
            if session is None:
                session = fanout.new_session(self.config.push.fanout_workers)
                self._sessions[host] = session
            return session

    def close(self) -> None:
        """Close the per-host sessions (app shutdown)."""
        with self._sessions_lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            try:
                session.close()
            except Exception:  # noqa: BLE001
                pass

    def _fanout(self, pending: list["_Pending"], now_utc: datetime) -> dict:
        """Send the queued payloads concurrently, inside the wall-clock budget.

        ``push.fanout_workers`` threads share one keep-alive session per
        push host. A payload whose turn comes after the budget ran out is
        NOT sent, and its subscription's state is written back to what it
        was before this observation (``_restore``): the decision was
        persisted first, so without the write-back the row would read
        "notified" for a warning nobody attempted — and the next
        observation, which is what retries it, would not. A warning that
        WAS sent keeps its persisted state, success or not.

        All store bookkeeping happens here, on the calling thread, after
        the sends: ``mark_delivered`` for a delivered warning, deletion on
        404/410, :meth:`PushStore.record_send` (garbage collection) for
        everything else.
        """
        sent = failed = removed = skipped = collected = 0
        if not pending:
            return {
                "sent": 0, "failed": 0, "removed": 0, "skipped": 0,
                "collected": 0,
            }
        deadline = time.monotonic() + self.config.push.fanout_budget_s

        def attempt(item: _Pending) -> tuple[bool, bool, bool] | None:
            if time.monotonic() >= deadline:
                return None
            return self._send_one(item.sub, item.payload)

        workers = max(1, min(int(self.config.push.fanout_workers), len(pending)))
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="push-fanout",
        ) as pool:
            futures = [pool.submit(attempt, item) for item in pending]
        results = []
        for future in futures:
            try:
                results.append(future.result())
            except Exception:  # noqa: BLE001 - _send_one never raises
                results.append((False, False, True))

        pc = self.config.push
        for item, result in zip(pending, results):
            endpoint = item.sub.endpoint
            if result is None:
                skipped += 1
                self._restore(item)
                continue
            ok, gone, transient = result
            if ok:
                sent += 1
                if item.notified_utc is not None:
                    try:
                        self.store.mark_delivered(endpoint, item.notified_utc)
                    except Exception as exc:  # noqa: BLE001 - bookkeeping
                        # only: the cost is one all-clear not sent.
                        _log.warning(
                            "push_mark_delivered_failed",
                            sub=sub_id(endpoint), error=type(exc).__name__,
                        )
            else:
                failed += 1
            try:
                if gone:
                    self.store.delete(endpoint)
                    removed += 1
                elif self.store.record_send(
                    endpoint,
                    ok=ok,
                    transient=transient,
                    now_utc=now_utc,
                    max_failures=pc.gc_max_failures,
                    stale_days=pc.gc_stale_days,
                ):
                    collected += 1
                    _log.info("push_subscription_collected", sub=sub_id(endpoint))
            except Exception as exc:  # noqa: BLE001 - bookkeeping only
                _log.warning(
                    "push_send_bookkeeping_failed",
                    sub=sub_id(endpoint), error=type(exc).__name__,
                )
        if skipped:
            _log.warning("push_fanout_budget_exhausted", skipped=skipped)
        return {
            "sent": sent, "failed": failed,
            "removed": removed, "skipped": skipped,
            "collected": collected,
        }

    def _restore(self, item: "_Pending") -> None:
        """Write back the state a skipped send's decision replaced."""
        prev = item.previous
        try:
            self.store.update_state(
                item.sub.endpoint,
                armed=prev.armed,
                streak=prev.streak,
                below_since_utc=prev.below_since_utc,
                last_eval_radar_ts=prev.last_eval_radar_ts,
                notified=prev.notified,
                below_streak=prev.below_streak,
                all_clear_sent=prev.all_clear_sent,
                expected_version=item.sub.version,
                **(
                    {"last_notified_utc": item.sub.last_notified_utc}
                    if item.notified_utc is not None else {}
                ),
            )
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "push_state_restore_failed",
                sub=sub_id(item.sub.endpoint), error=type(exc).__name__,
            )

    def _send_one(
        self, sub: Subscription, payload: dict,
    ) -> tuple[bool, bool, bool]:
        """One delivery → ``(ok, gone, transient)``. Never raises.

        The endpoint policy is re-checked at send time, so a row stored
        before a policy tightened is never POSTed to; such a row counts as
        a non-transient failure and is collected like any other dead row.
        """
        reason = validate_endpoint(
            sub.endpoint, self.config.push.allowed_endpoint_host_suffixes,
        )
        if reason is not None:
            _log.warning("push_send_refused", sub=sub_id(sub.endpoint))
            return False, False, False
        try:
            result = fanout.send(
                endpoint=sub.endpoint,
                p256dh=sub.p256dh,
                auth=sub.auth,
                payload=payload,
                vapid_private_pem=self.vapid_private_pem,
                vapid_subject=str(self.config.push.vapid_subject),
                ttl_s=self.config.push.ttl_s,
                session=self._session_for(sub.endpoint),
            )
        except Exception as exc:  # noqa: BLE001 - one dead push service must
            # not cost every other subscriber their notification.
            _log.warning(
                "push_send_error", sub=sub_id(sub.endpoint),
                error=type(exc).__name__,
            )
            return False, False, True
        if not result.ok:
            _log.info(
                "push_send_failed",
                sub=sub_id(sub.endpoint),
                status=result.status,
                gone=result.gone,
            )
        return (
            bool(result.ok),
            bool(result.gone),
            fanout.is_transient(result.status),
        )

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
        pc = self.config.push
        sent = failed = removed = 0
        for sub in targets:
            ok, gone, transient = self._send_one(
                sub,
                test_payload(
                    lang=sub.lang, lat=sub.lat, lon=sub.lon, sent_utc=now_utc,
                ),
            )
            if ok:
                sent += 1
            else:
                failed += 1
            try:
                if gone:
                    self.store.delete(sub.endpoint)
                    removed += 1
                elif self.store.record_send(
                    sub.endpoint, ok=ok, transient=transient, now_utc=now_utc,
                    max_failures=pc.gc_max_failures,
                    stale_days=pc.gc_stale_days,
                ):
                    removed += 1
            except Exception as exc:  # noqa: BLE001 - bookkeeping only
                _log.warning(
                    "push_send_bookkeeping_failed",
                    sub=sub_id(sub.endpoint), error=type(exc).__name__,
                )
        _log.info(
            "push_test_sent",
            targets=len(targets),
            sent=sent,
            failed=failed,
            removed=removed,
        )
        return {"sent": sent, "failed": failed, "removed": removed}


@dataclass(frozen=True)
class _Pending:
    """One queued delivery and what to write back if it is never attempted."""

    sub: Subscription
    payload: dict
    #: The warning's ``last_notified_utc`` stamp; None for an all-clear.
    notified_utc: datetime | None
    #: The state this observation's decision replaced.
    previous: SubState


__all__ = ["PushService"]
