"""The push rule's two timing constants, with one home.

A notification fires when the calibrated probability of rain within the
subscriber's horizon has sat at or above that horizon's threshold for
:data:`DEFAULT_PERSISTENCE_OBS` consecutive radar observations, and the
subscription then stays disarmed until :data:`DEFAULT_REARM_AFTER_MIN`
minutes of continuous below-threshold radar time have passed. The state
machine that does it is ``push.engine.evaluate`` in the sidecar; this
module is only the two numbers, and it exists because they were spelled
out in nine places and two of them disagreed.

**Why this module exists (decision DECIDE-14, 2026-09-13).** The push
engine for real subscribers required TWO observations
(``PushConfig.persistence_obs``), and so did the nightly threshold fit
that chose the percentages it warns at. Everything that *measured* the
rule required ONE: the station scoreboard job, the quality page's
served-rule hook, the manual fit script, the historical replay and the
benchmark. So the table in service was fitted for a rule the scoreboard
never scored, and a manual fit and a nightly fit produced different
tables from the same rows.

**Why one observation won.** Fitted on the same rows, the same model and
the same gauge onsets, F1 at the fitted threshold came out:

======  ==================  ===================
lead    one observation     two observations
======  ==================  ===================
20 min  0.244               0.182
30 min  0.326               0.263
45 min  0.397               0.362
60 min  0.343               0.379
======  ==================  ===================

One observation wins at three of the four horizons and is close at the
fourth. The mechanism is lead time, not skill: a second observation costs
another ~10 minutes of radar cadence on top of the 13–18 minutes the
composite is already old by the time the cycle computes, which at 20 and
30 minutes is most of the warning. The 16-member ensemble has already
done the averaging a persistence streak was there to do — that is the
same argument the public config recorded for the scoreboard's rule on
2026-09-02, now applied to the subscribers as well.

**How to change them.** Change the number here and every default follows:
``PushConfig`` (the only place an operator may override it, via
``push.persistence_obs`` / ``push.rearm_after_min`` in the sidecar's
config), ``push.engine.Rules``, ``threshold_sweep.SweepOptions``,
``served_rule.ServedRuleOptions``, the station scoreboard, the historical
replay's ``DEFAULT_RULES`` and the benchmark's CLI. ``station_eval.rules``
deliberately has no say: it carries the virtual subscriber's threshold and
horizon, and takes the timing from ``push.*`` so the measurement and the
service cannot drift apart again.
"""
from __future__ import annotations

__all__ = ["DEFAULT_PERSISTENCE_OBS", "DEFAULT_REARM_AFTER_MIN"]

#: Consecutive over-threshold radar observations required to fire.
DEFAULT_PERSISTENCE_OBS = 1

#: Minutes of continuous below-threshold radar time before a notified
#: subscription re-arms. Measured on the radar clock, not on wall time and
#: not on the poll cadence. The gauge scoreboard's onset definition
#: (``warning_score.DEFAULT_DRY_MIN``) matches it deliberately: a rain
#: event the rule could never have warned about twice is not evidence
#: about the forecast.
DEFAULT_REARM_AFTER_MIN = 60
