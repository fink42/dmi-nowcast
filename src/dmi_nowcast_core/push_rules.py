"""The push rule's timing constants, and the all-clear's, with one home.

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

**The all-clear (2026-09-23).** After a warning has been pushed, the
subscription stays disarmed until the re-arm. If, inside that window, the
decision probability sits BELOW the subscriber's threshold on
:data:`DEFAULT_ALLCLEAR_READINGS` consecutive radar observations, the
service sends ONE silent replacement notification ("rain no longer
expected") under the same tag, so it replaces the warning on the device
instead of stacking. Nothing more is sent until the re-arm. It is never
sent after an "already raining" consumption (nothing was sent, so there
is nothing to retract), and quiet hours do not defer it — it is silent.

Why two readings and not one: a single frame's dip below the threshold
is exactly the one-frame noise the rest of the rule is built to ignore,
and a retraction that turns out wrong costs more trust than the false
alarm it retracts. With two consecutive readings the 272-day replay
(``scripts/eta_revision_study.py``) retracts 32–57 % of the false alarms
(by horizon) and is wrong — rain arrived after the all-clear — on only
0.4–3.4 % of the hits: about 30 right retractions for every wrong one.

Changed the same way as the two numbers above: here, overridden only by
``push.allclear_enabled`` / ``push.allclear_readings`` in the sidecar
config. The all-clear changes
no arming decision, so the threshold fit's objective is untouched; the
scorer only REPORTS it (``warning_score.score_warnings``: ``right`` for
a retracted false alarm, ``wrong`` for a hit whose onset followed).
"""
from __future__ import annotations

__all__ = [
    "DEFAULT_ALLCLEAR_ENABLED",
    "DEFAULT_ALLCLEAR_READINGS",
    "DEFAULT_PERSISTENCE_OBS",
    "DEFAULT_REARM_AFTER_MIN",
]

#: Consecutive over-threshold radar observations required to fire.
DEFAULT_PERSISTENCE_OBS = 1

#: Minutes of continuous below-threshold radar time before a notified
#: subscription re-arms. Measured on the radar clock, not on wall time and
#: not on the poll cadence. The gauge scoreboard's onset definition
#: (``warning_score.DEFAULT_DRY_MIN``) matches it deliberately: a rain
#: event the rule could never have warned about twice is not evidence
#: about the forecast.
DEFAULT_REARM_AFTER_MIN = 60

#: Whether a pushed warning may be retracted by a silent all-clear.
DEFAULT_ALLCLEAR_ENABLED = True

#: Consecutive below-threshold radar observations, after a push and
#: before the re-arm, that trigger the one all-clear. Observations with no
#: probability (nodata, off coverage) neither count nor break the run —
#: exactly as the replay study skipped them.
DEFAULT_ALLCLEAR_READINGS = 2
