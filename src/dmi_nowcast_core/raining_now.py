"""raining_now state machine with hysteresis (plan §6.4, §14).

Pure-Python; no Home Assistant dependencies. The HA binary sensor in Phase 5
wraps this. Defaults follow plan §14:

    detection_threshold_mm_h = 0.1
    hysteresis_offset_mm_h   = 0.05

The off-state turns on at ``max >= threshold``; the on-state turns off at
``max < (threshold - hysteresis)``. NaN inputs preserve the previous state
(stale frame, don't flap).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RainingNowConfig:
    detection_threshold_mm_h: float = 0.1
    hysteresis_offset_mm_h: float = 0.05

    def __post_init__(self) -> None:
        if self.hysteresis_offset_mm_h < 0:
            raise ValueError("hysteresis_offset_mm_h must be non-negative")
        if self.hysteresis_offset_mm_h >= self.detection_threshold_mm_h:
            raise ValueError(
                "hysteresis_offset must be smaller than detection_threshold; "
                "otherwise the off-threshold goes ≤ 0 and the sensor latches on"
            )

    @property
    def off_threshold_mm_h(self) -> float:
        return self.detection_threshold_mm_h - self.hysteresis_offset_mm_h


@dataclass(frozen=True)
class RainingNowResult:
    state: bool
    changed: bool


class RainingNow:
    """Tracks whether it is currently raining at the home location."""

    def __init__(
        self,
        config: RainingNowConfig | None = None,
        *,
        initial_state: bool = False,
    ) -> None:
        self.config = config or RainingNowConfig()
        self._state = initial_state

    @property
    def state(self) -> bool:
        return self._state

    def update(self, max_mm_h: float) -> RainingNowResult:
        """Apply one observation; return the new state and whether it changed."""
        if math.isnan(max_mm_h):
            return RainingNowResult(state=self._state, changed=False)
        previous = self._state
        if not self._state and max_mm_h >= self.config.detection_threshold_mm_h:
            self._state = True
        elif self._state and max_mm_h < self.config.off_threshold_mm_h:
            self._state = False
        return RainingNowResult(state=self._state, changed=self._state != previous)
