"""
severity_engine/smoother.py
============================
Temporal smoothing for severity to prevent unrealistic jumps.

Strategy: Exponential Moving Average (EMA) + Hysteresis
─────────────────────────────────────────────────────────
1.  Raw severity is encoded as a numeric value:
        P4=1  P3=2  P2=3  P1=4

2.  An EMA is maintained per episode:
        smoothed_t = alpha * raw_t + (1 - alpha) * smoothed_{t-1}
    Default alpha = 0.30 (configurable in thresholds.yaml).

3.  Hysteresis prevents rapid de-escalation:
    - Severity CAN escalate immediately (worst-case wins).
    - Severity can only DE-ESCALATE after `hysteresis_steps`
      consecutive readings that all support the lower level.

4.  The final integer is rounded and clamped to [1, 4], then
    converted back to a label.

Result guarantee
─────────────────
    P4 → P4 → P3 → P3 → P2 → P1   ✅
    P4 → P1 → P4 → P2              ❌  (never happens)
"""
from __future__ import annotations

from collections import defaultdict, deque

from .utils import SEVERITY_ORDER, SEVERITY_LABELS, severity_to_int, int_to_severity


class SeverityState:
    """
    Mutable state carried between rows for a single episode.
    One instance per episode_id — created by EpisodeSmoother.
    """
    __slots__ = ("ema", "hysteresis_buffer", "last_severity")

    def __init__(self, alpha: float, hysteresis_steps: int) -> None:
        self.ema: float = 1.0                        # starts at P4 (no incident)
        self.hysteresis_buffer: deque[int] = deque(maxlen=hysteresis_steps)
        self.last_severity: int = 1                  # numeric P4


class EpisodeSmoother:
    """
    Manages per-episode SeverityState objects and applies smoothing.

    Usage:
        smoother = EpisodeSmoother(alpha=0.30, hysteresis_steps=3)
        smoothed = smoother.smooth(episode_id="ep_001", raw_severity="P3")
    """

    def __init__(self, alpha: float = 0.30, hysteresis_steps: int = 3) -> None:
        assert 0 < alpha <= 1.0, "alpha must be in (0, 1]"
        assert hysteresis_steps >= 1
        self.alpha = alpha
        self.hysteresis_steps = hysteresis_steps
        self._states: dict[str, SeverityState] = {}

    def smooth(self, episode_id: str, raw_severity: str) -> str:
        """
        Apply EMA + hysteresis smoothing and return the smoothed severity label.

        Args:
            episode_id:    Unique episode identifier (state is maintained per ID).
            raw_severity:  Raw P1/P2/P3/P4 label from the rule engine.

        Returns:
            Smoothed P1/P2/P3/P4 label.
        """
        state = self._get_or_create(episode_id)
        raw_num = severity_to_int(raw_severity)

        # 1. Update EMA
        state.ema = self.alpha * raw_num + (1 - self.alpha) * state.ema

        # 2. Round EMA to nearest integer severity level
        ema_rounded = max(1, min(4, round(state.ema)))

        # 3. Hysteresis: escalation is immediate, de-escalation needs buffer
        if ema_rounded > state.last_severity:
            # Escalation — apply immediately
            state.last_severity = ema_rounded
            state.hysteresis_buffer.clear()
        elif ema_rounded < state.last_severity:
            # De-escalation — add to buffer; only change after N consecutive lows
            state.hysteresis_buffer.append(ema_rounded)
            if len(state.hysteresis_buffer) == self.hysteresis_steps and \
               all(v <= ema_rounded for v in state.hysteresis_buffer):
                state.last_severity = ema_rounded
                state.hysteresis_buffer.clear()
            # else: stay at current severity
        else:
            # Same level — clear any pending de-escalation buffer
            state.hysteresis_buffer.clear()

        return int_to_severity(state.last_severity)

    def reset(self, episode_id: str) -> None:
        """Reset state for a specific episode (e.g. after incident resolution)."""
        if episode_id in self._states:
            del self._states[episode_id]

    def reset_all(self) -> None:
        """Clear all episode states."""
        self._states.clear()

    def _get_or_create(self, episode_id: str) -> SeverityState:
        if episode_id not in self._states:
            self._states[episode_id] = SeverityState(self.alpha, self.hysteresis_steps)
        return self._states[episode_id]
