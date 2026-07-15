"""tests/test_smoother.py — pytest tests for EMA + hysteresis temporal smoother."""
import pytest
from severity_engine.smoother import EpisodeSmoother


def make_smoother(alpha=0.3, hysteresis=3):
    return EpisodeSmoother(alpha=alpha, hysteresis_steps=hysteresis)


# ── Basic smoothing ───────────────────────────────────────────────────────────

def test_constant_p4_stays_p4():
    s = make_smoother()
    for _ in range(10):
        result = s.smooth("ep1", "P4")
    assert result == "P4"


def test_constant_p1_stays_p1():
    s = make_smoother()
    for _ in range(10):
        result = s.smooth("ep1", "P1")
    assert result == "P1"


# ── Escalation is immediate ───────────────────────────────────────────────────

def test_escalation_p4_to_p1_at_once():
    """With alpha=1.0, escalation should be instantaneous."""
    s = make_smoother(alpha=1.0, hysteresis=3)
    s.smooth("ep1", "P4")
    # Sudden P1 — with alpha=1 the EMA jumps immediately
    result = s.smooth("ep1", "P1")
    assert result == "P1"


def test_escalation_direction_is_immediate():
    """Severity should always be allowed to escalate immediately."""
    s = make_smoother(alpha=0.5, hysteresis=3)
    s.smooth("ep1", "P4")
    s.smooth("ep1", "P4")
    r = s.smooth("ep1", "P1")    # sharp jump
    # With alpha=0.5 the EMA may not reach 4 in one step — but
    # the hysteresis rule only blocks DE-escalation, not escalation.
    # With alpha=0.5: ema = 0.5*4 + 0.5*1 = 2.5 → round=3 → P2 at least
    assert r in ("P2", "P3", "P1")   # must be at least P2, no P4


# ── De-escalation requires hysteresis ────────────────────────────────────────

def test_de_escalation_requires_3_steps():
    """De-escalation from P1 to P4 must not happen in fewer than hysteresis_steps."""
    s = make_smoother(alpha=1.0, hysteresis=3)
    for _ in range(5):
        s.smooth("ep1", "P1")
    # Now feed P4 three times
    r1 = s.smooth("ep1", "P4")   # step 1: not yet de-escalated
    r2 = s.smooth("ep1", "P4")   # step 2
    r3 = s.smooth("ep1", "P4")   # step 3: should now de-escalate
    # After 3 consecutive lower readings it should de-escalate
    # r1 and r2 should still reflect a higher severity
    assert r1 in ("P1", "P2", "P3")
    assert r3 in ("P3", "P4")    # must have come down after 3 readings


def test_single_p4_does_not_deescalate_p1():
    """One low reading should not drop severity immediately."""
    s = make_smoother(alpha=1.0, hysteresis=3)
    for _ in range(5):
        s.smooth("ep1", "P1")
    r = s.smooth("ep1", "P4")   # single low reading
    # Should still be high — hysteresis prevents instant drop
    assert r in ("P1", "P2", "P3")


# ── No unrealistic jumps ──────────────────────────────────────────────────────

def test_no_p4_p1_p4_oscillation():
    """Severity should never oscillate P4→P1→P4 randomly."""
    s = make_smoother(alpha=0.3, hysteresis=3)
    sequence = ["P4", "P1", "P4", "P1", "P4"]
    results = [s.smooth("ep1", sev) for sev in sequence]
    # After seeing P1, severity must not instantly return to P4
    # Check that we never see P4 immediately after P1
    for i in range(1, len(results)):
        if results[i - 1] == "P1":
            assert results[i] != "P4", \
                f"Severity jumped from P1 to P4 at step {i}: {results}"


# ── Realistic progression ─────────────────────────────────────────────────────

def test_realistic_escalation_sequence():
    """P4→P3→P3→P2→P1 should be achievable."""
    s = make_smoother(alpha=1.0, hysteresis=1)  # instant response
    assert s.smooth("ep1", "P4") == "P4"
    assert s.smooth("ep1", "P3") == "P3"
    # P3 input again — same level
    assert s.smooth("ep1", "P3") == "P3"
    assert s.smooth("ep1", "P2") == "P2"
    assert s.smooth("ep1", "P1") == "P1"


# ── Per-episode isolation ─────────────────────────────────────────────────────

def test_episodes_are_independent():
    s = make_smoother()
    for _ in range(10):
        s.smooth("ep_a", "P1")
    s.smooth("ep_b", "P4")
    assert s.smooth("ep_b", "P4") == "P4"   # ep_b unaffected by ep_a


# ── Reset ─────────────────────────────────────────────────────────────────────

def test_reset_clears_state():
    s = make_smoother(alpha=1.0, hysteresis=3)
    for _ in range(5):
        s.smooth("ep1", "P1")
    s.reset("ep1")
    # After reset, a single P4 should start fresh
    result = s.smooth("ep1", "P4")
    assert result == "P4"
