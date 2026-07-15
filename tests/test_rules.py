"""tests/test_rules.py — pytest tests for the 6-rule severity decision engine."""
import pytest
from severity_engine.rules import classify_severity, classify_severity_with_details


def sev(cc=0, wc=0, bs=0, hrm=False, brg=False, mode="NONE"):
    return classify_severity(cc, wc, bs, hrm, brg, mode)


def sev_d(cc=0, wc=0, bs=0, hrm=False, brg=False, mode="NONE"):
    return classify_severity_with_details(cc, wc, bs, hrm, brg, mode)


# ── Rule 1a: critical_count >= 2 ──────────────────────────────────────────────

def test_rule1a_two_criticals():
    assert sev(cc=2) == "P1"

def test_rule1a_three_criticals():
    assert sev(cc=3) == "P1"

def test_rule1_detail():
    d = sev_d(cc=2)
    assert d["rule_matched"] == 1


# ── Rule 1b: critical_count >= 1 AND high_risk_mode ──────────────────────────

def test_rule1b_one_critical_high_risk():
    assert sev(cc=1, hrm=True) == "P1"

def test_rule1b_no_critical_high_risk_not_p1():
    # high_risk_mode alone without critical should NOT trigger rule 1
    assert sev(cc=0, hrm=True, wc=1, bs=1) != "P1"


# ── Rule 2: blast_radius_growing + critical_count >= 1 ───────────────────────

def test_rule2_cascade():
    assert sev(cc=1, brg=True) == "P1"

def test_rule2_no_critical_growing_not_p1():
    # Growing but no critical → should NOT be P1 via rule 2
    result = sev(cc=0, brg=True, wc=2, bs=2)
    assert result in ("P3", "P4")   # falls to rule 5 or 6

def test_rule2_detail():
    d = sev_d(cc=1, brg=True)
    assert d["rule_matched"] == 2


# ── Rule 3: critical_count == 1 AND NOT high_risk_mode ───────────────────────

def test_rule3_single_non_risk_critical():
    assert sev(cc=1, hrm=False) == "P2"

def test_rule3_detail():
    d = sev_d(cc=1, hrm=False)
    assert d["rule_matched"] == 3


# ── Rule 4: warning_count >= 3 AND high_risk_mode ────────────────────────────

def test_rule4_many_warnings_high_risk():
    assert sev(cc=0, wc=3, bs=3, hrm=True) == "P2"

def test_rule4_two_warnings_high_risk_not_enough():
    # 2 warnings in high_risk → rule 4 needs 3+
    result = sev(cc=0, wc=2, bs=2, hrm=True)
    assert result == "P3"   # falls to rule 5

def test_rule4_detail():
    d = sev_d(cc=0, wc=4, bs=4, hrm=True)
    assert d["rule_matched"] == 4


# ── Rule 5: warning_count >= 2 OR blast_size >= 4 ────────────────────────────

def test_rule5_two_warnings():
    assert sev(cc=0, wc=2, bs=2) == "P3"

def test_rule5_blast_size_four():
    assert sev(cc=0, wc=1, bs=4) == "P3"

def test_rule5_detail():
    d = sev_d(cc=0, wc=2, bs=2)
    assert d["rule_matched"] == 5


# ── Rule 6: fallback P4 ───────────────────────────────────────────────────────

def test_rule6_zero_anomalies():
    assert sev(cc=0, wc=0, bs=0) == "P4"

def test_rule6_one_warning():
    assert sev(cc=0, wc=1, bs=1) == "P4"

def test_rule6_detail():
    d = sev_d(cc=0, wc=0, bs=0)
    assert d["rule_matched"] == 6


# ── Failure mode severity floor ───────────────────────────────────────────────

def test_bad_deploy_floor_p2():
    """BAD_DEPLOY must never be below P2 regardless of indicators."""
    # Even with zero anomalies, BAD_DEPLOY should be at least P2
    result = sev(cc=0, wc=0, bs=0, mode="BAD_DEPLOY")
    assert result in ("P1", "P2")

def test_bad_deploy_floor_does_not_lower_p1():
    """P1 should remain P1 even for BAD_DEPLOY."""
    assert sev(cc=2, mode="BAD_DEPLOY") == "P1"


# ── Reference scenarios from spec document ────────────────────────────────────

def test_spec_early_memory_leak():
    """Early MEMORY_LEAK: heap + gc elevated, not yet critical → P3."""
    assert sev(cc=0, wc=2, bs=2) == "P3"

def test_spec_memory_leak_near_oom():
    """MEMORY_LEAK near OOM: 3 criticals, high_risk → P1."""
    assert sev(cc=3, hrm=True) == "P1"

def test_spec_isolated_db_slowdown():
    """Isolated DB_SLOWDOWN: 1 critical, not high-risk → P2."""
    assert sev(cc=1, hrm=False) == "P2"

def test_spec_cascading_failure():
    """CASCADING_FAILURE: blast_radius_growing + critical → P1."""
    assert sev(cc=1, wc=2, bs=3, hrm=True, brg=True) == "P1"

def test_spec_minor_queue_backup():
    """Minor QUEUE_BACKUP: 1 warning → P4."""
    assert sev(cc=0, wc=1, bs=1) == "P4"
