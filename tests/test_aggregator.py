"""tests/test_aggregator.py — pytest tests for the aggregator module."""
import pytest
from severity_engine.aggregator import aggregate, build_reason
from severity_engine.indicators import IndicatorLevel

HIGH_RISK = ["cpu_utilization", "memory_utilization", "error_rate",
             "root_span_error_rate", "distinct_error_services"]


def _agg(indicators, weights=None, prev_des=0.0, curr_des=0.0):
    return aggregate(indicators, weights or {}, HIGH_RISK, prev_des, curr_des)


# ── critical_count ────────────────────────────────────────────────────────────

def test_critical_count_zero():
    ind = {"cpu_utilization": IndicatorLevel.NORMAL, "p99_latency": IndicatorLevel.WARNING}
    assert _agg(ind)["critical_count"] == 0

def test_critical_count_one():
    ind = {"cpu_utilization": IndicatorLevel.CRITICAL, "p99_latency": IndicatorLevel.WARNING}
    assert _agg(ind)["critical_count"] == 1

def test_critical_count_multiple():
    ind = {
        "cpu_utilization": IndicatorLevel.CRITICAL,
        "memory_utilization": IndicatorLevel.CRITICAL,
        "error_rate": IndicatorLevel.CRITICAL,
    }
    assert _agg(ind)["critical_count"] == 3


# ── warning_count ─────────────────────────────────────────────────────────────

def test_warning_count_none():
    ind = {"cpu_utilization": IndicatorLevel.NORMAL}
    assert _agg(ind)["warning_count"] == 0

def test_warning_count_two():
    ind = {"p99_latency": IndicatorLevel.WARNING, "db_p99": IndicatorLevel.WARNING}
    assert _agg(ind)["warning_count"] == 2


# ── blast_size ────────────────────────────────────────────────────────────────

def test_blast_size_is_sum():
    ind = {
        "cpu_utilization": IndicatorLevel.CRITICAL,
        "p99_latency": IndicatorLevel.WARNING,
        "db_p99": IndicatorLevel.WARNING,
        "error_rate": IndicatorLevel.NORMAL,
    }
    result = _agg(ind)
    assert result["blast_size"] == 3    # 1 critical + 2 warning


# ── high_risk_mode ────────────────────────────────────────────────────────────

def test_high_risk_false_when_only_warning():
    ind = {"cpu_utilization": IndicatorLevel.WARNING}
    assert _agg(ind)["high_risk_mode"] is False

def test_high_risk_true_when_cpu_critical():
    ind = {"cpu_utilization": IndicatorLevel.CRITICAL}
    assert _agg(ind)["high_risk_mode"] is True

def test_high_risk_true_error_rate():
    ind = {"error_rate": IndicatorLevel.CRITICAL}
    assert _agg(ind)["high_risk_mode"] is True

def test_high_risk_false_for_non_risk_critical():
    ind = {"db_p99": IndicatorLevel.CRITICAL}   # not in high_risk list
    assert _agg(ind)["high_risk_mode"] is False


# ── blast_radius_growing ──────────────────────────────────────────────────────

def test_blast_radius_not_growing():
    assert _agg({}, prev_des=2.0, curr_des=2.0)["blast_radius_growing"] is False

def test_blast_radius_growing():
    assert _agg({}, prev_des=1.0, curr_des=3.0)["blast_radius_growing"] is True

def test_blast_radius_shrinking_not_growing():
    assert _agg({}, prev_des=3.0, curr_des=1.0)["blast_radius_growing"] is False


# ── weight escalation (WARNING → CRITICAL with high weight) ──────────────────

def test_weight_escalation_warning_to_critical():
    ind = {"p99_latency": IndicatorLevel.WARNING}
    # Weight >= 1.5 should escalate WARNING to CRITICAL
    result = _agg(ind, weights={"p99_latency": 1.5})
    assert result["critical_count"] == 1
    assert result["warning_count"] == 0

def test_weight_no_escalation_low_weight():
    ind = {"p99_latency": IndicatorLevel.WARNING}
    result = _agg(ind, weights={"p99_latency": 1.2})
    assert result["critical_count"] == 0
    assert result["warning_count"] == 1


# ── build_reason ─────────────────────────────────────────────────────────────

def test_build_reason_contains_failure_mode():
    ind = {"cpu_utilization": IndicatorLevel.CRITICAL}
    agg = _agg(ind)
    reason = build_reason(ind, "CPU_SATURATION", agg)
    assert "CPU_SATURATION" in reason

def test_build_reason_mentions_critical_feature():
    ind = {"error_rate": IndicatorLevel.CRITICAL}
    agg = _agg(ind)
    reason = build_reason(ind, "ERROR_STORM", agg)
    assert "error_rate" in reason
