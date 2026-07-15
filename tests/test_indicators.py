"""tests/test_indicators.py — pytest tests for all feature indicator functions.
   Values calibrated to the recalibrated thresholds.yaml (data-driven).
"""
import pytest
from severity_engine.indicators import IndicatorLevel, FEATURE_EVALUATORS, evaluate_all
from severity_engine.config import load_config

# Always reload config (prevents stale cache across test runs)
load_config.cache_clear()
CFG = load_config()


# ── Helpers ───────────────────────────────────────────────────────────────────

def lvl(feature: str, value: float) -> IndicatorLevel:
    return FEATURE_EVALUATORS[feature](value, CFG)


# ── CPU Utilization (warn=55, crit=80) ────────────────────────────────────────

def test_cpu_normal():        assert lvl("cpu_utilization", 40.0) == IndicatorLevel.NORMAL
def test_cpu_warning():       assert lvl("cpu_utilization", 60.0) == IndicatorLevel.WARNING
def test_cpu_at_boundary():   assert lvl("cpu_utilization", 55.0) == IndicatorLevel.WARNING
def test_cpu_critical():      assert lvl("cpu_utilization", 90.0) == IndicatorLevel.CRITICAL
def test_cpu_at_crit_bound(): assert lvl("cpu_utilization", 80.0) == IndicatorLevel.CRITICAL


# ── Memory Utilization (warn=0.30, crit=0.37) ────────────────────────────────

def test_mem_normal():    assert lvl("memory_utilization", 0.25) == IndicatorLevel.NORMAL
def test_mem_warning():   assert lvl("memory_utilization", 0.33) == IndicatorLevel.WARNING
def test_mem_critical():  assert lvl("memory_utilization", 0.40) == IndicatorLevel.CRITICAL


# ── GC Pause (warn=50, crit=100) ──────────────────────────────────────────────

def test_gc_normal():    assert lvl("gc_pause_p99", 30.0)  == IndicatorLevel.NORMAL
def test_gc_warning():   assert lvl("gc_pause_p99", 75.0)  == IndicatorLevel.WARNING
def test_gc_critical():  assert lvl("gc_pause_p99", 200.0) == IndicatorLevel.CRITICAL


# ── P99 Latency (warn=300, crit=700) ──────────────────────────────────────────

def test_p99_normal():   assert lvl("p99_latency", 200.0)  == IndicatorLevel.NORMAL
def test_p99_warning():  assert lvl("p99_latency", 500.0)  == IndicatorLevel.WARNING
def test_p99_critical(): assert lvl("p99_latency", 900.0)  == IndicatorLevel.CRITICAL


# ── DB P99 (warn=100, crit=200) ───────────────────────────────────────────────

def test_db_normal():   assert lvl("db_p99", 80.0)   == IndicatorLevel.NORMAL
def test_db_warning():  assert lvl("db_p99", 150.0)  == IndicatorLevel.WARNING
def test_db_critical(): assert lvl("db_p99", 300.0)  == IndicatorLevel.CRITICAL


# ── Error Rate (warn=0.05, crit=0.20) ────────────────────────────────────────

def test_err_normal():   assert lvl("error_rate", 0.02) == IndicatorLevel.NORMAL
def test_err_warning():  assert lvl("error_rate", 0.08) == IndicatorLevel.WARNING
def test_err_critical(): assert lvl("error_rate", 0.30) == IndicatorLevel.CRITICAL


# ── Cache Hit Rate (warn=0.85, crit=0.50, lower=worse) ───────────────────────

def test_cache_hit_normal():   assert lvl("cache_hit_rate", 0.95) == IndicatorLevel.NORMAL
def test_cache_hit_warning():  assert lvl("cache_hit_rate", 0.70) == IndicatorLevel.WARNING
def test_cache_hit_critical(): assert lvl("cache_hit_rate", 0.10) == IndicatorLevel.CRITICAL


# ── Queue Lag (warn=30, crit=100) ─────────────────────────────────────────────

def test_queue_normal():   assert lvl("queue_lag", 20.0)  == IndicatorLevel.NORMAL
def test_queue_warning():  assert lvl("queue_lag", 60.0)  == IndicatorLevel.WARNING
def test_queue_critical(): assert lvl("queue_lag", 200.0) == IndicatorLevel.CRITICAL


# ── Circuit Breaker (categorical) ─────────────────────────────────────────────

def test_cb_closed():    assert lvl("circuit_breaker_state", 0) == IndicatorLevel.NORMAL
def test_cb_half_open(): assert lvl("circuit_breaker_state", 1) == IndicatorLevel.WARNING
def test_cb_open():      assert lvl("circuit_breaker_state", 2) == IndicatorLevel.CRITICAL


# ── Log features ──────────────────────────────────────────────────────────────

def test_log_severity_normal():   assert lvl("log_max_severity", 1) == IndicatorLevel.NORMAL
def test_log_severity_warning():  assert lvl("log_max_severity", 2) == IndicatorLevel.WARNING
def test_log_severity_critical(): assert lvl("log_max_severity", 4) == IndicatorLevel.CRITICAL

def test_log_exception_warning(): assert lvl("log_has_exception", 1) == IndicatorLevel.WARNING
def test_log_exception_normal():  assert lvl("log_has_exception", 0) == IndicatorLevel.NORMAL


# ── Trace features ────────────────────────────────────────────────────────────

def test_root_span_err_normal():   assert lvl("root_span_error_rate", 0.05) == IndicatorLevel.NORMAL
def test_root_span_err_warning():  assert lvl("root_span_error_rate", 0.20) == IndicatorLevel.WARNING
def test_root_span_err_critical(): assert lvl("root_span_error_rate", 0.50) == IndicatorLevel.CRITICAL

def test_dup_trace_normal():   assert lvl("duplicate_trace_id_count", 2)  == IndicatorLevel.NORMAL
def test_dup_trace_warning():  assert lvl("duplicate_trace_id_count", 5)  == IndicatorLevel.WARNING
def test_dup_trace_critical(): assert lvl("duplicate_trace_id_count", 15) == IndicatorLevel.CRITICAL


# ── evaluate_all ─────────────────────────────────────────────────────────────

def test_evaluate_all_returns_dict():
    values = {
        "cpu_utilization": 90.0,    # CRITICAL (>80)
        "error_rate": 0.30,         # CRITICAL (>0.20)
        "memory_utilization": 0.25, # NORMAL  (<0.30)
        "p99_latency": 200.0,       # NORMAL  (<300)
    }
    result = evaluate_all(values, CFG)
    assert isinstance(result, dict)
    assert result["cpu_utilization"] == IndicatorLevel.CRITICAL
    assert result["error_rate"] == IndicatorLevel.CRITICAL
    assert result["memory_utilization"] == IndicatorLevel.NORMAL


def test_evaluate_all_skips_missing_features():
    """Missing features should be silently skipped."""
    result = evaluate_all({"nonexistent_feature": 999.0}, CFG)
    assert len(result) == 0
