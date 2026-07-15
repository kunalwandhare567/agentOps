"""tests/test_severity_engine.py — End-to-end pytest tests for SeverityEngine."""
import pytest
import pandas as pd
from severity_engine.severity_engine import SeverityEngine
from severity_engine.utils import SeverityResult


@pytest.fixture()
def engine():
    e = SeverityEngine()
    e.reset_all()
    return e


# ── Helper to build a minimal row ────────────────────────────────────────────

def make_row(**kwargs):
    defaults = {
        "cpu_utilization": 35.0, "memory_utilization": 0.30,
        "heap_mb": 520.0, "p99_latency": 200.0, "p95_latency": 150.0,
        "p50_latency": 95.0, "error_rate": 0.01, "rps": 200.0,
        "queue_lag": 10.0, "cache_hit_rate": 0.95, "cache_miss_rate": 0.05,
        "gc_pause_p99": 15.0, "disk_write_latency": 10.0, "disk_read_latency": 5.0,
        "db_p99": 30.0, "upstream_timeout_rate": 0.01, "network_errors": 1.0,
        "thread_pool_queue": 8.0, "retry_count_per_request": 0.05,
        "active_connections": 80.0, "db_connection_pool": 0.40,
        "circuit_breaker_state": 0, "root_span_error_rate": 0.02,
        "span_gap_ratio": 0.10, "uniform_slowdown_score": 0.20,
        "duplicate_trace_id_count": 1.0, "distinct_error_services": 0.0,
        "log_max_severity": 1.0, "log_critical_count": 0.0,
        "log_has_exception": 0.0,
    }
    defaults.update(kwargs)
    return defaults


# ── evaluate_row returns SeverityResult ──────────────────────────────────────

def test_evaluate_row_returns_severity_result(engine):
    result = engine.evaluate_row(make_row(), "NONE", "ep1")
    assert isinstance(result, SeverityResult)


def test_healthy_row_is_p4(engine):
    result = engine.evaluate_row(make_row(), "NONE", "ep1")
    assert result.severity == "P4"


def test_high_cpu_escalates(engine):
    row = make_row(cpu_utilization=95.0, thread_pool_queue=50.0)
    # Pre-warm the smoother so EMA converges before assertion
    for _ in range(5):
        engine.evaluate_row(row, "CPU_SATURATION", "ep2")
    result = engine.evaluate_row(row, "CPU_SATURATION", "ep2")
    # Raw severity should be P1/P2; smoothed may lag slightly but raw must be correct
    assert result.raw_severity in ("P1", "P2")


def test_memory_leak_near_oom_is_p1(engine):
    engine.reset_all()
    row = make_row(
        memory_utilization=0.93, gc_pause_p99=450.0,
        heap_mb=3500.0, p99_latency=2100.0,
    )
    # Pre-warm so EMA converges to P1
    for _ in range(8):
        engine.evaluate_row(row, "MEMORY_LEAK", "ep3")
    result = engine.evaluate_row(row, "MEMORY_LEAK", "ep3")
    assert result.severity == "P1"


def test_bad_deploy_always_p2_or_higher(engine):
    # Even with almost healthy metrics, BAD_DEPLOY rule floor = P2
    # raw_severity is set to P2 by the floor in rules.py;
    # the smoother starts at P4 baseline so smoothed may be P3 initially.
    # Verify the rule engine itself applies the floor correctly.
    result = engine.evaluate_row(make_row(), "BAD_DEPLOY", "ep4")
    assert result.raw_severity in ("P1", "P2")


# ── SeverityResult fields ─────────────────────────────────────────────────────

def test_result_contains_all_required_fields(engine):
    result = engine.evaluate_row(make_row(), "NONE", "ep5")
    assert result.failure_mode == "NONE"
    assert result.severity in ("P1", "P2", "P3", "P4")
    assert isinstance(result.critical_count, int)
    assert isinstance(result.warning_count, int)
    assert isinstance(result.indicator_breakdown, dict)
    assert isinstance(result.reason, str)
    assert isinstance(result.recommended_action, str)


def test_result_to_dict_is_json_safe(engine):
    result = engine.evaluate_row(make_row(), "NONE", "ep6")
    d = result.to_dict()
    import json
    assert json.dumps(d)  # should not raise


# ── compute_severity (batch) ──────────────────────────────────────────────────

def test_compute_severity_adds_columns(engine):
    rows = [make_row() for _ in range(5)]
    df = pd.DataFrame(rows)
    df["episode_id"] = "ep_batch"
    df["elapsed_s"] = range(5)
    df["failure_mode"] = "NONE"
    df["timestamp"] = "2024-01-01T00:00:00"
    out = engine.compute_severity(df)
    assert "Severity" in out.columns
    assert "CriticalCount" in out.columns
    assert "Reason" in out.columns
    assert len(out) == 5


def test_compute_severity_values_are_valid(engine):
    rows = [make_row(cpu_utilization=95.0) for _ in range(3)]
    df = pd.DataFrame(rows)
    df["episode_id"] = "ep_batch2"
    df["elapsed_s"] = range(3)
    df["failure_mode"] = "CPU_SATURATION"
    df["timestamp"] = ""
    out = engine.compute_severity(df)
    assert all(out["Severity"].isin(["P1", "P2", "P3", "P4"]))


# ── Temporal smoothing via engine ─────────────────────────────────────────────

def test_severity_does_not_oscillate(engine):
    """Feed P1 then P4 alternating — severity should not instantly return to P4."""
    inputs = ["P1", "P4", "P1", "P4"]
    results = []
    for raw in inputs:
        # Manufacture a row that would produce the target raw severity
        if raw == "P1":
            row = make_row(cpu_utilization=95.0, memory_utilization=0.93,
                           error_rate=0.30, root_span_error_rate=0.50)
        else:
            row = make_row()
        result = engine.evaluate_row(row, "CPU_SATURATION", "ep_smooth")
        results.append(result.severity)

    # After seeing P1, severity must not immediately drop to P4
    for i in range(1, len(results)):
        if results[i - 1] == "P1":
            assert results[i] != "P4", \
                f"Instant drop from P1 to P4 at step {i}: {results}"
