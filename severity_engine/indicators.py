"""
severity_engine/indicators.py
==============================
IndicatorLevel enum + one pure evaluation function per feature.

Each function signature:
    evaluate_<feature>(value: float, thresholds: dict) -> IndicatorLevel

All functions are stateless and independently testable.
"""
from __future__ import annotations

from enum import IntEnum
from typing import Any

from .config import load_config


# ── Indicator levels ─────────────────────────────────────────────────────────

class IndicatorLevel(IntEnum):
    """
    Ordered severity indicator for a single feature.
    Higher numeric value = worse condition.
    """
    NORMAL   = 0
    WARNING  = 1
    HIGH     = 2
    CRITICAL = 3

    def label(self) -> str:
        return self.name


# ── Generic evaluation helpers ────────────────────────────────────────────────

def _eval_higher_worse(value: float, warn: float, crit: float) -> IndicatorLevel:
    """Return CRITICAL / WARNING / NORMAL for a metric where higher = worse."""
    if value >= crit:
        return IndicatorLevel.CRITICAL
    if value >= warn:
        return IndicatorLevel.WARNING
    return IndicatorLevel.NORMAL


def _eval_lower_worse(value: float, warn: float, crit: float) -> IndicatorLevel:
    """Return CRITICAL / WARNING / NORMAL for a metric where lower = worse."""
    if value <= crit:
        return IndicatorLevel.CRITICAL
    if value <= warn:
        return IndicatorLevel.WARNING
    return IndicatorLevel.NORMAL


def _thresholds(feature: str, cfg: dict) -> tuple[float, float]:
    """Extract (warning_threshold, critical_threshold) for a feature."""
    fc = cfg["features"][feature]
    return float(fc["warning_threshold"]), float(fc["critical_threshold"])


# ── Per-feature evaluation functions ─────────────────────────────────────────

def evaluate_cpu_utilization(value: float, cfg: dict) -> IndicatorLevel:
    w, c = _thresholds("cpu_utilization", cfg)
    return _eval_higher_worse(value, w, c)


def evaluate_memory_utilization(value: float, cfg: dict) -> IndicatorLevel:
    w, c = _thresholds("memory_utilization", cfg)
    return _eval_higher_worse(value, w, c)


def evaluate_heap_mb(value: float, cfg: dict) -> IndicatorLevel:
    w, c = _thresholds("heap_mb", cfg)
    return _eval_higher_worse(value, w, c)


def evaluate_gc_pause_p99(value: float, cfg: dict) -> IndicatorLevel:
    w, c = _thresholds("gc_pause_p99", cfg)
    return _eval_higher_worse(value, w, c)


def evaluate_p99_latency(value: float, cfg: dict) -> IndicatorLevel:
    w, c = _thresholds("p99_latency", cfg)
    return _eval_higher_worse(value, w, c)


def evaluate_p95_latency(value: float, cfg: dict) -> IndicatorLevel:
    w, c = _thresholds("p95_latency", cfg)
    return _eval_higher_worse(value, w, c)


def evaluate_p50_latency(value: float, cfg: dict) -> IndicatorLevel:
    w, c = _thresholds("p50_latency", cfg)
    return _eval_higher_worse(value, w, c)


def evaluate_db_p99(value: float, cfg: dict) -> IndicatorLevel:
    w, c = _thresholds("db_p99", cfg)
    return _eval_higher_worse(value, w, c)


def evaluate_disk_write_latency(value: float, cfg: dict) -> IndicatorLevel:
    w, c = _thresholds("disk_write_latency", cfg)
    return _eval_higher_worse(value, w, c)


def evaluate_disk_read_latency(value: float, cfg: dict) -> IndicatorLevel:
    w, c = _thresholds("disk_read_latency", cfg)
    return _eval_higher_worse(value, w, c)


def evaluate_error_rate(value: float, cfg: dict) -> IndicatorLevel:
    w, c = _thresholds("error_rate", cfg)
    return _eval_higher_worse(value, w, c)


def evaluate_upstream_timeout_rate(value: float, cfg: dict) -> IndicatorLevel:
    w, c = _thresholds("upstream_timeout_rate", cfg)
    return _eval_higher_worse(value, w, c)


def evaluate_cache_miss_rate(value: float, cfg: dict) -> IndicatorLevel:
    w, c = _thresholds("cache_miss_rate", cfg)
    return _eval_higher_worse(value, w, c)


def evaluate_cache_hit_rate(value: float, cfg: dict) -> IndicatorLevel:
    """Lower cache_hit_rate = worse."""
    w, c = _thresholds("cache_hit_rate", cfg)
    return _eval_lower_worse(value, w, c)


def evaluate_queue_lag(value: float, cfg: dict) -> IndicatorLevel:
    w, c = _thresholds("queue_lag", cfg)
    return _eval_higher_worse(value, w, c)


def evaluate_thread_pool_queue(value: float, cfg: dict) -> IndicatorLevel:
    w, c = _thresholds("thread_pool_queue", cfg)
    return _eval_higher_worse(value, w, c)


def evaluate_network_errors(value: float, cfg: dict) -> IndicatorLevel:
    w, c = _thresholds("network_errors", cfg)
    return _eval_higher_worse(value, w, c)


def evaluate_retry_count_per_request(value: float, cfg: dict) -> IndicatorLevel:
    w, c = _thresholds("retry_count_per_request", cfg)
    return _eval_higher_worse(value, w, c)


def evaluate_active_connections(value: float, cfg: dict) -> IndicatorLevel:
    w, c = _thresholds("active_connections", cfg)
    return _eval_higher_worse(value, w, c)


def evaluate_db_connection_pool(value: float, cfg: dict) -> IndicatorLevel:
    w, c = _thresholds("db_connection_pool", cfg)
    return _eval_higher_worse(value, w, c)


def evaluate_circuit_breaker_state(value: float, cfg: dict) -> IndicatorLevel:
    """Categorical: 0=closed(normal), 1=half-open(warning), 2=open(critical)."""
    w, c = _thresholds("circuit_breaker_state", cfg)
    return _eval_higher_worse(value, w, c)


# ── Derived feature evaluation functions ─────────────────────────────────────

def evaluate_p99_minus_p50_ratio(value: float, cfg: dict) -> IndicatorLevel:
    w, c = _thresholds("p99_minus_p50_ratio", cfg)
    return _eval_higher_worse(value, w, c)


def evaluate_heap_growth_rate(value: float, cfg: dict) -> IndicatorLevel:
    w, c = _thresholds("heap_growth_rate", cfg)
    return _eval_higher_worse(value, w, c)


def evaluate_write_read_latency_ratio(value: float, cfg: dict) -> IndicatorLevel:
    w, c = _thresholds("write_read_latency_ratio", cfg)
    return _eval_higher_worse(value, w, c)


# ── Log feature evaluation functions ─────────────────────────────────────────

def evaluate_log_max_severity(value: float, cfg: dict) -> IndicatorLevel:
    """log_level numeric code: INFO=1, WARNING=2, ERROR=3, CRITICAL=4."""
    w, c = _thresholds("log_max_severity", cfg)
    return _eval_higher_worse(value, w, c)


def evaluate_log_critical_count(value: float, cfg: dict) -> IndicatorLevel:
    w, c = _thresholds("log_critical_count", cfg)
    return _eval_higher_worse(value, w, c)


def evaluate_log_has_exception(value: float, cfg: dict) -> IndicatorLevel:
    """Binary: 1=exception present → WARNING only."""
    w, c = _thresholds("log_has_exception", cfg)
    return _eval_higher_worse(value, w, c)


# ── Trace feature evaluation functions ───────────────────────────────────────

def evaluate_root_span_error_rate(value: float, cfg: dict) -> IndicatorLevel:
    w, c = _thresholds("root_span_error_rate", cfg)
    return _eval_higher_worse(value, w, c)


def evaluate_span_gap_ratio(value: float, cfg: dict) -> IndicatorLevel:
    w, c = _thresholds("span_gap_ratio", cfg)
    return _eval_higher_worse(value, w, c)


def evaluate_uniform_slowdown_score(value: float, cfg: dict) -> IndicatorLevel:
    w, c = _thresholds("uniform_slowdown_score", cfg)
    return _eval_higher_worse(value, w, c)


def evaluate_duplicate_trace_id_count(value: float, cfg: dict) -> IndicatorLevel:
    w, c = _thresholds("duplicate_trace_id_count", cfg)
    return _eval_higher_worse(value, w, c)


def evaluate_distinct_error_services(value: float, cfg: dict) -> IndicatorLevel:
    w, c = _thresholds("distinct_error_services", cfg)
    return _eval_higher_worse(value, w, c)


# ── Feature registry ──────────────────────────────────────────────────────────
# Maps feature name → evaluation function. Add new features here.

FEATURE_EVALUATORS: dict[str, Any] = {
    "cpu_utilization":          evaluate_cpu_utilization,
    "memory_utilization":       evaluate_memory_utilization,
    "heap_mb":                  evaluate_heap_mb,
    "gc_pause_p99":             evaluate_gc_pause_p99,
    "p99_latency":              evaluate_p99_latency,
    "p95_latency":              evaluate_p95_latency,
    "p50_latency":              evaluate_p50_latency,
    "db_p99":                   evaluate_db_p99,
    "disk_write_latency":       evaluate_disk_write_latency,
    "disk_read_latency":        evaluate_disk_read_latency,
    "error_rate":               evaluate_error_rate,
    "upstream_timeout_rate":    evaluate_upstream_timeout_rate,
    "cache_miss_rate":          evaluate_cache_miss_rate,
    "cache_hit_rate":           evaluate_cache_hit_rate,
    "queue_lag":                evaluate_queue_lag,
    "thread_pool_queue":        evaluate_thread_pool_queue,
    "network_errors":           evaluate_network_errors,
    "retry_count_per_request":  evaluate_retry_count_per_request,
    "active_connections":       evaluate_active_connections,
    "db_connection_pool":       evaluate_db_connection_pool,
    "circuit_breaker_state":    evaluate_circuit_breaker_state,
    # Derived
    "p99_minus_p50_ratio":      evaluate_p99_minus_p50_ratio,
    "heap_growth_rate":         evaluate_heap_growth_rate,
    "write_read_latency_ratio": evaluate_write_read_latency_ratio,
    # Log
    "log_max_severity":         evaluate_log_max_severity,
    "log_critical_count":       evaluate_log_critical_count,
    "log_has_exception":        evaluate_log_has_exception,
    # Trace
    "root_span_error_rate":     evaluate_root_span_error_rate,
    "span_gap_ratio":           evaluate_span_gap_ratio,
    "uniform_slowdown_score":   evaluate_uniform_slowdown_score,
    "duplicate_trace_id_count": evaluate_duplicate_trace_id_count,
    "distinct_error_services":  evaluate_distinct_error_services,
}


def evaluate_all(
    feature_values: dict[str, float],
    cfg: dict,
    weight_overrides: dict[str, float] | None = None,
) -> dict[str, IndicatorLevel]:
    """
    Evaluate every available feature and return a mapping of
    feature_name → IndicatorLevel.

    Weights can be passed in for failure-mode-specific boosting; they do
    NOT change the raw level but are used downstream in the aggregator.

    Args:
        feature_values:  Dict of feature_name → float value for this row.
        cfg:             Loaded YAML config dict.
        weight_overrides: Optional per-feature multipliers (from failure mode).

    Returns:
        Dict of feature_name → IndicatorLevel.
    """
    results: dict[str, IndicatorLevel] = {}
    for feature, evaluator in FEATURE_EVALUATORS.items():
        val = feature_values.get(feature)
        if val is None:
            continue
        try:
            level = evaluator(float(val), cfg)
            results[feature] = level
        except Exception:
            # Skip missing config keys gracefully; unknown features stay out
            pass
    return results
