"""
severity_engine/utils.py
========================
Shared data structures, logging setup, and helper utilities.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from typing import Any

# ── Logging ──────────────────────────────────────────────────────────────────

def get_logger(name: str = "severity_engine") -> logging.Logger:
    """Return a configured logger. Call once per module."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# ── SeverityResult ────────────────────────────────────────────────────────────

@dataclass
class SeverityResult:
    """
    Immutable result object produced by the SeverityEngine for every
    telemetry row (or aggregated window).

    Designed as a LangGraph-compatible payload:
    the fields map 1-to-1 onto the AgentState severity sub-object.
    """
    # Identifiers
    timestamp: str
    episode_id: str
    elapsed_s: float

    # Classification output (from XGBoost upstream)
    failure_mode: str

    # Severity output
    severity: str                          # "P1" / "P2" / "P3" / "P4"
    raw_severity: str                      # before temporal smoothing
    weighted_score: float                  # aggregated numeric score (for logging)

    # Aggregated signals
    critical_count: int
    warning_count: int
    blast_size: int
    high_risk_mode: bool
    blast_radius_growing: bool

    # Per-feature breakdown
    indicator_breakdown: dict[str, str] = field(default_factory=dict)
    # e.g. {"cpu_utilization": "CRITICAL", "memory_utilization": "WARNING", ...}

    # Human-readable outputs (for LLM / LangGraph explanation node)
    reason: str = ""
    recommended_action: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (JSON-safe)."""
        return {
            "timestamp": self.timestamp,
            "episode_id": self.episode_id,
            "elapsed_s": self.elapsed_s,
            "failure_mode": self.failure_mode,
            "severity": self.severity,
            "raw_severity": self.raw_severity,
            "weighted_score": self.weighted_score,
            "critical_count": self.critical_count,
            "warning_count": self.warning_count,
            "blast_size": self.blast_size,
            "high_risk_mode": self.high_risk_mode,
            "blast_radius_growing": self.blast_radius_growing,
            "indicator_breakdown": self.indicator_breakdown,
            "reason": self.reason,
            "recommended_action": self.recommended_action,
        }


# ── Severity numeric encoding / decoding ─────────────────────────────────────

SEVERITY_ORDER: dict[str, int] = {"P4": 1, "P3": 2, "P2": 3, "P1": 4}
SEVERITY_LABELS: dict[int, str] = {v: k for k, v in SEVERITY_ORDER.items()}


def severity_to_int(sev: str) -> int:
    return SEVERITY_ORDER.get(sev, 1)


def int_to_severity(val: int) -> str:
    clamped = max(1, min(4, round(val)))
    return SEVERITY_LABELS[clamped]


# ── Recommended actions per failure mode + severity ───────────────────────────

_ACTIONS: dict[str, dict[str, str]] = {
    "CPU_SATURATION": {
        "P1": "Scale out replicas immediately. Shed non-critical traffic.",
        "P2": "Increase CPU quota. Profile hotspot threads.",
        "P3": "Monitor CPU trend. Consider pre-emptive scaling.",
        "P4": "No action required. Watch for escalation.",
    },
    "MEMORY_LEAK": {
        "P1": "Trigger controlled rolling restart. Alert on-call SRE.",
        "P2": "Force GC. Capture heap dump for analysis.",
        "P3": "Enable heap profiling. Set OOM kill threshold alert.",
        "P4": "Log heap trend. No immediate action.",
    },
    "CACHE_STAMPEDE": {
        "P1": "Enable request coalescing / mutex on cache fill. Rate-limit cold reads.",
        "P2": "Warm cache proactively. Add jitter to TTL.",
        "P3": "Review cache sizing. Monitor hit-rate trend.",
        "P4": "No action. Normal cache miss variance.",
    },
    "LATENCY_SPIKE": {
        "P1": "Activate circuit breaker. Re-route traffic to healthy nodes.",
        "P2": "Investigate slow queries. Check downstream dependencies.",
        "P3": "Profile P99 endpoint. Check GC pauses.",
        "P4": "Log latency percentiles. No action needed.",
    },
    "DB_SLOWDOWN": {
        "P1": "Failover to read replica. Kill long-running queries.",
        "P2": "Analyse slow query log. Increase connection pool.",
        "P3": "Review index usage. Monitor db_p99 trend.",
        "P4": "No action. Minor DB variance.",
    },
    "CASCADING_FAILURE": {
        "P1": "Initiate incident bridge. Isolate failing services immediately.",
        "P2": "Enable bulkhead pattern. Stop cascade propagation.",
        "P3": "Identify blast origin service. Increase health-check frequency.",
        "P4": "Monitor service dependency graph.",
    },
    "DEPENDENCY_TIMEOUT": {
        "P1": "Open circuit breaker. Fall back to cached/degraded responses.",
        "P2": "Increase timeout margins. Alert dependency team.",
        "P3": "Review upstream SLA. Check network path.",
        "P4": "No action. Transient timeout.",
    },
    "RETRY_STORM": {
        "P1": "Apply exponential back-off immediately. Rate-limit retry clients.",
        "P2": "Add jitter to retry policy. Reduce max retry count.",
        "P3": "Audit retry configuration across services.",
        "P4": "No action. Normal retry activity.",
    },
    "DISK_IO_SATURATION": {
        "P1": "Offload writes to async queue. Consider disk upgrade/replacement.",
        "P2": "Throttle write-heavy operations. Monitor iops_utilization.",
        "P3": "Profile disk write paths. Review log rotation policy.",
        "P4": "No action. Minor disk variance.",
    },
    "ERROR_STORM": {
        "P1": "Drain request queue. Investigate error root cause. Alert on-call.",
        "P2": "Increase error budget burn-rate alert. Review recent deployments.",
        "P3": "Inspect error logs for pattern. Check downstream APIs.",
        "P4": "No action. Normal error baseline.",
    },
    "BAD_DEPLOY": {
        "P1": "Rollback deployment immediately. Freeze further rollouts.",
        "P2": "Pause rollout. Run canary health checks.",
        "P3": "Validate deployment health metrics. Prepare rollback plan.",
        "P4": "Monitor post-deployment metrics closely.",
    },
    "NONE": {
        "P1": "Investigate unexpected P1 with no known failure mode.",
        "P2": "Investigate unexpected P2 with NONE failure mode.",
        "P3": "Review feature thresholds for false positives.",
        "P4": "System healthy. No action required.",
    },
}


def get_recommended_action(failure_mode: str, severity: str) -> str:
    """Return a recommended action string for the given failure mode and severity."""
    mode_actions = _ACTIONS.get(failure_mode, _ACTIONS["NONE"])
    return mode_actions.get(severity, "Investigate and monitor closely.")
