"""
severity_engine/aggregator.py
==============================
Consumes the per-feature IndicatorLevel map and derives the
five inputs needed by the rule engine:

    critical_count       — features at CRITICAL
    warning_count        — features at WARNING or HIGH (not CRITICAL)
    blast_size           — all anomalous features (WARNING + HIGH + CRITICAL)
    high_risk_mode       — any CRITICAL in the high-risk category
    blast_radius_growing — distinct_error_services grew since last cycle

Weight overrides (from failure_mode_weights config) are applied here:
a weight ≥ 1.5 on a WARNING feature can "round up" its effective
contribution toward the CRITICAL bucket, making failure-mode-specific
escalation possible without changing the base threshold logic.
"""
from __future__ import annotations

from .config import get_high_risk_features
from .indicators import IndicatorLevel


# Effective weight threshold above which a WARNING is counted as CRITICAL
_WEIGHT_ESCALATION_THRESHOLD = 1.5


def aggregate(
    indicators: dict[str, IndicatorLevel],
    weight_overrides: dict[str, float],
    high_risk_features: list[str],
    prev_distinct_error_services: float = 0.0,
    curr_distinct_error_services: float = 0.0,
) -> dict:
    """
    Compute the five aggregated signals consumed by rules.py.

    Args:
        indicators:                    feature → IndicatorLevel mapping.
        weight_overrides:              Failure-mode weight multipliers.
        high_risk_features:            List of high-risk feature names (from config).
        prev_distinct_error_services:  Value from the previous telemetry step.
        curr_distinct_error_services:  Value from the current telemetry step.

    Returns:
        Dict with keys:
            critical_count, warning_count, blast_size,
            high_risk_mode, blast_radius_growing, weighted_score
    """
    critical_count = 0
    warning_count = 0
    weighted_score = 0.0

    for feature, level in indicators.items():
        weight = weight_overrides.get(feature, 1.0)
        effective_level = _apply_weight(level, weight)

        # Score contribution (weighted)
        weighted_score += effective_level * weight

        if effective_level >= IndicatorLevel.CRITICAL:
            critical_count += 1
        elif effective_level >= IndicatorLevel.WARNING:
            warning_count += 1

    blast_size = critical_count + warning_count

    # high_risk_mode: any CRITICAL feature in the high-risk category
    high_risk_mode = any(
        indicators.get(f, IndicatorLevel.NORMAL) >= IndicatorLevel.CRITICAL
        for f in high_risk_features
    )

    # blast_radius_growing: distinct_error_services strictly increased
    blast_radius_growing = (
        curr_distinct_error_services > prev_distinct_error_services
    )

    return {
        "critical_count": critical_count,
        "warning_count": warning_count,
        "blast_size": blast_size,
        "high_risk_mode": high_risk_mode,
        "blast_radius_growing": blast_radius_growing,
        "weighted_score": round(weighted_score, 3),
    }


def _apply_weight(level: IndicatorLevel, weight: float) -> IndicatorLevel:
    """
    Escalate a WARNING to CRITICAL if the failure-mode weight is high enough.

    Keeps the level clamped to CRITICAL maximum.
    """
    if level == IndicatorLevel.WARNING and weight >= _WEIGHT_ESCALATION_THRESHOLD:
        return IndicatorLevel.CRITICAL
    return level


def build_reason(
    indicators: dict[str, IndicatorLevel],
    failure_mode: str,
    agg: dict,
) -> str:
    """
    Build a concise human-readable reason string for the SeverityResult.
    """
    critical_features = [
        f for f, lvl in indicators.items() if lvl >= IndicatorLevel.CRITICAL
    ]
    warning_features = [
        f for f, lvl in indicators.items() if lvl == IndicatorLevel.WARNING
    ]

    parts: list[str] = [f"Failure mode: {failure_mode}."]

    if critical_features:
        parts.append(
            f"CRITICAL: {', '.join(critical_features)} "
            f"({agg['critical_count']} feature(s))."
        )
    if warning_features:
        parts.append(
            f"WARNING: {', '.join(warning_features[:5])} "
            f"({agg['warning_count']} feature(s))."
        )
    if agg["blast_radius_growing"]:
        parts.append("Blast radius expanding (distinct_error_services growing).")
    if agg["high_risk_mode"]:
        parts.append("High-risk category breach detected.")

    return " ".join(parts)
