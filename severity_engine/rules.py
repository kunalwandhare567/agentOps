"""
severity_engine/rules.py
=========================
Implements the 6-rule severity decision engine exactly as specified
in the Severity Detection Features document.

Rules are evaluated TOP-TO-BOTTOM; the FIRST matching rule wins.
No ML — deterministic business logic only.

Rule table
──────────────────────────────────────────────────────────────────────
 Priority │ Severity │ Condition
──────────────────────────────────────────────────────────────────────
    1     │   P1     │ critical_count >= 2
          │          │ OR (critical_count >= 1 AND high_risk_mode)
──────────────────────────────────────────────────────────────────────
    2     │   P1     │ blast_radius_growing AND critical_count >= 1
──────────────────────────────────────────────────────────────────────
    3     │   P2     │ critical_count == 1 AND NOT high_risk_mode
──────────────────────────────────────────────────────────────────────
    4     │   P2     │ warning_count >= 3 AND high_risk_mode
──────────────────────────────────────────────────────────────────────
    5     │   P3     │ warning_count >= 2 OR blast_size >= 4
──────────────────────────────────────────────────────────────────────
    6     │   P4     │ (fallback) warning_count in [0,1] AND blast_size < 4
──────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

from typing import NamedTuple


class _RuleResult(NamedTuple):
    severity: str
    rule_matched: int        # 1-6 (0 = no rule matched, shouldn't happen)
    rule_description: str


# Severity floor per failure mode (BAD_DEPLOY → never lower than P2)
_MODE_FLOOR: dict[str, int] = {
    "BAD_DEPLOY": 3,    # numeric: P4=1 P3=2 P2=3 P1=4
}

_SEVERITY_NUMERIC: dict[str, int] = {"P4": 1, "P3": 2, "P2": 3, "P1": 4}
_NUMERIC_SEVERITY: dict[int, str] = {v: k for k, v in _SEVERITY_NUMERIC.items()}


def classify_severity(
    critical_count: int,
    warning_count: int,
    blast_size: int,
    high_risk_mode: bool,
    blast_radius_growing: bool,
    failure_mode: str = "NONE",
) -> str:
    """
    Apply the 6-rule decision tree and return a severity label.

    Args:
        critical_count:       Count of features at CRITICAL.
        warning_count:        Count of features at WARNING / HIGH.
        blast_size:           Total anomalous features (WARNING + CRITICAL).
        high_risk_mode:       Any CRITICAL feature in the high-risk category.
        blast_radius_growing: distinct_error_services grew since last cycle.
        failure_mode:         Predicted failure mode (for floor enforcement).

    Returns:
        "P1" | "P2" | "P3" | "P4"
    """
    result = _apply_rules(
        critical_count, warning_count, blast_size,
        high_risk_mode, blast_radius_growing
    )
    severity = result.severity

    # Apply failure-mode-specific severity floor
    floor_num = _MODE_FLOOR.get(failure_mode, 1)
    severity_num = _SEVERITY_NUMERIC.get(severity, 1)
    if severity_num < floor_num:
        severity = _NUMERIC_SEVERITY[floor_num]

    return severity


def _apply_rules(
    critical_count: int,
    warning_count: int,
    blast_size: int,
    high_risk_mode: bool,
    blast_radius_growing: bool,
) -> _RuleResult:
    """Internal rule evaluation — returns full RuleResult for debugging."""

    # ── Rule 1 ──────────────────────────────────────────────────────────────
    if critical_count >= 2 or (critical_count >= 1 and high_risk_mode):
        return _RuleResult(
            severity="P1",
            rule_matched=1,
            rule_description=(
                "Multiple critical signals OR single critical in high-risk category."
            ),
        )

    # ── Rule 2 ──────────────────────────────────────────────────────────────
    if blast_radius_growing and critical_count >= 1:
        return _RuleResult(
            severity="P1",
            rule_matched=2,
            rule_description=(
                "Active cascade: blast radius expanding + critical signal present."
            ),
        )

    # ── Rule 3 ──────────────────────────────────────────────────────────────
    if critical_count == 1 and not high_risk_mode:
        return _RuleResult(
            severity="P2",
            rule_matched=3,
            rule_description="Single critical signal outside high-risk category.",
        )

    # ── Rule 4 ──────────────────────────────────────────────────────────────
    if warning_count >= 3 and high_risk_mode:
        return _RuleResult(
            severity="P2",
            rule_matched=4,
            rule_description=(
                "Multiple warnings clustered in high-risk category — "
                "emerging critical pattern."
            ),
        )

    # ── Rule 5 ──────────────────────────────────────────────────────────────
    if warning_count >= 2 or blast_size >= 4:
        return _RuleResult(
            severity="P3",
            rule_matched=5,
            rule_description="Moderate anomaly footprint across multiple features.",
        )

    # ── Rule 6 (fallback) ────────────────────────────────────────────────────
    return _RuleResult(
        severity="P4",
        rule_matched=6,
        rule_description="Minor or isolated anomaly — default severity.",
    )


def classify_severity_with_details(
    critical_count: int,
    warning_count: int,
    blast_size: int,
    high_risk_mode: bool,
    blast_radius_growing: bool,
    failure_mode: str = "NONE",
) -> dict:
    """
    Same as classify_severity() but returns a full details dict including
    which rule fired. Useful for debugging and explanation nodes.
    """
    result = _apply_rules(
        critical_count, warning_count, blast_size,
        high_risk_mode, blast_radius_growing
    )
    severity = result.severity

    floor_num = _MODE_FLOOR.get(failure_mode, 1)
    severity_num = _SEVERITY_NUMERIC.get(severity, 1)
    floored = False
    if severity_num < floor_num:
        severity = _NUMERIC_SEVERITY[floor_num]
        floored = True

    return {
        "severity": severity,
        "rule_matched": result.rule_matched,
        "rule_description": result.rule_description,
        "failure_mode_floor_applied": floored,
    }
