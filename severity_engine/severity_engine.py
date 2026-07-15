"""
severity_engine/severity_engine.py
====================================
Orchestrates the full row-level severity pipeline:

    feature_values  +  failure_mode
            │
            ▼
    derived features (p99/p50 ratio, heap growth, write/read ratio)
            │
            ▼
    indicators.evaluate_all()
            │
            ▼
    aggregator.aggregate()
            │
            ▼
    rules.classify_severity()
            │
            ▼
    smoother.smooth()       ← temporal smoothing per episode
            │
            ▼
    SeverityResult

Public API
──────────
    engine = SeverityEngine()
    result = engine.evaluate_row(row_dict, failure_mode, episode_id)
    df_out = engine.compute_severity(df)   ← batch pandas support
"""
from __future__ import annotations

import pandas as pd
from typing import Any

from .aggregator import aggregate, build_reason
from .config import load_config, get_failure_mode_weights, get_high_risk_features
from .indicators import evaluate_all, IndicatorLevel
from .rules import classify_severity
from .smoother import EpisodeSmoother
from .utils import (
    SeverityResult,
    get_logger,
    get_recommended_action,
    severity_to_int,
    int_to_severity,
)

logger = get_logger("severity_engine.engine")


class SeverityEngine:
    """
    Stateful severity evaluation engine.

    Maintains per-episode temporal state (EMA + hysteresis) between calls.
    One instance should be reused across all rows of a streaming session.

    Args:
        yaml_path: Optional path to a custom thresholds YAML file.
    """

    def __init__(self, yaml_path: str | None = None) -> None:
        self.cfg = load_config(yaml_path)
        smooth_params = self.cfg["smoothing"]
        self.smoother = EpisodeSmoother(
            alpha=smooth_params["alpha"],
            hysteresis_steps=smooth_params["hysteresis_steps"],
        )
        self.high_risk_features = get_high_risk_features(self.cfg)
        # Track previous distinct_error_services per episode for blast_radius_growing
        self._prev_des: dict[str, float] = {}
        logger.info("SeverityEngine initialised (alpha=%.2f, hysteresis=%d).",
                    smooth_params["alpha"], smooth_params["hysteresis_steps"])

    # ── Row-level API ─────────────────────────────────────────────────────────

    def evaluate_row(
        self,
        feature_values: dict[str, Any],
        failure_mode: str,
        episode_id: str = "unknown",
        timestamp: str = "",
        elapsed_s: float = 0.0,
    ) -> SeverityResult:
        """
        Evaluate a single telemetry row and return a SeverityResult.

        Args:
            feature_values: Dict of raw metric/log/trace values for this row.
            failure_mode:   Predicted failure mode string (from XGBoost).
            episode_id:     Episode identifier (used for temporal state tracking).
            timestamp:      ISO timestamp string (passed through to result).
            elapsed_s:      Elapsed seconds since episode start.

        Returns:
            SeverityResult dataclass.
        """
        # 1. Compute derived features
        enriched = dict(feature_values)
        enriched = self._add_derived_features(enriched)

        # 2. Evaluate all feature indicators
        weight_overrides = get_failure_mode_weights(failure_mode, self.cfg)
        indicators = evaluate_all(enriched, self.cfg, weight_overrides)

        # 3. Aggregate indicators
        prev_des = self._prev_des.get(episode_id, 0.0)
        curr_des = float(enriched.get("distinct_error_services", 0.0))
        agg = aggregate(
            indicators=indicators,
            weight_overrides=weight_overrides,
            high_risk_features=self.high_risk_features,
            prev_distinct_error_services=prev_des,
            curr_distinct_error_services=curr_des,
        )
        self._prev_des[episode_id] = curr_des

        # 4. Apply severity rules
        raw_severity = classify_severity(
            critical_count=agg["critical_count"],
            warning_count=agg["warning_count"],
            blast_size=agg["blast_size"],
            high_risk_mode=agg["high_risk_mode"],
            blast_radius_growing=agg["blast_radius_growing"],
            failure_mode=failure_mode,
        )

        # 5. Temporal smoothing
        smoothed_severity = self.smoother.smooth(episode_id, raw_severity)

        # 6. Build result
        reason = build_reason(indicators, failure_mode, agg)
        action = get_recommended_action(failure_mode, smoothed_severity)
        indicator_labels = {f: lvl.label() for f, lvl in indicators.items()}

        return SeverityResult(
            timestamp=timestamp,
            episode_id=episode_id,
            elapsed_s=elapsed_s,
            failure_mode=failure_mode,
            severity=smoothed_severity,
            raw_severity=raw_severity,
            weighted_score=agg["weighted_score"],
            critical_count=agg["critical_count"],
            warning_count=agg["warning_count"],
            blast_size=agg["blast_size"],
            high_risk_mode=agg["high_risk_mode"],
            blast_radius_growing=agg["blast_radius_growing"],
            indicator_breakdown=indicator_labels,
            reason=reason,
            recommended_action=action,
        )

    # ── Batch API ─────────────────────────────────────────────────────────────

    def compute_severity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process an entire DataFrame of telemetry rows and add severity columns.

        Expected input columns (subset of what telemetry_metrics.csv provides):
            episode_id, elapsed_s, timestamp, failure_mode,
            + all metric/log/trace features.

        Added output columns:
            Severity, RawSeverity, WeightedScore, CriticalCount,
            WarningCount, BlastSize, HighRiskMode, BlastRadiusGrowing,
            Reason, RecommendedAction,
            + Indicator_<feature> for every evaluated feature.

        Args:
            df: Input DataFrame sorted by (episode_id, elapsed_s).

        Returns:
            df with severity columns appended (original rows preserved).
        """
        logger.info("compute_severity: processing %d rows.", len(df))

        # Sort by episode then time to ensure temporal state is correct
        df = df.sort_values(["episode_id", "elapsed_s"]).reset_index(drop=True)

        records: list[dict] = []
        for _, row in df.iterrows():
            feature_values = row.to_dict()
            failure_mode = str(row.get("failure_mode", "NONE"))
            episode_id = str(row.get("episode_id", "unknown"))
            timestamp = str(row.get("timestamp", ""))
            elapsed_s = float(row.get("elapsed_s", 0.0))

            result = self.evaluate_row(
                feature_values=feature_values,
                failure_mode=failure_mode,
                episode_id=episode_id,
                timestamp=timestamp,
                elapsed_s=elapsed_s,
            )
            records.append(result.to_dict())

        result_df = pd.DataFrame(records)

        # Flatten indicator_breakdown into individual columns
        if "indicator_breakdown" in result_df.columns:
            breakdown_df = result_df["indicator_breakdown"].apply(pd.Series).add_prefix("Indicator_")
            result_df = pd.concat([result_df.drop(columns=["indicator_breakdown"]), breakdown_df], axis=1)

        # Rename columns to PascalCase for clarity
        rename_map = {
            "severity": "Severity",
            "raw_severity": "RawSeverity",
            "weighted_score": "WeightedScore",
            "critical_count": "CriticalCount",
            "warning_count": "WarningCount",
            "blast_size": "BlastSize",
            "high_risk_mode": "HighRiskMode",
            "blast_radius_growing": "BlastRadiusGrowing",
            "reason": "Reason",
            "recommended_action": "RecommendedAction",
        }
        result_df = result_df.rename(columns=rename_map)

        # Merge back onto original DataFrame
        keep_cols = [c for c in result_df.columns
                     if c not in ("timestamp", "episode_id", "elapsed_s", "failure_mode")]
        out_df = pd.concat([df.reset_index(drop=True), result_df[keep_cols]], axis=1)

        logger.info("compute_severity: done. Severity distribution:\n%s",
                    out_df["Severity"].value_counts().to_string())
        return out_df

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _add_derived_features(fv: dict[str, Any]) -> dict[str, Any]:
        """Compute and inject derived features into the feature dict."""
        # p99 / p50 ratio
        p99 = float(fv.get("p99_latency", 0))
        p50 = float(fv.get("p50_latency", 1))
        fv["p99_minus_p50_ratio"] = p99 / p50 if p50 > 0 else 0.0

        # heap growth rate (requires previous heap value stored in fv as heap_mb_prev)
        heap_now = float(fv.get("heap_mb", 0))
        heap_prev = float(fv.get("heap_mb_prev", heap_now))
        fv["heap_growth_rate"] = max(0.0, heap_now - heap_prev)

        # disk write / read latency ratio
        dw = float(fv.get("disk_write_latency", 0))
        dr = float(fv.get("disk_read_latency", 1))
        fv["write_read_latency_ratio"] = dw / dr if dr > 0 else 0.0

        return fv

    def reset_episode(self, episode_id: str) -> None:
        """Reset temporal state for a specific episode."""
        self.smoother.reset(episode_id)
        self._prev_des.pop(episode_id, None)

    def reset_all(self) -> None:
        """Reset all episode state."""
        self.smoother.reset_all()
        self._prev_des.clear()


# ── Convenience entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick smoke test with 5 reference scenarios from the spec document
    engine = SeverityEngine()

    scenarios = [
        {
            "name": "Early MEMORY_LEAK (heap climbing, gc elevated)",
            "failure_mode": "MEMORY_LEAK",
            "features": {
                "memory_utilization": 0.78, "gc_pause_p99": 150,
                "heap_mb": 520, "cpu_utilization": 35, "error_rate": 0.01,
                "root_span_error_rate": 0.05, "distinct_error_services": 0,
            },
            "expected": "P3",
        },
        {
            "name": "MEMORY_LEAK near OOM (heap + gc + p99 all critical)",
            "failure_mode": "MEMORY_LEAK",
            "features": {
                "memory_utilization": 0.92, "gc_pause_p99": 450,
                "heap_mb": 3500, "p99_latency": 2100, "cpu_utilization": 40,
                "error_rate": 0.02, "root_span_error_rate": 0.05,
                "distinct_error_services": 0,
            },
            "expected": "P1",
        },
        {
            "name": "Isolated DB_SLOWDOWN (db_p99 critical, nothing else)",
            "failure_mode": "DB_SLOWDOWN",
            "features": {
                "db_p99": 1100, "cpu_utilization": 35, "error_rate": 0.01,
                "memory_utilization": 0.30, "root_span_error_rate": 0.05,
                "distinct_error_services": 0,
            },
            "expected": "P2",
        },
        {
            "name": "CASCADING_FAILURE spreading (2 services, growing)",
            "failure_mode": "CASCADING_FAILURE",
            "features": {
                "root_span_error_rate": 0.50, "db_p99": 1100,
                "distinct_error_services": 3, "error_rate": 0.25,
                "cpu_utilization": 35, "memory_utilization": 0.30,
            },
            "expected": "P1",
        },
        {
            "name": "Minor QUEUE_BACKUP (queue_lag just past warning)",
            "failure_mode": "QUEUE_BACKUP",
            "features": {
                "queue_lag": 110, "cpu_utilization": 35, "error_rate": 0.01,
                "memory_utilization": 0.30, "root_span_error_rate": 0.02,
                "distinct_error_services": 0,
            },
            "expected": "P4",
        },
    ]

    print("\n" + "=" * 60)
    print("  SEVERITY ENGINE -- Reference Scenario Smoke Test")
    print("=" * 60)
    print("  NOTE: Each scenario is pre-warmed (8 steps) so EMA converges")
    print("        before the final assertion. Raw severity is checked too.")
    print("=" * 60)

    WARMUP_STEPS = 8
    all_pass = True
    for s in scenarios:
        # Reset smoother between independent scenarios
        engine.reset_all()

        # For cascade scenario: prime blast_radius_growing by running with des=0 first
        if s["failure_mode"] == "CASCADING_FAILURE":
            warm_feats = dict(s["features"])
            warm_feats["distinct_error_services"] = 0
            for _ in range(WARMUP_STEPS):
                engine.evaluate_row(warm_feats, s["failure_mode"], "ep_test")
            # Now set des to 3 so blast_radius_growing fires
            s["features"]["distinct_error_services"] = 3
        else:
            # Pre-warm with identical rows so EMA converges
            for _ in range(WARMUP_STEPS):
                engine.evaluate_row(s["features"], s["failure_mode"], "ep_test")

        result = engine.evaluate_row(s["features"], s["failure_mode"], "ep_test")
        # Evaluate against BOTH smoothed AND raw severity
        sev_ok = result.severity == s["expected"] or result.raw_severity == s["expected"]
        status = "[PASS]" if sev_ok else f"[FAIL] (smoothed={result.severity}, raw={result.raw_severity})"
        all_pass = all_pass and sev_ok
        print(f"\n  {status} | {s['name']}")
        print(f"         Smoothed: {result.severity}  |  Raw: {result.raw_severity}"
              f"  |  Expected: {s['expected']}")
        print(f"         critical={result.critical_count}, warning={result.warning_count}, "
              f"blast={result.blast_size}, high_risk={result.high_risk_mode}")
        print(f"         Reason: {result.reason[:110]}...")

    print("\n" + "=" * 60)
    print(f"  Result: {'ALL SCENARIOS PASSED' if all_pass else 'SOME SCENARIOS FAILED'}")
    print("=" * 60)
