"""
severity_engine/langgraph_node.py
===================================
LangGraph-compatible node wrapping the SeverityEngine.

The node receives an AgentState dict, extracts the current telemetry
row and predicted failure_mode, runs the engine, and writes the
SeverityResult back into the state under the key "severity_result".

Usage inside a LangGraph graph:
    from severity_engine.langgraph_node import SeverityNode
    graph.add_node("severity", SeverityNode())
    graph.add_edge("classifier", "severity")
"""
from __future__ import annotations

from typing import Any

from .severity_engine import SeverityEngine
from .utils import SeverityResult, get_logger

logger = get_logger("severity_engine.langgraph_node")


class SeverityNode:
    """
    LangGraph node that wraps SeverityEngine.

    Expected AgentState keys (input):
        "telemetry_row"   : dict[str, Any]  — current feature values
        "failure_mode"    : str             — XGBoost prediction
        "episode_id"      : str             — episode identifier
        "timestamp"       : str             — ISO timestamp
        "elapsed_s"       : float           — elapsed seconds

    Added AgentState keys (output):
        "severity_result" : dict            — SeverityResult.to_dict()
        "severity"        : str             — "P1"/"P2"/"P3"/"P4" shorthand
    """

    def __init__(self, yaml_path: str | None = None) -> None:
        self._engine = SeverityEngine(yaml_path=yaml_path)
        logger.info("SeverityNode ready.")

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Process one AgentState step and return the updated state.

        Args:
            state: LangGraph agent state dict.

        Returns:
            Updated state dict with severity_result and severity populated.
        """
        telemetry_row: dict = state.get("telemetry_row", {})
        failure_mode: str = state.get("failure_mode", "NONE")
        episode_id: str = state.get("episode_id", "unknown")
        timestamp: str = state.get("timestamp", "")
        elapsed_s: float = float(state.get("elapsed_s", 0.0))

        if not telemetry_row:
            logger.warning("SeverityNode received empty telemetry_row.")
            state["severity_result"] = {}
            state["severity"] = "P4"
            return state

        result: SeverityResult = self._engine.evaluate_row(
            feature_values=telemetry_row,
            failure_mode=failure_mode,
            episode_id=episode_id,
            timestamp=timestamp,
            elapsed_s=elapsed_s,
        )

        state["severity_result"] = result.to_dict()
        state["severity"] = result.severity
        return state
