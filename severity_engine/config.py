"""
severity_engine/config.py
=========================
Loads severity_config/thresholds.yaml and exposes a single
validated Config object used by every other module.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_YAML = Path(__file__).parent.parent / "severity_config" / "thresholds.yaml"


@lru_cache(maxsize=1)
def load_config(yaml_path: str | None = None) -> dict[str, Any]:
    """
    Load and cache the YAML configuration file.

    Args:
        yaml_path: Explicit path to a YAML file.  If None, uses the
                   default ``severity_config/thresholds.yaml`` relative
                   to this file's parent.

    Returns:
        Parsed configuration dictionary.
    """
    path = Path(yaml_path) if yaml_path else _DEFAULT_YAML
    if not path.exists():
        raise FileNotFoundError(
            f"Threshold configuration not found at: {path}\n"
            "Ensure severity_config/thresholds.yaml exists."
        )
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    _validate(cfg)
    return cfg


def _validate(cfg: dict[str, Any]) -> None:
    """Minimal sanity checks on the loaded config."""
    required_top_keys = {"smoothing", "high_risk_features", "features", "failure_mode_weights"}
    missing = required_top_keys - set(cfg.keys())
    if missing:
        raise ValueError(f"Missing required top-level config keys: {missing}")

    sm = cfg["smoothing"]
    assert 0.0 < sm["alpha"] <= 1.0, "smoothing.alpha must be in (0, 1]"
    assert sm["hysteresis_steps"] >= 1, "smoothing.hysteresis_steps must be >= 1"


def get_feature_config(feature: str, cfg: dict | None = None) -> dict[str, Any]:
    """Return the threshold sub-dict for a single feature."""
    cfg = cfg or load_config()
    features = cfg.get("features", {})
    if feature not in features:
        raise KeyError(f"Feature '{feature}' not found in threshold config.")
    return features[feature]


def get_failure_mode_weights(failure_mode: str, cfg: dict | None = None) -> dict[str, float]:
    """Return weight overrides for the given failure mode (empty dict if none)."""
    cfg = cfg or load_config()
    return cfg.get("failure_mode_weights", {}).get(failure_mode, {})


def get_smoothing_params(cfg: dict | None = None) -> dict[str, Any]:
    """Return the smoothing sub-dict."""
    cfg = cfg or load_config()
    return cfg["smoothing"]


def get_high_risk_features(cfg: dict | None = None) -> list[str]:
    """Return the list of high-risk feature names."""
    cfg = cfg or load_config()
    return cfg.get("high_risk_features", [])
