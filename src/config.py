"""Configuration loader for churn prediction project."""

from __future__ import annotations

import os
from typing import Any, Dict

import yaml

from .constants import (
    MODEL_PATH,
    FEATURE_COLUMNS_PATH,
    RUN_ID_PATH,
    PREPROCESSOR_PATH,
    PROCESSED_FEATURES_PATH,
    PROCESSED_TARGET_PATH,
)

DEFAULT_CONFIG: Dict[str, Any] = {
    "data": {
        "raw": "data/raw/customer_data.csv",
        "processed_features": PROCESSED_FEATURES_PATH,
        "processed_target": PROCESSED_TARGET_PATH,
    },
    "model": {
        "path": MODEL_PATH,
        "feature_columns": FEATURE_COLUMNS_PATH,
        "run_id": RUN_ID_PATH,
        "preprocessor": PREPROCESSOR_PATH,
    },
}


def load_config(config_path: str = "config.yml") -> Dict[str, Any]:
    """Load configuration from a YAML file if it exists.

    Parameters
    ----------
    config_path : str, optional
        Path to the YAML configuration file. Defaults to ``"config.yml"``.

    Returns
    -------
    Dict[str, Any]
        Dictionary with configuration values. Missing keys fall back to
        ``DEFAULT_CONFIG``.
    """
    if os.path.exists(config_path):
        with open(config_path) as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg = DEFAULT_CONFIG.copy()
        for section, params in user_cfg.items():
            if isinstance(params, dict):
                base = cfg.setdefault(section, {})
                base.update(params)
            else:
                cfg[section] = params
        return cfg
    return DEFAULT_CONFIG
