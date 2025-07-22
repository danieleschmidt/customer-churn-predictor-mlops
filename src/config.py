"""Configuration loader for churn prediction project."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import yaml

from .path_config import PathConfig, get_raw_data_path
from .constants import (
    get_model_path_constant,
    get_feature_columns_path_constant,
    get_run_id_path_constant,
    get_preprocessor_path_constant,
    get_processed_features_path_constant,
    get_processed_target_path_constant,
    # Backwards compatibility
    MODEL_PATH,
    FEATURE_COLUMNS_PATH,
    RUN_ID_PATH,
    PREPROCESSOR_PATH,
    PROCESSED_FEATURES_PATH,
    PROCESSED_TARGET_PATH,
)

def get_default_config(path_config: Optional[PathConfig] = None) -> Dict[str, Any]:
    """
    Get default configuration dictionary.
    
    Args:
        path_config: PathConfig instance to use for generating paths.
                    If None, uses environment-based defaults.
    
    Returns:
        Dictionary with default configuration values.
    """
    return {
        "data": {
            "raw": get_raw_data_path("customer_data.csv", config=path_config),
            "processed_features": get_processed_features_path_constant(config=path_config),
            "processed_target": get_processed_target_path_constant(config=path_config),
        },
        "model": {
            "path": get_model_path_constant(config=path_config),
            "feature_columns": get_feature_columns_path_constant(config=path_config),
            "run_id": get_run_id_path_constant(config=path_config),
            "preprocessor": get_preprocessor_path_constant(config=path_config),
        },
    }

# Backwards compatibility - static config using legacy constants
DEFAULT_CONFIG: Dict[str, Any] = {
    "data": {
        "raw": get_raw_data_path("customer_data.csv"),
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


def load_config(
    config_path: str = "config.yml", 
    path_config: Optional[PathConfig] = None
) -> Dict[str, Any]:
    """Load configuration from a YAML file if it exists.

    Parameters
    ----------
    config_path : str, optional
        Path to the YAML configuration file. Defaults to ``"config.yml"``.
    path_config : PathConfig, optional
        PathConfig instance to use for generating default paths.
        If None, uses legacy DEFAULT_CONFIG for backwards compatibility.

    Returns
    -------
    Dict[str, Any]
        Dictionary with configuration values. Missing keys fall back to
        default configuration.
    """
    # Choose base config based on whether path_config is provided
    if path_config is not None:
        base_cfg = get_default_config(path_config)
    else:
        base_cfg = DEFAULT_CONFIG
    
    if os.path.exists(config_path):
        with open(config_path) as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg = base_cfg.copy()
        for section, params in user_cfg.items():
            if isinstance(params, dict):
                base = cfg.setdefault(section, {})
                base.update(params)
            else:
                cfg[section] = params
        return cfg
    return base_cfg
