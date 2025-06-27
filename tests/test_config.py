import yaml

from src.config import load_config, DEFAULT_CONFIG


def test_load_config_defaults(tmp_path):
    cfg = load_config("nonexistent.yml")
    assert cfg == DEFAULT_CONFIG


def test_load_config_override(tmp_path):
    config_path = tmp_path / "cfg.yml"
    with open(config_path, "w") as f:
        yaml.dump({"data": {"raw": "custom.csv"}}, f)
    cfg = load_config(str(config_path))
    assert cfg["data"]["raw"] == "custom.csv"
    # other values fall back to defaults
    assert cfg["model"]["path"] == DEFAULT_CONFIG["model"]["path"]
