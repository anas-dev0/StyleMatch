"""Read the app settings: a YAML file, with the data folder overridable by an env variable."""

import os
from pathlib import Path

import yaml

DEFAULT_CONFIG = "configs/app/app.yaml"


def load_settings(path: str | None = None, env=None) -> dict:
    """Settings dict. Env: `LOOKALIKE_APP_CONFIG` picks the file, `LOOKALIKE_DATA_DIR` the data."""
    env = os.environ if env is None else env
    path = path or env.get("LOOKALIKE_APP_CONFIG", DEFAULT_CONFIG)
    with open(path, encoding="utf-8") as f:
        settings = yaml.safe_load(f)
    settings["data_dir"] = Path(env.get("LOOKALIKE_DATA_DIR") or settings["data_dir"])
    qdrant = settings.setdefault("qdrant", {})
    qdrant["url"] = env.get("LOOKALIKE_QDRANT_URL") or qdrant.get("url")
    return settings
