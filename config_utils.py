from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a top-level mapping: {path}")
    return data


def merge_config_into_args(args, config: Dict[str, Any], parser):
    defaults = {
        action.dest: action.default
        for action in parser._actions
        if getattr(action, "dest", None) not in (None, "help")
    }

    for key, value in config.items():
        if not hasattr(args, key):
            continue
        current_value = getattr(args, key)
        if current_value == defaults.get(key):
            setattr(args, key, value)
    return args
