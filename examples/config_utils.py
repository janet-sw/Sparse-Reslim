"""Small, dependency-light helpers for loading experiment YAML files."""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any

import yaml


_ENV_VALUE = re.compile(
    r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-?(?P<default>[^}]*))?\}"
)


def _expand_environment(value: Any) -> Any:
    """Recursively expand ${NAME} and ${NAME:-default} in YAML values."""

    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name = match.group("name")
            default = match.group("default")
            if name in os.environ and os.environ[name]:
                return os.environ[name]
            if default is not None:
                return default
            return match.group(0)

        return _ENV_VALUE.sub(replace, value)
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    return value


def load_yaml_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a YAML mapping and expand environment-backed values."""

    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return _expand_environment(config)


def batch_size_per_rank(config: dict[str, Any], world_size: int) -> int:
    """Convert the paper's global batch size to a per-rank batch size."""

    global_batch_size = int(config["trainer"]["batch_size"])
    if global_batch_size < 1 or world_size < 1:
        raise ValueError("batch size and world size must be positive")
    if global_batch_size % world_size:
        raise ValueError(
            f"Global batch size {global_batch_size} is not divisible by "
            f"world size {world_size}"
        )
    return global_batch_size // world_size
