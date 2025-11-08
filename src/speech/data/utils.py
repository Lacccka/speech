"""Utility helpers for dataset preparation scripts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

try:  # pragma: no cover - optional dependency
    import yaml
except Exception:  # pragma: no cover - allow working without yaml
    yaml = None


def validate_existing_path(path: Path) -> Path:
    """Ensure that a file or directory exists and return a resolved path."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved}")
    return resolved


def prepare_output_dir(path: Path) -> Path:
    """Create an output directory if it does not exist."""

    resolved = Path(path).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def load_config(config_path: Path) -> Mapping[str, Any]:
    """Load a configuration file supporting YAML and JSON formats."""

    path = validate_existing_path(config_path)
    suffix = path.suffix.lower()

    if suffix in {".yaml", ".yml"}:
        if yaml is None:  # pragma: no cover - safety net
            raise RuntimeError("PyYAML is required to load YAML configuration files")
        with path.open("r", encoding="utf8") as handle:
            return yaml.safe_load(handle) or {}

    if suffix == ".json":
        with path.open("r", encoding="utf8") as handle:
            return json.load(handle)

    raise ValueError(f"Unsupported configuration format: {path.suffix}")


def save_manifest(entries: Iterable[Mapping[str, Any]], output_path: Path) -> Path:
    """Persist a manifest file listing the prepared dataset entries."""

    output = prepare_output_dir(Path(output_path).parent) / Path(output_path).name
    with output.open("w", encoding="utf8") as handle:
        for entry in entries:
            handle.write(json.dumps(dict(entry), ensure_ascii=False) + "\n")
    return output
