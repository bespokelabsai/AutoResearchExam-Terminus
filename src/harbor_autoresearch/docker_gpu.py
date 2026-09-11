"""Prepare Docker-only task copies with NVIDIA GPU reservations."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import yaml

_COMPOSE_PATHS = (
    Path("environment/docker-compose.yaml"),
    Path("tests/docker-compose.yaml"),
)


def stage_docker_gpu_task(
    source: Path,
    destination: Path,
    *,
    agent_gpu_count: int,
    verifier_gpu_count: int,
) -> Path:
    """Copy one task and reserve each Docker environment's declared GPUs."""
    reservations = (
        (_COMPOSE_PATHS[0], agent_gpu_count),
        (_COMPOSE_PATHS[1], verifier_gpu_count),
    )
    if all(gpu_count < 1 for _, gpu_count in reservations):
        raise ValueError("at least one GPU count must be positive")

    for relative_path, gpu_count in reservations:
        if gpu_count < 1:
            continue
        current = source
        for part in relative_path.parts:
            current /= part
            if current.is_symlink():
                raise ValueError(
                    f"Docker Compose paths cannot contain symlinks: {current}"
                )

    shutil.copytree(source, destination, symlinks=True)
    for relative_path, gpu_count in reservations:
        if gpu_count < 1:
            continue
        _merge_gpu_reservation(destination / relative_path, gpu_count=gpu_count)
    return destination


def _merge_gpu_reservation(path: Path, *, gpu_count: int) -> None:
    document: dict[str, Any]
    if path.is_symlink():
        raise ValueError(f"Docker Compose paths cannot contain symlinks: {path}")
    if path.exists():
        try:
            loaded = yaml.safe_load(path.read_text())
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid Docker Compose YAML: {path}") from exc
        if loaded is None:
            document = {}
        elif isinstance(loaded, dict):
            document = loaded
        else:
            raise ValueError(f"Docker Compose root must be a mapping: {path}")
    else:
        document = {}

    current: dict[str, Any] = document
    for key in ("services", "main", "deploy", "resources", "reservations"):
        value = current.setdefault(key, {})
        if not isinstance(value, dict):
            raise ValueError(f"Docker Compose {key!r} must be a mapping: {path}")
        current = value

    devices = current.setdefault("devices", [])
    if not isinstance(devices, list):
        raise ValueError(f"Docker Compose 'devices' must be a list: {path}")
    if any(not isinstance(device, dict) for device in devices):
        raise ValueError(f"Docker Compose device entries must be mappings: {path}")

    current["devices"] = [
        device
        for device in devices
        if not (
            device.get("driver") == "nvidia"
            and "gpu" in (device.get("capabilities") or [])
        )
    ]
    current["devices"].append(
        {
            "driver": "nvidia",
            "count": gpu_count,
            "capabilities": ["gpu"],
        }
    )
    path.write_text(yaml.safe_dump(document, sort_keys=False))
