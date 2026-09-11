from __future__ import annotations

import os
import tomllib
from pathlib import Path

import pytest
from harbor.models.task.task import Task

from harbor_autoresearch.plugin import TimedWindowPlugin


def _exam_root() -> Path:
    configured = os.environ.get("AUTORESEARCH_EXAM_PATH")
    if not configured:
        pytest.skip("set AUTORESEARCH_EXAM_PATH to run the external contract check")
    return Path(configured)


def test_current_exam_tasks_satisfy_the_timed_window_contract() -> None:
    root = _exam_root()
    tasks = sorted(path.parent for path in root.glob("*/task.toml"))

    assert len(tasks) == 29
    for task in tasks:
        config = tomllib.loads((task / "task.toml").read_text())
        assert (task / "instruction.md").is_file()
        assert (task / "environment" / "Dockerfile").is_file()
        assert (task / "tests" / "Dockerfile").is_file()
        assert (task / "tests" / "intermediate.sh").is_file()
        assert (task / "tests" / "test.sh").is_file()
        assert config["verifier"]["environment_mode"] == "separate"
        assert config["artifacts"]
        TimedWindowPlugin._validate_task(Task(task))
