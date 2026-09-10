from __future__ import annotations

from pathlib import Path

from harbor.models.environment_type import EnvironmentType
from harbor.models.task.task import Task
from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
)
from harbor.tasks.client import TaskDownloadResult

from harbor_autoresearch.agent import TimedWindowAgent
from harbor_autoresearch.config import TimedWindowConfig
from harbor_autoresearch.trial import TimedWindowTrial


def test_trial_constructs_against_the_supported_harbor_release(
    tmp_path: Path,
) -> None:
    task_dir = tmp_path / "task"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "tests").mkdir()
    (task_dir / "instruction.md").write_text("Improve the output.\n")
    (task_dir / "environment" / "Dockerfile").write_text(
        "FROM python:3.12-slim\nWORKDIR /app\n"
    )
    (task_dir / "tests" / "Dockerfile").write_text(
        "FROM python:3.12-slim\nCOPY . /tests\n"
    )
    (task_dir / "tests" / "intermediate.sh").write_text("#!/bin/sh\nexit 0\n")
    (task_dir / "tests" / "test.sh").write_text("#!/bin/sh\nexit 0\n")
    (task_dir / "task.toml").write_text(
        'artifacts = ["/app/output"]\n'
        '[agent]\nuser = "root"\n'
        '[verifier]\nenvironment_mode = "separate"\n'
        '[environment]\nnetwork_mode = "public"\n'
    )

    task_config = TaskConfig(path=task_dir)
    trial_config = TrialConfig(
        task=task_config,
        trials_dir=tmp_path / "trials",
        agent=AgentConfig(
            name="harbor_autoresearch.agent:TimedWindowAgent",
            model_name="openai/test-model",
            kwargs={
                "max_turns": 2,
                "min_time_per_iteration": 0,
            },
        ),
        environment=EnvironmentConfig(type=EnvironmentType.DOCKER),
    )
    task = Task(task_dir)
    download = TaskDownloadResult(
        path=task_dir,
        download_time_sec=0,
        cached=True,
        content_hash="fixture",
    )

    trial = TimedWindowTrial(
        trial_config,
        _task=task,
        _task_download_result=download,
        timed_window_config=TimedWindowConfig(2, 120, 0),
    )

    try:
        assert isinstance(trial.agent, TimedWindowAgent)
        assert trial.timed_window_config.max_duration_seconds == 120
    finally:
        trial._close_logger_handler()
