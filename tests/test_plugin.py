from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from harbor.models.environment_type import EnvironmentType
from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
)
from harbor.tasks.client import TaskDownloadResult
from harbor.trial.trial import Trial

import harbor_autoresearch.trial as trial_module
from harbor_autoresearch.plugin import TimedWindowPlugin


def _valid_job(
    tmp_path: Path,
    *,
    run_error: Exception | None = None,
    backend: EnvironmentType = EnvironmentType.DOCKER,
    storage_mb: int = 1,
) -> SimpleNamespace:
    task_dir = tmp_path / "task"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "tests").mkdir()
    (task_dir / "instruction.md").write_text("Improve the submitted program.")
    (task_dir / "environment" / "Dockerfile").write_text("FROM scratch\n")
    (task_dir / "tests" / "Dockerfile").write_text("FROM scratch\n")
    (task_dir / "tests" / "intermediate.sh").write_text("#!/bin/sh\n")
    (task_dir / "tests" / "test.sh").write_text("#!/bin/sh\n")
    (task_dir / "task.toml").write_text(
        'artifacts = ["/workspace/output"]\n'
        '[verifier]\nenvironment_mode = "separate"\n'
        f"[environment]\nstorage_mb = {storage_mb}\n"
    )
    task_config = TaskConfig(path=task_dir)
    trial_config = TrialConfig(
        task=task_config,
        trials_dir=tmp_path / "trials",
        agent=AgentConfig(
            name="autoresearchexam-terminus",
            model_name="provider/model",
            kwargs={
                "min_time_per_iteration": 0,
            },
        ),
        environment=EnvironmentConfig(type=backend),
    )
    download = TaskDownloadResult(
        path=task_dir,
        download_time_sec=0,
        cached=True,
        content_hash="abc",
    )

    async def run() -> object:
        if run_error is not None:
            raise run_error
        return object()

    return SimpleNamespace(
        config=SimpleNamespace(
            environment=trial_config.environment,
            source_jobs=[],
            is_regrade=False,
        ),
        _trial_configs=[trial_config],
        _task_download_results={task_config.get_task_id(): download},
        job_dir=tmp_path / "job",
        run=run,
    )


def test_plugin_defaults_iteration_minimum_to_zero() -> None:
    trial_config = SimpleNamespace(
        agent=SimpleNamespace(
            kwargs={"auto_summarization": False},
        )
    )
    plugin = TimedWindowPlugin(max_iterations=2, max_duration_seconds=120)

    window = plugin._window_config(trial_config)

    assert window.min_time_per_iteration == 0
    assert window.auto_summarize is False


@pytest.mark.asyncio
async def test_plugin_accepts_and_preserves_the_public_agent_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _valid_job(tmp_path)
    agent = job._trial_configs[0].agent
    agent.name = "autoresearchexam-terminus"
    agent.import_path = None
    monkeypatch.setattr(
        "harbor_autoresearch.plugin.EnvironmentFactory.run_preflight",
        lambda **_: None,
    )
    plugin = TimedWindowPlugin(max_iterations=2, max_duration_seconds=120)

    await plugin.on_job_start(job)
    try:
        assert agent.name == "autoresearchexam-terminus"
        assert agent.import_path == ("harbor_autoresearch.agent:AutoResearchExamAgent")
    finally:
        await plugin.on_job_end(object())


@pytest.mark.asyncio
async def test_plugin_forwards_all_public_agent_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _valid_job(tmp_path)
    agent = job._trial_configs[0].agent
    agent.name = "autoresearchexam-terminus"
    agent.import_path = None
    monkeypatch.setattr(
        "harbor_autoresearch.plugin.EnvironmentFactory.run_preflight",
        lambda **_: None,
    )
    plugin = TimedWindowPlugin(
        max_iterations=2,
        max_duration_seconds=120,
        max_turns=2_000,
        reasoning_effort="high",
        output_token_budget=12_345,
        auto_summarization=False,
        use_responses_api=True,
    )

    await plugin.on_job_start(job)
    try:
        assert agent.kwargs == {
            "min_time_per_iteration": 0,
            "max_turns": 2_000,
            "reasoning_effort": "high",
            "output_token_budget": 12_345,
            "auto_summarization": False,
            "use_responses_api": True,
        }
    finally:
        await plugin.on_job_end(object())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("setting", "agent_value", "plugin_value"),
    [
        ("min_time_per_iteration", 1, 2),
        ("max_turns", 5, 6),
        ("reasoning_effort", "medium", "high"),
        ("output_token_budget", 100, 200),
        ("auto_summarization", False, True),
        ("use_responses_api", False, True),
    ],
)
async def test_plugin_rejects_conflicting_agent_settings_without_overwriting_them(
    tmp_path: Path,
    setting: str,
    agent_value: object,
    plugin_value: object,
) -> None:
    job = _valid_job(tmp_path)
    agent = job._trial_configs[0].agent
    agent.name = "autoresearchexam-terminus"
    agent.import_path = None
    agent.kwargs = {setting: agent_value}
    plugin = TimedWindowPlugin(
        max_iterations=2,
        max_duration_seconds=120,
        **{setting: plugin_value},
    )

    with pytest.raises(ValueError, match=f"Conflicting {setting}"):
        await plugin.on_job_start(job)

    assert agent.name == "autoresearchexam-terminus"
    assert agent.import_path is None
    assert agent.kwargs == {setting: agent_value}


@pytest.mark.asyncio
async def test_failed_preflight_does_not_mutate_agent_configuration(
    tmp_path: Path,
) -> None:
    job = _valid_job(tmp_path)
    agent = job._trial_configs[0].agent
    task_dir = job._trial_configs[0].task.path
    assert task_dir is not None
    (task_dir / "tests" / "Dockerfile").unlink()
    plugin = TimedWindowPlugin(
        max_iterations=2,
        max_duration_seconds=120,
        max_turns=2_000,
    )

    with pytest.raises(ValueError, match="tests/Dockerfile"):
        await plugin.on_job_start(job)

    assert agent.name == "autoresearchexam-terminus"
    assert agent.import_path is None
    assert agent.kwargs == {"min_time_per_iteration": 0}


@pytest.mark.asyncio
async def test_failed_preflight_does_not_mutate_modal_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _valid_job(tmp_path, backend=EnvironmentType.MODAL)
    environment = job.config.environment

    def fail_preflight(**_: object) -> None:
        raise RuntimeError("preflight failed")

    monkeypatch.setattr(
        "harbor_autoresearch.plugin.EnvironmentFactory.run_preflight",
        fail_preflight,
    )
    plugin = TimedWindowPlugin(max_iterations=2, max_duration_seconds=120)

    with pytest.raises(RuntimeError, match="preflight failed"):
        await plugin.on_job_start(job)

    assert environment.kwargs == {}


def test_plugin_rejects_unknown_settings() -> None:
    with pytest.raises(TypeError, match="unexpected_keyword"):
        TimedWindowPlugin(
            max_iterations=2,
            max_duration_seconds=120,
            unexpected_keyword=True,
        )


@pytest.mark.asyncio
async def test_plugin_rejects_unsupported_backend_before_installing_factory() -> None:
    job = SimpleNamespace(
        config=SimpleNamespace(
            environment=SimpleNamespace(
                type=EnvironmentType.E2B,
                import_path=None,
            )
        )
    )
    plugin = TimedWindowPlugin(max_iterations=2, max_duration_seconds=120)

    with pytest.raises(ValueError, match="docker or modal"):
        await plugin.on_job_start(job)

    assert plugin.is_installed is False


@pytest.mark.asyncio
async def test_plugin_restores_trial_factory_when_job_run_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _valid_job(tmp_path, run_error=RuntimeError("job failed"))
    original = Trial.__dict__["create"]
    plugin = TimedWindowPlugin(max_iterations=2, max_duration_seconds=120)
    monkeypatch.setattr(
        "harbor_autoresearch.plugin.EnvironmentFactory.run_preflight",
        lambda **_: None,
    )

    await plugin.on_job_start(job)
    assert Trial.__dict__["create"] is not original

    with pytest.raises(RuntimeError, match="job failed"):
        await job.run()

    assert Trial.__dict__["create"] is original
    assert plugin.is_installed is False


@pytest.mark.asyncio
async def test_second_plugin_rejection_does_not_corrupt_active_plugin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_job = _valid_job(tmp_path / "first")
    second_job = _valid_job(tmp_path / "second")
    monkeypatch.setattr(
        "harbor_autoresearch.plugin.EnvironmentFactory.run_preflight",
        lambda **_: None,
    )
    first = TimedWindowPlugin(max_iterations=2, max_duration_seconds=120)
    second = TimedWindowPlugin(max_iterations=2, max_duration_seconds=120)
    original_create = Trial.__dict__["create"]

    await first.on_job_start(first_job)
    installed_create = Trial.__dict__["create"]
    installed_run = first_job.run
    try:
        with pytest.raises(RuntimeError, match="Only one"):
            await second.on_job_start(second_job)

        assert first.is_installed is True
        assert second.is_installed is False
        assert Trial.__dict__["create"] is installed_create
        assert first_job.run is installed_run
        assert second_job.run is not installed_run
    finally:
        await first.on_job_end(object())

    assert Trial.__dict__["create"] is original_create


@pytest.mark.asyncio
async def test_plugin_extends_modal_lifetime_on_job_and_derived_trials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _valid_job(tmp_path, backend=EnvironmentType.MODAL)
    monkeypatch.setattr(
        "harbor_autoresearch.plugin.EnvironmentFactory.run_preflight",
        lambda **_: None,
    )
    plugin = TimedWindowPlugin(max_iterations=2, max_duration_seconds=120)

    await plugin.on_job_start(job)

    # The global budget includes completed iterations. Only the final accepted
    # submission can overrun it while collecting artifacts and grading twice.
    expected = 120 + 2 * 600 + 600
    assert job.config.environment.kwargs["sandbox_timeout_secs"] == expected
    assert job._trial_configs[0].environment.kwargs["sandbox_timeout_secs"] == expected
    await plugin.on_job_end(object())


@pytest.mark.asyncio
async def test_plugin_rejects_short_explicit_modal_lifetime(
    tmp_path: Path,
) -> None:
    job = _valid_job(tmp_path, backend=EnvironmentType.MODAL)
    job.config.environment.kwargs["sandbox_timeout_secs"] = 1_919
    plugin = TimedWindowPlugin(max_iterations=2, max_duration_seconds=120)

    with pytest.raises(ValueError, match="at least 1920"):
        await plugin.on_job_start(job)

    assert plugin.is_installed is False


def test_modal_rejects_when_finalization_would_exceed_provider_maximum() -> None:
    environment = SimpleNamespace(kwargs={})
    plugin = TimedWindowPlugin(max_iterations=96, max_duration_seconds=28_800)

    with pytest.raises(ValueError, match="required Modal sandbox lifetime.*86400"):
        plugin._configure_modal_lifetime(
            [environment],
            required_lifetime=432_000,
        )

    assert "sandbox_timeout_secs" not in environment.kwargs


def test_modal_rejects_an_agent_budget_longer_than_one_sandbox() -> None:
    environment = SimpleNamespace(kwargs={})
    plugin = TimedWindowPlugin(max_iterations=1, max_duration_seconds=86_401)

    with pytest.raises(ValueError, match="wall-clock budget.*86400"):
        plugin._configure_modal_lifetime(
            [environment],
            required_lifetime=86_401,
        )


@pytest.mark.asyncio
async def test_modal_lifetime_uses_the_effective_verifier_timeout(
    tmp_path: Path,
) -> None:
    job = _valid_job(tmp_path, backend=EnvironmentType.MODAL)
    job._trial_configs[0].verifier.override_timeout_sec = 900
    # 120 + two 900s graders + 600s artifact allowance = 2520.
    job.config.environment.kwargs["sandbox_timeout_secs"] = 2_519
    plugin = TimedWindowPlugin(max_iterations=2, max_duration_seconds=120)

    with pytest.raises(ValueError, match="at least 2520"):
        await plugin.on_job_start(job)

    assert plugin.is_installed is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("backend", "available_mb", "required_mb"),
    [
        (EnvironmentType.DOCKER, 32_299, 32_300),
        (EnvironmentType.MODAL, 31_999, 32_000),
    ],
)
async def test_storage_preflight_checks_host_space_for_both_backends(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: EnvironmentType,
    available_mb: int,
    required_mb: int,
) -> None:
    job = _valid_job(tmp_path, backend=backend, storage_mb=300)
    monkeypatch.setattr(
        "harbor_autoresearch.plugin.EnvironmentFactory.run_preflight",
        lambda **_: None,
    )
    monkeypatch.setattr(
        "harbor_autoresearch.plugin.shutil.disk_usage",
        lambda _: SimpleNamespace(free=available_mb * 1024 * 1024),
    )
    plugin = TimedWindowPlugin(max_iterations=500, max_duration_seconds=120)

    with pytest.raises(ValueError, match=f"requires {required_mb} MiB"):
        await plugin.on_job_start(job)

    assert plugin.is_installed is False


@pytest.mark.asyncio
async def test_modal_host_estimate_does_not_multiply_remote_disk_by_iterations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _valid_job(
        tmp_path,
        backend=EnvironmentType.MODAL,
        storage_mb=300 * 1024,
    )
    observed_paths: list[Path] = []
    monkeypatch.setattr(
        "harbor_autoresearch.plugin.EnvironmentFactory.run_preflight",
        lambda **_: None,
    )

    def disk_usage(path: Path) -> SimpleNamespace:
        observed_paths.append(path)
        return SimpleNamespace(free=128 * 1024 * 1024)

    monkeypatch.setattr("harbor_autoresearch.plugin.shutil.disk_usage", disk_usage)
    plugin = TimedWindowPlugin(max_iterations=2, max_duration_seconds=120)

    await plugin.on_job_start(job)
    try:
        assert observed_paths
    finally:
        await plugin.on_job_end(object())


@pytest.mark.asyncio
async def test_plugin_timing_is_injected_before_factory_builds_the_trial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _valid_job(tmp_path)
    trial_config = job._trial_configs[0]
    trial_config.agent.kwargs.clear()
    download = job._task_download_results[trial_config.task.get_task_id()]
    created: list[object] = []

    class FakeTimedTrial:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.window = kwargs["timed_window_config"]
            created.append(self)

        async def _create_bridge(self) -> None:
            return None

    async def load_task(cls: type[Trial], config: object) -> tuple[object, object]:
        return object(), download

    monkeypatch.setattr(trial_module, "TimedWindowTrial", FakeTimedTrial)
    monkeypatch.setattr(Trial, "_load_task", classmethod(load_task))
    monkeypatch.setattr(Trial, "_resolve_agent_skills", lambda config: None)
    monkeypatch.setattr(
        "harbor_autoresearch.plugin.EnvironmentFactory.run_preflight",
        lambda **_: None,
    )
    plugin = TimedWindowPlugin(
        max_iterations=2,
        max_duration_seconds=120,
        min_time_per_iteration=0,
        auto_summarization=False,
    )
    await plugin.on_job_start(job)

    result = await Trial.create(trial_config)

    assert result is created[0]
    assert result.window.min_time_per_iteration == 0
    assert result.window.auto_summarize is False
    assert trial_config.agent.kwargs == {
        "min_time_per_iteration": 0,
        "auto_summarization": False,
    }
    await plugin.on_job_end(object())


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["Dockerfile", "intermediate.sh", "test.sh"])
async def test_plugin_rejects_missing_required_grader_files(
    tmp_path: Path,
    missing: str,
) -> None:
    job = _valid_job(tmp_path)
    task_dir = job._trial_configs[0].task.path
    assert task_dir is not None
    (task_dir / "tests" / missing).unlink()
    plugin = TimedWindowPlugin(max_iterations=2, max_duration_seconds=120)

    with pytest.raises(ValueError, match=f"tests/{missing}"):
        await plugin.on_job_start(job)

    assert plugin.is_installed is False


@pytest.mark.asyncio
async def test_plugin_rejects_a_prebuilt_verifier_image(tmp_path: Path) -> None:
    job = _valid_job(tmp_path)
    task_dir = job._trial_configs[0].task.path
    assert task_dir is not None
    with (task_dir / "task.toml").open("a") as handle:
        handle.write('\n[verifier.environment]\ndocker_image = "unsafe:latest"\n')
    plugin = TimedWindowPlugin(max_iterations=2, max_duration_seconds=120)

    with pytest.raises(ValueError, match="prebuilt verifier images"):
        await plugin.on_job_start(job)

    assert plugin.is_installed is False
