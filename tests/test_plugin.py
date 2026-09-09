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
            name="harbor_autoresearch.agent:TimedWindowAgent",
            model_name="provider/model",
            kwargs={
                "min_time_per_iteration": 1,
                "max_time_per_iteration": 1,
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

    # 120s agent budget + two iterations of two 600s verifiers and a 600s
    # artifact allowance.
    expected = 120 + 2 * (2 * 600 + 600)
    assert job.config.environment.kwargs["sandbox_timeout_secs"] == expected
    assert job._trial_configs[0].environment.kwargs["sandbox_timeout_secs"] == expected
    await plugin.on_job_end(object())


@pytest.mark.asyncio
async def test_plugin_rejects_short_explicit_modal_lifetime(
    tmp_path: Path,
) -> None:
    job = _valid_job(tmp_path, backend=EnvironmentType.MODAL)
    job.config.environment.kwargs["sandbox_timeout_secs"] = 3_719
    plugin = TimedWindowPlugin(max_iterations=2, max_duration_seconds=120)

    with pytest.raises(ValueError, match="at least 3720"):
        await plugin.on_job_start(job)

    assert plugin.is_installed is False


def test_modal_lifetime_is_capped_at_the_provider_maximum() -> None:
    environment = SimpleNamespace(kwargs={})
    plugin = TimedWindowPlugin(max_iterations=96, max_duration_seconds=28_800)

    plugin._configure_modal_lifetime(
        [environment],
        required_lifetime=432_000,
    )

    assert environment.kwargs["sandbox_timeout_secs"] == 86_400


def test_modal_rejects_an_agent_budget_longer_than_one_sandbox() -> None:
    environment = SimpleNamespace(kwargs={})
    plugin = TimedWindowPlugin(max_iterations=1, max_duration_seconds=86_401)

    with pytest.raises(ValueError, match="agent time.*86400"):
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
    # 120 + 2 * (2 * 900 + 600) = 4920.
    job.config.environment.kwargs["sandbox_timeout_secs"] = 4_919
    plugin = TimedWindowPlugin(max_iterations=2, max_duration_seconds=120)

    with pytest.raises(ValueError, match="at least 4920"):
        await plugin.on_job_start(job)

    assert plugin.is_installed is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("backend", "available_mb", "required_mb"),
    [
        (EnvironmentType.DOCKER, 427, 428),
        (EnvironmentType.MODAL, 127, 128),
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
    plugin = TimedWindowPlugin(max_iterations=500, max_duration_seconds=120)

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
        min_time_per_iteration=1,
        max_time_per_iteration=1,
        auto_summarize=False,
    )
    await plugin.on_job_start(job)

    result = await Trial.create(trial_config)

    assert result is created[0]
    assert result.window.min_time_per_iteration == 1
    assert result.window.max_time_per_iteration == 1
    assert result.window.auto_summarize is False
    assert trial_config.agent.kwargs == {
        "min_time_per_iteration": 1,
        "max_time_per_iteration": 1,
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
