from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
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
    gpus: int = 0,
    verifier_gpus: int | None = None,
) -> SimpleNamespace:
    task_dir = tmp_path / "task"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "tests").mkdir()
    (task_dir / "instruction.md").write_text("Improve the submitted program.")
    (task_dir / "environment" / "Dockerfile").write_text("FROM scratch\n")
    (task_dir / "tests" / "Dockerfile").write_text("FROM scratch\n")
    (task_dir / "tests" / "intermediate.sh").write_text("#!/bin/sh\n")
    (task_dir / "tests" / "test.sh").write_text("#!/bin/sh\n")
    verifier_environment = (
        ""
        if verifier_gpus is None
        else f"[verifier.environment]\ngpus = {verifier_gpus}\n"
    )
    (task_dir / "task.toml").write_text(
        'artifacts = ["/workspace/output"]\n'
        '[verifier]\nenvironment_mode = "separate"\n'
        f"{verifier_environment}"
        f"[environment]\nstorage_mb = {storage_mb}\ngpus = {gpus}\n"
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


@pytest.mark.asyncio
@pytest.mark.parametrize("example_index", [None, 0, 1])
async def test_default_plugin_creates_a_day_long_trial_with_repeated_experiments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    example_index: int | None,
) -> None:
    job = _valid_job(tmp_path)
    monkeypatch.setattr(
        "harbor_autoresearch.plugin.EnvironmentFactory.run_preflight",
        lambda **_: None,
    )
    monkeypatch.setattr(TimedWindowPlugin, "_validate_storage", lambda *args: None)
    plugin_kwargs = {}
    if example_index is not None:
        readme = (Path(__file__).parents[1] / "README.md").read_text()
        examples = [
            block
            for block in re.findall(r"```bash\n(.*?)```", readme, re.S)
            if block.startswith("uv run harbor run")
        ]
        plugin_kwargs = {
            key: int(value) if value.isdigit() else value
            for key, value in re.findall(r"--pk (\w+)=(\S+)", examples[example_index])
        }
        assert plugin_kwargs == {
            "max_iterations": 1_000,
            "max_duration_seconds": 86_400,
            "min_time_per_iteration": 0,
            "max_turns": 10_000,
            "max_tokens": 32_000,
            "reasoning_effort": "max",
        }
    plugin = TimedWindowPlugin(**plugin_kwargs)
    created = None

    await plugin.on_job_start(job)
    try:
        created = await Trial.create(job._trial_configs[0])
        assert created.timed_window_config.max_duration_seconds == 86_400
        assert created.timed_window_config.max_iterations == 1_000
        assert created.agent.remaining_turns == 10_000
        assert created.agent._reasoning_effort == "max"
        assert created.agent._llm_call_kwargs["max_tokens"] == 32_000
        assert created.agent._output_token_budget is None
        assert created.task.config.verifier.timeout_sec == 600
    finally:
        if created is not None:
            created._close_logger_handler()
        await plugin.on_job_end(object())


@pytest.mark.asyncio
async def test_default_modal_budget_is_rejected_without_silently_shortening_it(
    tmp_path: Path,
) -> None:
    job = _valid_job(tmp_path, backend=EnvironmentType.MODAL)
    plugin = TimedWindowPlugin()

    with pytest.raises(ValueError, match="required Modal sandbox lifetime.*86400"):
        await plugin.on_job_start(job)

    assert plugin.max_duration_seconds == 86_400
    assert plugin.is_installed is False


@pytest.mark.asyncio
async def test_docker_gpu_task_uses_staged_agent_and_verifier_compose_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _valid_job(tmp_path, gpus=2)
    trial_config = job._trial_configs[0]
    task_id = trial_config.task.get_task_id()
    source_path = job._task_download_results[task_id].path
    monkeypatch.setattr(
        "harbor_autoresearch.plugin.EnvironmentFactory.run_preflight",
        lambda **_: None,
    )
    plugin = TimedWindowPlugin(max_iterations=2, max_duration_seconds=120)
    created = None

    await plugin.on_job_start(job)
    staged_path = job._task_download_results[task_id].path
    try:
        created = await Trial.create(trial_config)

        assert created is not None
        assert staged_path != source_path
        assert trial_config.environment.override_gpus is None
        assert not (source_path / "environment" / "docker-compose.yaml").exists()
        assert not (source_path / "tests" / "docker-compose.yaml").exists()
        for directory in ("environment", "tests"):
            compose_path = staged_path / directory / "docker-compose.yaml"
            compose = yaml.safe_load(compose_path.read_text())
            assert compose["services"]["main"]["deploy"]["resources"]["reservations"][
                "devices"
            ] == [
                {
                    "driver": "nvidia",
                    "count": 2,
                    "capabilities": ["gpu"],
                }
            ]
        assert created.task.paths.task_dir == staged_path
        assert created.task.config.environment.gpus == 0
    finally:
        if created is not None:
            created._close_logger_handler()
        await plugin.on_job_end(object())

    assert not staged_path.exists()


@pytest.mark.asyncio
async def test_docker_gpu_task_uses_each_environment_gpu_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _valid_job(tmp_path, gpus=2, verifier_gpus=1)
    trial_config = job._trial_configs[0]
    task_id = trial_config.task.get_task_id()
    monkeypatch.setattr(
        "harbor_autoresearch.plugin.EnvironmentFactory.run_preflight",
        lambda **_: None,
    )
    plugin = TimedWindowPlugin(max_iterations=2, max_duration_seconds=120)

    await plugin.on_job_start(job)
    try:
        staged_path = job._task_download_results[task_id].path
        agent = yaml.safe_load(
            (staged_path / "environment" / "docker-compose.yaml").read_text()
        )
        verifier = yaml.safe_load(
            (staged_path / "tests" / "docker-compose.yaml").read_text()
        )

        assert (
            agent["services"]["main"]["deploy"]["resources"]["reservations"]["devices"][
                0
            ]["count"]
            == 2
        )
        assert (
            verifier["services"]["main"]["deploy"]["resources"]["reservations"][
                "devices"
            ][0]["count"]
            == 1
        )
    finally:
        await plugin.on_job_end(object())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("backend", "gpus"),
    [
        (EnvironmentType.DOCKER, 0),
        (EnvironmentType.MODAL, 2),
    ],
)
async def test_gpu_staging_does_not_change_cpu_or_modal_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: EnvironmentType,
    gpus: int,
) -> None:
    job = _valid_job(tmp_path, backend=backend, gpus=gpus)
    trial_config = job._trial_configs[0]
    task_id = trial_config.task.get_task_id()
    original_download = job._task_download_results[task_id]
    original_environment = trial_config.environment
    monkeypatch.setattr(
        "harbor_autoresearch.plugin.EnvironmentFactory.run_preflight",
        lambda **_: None,
    )
    plugin = TimedWindowPlugin(max_iterations=2, max_duration_seconds=120)

    await plugin.on_job_start(job)
    try:
        assert job._task_download_results[task_id] is original_download
        assert trial_config.environment is original_environment
        assert trial_config.environment.override_gpus is None
        assert not (
            original_download.path / "environment" / "docker-compose.yaml"
        ).exists()
        assert not (original_download.path / "tests" / "docker-compose.yaml").exists()
    finally:
        await plugin.on_job_end(object())


@pytest.mark.asyncio
async def test_modal_gpu_task_keeps_the_original_direct_gpu_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _valid_job(tmp_path, backend=EnvironmentType.MODAL, gpus=1)
    trial_config = job._trial_configs[0]
    task_id = trial_config.task.get_task_id()
    original_download = job._task_download_results[task_id]
    monkeypatch.setattr(
        "harbor_autoresearch.plugin.EnvironmentFactory.run_preflight",
        lambda **_: None,
    )
    plugin = TimedWindowPlugin(max_iterations=2, max_duration_seconds=120)
    created = None

    await plugin.on_job_start(job)
    try:
        created = await Trial.create(trial_config)

        assert created.task.paths.task_dir == original_download.path
        assert created.task.config.environment.gpus == 1
        assert created.agent_environment._compose_mode is False
        assert type(created.agent_environment._strategy).__name__ == "_ModalDirect"
    finally:
        if created is not None:
            created._close_logger_handler()
        await plugin.on_job_end(object())


@pytest.mark.asyncio
async def test_docker_gpu_staging_preserves_existing_compose_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _valid_job(tmp_path, gpus=2)
    trial_config = job._trial_configs[0]
    task_id = trial_config.task.get_task_id()
    source_path = job._task_download_results[task_id].path
    compose_path = source_path / "environment" / "docker-compose.yaml"
    original = {
        "services": {
            "main": {
                "environment": {"EXISTING": "yes"},
                "deploy": {
                    "resources": {
                        "reservations": {
                            "devices": [
                                {
                                    "driver": "custom",
                                    "count": 1,
                                    "capabilities": ["accelerator"],
                                },
                                {
                                    "driver": "nvidia",
                                    "count": 99,
                                    "capabilities": ["gpu"],
                                },
                            ]
                        }
                    }
                },
            },
            "sidecar": {"image": "example/sidecar:latest"},
        }
    }
    original_text = yaml.safe_dump(original, sort_keys=False)
    compose_path.write_text(original_text)
    monkeypatch.setattr(
        "harbor_autoresearch.plugin.EnvironmentFactory.run_preflight",
        lambda **_: None,
    )
    plugin = TimedWindowPlugin(max_iterations=2, max_duration_seconds=120)

    await plugin.on_job_start(job)
    try:
        staged_path = job._task_download_results[task_id].path
        staged = yaml.safe_load(
            (staged_path / "environment" / "docker-compose.yaml").read_text()
        )

        assert compose_path.read_text() == original_text
        assert staged["services"]["main"]["environment"] == {"EXISTING": "yes"}
        assert staged["services"]["sidecar"] == {"image": "example/sidecar:latest"}
        assert staged["services"]["main"]["deploy"]["resources"]["reservations"][
            "devices"
        ] == [
            {
                "driver": "custom",
                "count": 1,
                "capabilities": ["accelerator"],
            },
            {"driver": "nvidia", "count": 2, "capabilities": ["gpu"]},
        ]
    finally:
        await plugin.on_job_end(object())


@pytest.mark.asyncio
async def test_docker_gpu_staging_rejects_invalid_compose_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _valid_job(tmp_path, gpus=1)
    trial_config = job._trial_configs[0]
    task_id = trial_config.task.get_task_id()
    original_download = job._task_download_results[task_id]
    original_environment = trial_config.environment
    compose_path = original_download.path / "tests" / "docker-compose.yaml"
    compose_path.write_text("- not-a-compose-mapping\n")
    monkeypatch.setattr(
        "harbor_autoresearch.plugin.EnvironmentFactory.run_preflight",
        lambda **_: None,
    )
    plugin = TimedWindowPlugin(max_iterations=2, max_duration_seconds=120)

    with pytest.raises(ValueError, match="root must be a mapping"):
        await plugin.on_job_start(job)

    assert plugin.is_installed is False
    assert job._task_download_results[task_id] is original_download
    assert trial_config.environment is original_environment
    assert compose_path.read_text() == "- not-a-compose-mapping\n"


@pytest.mark.asyncio
async def test_docker_gpu_staging_rejects_a_compose_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _valid_job(tmp_path, gpus=1)
    trial_config = job._trial_configs[0]
    task_id = trial_config.task.get_task_id()
    original_download = job._task_download_results[task_id]
    environment_dir = original_download.path / "environment"
    (environment_dir / "Dockerfile").unlink()
    environment_dir.rmdir()
    external_environment = tmp_path / "external-environment"
    external_environment.mkdir()
    (external_environment / "Dockerfile").write_text("FROM scratch\n")
    external_compose = external_environment / "docker-compose.yaml"
    external_compose.write_text("services: {}\n")
    environment_dir.symlink_to(external_environment, target_is_directory=True)
    monkeypatch.setattr(
        "harbor_autoresearch.plugin.EnvironmentFactory.run_preflight",
        lambda **_: None,
    )
    plugin = TimedWindowPlugin(max_iterations=2, max_duration_seconds=120)

    with pytest.raises(ValueError, match="Compose paths cannot contain symlinks"):
        await plugin.on_job_start(job)

    assert job._task_download_results[task_id] is original_download
    assert external_compose.read_text() == "services: {}\n"


@pytest.mark.asyncio
async def test_docker_gpu_staging_rejects_a_nonzero_harbor_gpu_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _valid_job(tmp_path, gpus=1)
    trial_config = job._trial_configs[0]
    trial_config.environment.override_gpus = 2
    task_id = trial_config.task.get_task_id()
    original_download = job._task_download_results[task_id]
    monkeypatch.setattr(
        "harbor_autoresearch.plugin.EnvironmentFactory.run_preflight",
        lambda **_: None,
    )
    plugin = TimedWindowPlugin(max_iterations=2, max_duration_seconds=120)

    with pytest.raises(ValueError, match="remove the nonzero --override-gpus"):
        await plugin.on_job_start(job)

    assert plugin.is_installed is False
    assert job._task_download_results[task_id] is original_download
    assert trial_config.environment.override_gpus == 2
    assert not (original_download.path / "environment" / "docker-compose.yaml").exists()


@pytest.mark.asyncio
async def test_docker_gpu_staging_respects_an_explicit_zero_gpu_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _valid_job(tmp_path, gpus=1)
    trial_config = job._trial_configs[0]
    trial_config.environment.override_gpus = 0
    task_id = trial_config.task.get_task_id()
    original_download = job._task_download_results[task_id]
    monkeypatch.setattr(
        "harbor_autoresearch.plugin.EnvironmentFactory.run_preflight",
        lambda **_: None,
    )
    plugin = TimedWindowPlugin(max_iterations=2, max_duration_seconds=120)

    await plugin.on_job_start(job)
    try:
        assert job._task_download_results[task_id] is original_download
        assert not (
            original_download.path / "environment" / "docker-compose.yaml"
        ).exists()
        assert not (original_download.path / "tests" / "docker-compose.yaml").exists()
    finally:
        await plugin.on_job_end(object())


@pytest.mark.asyncio
async def test_docker_gpu_staging_is_restored_when_the_job_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _valid_job(tmp_path, gpus=1, run_error=RuntimeError("job failed"))
    trial_config = job._trial_configs[0]
    task_id = trial_config.task.get_task_id()
    original_download = job._task_download_results[task_id]
    original_environment = trial_config.environment
    monkeypatch.setattr(
        "harbor_autoresearch.plugin.EnvironmentFactory.run_preflight",
        lambda **_: None,
    )
    plugin = TimedWindowPlugin(max_iterations=2, max_duration_seconds=120)

    await plugin.on_job_start(job)
    staged_path = job._task_download_results[task_id].path
    with pytest.raises(RuntimeError, match="job failed"):
        await job.run()

    assert not staged_path.exists()
    assert job._task_download_results[task_id] is original_download
    assert trial_config.environment is original_environment
    assert plugin.is_installed is False


@pytest.mark.asyncio
async def test_mixed_docker_job_only_stages_the_gpu_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cpu_job = _valid_job(tmp_path / "cpu", gpus=0)
    gpu_job = _valid_job(tmp_path / "gpu", gpus=2)
    cpu_config = cpu_job._trial_configs[0]
    gpu_config = gpu_job._trial_configs[0]
    shared_environment = cpu_config.environment
    gpu_config.environment = shared_environment
    downloads = {
        **cpu_job._task_download_results,
        **gpu_job._task_download_results,
    }
    cpu_id = cpu_config.task.get_task_id()
    gpu_id = gpu_config.task.get_task_id()
    cpu_download = downloads[cpu_id]
    gpu_download = downloads[gpu_id]
    job = SimpleNamespace(
        config=SimpleNamespace(
            environment=shared_environment,
            source_jobs=[],
            is_regrade=False,
        ),
        _trial_configs=[cpu_config, gpu_config],
        _task_download_results=downloads,
        job_dir=tmp_path / "job",
        run=cpu_job.run,
    )
    monkeypatch.setattr(
        "harbor_autoresearch.plugin.EnvironmentFactory.run_preflight",
        lambda **_: None,
    )
    plugin = TimedWindowPlugin(max_iterations=2, max_duration_seconds=120)

    await plugin.on_job_start(job)
    try:
        assert downloads[cpu_id] is cpu_download
        assert downloads[gpu_id].path != gpu_download.path
        assert cpu_config.environment is shared_environment
        assert cpu_config.environment.override_gpus is None
        assert gpu_config.environment is shared_environment
        assert gpu_config.environment.override_gpus is None
    finally:
        await plugin.on_job_end(object())

    assert downloads[gpu_id] is gpu_download
    assert gpu_config.environment is shared_environment


def test_plugin_defaults_iteration_minimum_to_zero() -> None:
    trial_config = SimpleNamespace(
        agent=SimpleNamespace(
            kwargs={},
        )
    )
    plugin = TimedWindowPlugin(max_iterations=2, max_duration_seconds=120)

    window = plugin._window_config(trial_config)

    assert plugin.auto_summarization is True
    assert plugin.llm_backend is None
    assert plugin.max_turns == 10_000
    assert plugin.max_tokens == 32_000
    assert plugin.reasoning_effort == "max"
    assert plugin.output_token_budget is None
    assert window.min_time_per_iteration == 0
    assert window.auto_summarize is True


@pytest.mark.parametrize("value", [None, True, 1.5, "32000"])
def test_plugin_rejects_noninteger_response_limits(value: object) -> None:
    with pytest.raises(TypeError, match="max_tokens must be an integer"):
        TimedWindowPlugin(max_tokens=value)


@pytest.mark.parametrize("value", [0, -1])
def test_plugin_rejects_nonpositive_response_limits(value: int) -> None:
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        TimedWindowPlugin(max_tokens=value)


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
        assert "llm_backend" not in agent.kwargs
        assert agent.kwargs["max_turns"] == 10_000
        assert agent.kwargs["auto_summarization"] is True
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
        max_tokens=8_000,
        reasoning_effort="high",
        output_token_budget=12_345,
        auto_summarization=False,
        llm_backend="tinker",
    )

    await plugin.on_job_start(job)
    try:
        assert agent.kwargs == {
            "llm_backend": "tinker",
            "min_time_per_iteration": 0,
            "max_turns": 2_000,
            "max_tokens": 8_000,
            "reasoning_effort": "high",
            "output_token_budget": 12_345,
            "auto_summarization": False,
        }
    finally:
        await plugin.on_job_end(object())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("setting", "agent_value", "plugin_value"),
    [
        ("min_time_per_iteration", 1, 2),
        ("reasoning_effort", "medium", "high"),
        ("output_token_budget", 100, 200),
        ("auto_summarization", False, True),
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
async def test_plugin_rejects_conflicting_llm_backend_without_mutating_agent(
    tmp_path: Path,
) -> None:
    job = _valid_job(tmp_path)
    agent = job._trial_configs[0].agent
    agent.kwargs = {"llm_backend": "litellm"}
    plugin = TimedWindowPlugin(
        max_iterations=2,
        max_duration_seconds=120,
        llm_backend="tinker",
    )

    with pytest.raises(ValueError, match="Conflicting llm_backend"):
        await plugin.on_job_start(job)

    assert agent.kwargs == {"llm_backend": "litellm"}
    assert plugin.is_installed is False


@pytest.mark.asyncio
async def test_plugin_preserves_agent_llm_backend_when_plugin_does_not_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _valid_job(tmp_path)
    agent = job._trial_configs[0].agent
    agent.kwargs = {"llm_backend": "tinker"}
    monkeypatch.setattr(
        "harbor_autoresearch.plugin.EnvironmentFactory.run_preflight",
        lambda **_: None,
    )
    plugin = TimedWindowPlugin(max_iterations=2, max_duration_seconds=120)

    await plugin.on_job_start(job)
    try:
        assert agent.kwargs["llm_backend"] == "tinker"
    finally:
        await plugin.on_job_end(object())


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
async def test_max_turns_must_be_supplied_through_plugin_kwargs(
    tmp_path: Path,
) -> None:
    job = _valid_job(tmp_path)
    agent = job._trial_configs[0].agent
    agent.kwargs["max_turns"] = 2_000
    plugin = TimedWindowPlugin(max_iterations=2, max_duration_seconds=120)

    with pytest.raises(ValueError, match="max_turns.*plugin kwargs"):
        await plugin.on_job_start(job)

    assert plugin.is_installed is False


@pytest.mark.parametrize("llm_backend", ["custom", "Tinker", ""])
def test_plugin_rejects_unsupported_llm_backend(llm_backend: str) -> None:
    with pytest.raises(
        ValueError, match="llm_backend must be either litellm or tinker"
    ):
        TimedWindowPlugin(
            max_iterations=2,
            max_duration_seconds=120,
            llm_backend=llm_backend,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_name", ["max_episodes", "episodes"])
async def test_legacy_turn_limits_cannot_bypass_plugin_cap(
    tmp_path: Path,
    legacy_name: str,
) -> None:
    job = _valid_job(tmp_path)
    job._trial_configs[0].agent.kwargs[legacy_name] = 50_001
    plugin = TimedWindowPlugin(max_iterations=2, max_duration_seconds=120)

    with pytest.raises(ValueError, match=f"{legacy_name}.*max_turns"):
        await plugin.on_job_start(job)

    assert plugin.is_installed is False


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


def test_plugin_rejects_max_turns_above_public_limit_at_construction() -> None:
    with pytest.raises(ValueError, match="max_turns must be between 1 and 50000"):
        TimedWindowPlugin(
            max_iterations=2,
            max_duration_seconds=120,
            max_turns=50_001,
        )


def test_plugin_accepts_max_turns_at_public_limit() -> None:
    plugin = TimedWindowPlugin(
        max_iterations=2,
        max_duration_seconds=120,
        max_turns=50_000,
    )

    assert plugin.max_turns == 50_000


def test_plugin_rejects_nonpositive_max_turns_at_construction() -> None:
    with pytest.raises(ValueError, match="max_turns must be between 1 and 50000"):
        TimedWindowPlugin(
            max_iterations=2,
            max_duration_seconds=120,
            max_turns=0,
        )


@pytest.mark.parametrize("value", [None, True, 1.5, "50000"])
def test_plugin_rejects_noninteger_max_turns_at_construction(value: object) -> None:
    with pytest.raises(TypeError, match="max_turns must be an integer"):
        TimedWindowPlugin(
            max_iterations=2,
            max_duration_seconds=120,
            max_turns=value,
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
async def test_storage_preflight_caps_static_artifact_reserve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _valid_job(tmp_path, storage_mb=300)
    monkeypatch.setattr(
        "harbor_autoresearch.plugin.EnvironmentFactory.run_preflight",
        lambda **_: None,
    )
    monkeypatch.setattr(
        "harbor_autoresearch.plugin.shutil.disk_usage",
        lambda _: SimpleNamespace(free=32_300 * 1024 * 1024),
    )
    plugin = TimedWindowPlugin(max_iterations=5_000, max_duration_seconds=120)

    await plugin.on_job_start(job)
    try:
        assert plugin.is_installed is True
    finally:
        await plugin.on_job_end(object())


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
        "max_turns": 10_000,
        "max_tokens": 32_000,
        "reasoning_effort": "max",
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
