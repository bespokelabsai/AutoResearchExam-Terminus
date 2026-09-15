"""Harbor job plugin that installs the timed-window trial locally."""

from __future__ import annotations

import math
import shutil
import tempfile
import tomllib
from copy import deepcopy
from pathlib import Path
from typing import Any

from harbor.environments.factory import EnvironmentFactory
from harbor.models.task.config import TaskOS
from harbor.models.task.task import Task
from harbor.models.task.verifier_mode import (
    VerifierEnvironmentMode,
    resolve_effective_verifier_env_config,
    resolve_task_verifier_mode,
)
from harbor.publisher.packager import Packager
from harbor.tasks.client import TaskDownloadResult
from harbor.trial.trial import Trial

from .config import (
    DEFAULT_MAX_DURATION_SECONDS,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_TURNS,
    TimedWindowConfig,
    validate_backend,
)
from .docker_gpu import stage_docker_gpu_task

_AGENT_NAME = "autoresearchexam-terminus"
_AGENT_IMPORT_PATH = "harbor_autoresearch.agent:AutoResearchExamAgent"
_LLM_BACKENDS = {"litellm", "tinker"}
_MODAL_STORAGE_LIMIT_MB = 512 * 1024
_MODAL_MAX_SANDBOX_TIMEOUT_SECONDS = 24 * 60 * 60
# The final accepted submission can outlive the research deadline while its two
# verifiers finish. Reserve both timeouts plus artifact collection and archiving.
_ARTIFACT_ALLOWANCE_SECONDS = 10 * 60
# Static preflight reserves a small host allowance per retained iteration, up
# to the previous 500-iteration maximum. Artifact size is unknowable before
# execution, so the trial also checks free space before every snapshot.
_RETAINED_ITERATION_ALLOWANCE_MB = 64
_MAX_RETAINED_ARTIFACT_PREFLIGHT_MB = 500 * _RETAINED_ITERATION_ALLOWANCE_MB
_active_plugin: TimedWindowPlugin | None = None


class TimedWindowPlugin:
    """Install the timed-window trial for the duration of one Harbor job."""

    def __init__(
        self,
        *,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        max_duration_seconds: int = DEFAULT_MAX_DURATION_SECONDS,
        min_time_per_iteration: int | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        reasoning_effort: str | None = "max",
        output_token_budget: int | None = None,
        auto_summarization: bool = True,
        llm_backend: str | None = None,
    ) -> None:
        self.max_iterations = max_iterations
        self.max_duration_seconds = max_duration_seconds
        self.min_time_per_iteration = min_time_per_iteration
        if isinstance(max_turns, bool) or not isinstance(max_turns, int):
            raise TypeError("max_turns must be an integer")
        if not 1 <= max_turns <= 50_000:
            raise ValueError("max_turns must be between 1 and 50000")
        self.max_turns = max_turns
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            raise TypeError("max_tokens must be an integer")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.output_token_budget = output_token_budget
        self.auto_summarization = auto_summarization
        if llm_backend is not None and llm_backend not in _LLM_BACKENDS:
            raise ValueError("llm_backend must be either litellm or tinker")
        self.llm_backend = llm_backend
        self._original_create: Any = None
        self._installed_create: Any = None
        self._job: Any = None
        self._original_job_run: Any = None
        self._installed_job_run: Any = None
        self._docker_gpu_staging: tempfile.TemporaryDirectory[str] | None = None
        self._staged_downloads: dict[Any, TaskDownloadResult] = {}
        self._original_downloads: list[tuple[dict[Any, Any], Any, Any]] = []

    @property
    def is_installed(self) -> bool:
        return self._original_create is not None

    async def on_job_start(self, job: Any) -> None:
        """Validate all inputs before installing the process-local factory."""
        global _active_plugin

        if self.is_installed:
            raise RuntimeError("TimedWindowPlugin was attached more than once")
        if _active_plugin is not None:
            raise RuntimeError(
                "Only one timed-window job can run in a Python process at a time"
            )

        backend = self._backend_name(job.config.environment)
        validate_backend(backend)
        if getattr(job.config, "is_regrade", False) or getattr(
            job.config, "source_jobs", ()
        ):
            raise ValueError("Timed-window runs do not support regrade jobs")

        trial_configs = list(getattr(job, "_trial_configs", ()))
        if not trial_configs:
            raise ValueError("Timed-window jobs must contain at least one trial")
        prepared_agents: list[tuple[Any, Any]] = []
        trial_tasks: list[tuple[Any, Task, TimedWindowConfig]] = []
        for config in trial_configs:
            prepared_agent = deepcopy(config.agent)
            self._normalize_agent(prepared_agent)
            self._apply_agent_overrides(prepared_agent)
            self._validate_trial_config(config, agent=prepared_agent)
            download = job._task_download_results[config.task.get_task_id()]
            task = Task(
                task_dir=download.path,
                extra_instruction_paths=config.extra_instruction_paths,
                extra_instructions=config.extra_instructions,
                disable_verification=config.verifier.disable,
            )
            self._validate_task(task)
            window = self._window_config(config, agent=prepared_agent)
            prepared_agents.append((config.agent, prepared_agent))
            trial_tasks.append((config, task, window))

        prepared_environments: list[tuple[Any, Any]] = []
        if backend == "modal":
            required_lifetime = max(
                self._modal_minimum_lifetime(config, task, window)
                for config, task, window in trial_tasks
            )
            environments = [
                job.config.environment,
                *(config.environment for config in trial_configs),
            ]
            prepared_environment_values = [
                deepcopy(environment) for environment in environments
            ]
            self._configure_modal_lifetime(
                prepared_environment_values,
                required_lifetime=required_lifetime,
            )
            prepared_environments = list(
                zip(environments, prepared_environment_values, strict=True)
            )

        EnvironmentFactory.run_preflight(
            type=job.config.environment.type,
            import_path=job.config.environment.import_path,
        )
        self._validate_storage(backend, trial_tasks, Path(job.job_dir))

        docker_gpu_preparation = None
        if backend == "docker":
            docker_gpu_preparation = self._prepare_docker_gpu_tasks(
                job,
                trial_tasks,
            )

        original_create = Trial.__dict__["create"]
        plugin = self

        async def create_timed_trial(cls: type[Trial], config: Any) -> Any:
            from .trial import TimedWindowTrial

            cls._resolve_agent_skills(config)
            download_result = plugin._staged_downloads.get(config.task.get_task_id())
            if download_result is None:
                task, download_result = await cls._load_task(config)
            else:
                task = Task(
                    task_dir=download_result.path,
                    extra_instruction_paths=config.extra_instruction_paths,
                    extra_instructions=config.extra_instructions,
                    disable_verification=config.verifier.disable,
                )
                # Harbor 0.22 rejects task-level Docker GPU allocation before
                # Compose starts. The staged Compose files own that allocation.
                task.config.environment.gpus = 0
                if task.config.verifier.environment is not None:
                    task.config.verifier.environment.gpus = 0
            trial = TimedWindowTrial(
                config,
                _task=task,
                _task_download_result=download_result,
                timed_window_config=plugin._window_config(config),
            )
            await trial._create_bridge()
            return trial

        installed_create = classmethod(create_timed_trial)
        self._original_create = original_create
        self._installed_create = installed_create
        self._job = job
        self._original_job_run = job.run
        if docker_gpu_preparation is not None:
            staging, staged_downloads = docker_gpu_preparation
            self._docker_gpu_staging = staging
            self._staged_downloads = staged_downloads

        async def run_with_restoration() -> Any:
            original_job_run = self._original_job_run
            try:
                return await original_job_run()
            finally:
                self._restore()

        self._installed_job_run = run_with_restoration
        try:
            for agent, prepared_agent in prepared_agents:
                agent.name = prepared_agent.name
                agent.import_path = prepared_agent.import_path
                agent.kwargs = prepared_agent.kwargs
            for environment, prepared_environment in prepared_environments:
                environment.kwargs = prepared_environment.kwargs
            if docker_gpu_preparation is not None:
                for task_id, staged_download in self._staged_downloads.items():
                    original_download = job._task_download_results[task_id]
                    self._original_downloads.append(
                        (job._task_download_results, task_id, original_download)
                    )
                    job._task_download_results[task_id] = staged_download
            _active_plugin = self
            Trial.create = installed_create  # ty: ignore[invalid-assignment]
            job.run = run_with_restoration
        except BaseException:
            self._restore()
            raise

    async def on_job_end(self, job_result: Any) -> None:
        """Restore Harbor even when finalization is invoked more than once."""
        self._restore()

    def _restore(self) -> None:
        global _active_plugin

        if (
            self._original_create is not None
            and Trial.__dict__.get("create") is self._installed_create
        ):
            Trial.create = self._original_create
        if (
            self._job is not None
            and getattr(self._job, "run", None) is self._installed_job_run
        ):
            self._job.run = self._original_job_run
        if _active_plugin is self:
            _active_plugin = None
        for downloads, task_id, download in reversed(self._original_downloads):
            downloads[task_id] = download
        if self._docker_gpu_staging is not None:
            self._docker_gpu_staging.cleanup()
        self._original_create = None
        self._installed_create = None
        self._job = None
        self._original_job_run = None
        self._installed_job_run = None
        self._docker_gpu_staging = None
        self._staged_downloads = {}
        self._original_downloads = []

    @staticmethod
    def _prepare_docker_gpu_tasks(
        job: Any,
        trial_tasks: list[tuple[Any, Task, TimedWindowConfig]],
    ) -> tuple[tempfile.TemporaryDirectory[str], dict[Any, TaskDownloadResult]] | None:
        gpu_tasks: dict[Any, tuple[TaskDownloadResult, int, int]] = {}
        for config, task, _ in trial_tasks:
            agent_gpu_count = int(task.config.environment.gpus or 0)
            verifier_environment = resolve_effective_verifier_env_config(
                task.config, None
            )
            verifier_gpu_count = int(
                (verifier_environment.gpus or 0)
                if verifier_environment is not None
                else 0
            )
            if agent_gpu_count == 0 and verifier_gpu_count == 0:
                continue
            if config.environment.override_gpus == 0:
                continue
            if config.environment.override_gpus is not None:
                raise ValueError(
                    "Docker GPU tasks manage GPU allocation through task Compose "
                    "files; remove the nonzero --override-gpus value"
                )
            task_id = config.task.get_task_id()
            download = job._task_download_results[task_id]
            gpu_tasks[task_id] = (
                download,
                agent_gpu_count,
                verifier_gpu_count,
            )

        if not gpu_tasks:
            return None

        staging = tempfile.TemporaryDirectory(prefix="harbor-autoresearch-docker-gpu-")
        staging_path = Path(staging.name)
        staged_downloads: dict[Any, TaskDownloadResult] = {}
        try:
            for index, (
                task_id,
                (download, agent_gpu_count, verifier_gpu_count),
            ) in enumerate(gpu_tasks.items()):
                destination = staging_path / f"{index:04d}-{download.path.name}"
                stage_docker_gpu_task(
                    download.path,
                    destination,
                    agent_gpu_count=agent_gpu_count,
                    verifier_gpu_count=verifier_gpu_count,
                )
                content_hash, _ = Packager.compute_content_hash(destination)
                staged_downloads[task_id] = download.model_copy(
                    update={
                        "path": destination,
                        "cached": False,
                        "content_hash": content_hash,
                    }
                )
        except BaseException:
            staging.cleanup()
            raise
        return staging, staged_downloads

    def _window_config(
        self,
        trial_config: Any,
        *,
        agent: Any | None = None,
    ) -> TimedWindowConfig:
        kwargs = (agent or trial_config.agent).kwargs
        minimum = self.min_time_per_iteration
        if minimum is None:
            minimum = kwargs.get("min_time_per_iteration", 0)
        auto_summarize = kwargs.get("auto_summarization", True)
        return TimedWindowConfig(
            max_iterations=self.max_iterations,
            max_duration_seconds=self.max_duration_seconds,
            min_time_per_iteration=minimum,
            auto_summarize=auto_summarize,
        )

    def _apply_agent_overrides(self, agent: Any) -> None:
        kwargs = agent.kwargs
        if "max_turns" in kwargs:
            raise ValueError("max_turns must be supplied through plugin kwargs")
        for legacy_name in ("max_episodes", "episodes"):
            if legacy_name in kwargs:
                raise ValueError(
                    f"{legacy_name} is not supported; use max_turns through "
                    "plugin kwargs"
                )
        overrides = {
            "llm_backend": self.llm_backend,
            "min_time_per_iteration": self.min_time_per_iteration,
            "max_turns": self.max_turns,
            "max_tokens": self.max_tokens,
            "reasoning_effort": self.reasoning_effort,
            "output_token_budget": self.output_token_budget,
            "auto_summarization": self.auto_summarization,
        }
        for name, value in overrides.items():
            if value is None:
                continue
            existing = kwargs.get(name)
            if existing is not None and existing != value:
                raise ValueError(
                    f"Conflicting {name} values in agent and plugin kwargs"
                )
            kwargs[name] = value

    @staticmethod
    def _normalize_agent(agent: Any) -> None:
        """Resolve the public shorthand while preserving its result display name."""
        configured_name = agent.name
        configured_import_path = agent.import_path
        if configured_name in {_AGENT_NAME, _AGENT_IMPORT_PATH}:
            if configured_import_path not in {None, _AGENT_IMPORT_PATH}:
                raise ValueError(
                    f"{_AGENT_NAME} cannot be combined with another import path"
                )
            agent.name = _AGENT_NAME
            agent.import_path = _AGENT_IMPORT_PATH
        elif configured_import_path == _AGENT_IMPORT_PATH and configured_name is None:
            agent.name = _AGENT_NAME

    @staticmethod
    def _backend_name(environment: Any) -> str:
        if environment.import_path is not None:
            return "custom"
        value = environment.type
        return str(getattr(value, "value", value))

    def _configure_modal_lifetime(
        self,
        environments: list[Any],
        *,
        required_lifetime: int,
    ) -> None:
        if self.max_duration_seconds > _MODAL_MAX_SANDBOX_TIMEOUT_SECONDS:
            raise ValueError(
                "Modal wall-clock budget cannot exceed its 86400-second sandbox limit"
            )
        if required_lifetime > _MODAL_MAX_SANDBOX_TIMEOUT_SECONDS:
            raise ValueError(
                "The required Modal sandbox lifetime exceeds 86400 seconds; "
                "reduce the wall-clock budget or verifier timeout"
            )
        supported_lifetime = required_lifetime
        for environment in environments:
            configured = environment.kwargs.get("sandbox_timeout_secs")
            if configured is None:
                environment.kwargs["sandbox_timeout_secs"] = supported_lifetime
                continue
            if (
                isinstance(configured, bool)
                or not isinstance(configured, (int, float))
                or not math.isfinite(configured)
            ):
                raise ValueError("Modal sandbox_timeout_secs must be a finite number")
            if configured > _MODAL_MAX_SANDBOX_TIMEOUT_SECONDS:
                raise ValueError(
                    "Modal sandbox_timeout_secs cannot exceed 86400 seconds"
                )
            if configured < supported_lifetime:
                raise ValueError(
                    "Modal sandbox_timeout_secs must be at least "
                    f"{supported_lifetime} seconds"
                )

    @staticmethod
    def _modal_minimum_lifetime(
        trial_config: Any,
        task: Task,
        window: TimedWindowConfig,
    ) -> int:
        base_timeout = (
            trial_config.verifier.override_timeout_sec
            or task.config.verifier.timeout_sec
        )
        max_timeout = trial_config.verifier.max_timeout_sec
        multiplier = trial_config.verifier_timeout_multiplier
        if multiplier is None:
            multiplier = trial_config.timeout_multiplier
        verifier_timeout = (
            min(
                base_timeout,
                max_timeout if max_timeout is not None else math.inf,
            )
            * multiplier
        )
        finalization_allowance = 2 * verifier_timeout + _ARTIFACT_ALLOWANCE_SECONDS
        return math.ceil(window.max_duration_seconds + finalization_allowance)

    @staticmethod
    def _validate_trial_config(config: Any, *, agent: Any | None = None) -> None:
        validate_backend(TimedWindowPlugin._backend_name(config.environment))
        if config.source_trial is not None:
            raise ValueError("Timed-window runs do not support regrade trials")
        if config.user_agent is not None:
            raise ValueError("Timed-window runs do not support simulated users")
        if config.install_only:
            raise ValueError("Timed-window runs do not support install-only mode")
        if config.verifier.disable:
            raise ValueError("Timed-window runs require verification")
        if config.verifier.import_path is not None:
            raise ValueError("Timed-window runs require Harbor's standard verifier")
        if config.artifacts:
            raise ValueError("Artifacts must be declared by the task")

        agent = agent or config.agent
        if agent.name != _AGENT_NAME or agent.import_path != _AGENT_IMPORT_PATH:
            raise ValueError(f"Timed-window runs require agent {_AGENT_NAME!r}")
        if not agent.model_name:
            raise ValueError("Timed-window runs require a model")

    @staticmethod
    def _validate_task(task: Task) -> None:
        if task.has_steps:
            raise ValueError("Timed-window runs require single-step tasks")
        if task.config.environment.os != TaskOS.LINUX:
            raise ValueError("Timed-window runs require Linux tasks")
        if resolve_task_verifier_mode(task.config) != VerifierEnvironmentMode.SEPARATE:
            raise ValueError("Timed-window runs require a separate verifier")
        verifier_environment = resolve_effective_verifier_env_config(task.config, None)
        if verifier_environment is None:
            raise ValueError("Timed-window runs require a verifier environment")
        if verifier_environment.docker_image:
            raise ValueError(
                "Timed-window runs require a build-based verifier; "
                "prebuilt verifier images are not supported"
            )
        if not task.config.artifacts:
            raise ValueError("Timed-window tasks must declare artifacts")

        tests_dir = task.paths.tests_dir
        required = ("Dockerfile", "intermediate.sh", "test.sh")
        for name in required:
            path = tests_dir / name
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"Timed-window tasks require tests/{name}")
        if any(path.is_symlink() for path in tests_dir.rglob("*")):
            raise ValueError("Timed-window verifier files cannot be symlinks")

    @classmethod
    def _validate_storage(
        cls,
        backend: str,
        trial_tasks: list[tuple[Any, Task, TimedWindowConfig]],
        job_dir: Path,
    ) -> None:
        if backend == "modal" and any(
            cls._task_storage_mb(task) > _MODAL_STORAGE_LIMIT_MB
            for _, task, _ in trial_tasks
        ):
            raise ValueError("Task storage requirement exceeds Modal's sandbox quota")

        required_mb = 0
        for _, task, window in trial_tasks:
            if backend == "docker":
                # A task environment exists once, not once per iteration.
                required_mb += cls._task_storage_mb(task)
            possible_iterations = window.max_iterations
            if window.min_time_per_iteration > 0:
                possible_iterations = min(
                    possible_iterations,
                    math.ceil(
                        window.max_duration_seconds
                        / (window.min_time_per_iteration * 60)
                    ),
                )
            retained_artifact_mb = (
                possible_iterations * _RETAINED_ITERATION_ALLOWANCE_MB
            )
            required_mb += min(
                retained_artifact_mb,
                _MAX_RETAINED_ARTIFACT_PREFLIGHT_MB,
            )

        existing = job_dir
        while not existing.exists() and existing != existing.parent:
            existing = existing.parent
        available_bytes = shutil.disk_usage(existing).free
        if required_mb * 1024 * 1024 > available_bytes:
            raise ValueError(
                f"{backend.capitalize()} host has insufficient free space: "
                f"run requires "
                f"{required_mb} MiB"
            )

    @staticmethod
    def _task_storage_mb(task: Task) -> int:
        raw = tomllib.loads(task.paths.config_path.read_text())
        environment = raw.get("environment", {})
        candidates = [int(task.config.environment.storage_mb or 0)]
        storage_mb = environment.get("storage_mb")
        if isinstance(storage_mb, (int, float)) and not isinstance(storage_mb, bool):
            candidates.append(int(storage_mb))
        boot_disk_gb = environment.get("boot_disk_gb")
        if isinstance(boot_disk_gb, (int, float)) and not isinstance(
            boot_disk_gb, bool
        ):
            candidates.append(int(boot_disk_gb * 1024))
        return max(candidates)
