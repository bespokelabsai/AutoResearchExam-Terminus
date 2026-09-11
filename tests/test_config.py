import pytest

from harbor_autoresearch.config import TimedWindowConfig, validate_backend


def test_accepts_user_selected_timing_at_supported_boundaries() -> None:
    config = TimedWindowConfig(
        max_iterations=5_000,
        max_duration_seconds=172_800,
        min_time_per_iteration=0,
    )

    assert config.max_iterations == 5_000
    assert config.max_duration_seconds == 172_800
    assert config.min_time_per_iteration == 0
    assert config.auto_summarize is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_iterations", 0),
        ("max_iterations", 5_001),
        ("max_duration_seconds", 0),
        ("max_duration_seconds", 172_801),
        ("min_time_per_iteration", -1),
    ],
)
def test_rejects_timing_outside_supported_ranges(field: str, value: int) -> None:
    values = {
        "max_iterations": 4,
        "max_duration_seconds": 600,
        "min_time_per_iteration": 1,
    }
    values[field] = value

    with pytest.raises(ValueError, match=field):
        TimedWindowConfig(**values)


@pytest.mark.parametrize("value", [True, 1.5, "5"])
def test_timing_values_are_integers_not_coercions(value: object) -> None:
    with pytest.raises(TypeError, match="max_iterations"):
        TimedWindowConfig(
            max_iterations=value,
            max_duration_seconds=600,
            min_time_per_iteration=1,
        )


def test_iteration_minimum_must_fit_in_global_budget() -> None:
    with pytest.raises(ValueError, match="min_time_per_iteration"):
        TimedWindowConfig(4, 299, 5)


def test_serializes_the_exact_validated_user_configuration() -> None:
    config = TimedWindowConfig(4, 600, 0, auto_summarize=False)

    assert config.as_dict() == {
        "max_iterations": 4,
        "max_duration_seconds": 600,
        "min_time_per_iteration": 0,
        "auto_summarize": False,
    }

    with pytest.raises(TypeError, match="auto_summarize"):
        TimedWindowConfig(4, 600, 0, auto_summarize=1)


def test_backend_allowlist_accepts_only_local_docker_and_modal() -> None:
    assert validate_backend("docker") == "docker"
    assert validate_backend("modal") == "modal"

    for backend in ("daytona", "Docker", "package.module:Backend", ""):
        with pytest.raises(ValueError, match="docker.*modal"):
            validate_backend(backend)
