"""Validated settings for timed, repeated grading runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TimedWindowConfig:
    """All timing policy supplied by the user for one run."""

    max_iterations: int
    max_duration_seconds: int
    min_time_per_iteration: int
    max_time_per_iteration: int
    auto_summarize: bool = True

    def __post_init__(self) -> None:
        for field_name in (
            "max_iterations",
            "max_duration_seconds",
            "min_time_per_iteration",
            "max_time_per_iteration",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
        if not isinstance(self.auto_summarize, bool):
            raise TypeError("auto_summarize must be a boolean")
        if not 1 <= self.max_iterations <= 500:
            raise ValueError("max_iterations must be between 1 and 500")
        if not 1 <= self.max_duration_seconds <= 172_800:
            raise ValueError("max_duration_seconds must be between 1 and 172800")
        if self.min_time_per_iteration < 1:
            raise ValueError("min_time_per_iteration must be positive")
        if self.max_time_per_iteration < 1:
            raise ValueError("max_time_per_iteration must be positive")
        if self.min_time_per_iteration > self.max_time_per_iteration:
            raise ValueError(
                "min_time_per_iteration cannot exceed max_time_per_iteration"
            )
        if self.max_time_per_iteration * 60 > self.max_duration_seconds:
            raise ValueError(
                "max_time_per_iteration must fit within max_duration_seconds"
            )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_backend(name: str) -> str:
    if name not in {"docker", "modal"}:
        raise ValueError("backend must be either docker or modal")
    return name
