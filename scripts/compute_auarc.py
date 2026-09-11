#!/usr/bin/env python3
"""Compute difficulty-adjusted hidden-test AUARC for one task or a dataset."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_BLOG_TIME_POINTS = (
    ("15m", 15 * 60),
    ("1h", 60 * 60),
    ("4h", 4 * 60 * 60),
    ("12h", 12 * 60 * 60),
    ("24h", 24 * 60 * 60),
)
_REWARD_MAPS_PATH = Path(__file__).with_name("reward_maps.json")


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _summary_inputs(summary: Mapping[str, Any]) -> tuple[float, Sequence[Any]]:
    configuration = summary.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError("configuration must be an object")
    duration = _finite_number(
        configuration.get("max_duration_seconds"),
        "configuration.max_duration_seconds",
    )
    if duration <= 0:
        raise ValueError("configuration.max_duration_seconds must be positive")

    scores = summary.get("scores")
    if not isinstance(scores, Sequence) or isinstance(scores, (str, bytes)):
        raise ValueError("scores must be an array")
    return duration, scores


def _auarc_at_time(
    scores: Sequence[Any],
    duration: float,
    *,
    require_checkpoint: bool,
) -> float:
    area = 0.0
    previous_time = 0.0
    held_test_score = 0.0
    best_intermediate_score = -math.inf
    selected_a_checkpoint = False

    for index, row in enumerate(scores):
        if not isinstance(row, Mapping):
            raise ValueError(f"scores[{index}] must be an object")
        elapsed = _finite_number(
            row.get("wall_elapsed_seconds"),
            f"scores[{index}].wall_elapsed_seconds",
        )
        if elapsed < previous_time:
            raise ValueError("scores must be ordered by wall_elapsed_seconds")

        event_time = min(elapsed, duration)
        area += held_test_score * (event_time - previous_time)
        previous_time = event_time
        if elapsed > duration:
            break

        intermediate = row.get("intermediate_score")
        if (
            isinstance(intermediate, (int, float))
            and not isinstance(intermediate, bool)
            and math.isfinite(intermediate)
            and intermediate > best_intermediate_score
        ):
            test_score = _finite_number(
                row.get("test_score"),
                f"scores[{index}].test_score",
            )
            best_intermediate_score = float(intermediate)
            held_test_score = test_score
            selected_a_checkpoint = True

        if previous_time == duration:
            break

    if not selected_a_checkpoint and require_checkpoint:
        raise ValueError("summary has no usable scored checkpoint")

    area += held_test_score * (duration - previous_time)
    return area / duration


def compute_hidden_test_auarc(summary: Mapping[str, Any]) -> float:
    """Return the full-window hidden reward selected by public reward."""
    duration, scores = _summary_inputs(summary)
    return _auarc_at_time(scores, duration, require_checkpoint=True)


def compute_auarc_report(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Return blog time points and the full-window hidden test AUARC."""
    duration, scores = _summary_inputs(summary)
    final_auarc = _auarc_at_time(scores, duration, require_checkpoint=True)
    return {
        "metric": "hidden_test_auarc",
        "auarc_by_time": [
            {
                "label": label,
                "elapsed_seconds": elapsed_seconds,
                "auarc": _auarc_at_time(
                    scores,
                    elapsed_seconds,
                    require_checkpoint=False,
                ),
            }
            for label, elapsed_seconds in _BLOG_TIME_POINTS
            if elapsed_seconds < duration
        ],
        "final": {"elapsed_seconds": duration, "auarc": final_auarc},
    }


def _load_reward_maps() -> list[Mapping[str, Any]]:
    payload = json.loads(_REWARD_MAPS_PATH.read_text())
    tasks = payload.get("tasks") if isinstance(payload, Mapping) else None
    if not isinstance(tasks, list):
        raise ValueError("reward_maps.json must contain a tasks array")
    return tasks


def _task_aliases(task: Mapping[str, Any]) -> set[str]:
    aliases = {str(task.get("name", "")), str(task.get("canonical_id", ""))}
    canonical_id = str(task.get("canonical_id", ""))
    if "/" in canonical_id:
        aliases.add(canonical_id.split("/", 1)[1])
    return aliases - {""}


def _resolve_task(name: str) -> Mapping[str, Any]:
    matches = [task for task in _load_reward_maps() if name in _task_aliases(task)]
    if len(matches) != 1:
        raise ValueError(f"unknown task name: {name}")
    return matches[0]


def _apply_progress(config: Mapping[str, Any], metric: float, units: float) -> float:
    m0 = _finite_number(config.get("m0"), "reward map m0")
    direction = _finite_number(config.get("direction", 1), "reward map direction")
    baseline_reward = _finite_number(
        config.get("baseline_reward", 0), "reward map baseline_reward"
    )
    improvement = direction * (metric - m0)
    if baseline_reward > 0 and improvement < 0:
        m_floor = _finite_number(config.get("m_floor"), "reward map m_floor")
        width = direction * (m0 - m_floor)
        if width <= 0:
            return 0.0
        return baseline_reward * max(0.0, direction * (metric - m_floor)) / width
    units = max(0.0, units)
    squashed = 1.0 if math.isinf(units) else units / (1.0 + units)
    return baseline_reward + (1.0 - baseline_reward) * squashed


def _difficulty_reward(task: Mapping[str, Any], raw_metric: Any, name: str) -> float:
    metric = _finite_number(raw_metric, name)
    config = task.get("map")
    if not isinstance(config, Mapping):
        raise ValueError(f"{task.get('name')} has no reward map")
    m0 = _finite_number(config.get("m0"), "reward map m0")
    direction = _finite_number(config.get("direction", 1), "reward map direction")
    kind = config.get("kind")
    if kind == "rational":
        m_ref = _finite_number(config.get("m_ref"), "reward map m_ref")
        span = abs(m_ref - m0)
        if span <= 0:
            raise ValueError("reward map reference must differ from its baseline")
        units = max(0.0, direction * (metric - m0)) / span
    elif kind in {"gap_ratio", "gap_halvings"}:
        bound = _finite_number(config.get("bound"), "reward map bound")
        if direction * (metric - bound) >= 0:
            return 1.0
        gap = abs(metric - bound)
        if gap == 0:
            units = math.inf
        elif kind == "gap_halvings":
            units = max(0.0, math.log2(abs(m0 - bound) / gap))
        else:
            m_ref = _finite_number(config.get("m_ref"), "reward map m_ref")
            scale = math.log(abs(m0 - bound) / abs(m_ref - bound))
            if scale <= 0:
                raise ValueError("reward map reference must improve on its baseline")
            units = max(0.0, math.log(abs(m0 - bound) / gap) / scale)
    else:
        raise ValueError(f"unsupported reward map kind: {kind}")
    return min(1.0, max(0.0, _apply_progress(config, metric, units)))


def _adjusted_test_score(
    task: Mapping[str, Any], row: Mapping[str, Any], index: int
) -> float | None:
    raw_metric = row.get("test_raw_metric")
    if (
        isinstance(raw_metric, (int, float))
        and not isinstance(raw_metric, bool)
        and math.isfinite(raw_metric)
    ):
        return _difficulty_reward(task, raw_metric, f"scores[{index}].test_raw_metric")
    reported_reward = row.get("test_score")
    if reported_reward == 0 and not isinstance(reported_reward, bool):
        return 0.0
    if reported_reward is None:
        return None
    raise ValueError(f"scores[{index}].test_raw_metric must be a finite number")


def _difficulty_adjusted_summary(
    summary: Mapping[str, Any], task: Mapping[str, Any]
) -> dict[str, Any]:
    duration, scores = _summary_inputs(summary)
    adjusted_scores: list[dict[str, Any]] = []
    for index, row in enumerate(scores):
        if not isinstance(row, Mapping):
            raise ValueError(f"scores[{index}] must be an object")
        adjusted_scores.append(
            {
                **row,
                "test_score": _adjusted_test_score(task, row, index),
            }
        )
    return {
        **summary,
        "configuration": {"max_duration_seconds": duration},
        "scores": adjusted_scores,
    }


def _task_metadata(task: Mapping[str, Any]) -> dict[str, str]:
    return {key: str(task[key]) for key in ("name", "canonical_id", "reward_name")}


def compute_difficulty_adjusted_report(
    summary: Mapping[str, Any], task_name: str
) -> dict[str, Any]:
    task = _resolve_task(task_name)
    report = compute_auarc_report(_difficulty_adjusted_summary(summary, task))
    report["metric"] = "difficulty_adjusted_hidden_test_auarc"
    report["task"] = _task_metadata(task)
    return report


def _summary_task_name(summary: Mapping[str, Any], path: Path) -> str:
    configuration = summary.get("configuration")
    if isinstance(configuration, Mapping):
        for key in ("task_name", "canonical_task_id"):
            value = configuration.get(key)
            if isinstance(value, str) and value:
                return value
    known_names = {
        alias: str(task["name"])
        for task in _load_reward_maps()
        for alias in _task_aliases(task)
    }
    for parent in path.parents:
        if parent.name in known_names:
            return known_names[parent.name]
    raise ValueError(
        f"{path}: missing task name; use a summary produced by this version"
    )


def compute_all_tasks_report(root: Path) -> dict[str, Any]:
    summary_paths = sorted(root.rglob("autoresearch/summary.json"))
    if not summary_paths:
        raise ValueError(f"no summary.json files found under {root}")
    grouped: dict[str, list[tuple[Path, float]]] = {}
    metadata: dict[str, dict[str, str]] = {}
    for path in summary_paths:
        payload = json.loads(path.read_text())
        if not isinstance(payload, Mapping):
            raise ValueError(f"{path}: summary must be a JSON object")
        task_name = _summary_task_name(payload, path)
        report = compute_difficulty_adjusted_report(payload, task_name)
        task = report["task"]
        canonical_id = task["canonical_id"]
        grouped.setdefault(canonical_id, []).append((path, report["final"]["auarc"]))
        metadata[canonical_id] = task
    tasks: list[dict[str, Any]] = []
    task_scores: list[float] = []
    for canonical_id in sorted(grouped):
        observations = grouped[canonical_id]
        task_score = statistics.fmean(score for _, score in observations)
        task_scores.append(task_score)
        tasks.append(
            {
                "task": metadata[canonical_id],
                "summary_count": len(observations),
                "final_test_auarc": task_score,
                "summaries": [str(path) for path, _ in observations],
            }
        )
    return {
        "metric": "difficulty_adjusted_hidden_test_auarc",
        "tasks": tasks,
        "aggregate": {
            "task_count": len(tasks),
            "summary_count": sum(
                len(observations) for observations in grouped.values()
            ),
            "mean_final_test_auarc": statistics.fmean(task_scores),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute final hidden-test AUARC for one task or a dataset run."
    )
    parser.add_argument("path", type=Path, help="summary.json or a directory")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--task-name", help="AutoResearchExam task name or ID")
    mode.add_argument(
        "--all", action="store_true", help="recursively aggregate all summary files"
    )
    mode.add_argument(
        "--plain", action="store_true", help="use already-reported rewards"
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        if args.all:
            if not args.path.is_dir():
                raise ValueError("--all requires a directory")
            report = compute_all_tasks_report(args.path)
        else:
            payload = json.loads(args.path.read_text())
            if not isinstance(payload, Mapping):
                raise ValueError("summary must be a JSON object")
            configuration = payload.get("configuration")
            saved_task_name = (
                configuration.get("task_name")
                if isinstance(configuration, Mapping)
                else None
            )
            task_name = args.task_name or saved_task_name
            if args.plain:
                report = compute_auarc_report(payload)
            elif isinstance(task_name, str) and task_name:
                report = compute_difficulty_adjusted_report(payload, task_name)
            else:
                raise ValueError("task name missing; pass --task-name or --plain")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
