from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "compute_auarc.py"
REWARD_MAPS = SCRIPT.with_name("reward_maps.json")


def test_saved_reward_maps_cover_the_full_dataset() -> None:
    tasks = json.loads(REWARD_MAPS.read_text())["tasks"]

    assert len(tasks) == 29
    assert len({task["name"] for task in tasks}) == 29
    assert all(task["canonical_id"] == f"bespokelabs/{task['name']}" for task in tasks)
    assert all(task["reward_name"] for task in tasks)


def test_cli_computes_hidden_test_auarc_from_validation_selected_curve(
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "configuration": {"max_duration_seconds": 100},
                "scores": [
                    {
                        "iteration": 1,
                        "wall_elapsed_seconds": 20,
                        "intermediate_score": 0.4,
                        "test_score": 0.8,
                    },
                    {
                        "iteration": 2,
                        "wall_elapsed_seconds": 60,
                        "intermediate_score": 0.6,
                        "test_score": 0.4,
                    },
                    {
                        "iteration": 3,
                        "wall_elapsed_seconds": 80,
                        "intermediate_score": 0.5,
                        "test_score": 0.9,
                    },
                ],
            }
        )
    )

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), str(summary_path), "--plain"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "metric": "hidden_test_auarc",
        "auarc_by_time": [],
        "final": {"elapsed_seconds": 100, "auarc": 0.48},
    }
    assert completed.stderr == ""


def test_cli_returns_the_blog_time_points_and_final_auarc(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "configuration": {"max_duration_seconds": 86_400},
                "scores": [
                    {
                        "iteration": 1,
                        "wall_elapsed_seconds": 1_800,
                        "intermediate_score": 0.2,
                        "test_score": 0.4,
                    },
                    {
                        "iteration": 2,
                        "wall_elapsed_seconds": 7_200,
                        "intermediate_score": 0.5,
                        "test_score": 0.8,
                    },
                    {
                        "iteration": 3,
                        "wall_elapsed_seconds": 21_600,
                        "intermediate_score": 0.4,
                        "test_score": 0.9,
                    },
                    {
                        "iteration": 4,
                        "wall_elapsed_seconds": 64_800,
                        "intermediate_score": 0.6,
                        "test_score": 0.6,
                    },
                ],
            }
        )
    )

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), str(summary_path), "--plain"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert [point["label"] for point in payload["auarc_by_time"]] == [
        "15m",
        "1h",
        "4h",
        "12h",
    ]
    assert [point["elapsed_seconds"] for point in payload["auarc_by_time"]] == [
        900,
        3_600,
        14_400,
        43_200,
    ]
    assert [point["auarc"] for point in payload["auarc_by_time"]] == pytest.approx(
        [0.0, 0.2, 0.55, 0.716666666667]
    )
    assert payload["final"] == pytest.approx(
        {"elapsed_seconds": 86_400, "auarc": 0.708333333333}
    )


def test_cli_uses_only_blog_time_points_before_a_shorter_run_ends(
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "configuration": {"max_duration_seconds": 21_600},
                "scores": [
                    {
                        "iteration": 1,
                        "wall_elapsed_seconds": 3_600,
                        "intermediate_score": 0.5,
                        "test_score": 0.5,
                    }
                ],
            }
        )
    )

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), str(summary_path), "--plain"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert [point["label"] for point in payload["auarc_by_time"]] == [
        "15m",
        "1h",
        "4h",
    ]
    assert [point["auarc"] for point in payload["auarc_by_time"]] == pytest.approx(
        [0.0, 0.0, 0.375]
    )
    assert payload["final"] == pytest.approx(
        {"elapsed_seconds": 21_600, "auarc": 0.416666666667}
    )


def test_cli_keeps_earlier_ties_and_ignores_events_after_the_window(
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "configuration": {"max_duration_seconds": 100},
                "scores": [
                    {
                        "iteration": 1,
                        "wall_elapsed_seconds": 20,
                        "intermediate_score": 0.5,
                        "test_score": 0.4,
                    },
                    {
                        "iteration": 2,
                        "wall_elapsed_seconds": 70,
                        "intermediate_score": 0.5,
                        "test_score": 0.9,
                    },
                    {
                        "iteration": 3,
                        "wall_elapsed_seconds": 120,
                        "intermediate_score": 0.8,
                        "test_score": 1.0,
                    },
                ],
            }
        )
    )

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), str(summary_path), "--plain"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["final"] == {
        "elapsed_seconds": 100,
        "auarc": 0.32,
    }


def test_cli_rejects_a_selected_checkpoint_without_a_test_score(
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "configuration": {"max_duration_seconds": 100},
                "scores": [
                    {
                        "iteration": 1,
                        "wall_elapsed_seconds": 20,
                        "intermediate_score": 0.5,
                        "test_score": None,
                    }
                ],
            }
        )
    )

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), str(summary_path), "--plain"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "scores[0].test_score must be a number" in completed.stderr


def test_cli_rejects_a_summary_without_an_in_window_grade(
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "configuration": {"max_duration_seconds": 100},
                "scores": [
                    {
                        "iteration": 1,
                        "wall_elapsed_seconds": 101,
                        "intermediate_score": 0.8,
                        "test_score": 0.9,
                    }
                ],
            }
        )
    )

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), str(summary_path), "--plain"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "no usable scored checkpoint" in completed.stderr


def test_cli_applies_task_reward_map_to_raw_metrics_before_auarc(
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "configuration": {"max_duration_seconds": 100},
                "scores": [
                    {
                        "iteration": 1,
                        "wall_elapsed_seconds": 20,
                        "intermediate_score": 0.1,
                        "intermediate_raw_metric": 18.0,
                        "test_score": 0.1,
                        "test_raw_metric": 35.0,
                    }
                ],
            }
        )
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(summary_path),
            "--task-name",
            "cpu-llm-decode-throughput",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["metric"] == "difficulty_adjusted_hidden_test_auarc"
    assert payload["task"] == {
        "name": "cpu-llm-decode-throughput",
        "canonical_id": "bespokelabs/cpu-llm-decode-throughput",
        "reward_name": "Geometric-mean throughput speedup (×)",
    }
    assert payload["final"] == pytest.approx(
        {"elapsed_seconds": 100, "auarc": 0.5333333333333333}
    )


def test_cli_recursively_reports_tasks_and_equal_weight_dataset_mean(
    tmp_path: Path,
) -> None:
    examples = {
        "cpu-llm-decode-throughput": 18.0,
        "hicard-latent-encoder": 27.0,
    }
    for task_name, raw_metric in examples.items():
        summary_path = tmp_path / task_name / "trial" / "autoresearch" / "summary.json"
        summary_path.parent.mkdir(parents=True)
        summary_path.write_text(
            json.dumps(
                {
                    "configuration": {
                        "max_duration_seconds": 100,
                        "task_name": task_name,
                    },
                    "scores": [
                        {
                            "iteration": 1,
                            "wall_elapsed_seconds": 20,
                            "intermediate_score": 0.5,
                            "intermediate_raw_metric": raw_metric,
                            "test_raw_metric": raw_metric,
                        }
                    ],
                }
            )
        )

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path), "--all"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert [row["task"]["name"] for row in payload["tasks"]] == [
        "cpu-llm-decode-throughput",
        "hicard-latent-encoder",
    ]
    assert [row["final_test_auarc"] for row in payload["tasks"]] == pytest.approx(
        [0.4, 0.4]
    )
    assert payload["aggregate"] == pytest.approx(
        {"task_count": 2, "summary_count": 2, "mean_final_test_auarc": 0.4}
    )


def test_difficulty_adjustment_keeps_validation_selected_checkpoint(
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "configuration": {"max_duration_seconds": 100},
                "scores": [
                    {
                        "wall_elapsed_seconds": 20,
                        "intermediate_score": 0.5,
                        "intermediate_raw_metric": 35.0,
                        "test_raw_metric": 18.0,
                    },
                    {
                        "wall_elapsed_seconds": 60,
                        "intermediate_score": 0.4,
                        "intermediate_raw_metric": 69.0,
                        "test_raw_metric": 35.0,
                    },
                ],
            }
        )
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(summary_path),
            "--task-name",
            "cpu-llm-decode-throughput",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["final"]["auarc"] == pytest.approx(0.4)


def test_difficulty_adjustment_preserves_zero_reward_without_raw_metric(
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "configuration": {"max_duration_seconds": 100},
                "scores": [
                    {
                        "wall_elapsed_seconds": 20,
                        "intermediate_score": 0,
                        "test_score": 0,
                        "test_raw_metric": None,
                    }
                ],
            }
        )
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(summary_path),
            "--task-name",
            "cpu-llm-decode-throughput",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["final"]["auarc"] == 0


def test_cli_uses_task_name_saved_in_summary(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "configuration": {
                    "max_duration_seconds": 100,
                    "task_name": "bespokelabs/cpu-llm-decode-throughput",
                },
                "scores": [
                    {
                        "wall_elapsed_seconds": 20,
                        "intermediate_score": 0.5,
                        "test_raw_metric": 18.0,
                    }
                ],
            }
        )
    )

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), str(summary_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["task"]["name"] == ("cpu-llm-decode-throughput")
