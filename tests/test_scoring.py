import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from harbor_autoresearch.scoring import (
    parse_reward,
    read_raw_metric,
    select_public_best,
)


def test_public_best_selects_finite_maximum_and_keeps_earliest_tie() -> None:
    best_score, best_iteration = select_public_best(None, None, 0.4, 1)
    best_score, best_iteration = select_public_best(best_score, best_iteration, 0.7, 2)
    best_score, best_iteration = select_public_best(best_score, best_iteration, 0.7, 3)

    assert (best_score, best_iteration) == (0.7, 2)


@pytest.mark.parametrize("candidate", [None, float("nan"), float("inf"), True])
def test_public_best_ignores_values_that_are_not_finite_scores(
    candidate: object,
) -> None:
    assert select_public_best(0.4, 2, candidate, 3) == (0.4, 2)


def test_public_best_uses_earliest_tied_iteration_regardless_of_input_order() -> None:
    assert select_public_best(0.7, 3, 0.7, 2) == (0.7, 2)


def test_reward_parser_prefers_named_reward_and_supports_single_metric() -> None:
    result = SimpleNamespace(rewards={"reward": 0.8, "accuracy": 0.9})

    assert parse_reward(result) == 0.8
    assert parse_reward({"accuracy": 0.6}) == 0.6
    assert parse_reward({"rewards": {"reward": 0.4}}) == 0.4


@pytest.mark.parametrize(
    "result",
    [
        None,
        SimpleNamespace(rewards={}),
        SimpleNamespace(rewards={"a": 0.2, "b": 0.3}),
        SimpleNamespace(rewards={"reward": float("nan")}),
        SimpleNamespace(rewards={"reward": True}),
        SimpleNamespace(rewards={"reward": "0.5"}),
    ],
)
def test_reward_parser_returns_none_when_there_is_no_unambiguous_finite_reward(
    result: object,
) -> None:
    assert parse_reward(result) is None


def test_raw_metric_reader_uses_only_the_named_finite_number(tmp_path) -> None:
    verifier_dir = tmp_path / "verifier"
    verifier_dir.mkdir()
    metric_path = verifier_dir / "metric.json"

    metric_path.write_text('{"metric": -3.25, "accuracy": 0.9}')
    assert read_raw_metric(verifier_dir) == -3.25

    metric_path.write_text('{"accuracy": 0.9}')
    assert read_raw_metric(verifier_dir) is None


@pytest.mark.parametrize(
    "body",
    [
        "not json",
        "[]",
        '{"metric": true}',
        '{"metric": "0.4"}',
        '{"metric": NaN}',
        '{"metric": Infinity}',
        '{"metric": 1' + "0" * 400 + "}",
    ],
)
def test_raw_metric_reader_never_raises_for_unusable_input(tmp_path, body: str) -> None:
    verifier_dir = tmp_path / "verifier"
    verifier_dir.mkdir()
    (verifier_dir / "metric.json").write_text(body)

    assert read_raw_metric(verifier_dir) is None


def test_raw_metric_reader_handles_missing_and_deeply_nested_files(tmp_path) -> None:
    assert read_raw_metric(tmp_path / "missing") is None

    verifier_dir = tmp_path / "verifier"
    verifier_dir.mkdir()
    (verifier_dir / "metric.json").write_text("[" * 100_000 + "]" * 100_000)
    assert read_raw_metric(verifier_dir) is None


def test_raw_metric_reader_rejects_symlinks_and_oversized_files(tmp_path) -> None:
    source = tmp_path / "source.json"
    source.write_text('{"metric": 0.7}')

    linked_dir = tmp_path / "linked"
    linked_dir.mkdir()
    (linked_dir / "metric.json").symlink_to(source)
    assert read_raw_metric(linked_dir) is None

    large_dir = tmp_path / "large"
    large_dir.mkdir()
    (large_dir / "metric.json").write_text(
        '{"metric": 0.8, "padding": "' + "x" * 70_000 + '"}'
    )
    assert read_raw_metric(large_dir) is None


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO requires POSIX")
def test_raw_metric_reader_does_not_block_on_a_fifo(tmp_path) -> None:
    verifier_dir = tmp_path / "verifier"
    verifier_dir.mkdir()
    os.mkfifo(verifier_dir / "metric.json")

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; "
                "from harbor_autoresearch.scoring import read_raw_metric; "
                "assert read_raw_metric(Path(__import__('sys').argv[1])) is None"
            ),
            str(verifier_dir),
        ],
        check=False,
        timeout=2,
    )
    assert completed.returncode == 0
