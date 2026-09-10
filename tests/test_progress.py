import json
import stat
from types import SimpleNamespace

import pytest

from harbor_autoresearch.progress import ProgressStore
from harbor_autoresearch.results import GraderRecord, IterationRecord
from harbor_autoresearch.scoring import parse_reward, select_public_best


def test_progress_store_appends_independent_fsynced_jsonl_streams(
    tmp_path, monkeypatch
) -> None:
    fsync_calls: list[int] = []
    monkeypatch.setattr("harbor_autoresearch.progress.os.fsync", fsync_calls.append)
    store = ProgressStore(tmp_path / "autoresearch")

    store.append_public({"iteration": 1, "score": 0.4})
    store.append_public({"iteration": 2, "score": 0.7})
    store.append_private({"iteration": 1, "score": 0.9})

    public = [json.loads(line) for line in store.public_path.read_text().splitlines()]
    private = [json.loads(line) for line in store.private_path.read_text().splitlines()]
    assert public == [
        {"iteration": 1, "score": 0.4},
        {"iteration": 2, "score": 0.7},
    ]
    assert private == [{"iteration": 1, "score": 0.9}]
    assert len(fsync_calls) >= 3


def test_summary_is_atomically_replaced_and_bad_json_keeps_previous_file(
    tmp_path,
) -> None:
    store = ProgressStore(tmp_path / "autoresearch")
    store.write_summary({"stop_reason": "running", "selected": 1})
    store.write_summary({"stop_reason": "complete", "selected": 2})

    assert json.loads(store.summary_path.read_text()) == {
        "selected": 2,
        "stop_reason": "complete",
    }

    with pytest.raises(ValueError):
        store.write_summary({"selected_score": float("nan")})

    assert json.loads(store.summary_path.read_text())["selected"] == 2
    assert list(store.root.glob(".summary-*.tmp")) == []


def test_iteration_model_is_split_between_public_and_private_streams(
    tmp_path,
) -> None:
    store = ProgressStore(tmp_path / "autoresearch")
    record = IterationRecord(
        attempt_id="attempt",
        trial_id="trial",
        iteration=1,
        started_at="start",
        submitted_at="submitted",
        finished_at="finish",
        agent_elapsed_seconds=1.0,
        wall_elapsed_seconds=2.0,
        wall_clock_budget_seconds=60.0,
        wall_clock_remaining_seconds=58.0,
        agent_effort_seconds=1.0,
        evaluation_seconds=0.5,
        min_time_per_iteration=0,
        turns=1,
        submission_summary=None,
        intermediate=GraderRecord(score=0.5, stdout="visible"),
        test=GraderRecord(score=0.9, stdout="PRIVATE-SENTINEL"),
        artifact_path="artifact.tgz",
        artifact_sha256="abc",
        public_best_score=0.5,
        public_best_at_record_time=True,
    )

    store.append_public(record)
    store.append_private(record)

    assert "PRIVATE-SENTINEL" not in store.public_path.read_text()
    assert "PRIVATE-SENTINEL" in store.private_path.read_text()


def test_invalid_reward_entries_remain_writable_and_unselectable(tmp_path) -> None:
    public_result = SimpleNamespace(
        rewards={
            "reward": float("nan"),
            "positive_infinity": float("inf"),
            "negative_infinity": float("-inf"),
            "message": "not numeric",
            "usable_auxiliary": 0.25,
        }
    )
    intermediate = GraderRecord(
        score=parse_reward(public_result), rewards=public_result.rewards
    )
    assert intermediate.score is None
    assert select_public_best(None, None, intermediate.score, 1) == (None, None)
    record = IterationRecord(
        attempt_id="attempt",
        trial_id="trial",
        iteration=1,
        started_at="start",
        submitted_at="submitted",
        finished_at="finish",
        agent_elapsed_seconds=1.0,
        wall_elapsed_seconds=2.0,
        wall_clock_budget_seconds=60.0,
        wall_clock_remaining_seconds=58.0,
        agent_effort_seconds=1.0,
        evaluation_seconds=0.5,
        min_time_per_iteration=0,
        turns=1,
        submission_summary=None,
        intermediate=intermediate,
        test=GraderRecord(
            score=None,
            rewards={"reward": float("inf"), "bad": object()},
        ),
        artifact_path="artifact.tgz",
        artifact_sha256="abc",
        public_best_score=None,
        public_best_at_record_time=False,
    )
    store = ProgressStore(tmp_path / "autoresearch")

    store.append_public(record)
    store.append_private(record)

    public = json.loads(store.public_path.read_text())
    private = json.loads(store.private_path.read_text())
    assert public["intermediate"]["score"] is None
    assert public["intermediate"]["rewards"] == {
        "reward": None,
        "positive_infinity": None,
        "negative_infinity": None,
        "message": None,
        "usable_auxiliary": 0.25,
    }
    assert private["test"]["rewards"] == {"reward": None, "bad": None}


def test_progress_files_are_private_even_when_existing_modes_are_broad(
    tmp_path,
) -> None:
    root = tmp_path / "autoresearch"
    root.mkdir(mode=0o777)
    root.chmod(0o777)
    private_path = root / "private-iterations.jsonl"
    private_path.write_text("", encoding="utf-8")
    private_path.chmod(0o666)

    store = ProgressStore(root)
    store.append_public({"iteration": 1})
    store.append_private({"iteration": 1, "test_score": 0.9})
    store.write_summary({"selected_test_score": 0.9})

    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert stat.S_IMODE(store.public_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(store.private_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(store.summary_path.stat().st_mode) == 0o600
