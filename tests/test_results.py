import json

from harbor_autoresearch.results import (
    GraderRecord,
    IterationRecord,
    IterationScores,
    SummaryRecord,
)


def test_grader_record_preserves_mapped_and_raw_scores() -> None:
    record = GraderRecord(
        score=0.7,
        raw_metric=9.25,
        rewards={"reward": 0.7, "accuracy": 0.8},
        stdout="details",
        stderr="warning",
        error=None,
        started_at="2026-09-09T10:00:00+00:00",
        finished_at="2026-09-09T10:00:03+00:00",
    )

    assert record.as_dict() == {
        "score": 0.7,
        "raw_metric": 9.25,
        "rewards": {"reward": 0.7, "accuracy": 0.8},
        "stdout": "details",
        "stderr": "warning",
        "error": None,
        "started_at": "2026-09-09T10:00:00+00:00",
        "finished_at": "2026-09-09T10:00:03+00:00",
    }


def test_iteration_record_separates_visible_and_private_grader_data() -> None:
    record = IterationRecord(
        attempt_id="attempt-1",
        trial_id="trial-1",
        iteration=2,
        started_at="start",
        finished_at="finish",
        agent_elapsed_seconds=12.5,
        wall_elapsed_seconds=20.0,
        turns=3,
        submission_summary="improved parser",
        intermediate=GraderRecord(score=0.8, stdout="visible"),
        test=GraderRecord(score=0.4, stdout="PRIVATE-SENTINEL"),
        artifact_path="iterations/0002/artifact.tar.gz",
        artifact_sha256="abc123",
        public_best_score=0.8,
        public_best_at_record_time=True,
    )

    public = record.as_public_dict()
    private = record.as_private_dict()

    assert public["intermediate"]["score"] == 0.8
    assert public["public_best_at_record_time"] is True
    assert "selected" not in public
    assert "test" not in public
    assert "PRIVATE-SENTINEL" not in json.dumps(public)
    assert private == {
        "attempt_id": "attempt-1",
        "trial_id": "trial-1",
        "iteration": 2,
        "test": GraderRecord(score=0.4, stdout="PRIVATE-SENTINEL").as_dict(),
    }


def test_summary_records_configuration_all_scores_and_final_selection() -> None:
    summary = SummaryRecord(
        configuration={"max_iterations": 2, "model": "provider/model"},
        stop_reason="max_iterations",
        error=None,
        scores=(
            IterationScores(1, 0.4, 4.0, 0.9, 9.0, False, "first.tgz"),
            IterationScores(2, 0.8, 8.0, 0.6, 6.0, True, "second.tgz"),
        ),
        public_best_score=0.8,
        selected_iteration=2,
        selected_test_score=0.6,
        selected_artifact_path="second.tgz",
    )

    payload = summary.as_dict()

    assert payload["configuration"]["model"] == "provider/model"
    assert payload["scores"][0]["test_score"] == 0.9
    assert payload["selected_iteration"] == 2
    assert payload["selected_test_score"] == 0.6
