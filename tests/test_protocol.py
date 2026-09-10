from __future__ import annotations

from types import SimpleNamespace

from harbor_autoresearch.protocol import (
    AUTORESEARCH_PROTOCOL,
    extract_submission_summary,
    format_iteration_feedback,
    format_phase_instruction,
)


def test_protocol_describes_the_repeated_experiment_contract() -> None:
    assert "one clear hypothesis" in AUTORESEARCH_PROTOCOL
    assert "EXPERIMENT_SUMMARY" in AUTORESEARCH_PROTOCOL
    assert "results.tsv" in AUTORESEARCH_PROTOCOL
    assert "Do not stop to ask" in AUTORESEARCH_PROTOCOL


def test_feedback_contains_only_bounded_intermediate_information() -> None:
    feedback = format_iteration_feedback(
        iteration=3,
        score=0.42,
        error=None,
        stdout="discard-this-" + "useful output",
        max_stdout_chars=13,
    )

    assert "Experiment 3" in feedback
    assert "0.42" in feedback
    assert "error: none" in feedback
    assert "useful output" in feedback
    assert "discard-this" not in feedback


def test_phase_instruction_reports_global_budget_and_previous_timing() -> None:
    instruction = format_phase_instruction(
        instruction="Visible intermediate score: 0.42",
        phase=3,
        total_budget_seconds=21_600,
        remaining_seconds=20_123.456,
        previous_iteration_seconds=321.25,
        previous_agent_effort_seconds=250.5,
        previous_non_agent_seconds=50.25,
        previous_validation_seconds=20.5,
    )

    assert "Visible intermediate score: 0.42" in instruction
    assert "phase: 3" in instruction
    assert "total autoresearch budget: 21600.0 seconds" in instruction
    assert "wall-clock time remaining before this phase: 20123.5 seconds" in instruction
    assert "previous iteration wall-clock: 321.2 seconds" in instruction
    assert "previous agent effort: 250.5 seconds" in instruction
    assert "previous non-agent processing: 50.2 seconds" in instruction
    assert "previous validation grading: 20.5 seconds" in instruction
    assert "Only visible validation score/output is available to you." in instruction
    assert "PRIVATE-SENTINEL" not in instruction


def test_summary_prefers_latest_marked_submission_over_confirmation() -> None:
    steps = [
        SimpleNamespace(
            source="agent",
            message=(
                "done\nEXPERIMENT_SUMMARY\n"
                "hypothesis: use a wider model\n"
                "changes: increased width\n"
                "expected_effect: improve score"
            ),
        ),
        SimpleNamespace(source="agent", message="Confirmed complete."),
    ]

    summary = extract_submission_summary(steps)

    assert summary.startswith("EXPERIMENT_SUMMARY")
    assert "wider model" in summary
    assert "Confirmed" not in summary


def test_summary_falls_back_to_latest_agent_message_and_is_bounded() -> None:
    steps = [
        SimpleNamespace(source="user", message="ignore"),
        SimpleNamespace(source="agent", message="0123456789"),
    ]

    assert extract_submission_summary(steps, max_chars=4) == "6789"
