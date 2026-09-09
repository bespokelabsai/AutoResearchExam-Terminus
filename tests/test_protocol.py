from __future__ import annotations

from types import SimpleNamespace

from harbor_autoresearch.protocol import (
    AUTORESEARCH_PROTOCOL,
    extract_submission_summary,
    format_iteration_feedback,
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
