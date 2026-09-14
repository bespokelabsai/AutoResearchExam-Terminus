# Use your own harness

You can run [AutoResearchExam](https://github.com/bespokelabsai/AutoResearchExam)
with Claude Code, Codex, or another agent harness. Export the scores below to use
the existing AUARC script. You do not need a new scoring integration.

## Run the task

The [official Terminus harness](../README.md) provides the complete run loop.
For another harness, implement this loop in your runner:

1. Build the task's agent environment and give the agent `instruction.md`.
   Preserve the task's data, resource limits, and artifact paths in `task.toml`.
   Set the research budget before starting; the official default is 24 hours
   (86400 seconds).
2. Run your agent in that environment. At each submission, pause the agent and
   save an immutable copy of the declared artifacts.
3. Grade that copy with `tests/intermediate.sh`, then `tests/test.sh`, each in a
   separate verifier environment built from the task's `tests/` directory.
   Both graders must receive the same saved artifact.
4. Record both results and the elapsed time after both graders finish. Return
   only public validation feedback and remaining time to the agent, then resume
   it until the budget or your declared experiment limit is reached.

Keep private test code, data, scores, and logs outside the agent environment.
Do not mount the full task checkout or results directory into it. Enforce this
with environment and network access controls; a prompt asking the agent not to
read tests is insufficient.

Use the same task revision, resources, and research budget when comparing
harnesses. Report the harness, model, and any additional stopping limits.
For the 24-hour benchmark, keep `max_duration_seconds=86400` even if a run stops
early. Record the actual runtime and stop reason separately.

## Save timestamped scores

Write one JSON file per run. Start with
[`custom_harness_summary.json`](custom_harness_summary.json), a synthetic
100-second example for the default command below.

| Field | Value |
| --- | --- |
| `configuration.max_duration_seconds` | The full research budget in seconds, even if the agent stops early. |
| `configuration.task_name` | The task directory name, required for official scoring unless passed as `--task-name`. |
| `scores[].wall_elapsed_seconds` | Seconds since the research window began, recorded after artifact collection and both graders finish. |
| `scores[].intermediate_score` | Public grader's `/logs/verifier/reward.txt`. Higher is better. Use this reward, not a raw loss or error. |
| `scores[].test_score` | Optional private grader's `/logs/verifier/reward.txt`. Needed only for rows without a raw metric, as described below. |
| `scores[].test_raw_metric` | Private grader's `metric` from `/logs/verifier/metric.json`. Used for official scoring. |

Default mode needs a finite `test_raw_metric` for every row, including
unselected rows. If it is missing, `test_score` must be 0, or `null` on a row
that is never selected.

Use a monotonic clock. Start it after environment and workspace setup, just
before the first agent phase. Include agent work, artifact collection, public
validation, private testing, and all overhead during the window. Do not use
submission time or cumulative agent time as `wall_elapsed_seconds`.

Keep rows in nondecreasing elapsed time, preserving submission order for equal
timestamps. Save this JSON outside the agent environment. Retain each artifact
and its iteration ID alongside your records so both scores can be traced to it.

## Compute AUARC

Compute AUARC on final benchmark rewards. The default command converts
`test_raw_metric` automatically using `configuration.task_name`.
Run it from this repository's root with Python 3.12 or newer:

```bash
python3 scripts/compute_auarc.py scripts/custom_harness_summary.json
```

This example returns a final AUARC of approximately 0.467. Your runner supplies
the JSON; the script handles reward conversion, checkpoint selection, and AUARC.
Always put the raw metric in `test_raw_metric`, never an already mapped reward.

AUARC is the time average of the private reward of the best public checkpoint
so far. Report test performance, but select checkpoints solely by validation.
Keep test results hidden from the agent throughout the run and checkpoint
selection. Never select by test score or take a running maximum of test scores.
The reward starts at zero. A strictly higher public reward replaces the
selected checkpoint; ties keep the earlier row. The selected private reward can
decrease. Hold it until the next improvement or the full budget ends.

Rows after the budget do not contribute. A row exactly at the deadline adds no
area. Terminus can finish grading after the deadline and retain that row in its
summary; do not move its timestamp earlier. The script requires at least one
usable scored checkpoint at or before the deadline, otherwise it reports an
error. Missing public scores do not select a checkpoint. A selected checkpoint
needs a finite private score; do not invent one to suppress an error.

For multiple runs, save files as `<run>/autoresearch/summary.json` under a common
directory and run `python3 scripts/compute_auarc.py <directory> --all`. This
averages runs within each task, then gives each included task equal weight.
