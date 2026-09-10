# Harbor AutoResearch

This package adds repeated research runs to Harbor. One agent session stays open while
the agent tests several changes. One wall clock limits the full run. The agent sees the
validation result and remaining time after each submitted change. Private test results
stay outside the agent environment.

## Install

```bash
pip install -e .
```

For Modal:

```bash
pip install -e '.[modal]'
```

## Run

```bash
harbor run --repo bespokelabsai/AutoResearchExam -p . \
  -i cpu-decoder-graph-executor \
  -a harbor_autoresearch.agent:TimedWindowAgent \
  -m <model> -e docker \
  --plugin autoresearch-timed \
  --pk max_iterations=500 \
  --pk max_duration_seconds=86400 \
  --pk min_time_per_iteration=0 \
  --ak max_turns=500 \
  --ak reasoning_effort=high
```

Choose `-e modal` for Modal execution.

Modal limits one persistent sandbox to 24 hours. The plugin rejects a longer wall clock
budget and reserves time for the last submitted change to finish grading.

## Run settings

You can set these values for each run:

| Setting | Harbor option |
| --- | --- |
| Model | `-m <model>` |
| Model effort | `--ak reasoning_effort=<level>` |
| Backend | `-e docker` or `-e modal` |
| Total wall time in seconds | `--pk max_duration_seconds=<seconds>` |
| Maximum experiment count | `--pk max_iterations=<count>` |
| Minimum minutes per experiment, default 0 | `--pk min_time_per_iteration=<minutes>` |
| Total agent turns | `--ak max_turns=<count>` |
| Total output tokens | `--ak output_token_budget=<count>` |
| Session summary between experiments | `--pk auto_summarize=true` or `false` |

The total wall time includes agent work, artifact collection, grading, and harness work.
There is no maximum time for one experiment. Each agent phase can use all time left in
the full run. A positive minimum prevents an early submission. A minimum of zero allows
the agent to submit at once.

The harness adds a timing block to every agent phase. It includes the full budget, time
left, and the previous phase timing. After a submitted change is graded, the next phase
also includes the validation score, validation error, and validation output. It never
includes private test data. If the deadline occurs during agent work, the harness grades
that work once and does not start another phase. Grading may finish after the deadline.

The task must use a separate verifier. It must contain `tests/Dockerfile`,
`tests/intermediate.sh`, and `tests/test.sh`. The intermediate script provides the
feedback that the agent sees. The test script stays private. The package chooses the
highest finite intermediate score and uses the private score for that same artifact as
the final Harbor reward. An earlier experiment wins when intermediate scores tie.

## Results

Each trial contains an `autoresearch` directory. It includes:

* `iterations.jsonl` with public validation results, submission times, and wall clock
  timing for each experiment.
* `private-iterations.jsonl` with every private test result and its grader times.
* `summary.json` with all scores, the selected experiment, the final score, timing, the
  model, the backend, and the stop reason.
* `iterations/<number>/artifact.tar.gz` with the exact submitted artifacts.
* `iterations/<number>/intermediate` and `iterations/<number>/test` with grader logs.

Read scores while a run is active:

```bash
tail -f jobs/<job>/<trial>/autoresearch/iterations.jsonl
```

Read the selected intermediate and test scores after the run:

```bash
jq '{public_best_score, selected_iteration, selected_test_score, scores}' \
  jobs/<job>/<trial>/autoresearch/summary.json
```

The result directory and files use private host permissions because the summary and
private stream contain test scores. They are never mounted into the agent environment.

Open the jobs directory with `harbor view` to use Harbor's normal result viewer.
