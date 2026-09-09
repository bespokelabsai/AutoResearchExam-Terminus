# Harbor AutoResearch

This package adds repeated research runs to Harbor. One agent session stays open while
the agent tests several changes. Each change gets a time window. The agent sees the
validation result after each change. Private test results stay outside the agent
environment.

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
  --pk min_time_per_iteration=5 \
  --pk max_time_per_iteration=30
```

Choose `-e modal` for Modal execution.

## Run settings

You can set these values for each run:

| Setting | Harbor option |
| --- | --- |
| Model | `-m <model>` |
| Backend | `-e docker` or `-e modal` |
| Total agent time in seconds | `--pk max_duration_seconds=<seconds>` |
| Maximum experiment count | `--pk max_iterations=<count>` |
| Minimum minutes per experiment | `--pk min_time_per_iteration=<minutes>` |
| Maximum minutes per experiment | `--pk max_time_per_iteration=<minutes>` |
| Total agent turns | `--ak max_turns=<count>` |
| Total output tokens | `--ak output_token_budget=<count>` |
| Session summary between experiments | `--pk auto_summarize=true` or `false` |

Only agent work counts toward the total agent time. Image builds, artifact copies, and
grading do not count. A maximum experiment window ends the current agent phase, grades
the current files, and starts the next experiment. The minimum window prevents an early
submission.

The task must use a separate verifier. It must contain `tests/Dockerfile`,
`tests/intermediate.sh`, and `tests/test.sh`. The intermediate script provides the
feedback that the agent sees. The test script stays private. The package chooses the
highest finite intermediate score and uses the private score for that same artifact as
the final Harbor reward. An earlier experiment wins when intermediate scores tie.

## Results

Each trial contains an `autoresearch` directory. It includes:

* `iterations.jsonl` with public validation results.
* `private-iterations.jsonl` with private test results.
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
