# AutoResearchExam Harness

This package adds a timed research window to Harbor for the
[AutoResearchExam](https://github.com/bespokelabsai/AutoResearchExam) tasks. One
agent session can submit many experiments during the window. The agent receives
the public validation result and the remaining time after each submission. The
private test result stays outside the agent environment.

The harness supports local Docker and Modal.

## Requirements

You need Python 3.12 and Harbor 0.22. For local runs, install Docker and make
sure its service is running. For Modal runs, sign in to Modal before starting a
job.

Your model provider API key must be available in the environment where you run
Harbor. See the Harbor documentation for the environment variable required by
your provider.

## Install

Clone this repository and install it in your Harbor environment:

```bash
git clone https://github.com/bespokelabsai/AutoResearchExam-Terminus.git
cd AutoResearchExam-Terminus
pip install .
```

Install the Modal support when you plan to use Modal:

```bash
pip install ".[modal]"
```

Check that Harbor can find the plugin:

```bash
harbor plugins list
```

## Download the tasks

The dataset is public on Harbor Hub. Harbor downloads and caches each selected
task when a run starts. To download the full dataset into Harbor's cache first:

```bash
harbor dataset download bespokelabs/autoresearch-exam@latest --cache
```

## Run one task

```bash
harbor run \
  -d bespokelabs/autoresearch-exam \
  -i cpu-decoder-graph-executor \
  -a autoresearchexam-terminus \
  -m openai/gpt-5.6-sol \
  -e docker \
  --plugin autoresearch-exam \
  --pk max_iterations=500 \
  --pk max_duration_seconds=86400 \
  --pk min_time_per_iteration=0 \
  --pk max_turns=2000 \
  --pk reasoning_effort=high
```

Use `-e modal` to run the task on Modal. Replace the task name and model with the
ones you want to use.

## Run all tasks

Remove the `-i` task filter to run every task in the Harbor Hub dataset:

```bash
harbor run \
  -d bespokelabs/autoresearch-exam \
  -a autoresearchexam-terminus \
  -m openai/gpt-5.6-sol \
  -e docker \
  --plugin autoresearch-exam \
  --pk max_iterations=500 \
  --pk max_duration_seconds=86400 \
  --pk min_time_per_iteration=0 \
  --pk max_turns=2000 \
  --pk reasoning_effort=high
```

The plugin settings are:

| Setting | Meaning |
| --- | --- |
| `max_iterations` | Maximum number of submitted experiments. The allowed range is 1 to 5000. |
| `max_duration_seconds` | Total research window in seconds. |
| `min_time_per_iteration` | Minimum agent work time before each submission, in minutes. Zero permits an immediate submission. |
| `max_turns` | Maximum model turns shared by the full agent session. The allowed range is 1 to 50000. |
| `reasoning_effort` | Reasoning effort sent to the model provider. The provider must support the selected value. |
| `output_token_budget` | Maximum output tokens shared by the full agent session. The default is `None`, which means there is no limit. |
| `auto_summarization` | Whether to summarize the session between experiments. The default is `true`. |

The total research window includes agent work, artifact collection, public
validation, private testing, and harness work. One experiment can use all the
time that remains. The harness does not interrupt an experiment with reminder
messages.

After each submission, the agent receives:

* The public validation score.
* The public validation output or error.
* The previous iteration timing and the time remaining in the research window.

The harness runs the private test for every accepted submission. It records the
private result on the host, but it never sends the private score or private test
output to the agent. The harness selects the experiment with the highest finite
public score. It uses the private score for that same experiment as the final
Harbor reward. If public scores tie, the earlier experiment wins.

Local Docker can use a research window of up to 172800 seconds. Modal limits a
sandbox to 86400 seconds. A Modal run must also leave enough time for artifact
collection and the final public and private graders, so use a research window
shorter than 86400 seconds.

## Results

Harbor writes each job under `jobs/`. Every trial has an `autoresearch`
directory with these files:

* `iterations.jsonl` contains each public validation result, submission time,
  and timing record.
* `private-iterations.jsonl` contains each private test result and grader time.
* `summary.json` contains the selected experiment, final score, all recorded
  scores, timing, model, backend, and stop reason.
* `iterations/<number>/artifact.tar.gz` contains the exact submitted artifact.
* `iterations/<number>/intermediate` contains the public grader logs.
* `iterations/<number>/test` contains the private grader logs.

Watch public scores during a run:

```bash
tail -f jobs/<job>/<trial>/autoresearch/iterations.jsonl
```

Read the selected public and private scores after a run:

```bash
jq '{public_best_score, selected_iteration, selected_test_score, scores}' \
  jobs/<job>/<trial>/autoresearch/summary.json
```

The harness uses private host permissions for result files that contain private
scores. It never mounts those files into the agent environment. You can also use
`harbor view` with the jobs directory to open Harbor's result viewer.

## Task contract

Each task must use a separate verifier environment. Its `tests/` directory must
contain `Dockerfile`, `intermediate.sh`, and `test.sh`. The harness gives the
output from `intermediate.sh` to the agent. It keeps `test.sh` and its output
private.
