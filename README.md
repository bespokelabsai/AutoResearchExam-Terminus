# AutoResearchExam Harness

This is the official Terminus harness for
[AutoResearchExam](https://github.com/bespokelabsai/AutoResearchExam).
It adds a timed research window to Harbor. One
agent session can submit many experiments during the window. The agent receives
the public validation result and the remaining time after each submission. The
private test result stays outside the agent environment.

The benchmark budget and AUARC horizon are 24 hours (86400 seconds) by default.
The examples below use local Docker.

To use Claude Code, Codex, or another harness, follow the
[custom harness guide](scripts/custom_harness.md). The AUARC script accepts
timestamped scores from any harness.

## Install

Clone this repository and create its environment with
[uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/bespokelabsai/AutoResearchExam-Terminus.git
cd AutoResearchExam-Terminus
uv sync --python 3.12 --extra modal
```

Add `--extra tinker` when using the Tinker LLM backend.

## Download the tasks

The dataset is public on Harbor Hub. Harbor downloads and caches each selected
task when a run starts. To download the full dataset into Harbor's cache first:

```bash
uv run harbor dataset download bespokelabs/autoresearch-exam@latest --cache
```

## Run one task

```bash
uv run harbor run \
  -d bespokelabs/autoresearch-exam \
  -i bespokelabs/cpu-decoder-graph-executor \
  -a autoresearchexam-terminus \
  -m openai/gpt-5.6-sol \
  -e docker \
  --plugin autoresearch-exam \
  --pk max_iterations=5000 \
  --pk max_duration_seconds=86400 \
  --pk min_time_per_iteration=0 \
  --pk max_turns=50000 \
  --pk reasoning_effort=high
```

Replace the task name and model with the ones you want to use.

For Modal, use a shorter run, such as 22 hours for this CPU example, to leave
room before its [24-hour sandbox timeout](https://modal.com/docs/guide/sandboxes#timeouts).

The GPU tasks use one GPU, as set in `task.toml`. Docker runs use a temporary
task copy with NVIDIA reservations (`docker-compose.yaml` to support local GPUs)
for both the agent and verifier containers. Modal runs use the original task and
request the GPU from Modal.

## Run all tasks

Remove the `-i` task filter to run every task in the Harbor Hub dataset:

```bash
uv run harbor run \
  -d bespokelabs/autoresearch-exam \
  -a autoresearchexam-terminus \
  -m openai/gpt-5.6-sol \
  -e docker \
  --plugin autoresearch-exam \
  --pk max_iterations=5000 \
  --pk max_duration_seconds=86400 \
  --pk min_time_per_iteration=0 \
  --pk max_turns=50000 \
  --pk reasoning_effort=high
```

The harness settings are:

| Setting | Meaning |
| --- | --- |
| `max_iterations` | Maximum number of submitted experiments. The default is 5000. The allowed range is 1 to 5000. |
| `max_duration_seconds` | Total research window in seconds. The default is 86400 (24 hours). |
| `min_time_per_iteration` | Minimum agent work time before each submission, in minutes. The default is 0, which permits an immediate submission. |
| `llm_backend` | LLM backend used by Terminus 2. The default is `litellm`; the alternative is `tinker`. |
| `max_turns` | Maximum model turns shared by the full agent session. The allowed range is 1 to 50000. The default is 50000. |
| `reasoning_effort` | Reasoning effort sent to the model provider. The provider must support the selected value. |
| `output_token_budget` | Maximum output tokens shared by the full agent session. The default is `None`, which means there is no limit. |
| `auto_summarization` | Whether to summarize the session between experiments. The default is `true`. |

An agent-level `--ak llm_backend=tinker` setting is also preserved when the
plugin setting is omitted.

The total research window (`max_duration_seconds`) includes agent work, public
validation, and private testing. The experiment or turn limit can end a run
earlier. The default limits are 5000 experiments and 50000 turns, with no output
token budget. Task build and grader timeouts are separate limits.

After each submission, the agent receives:

* The public validation score.
* The public validation output or error.
* The previous iteration timing and the time remaining in the research window.

The harness runs the private test for every accepted submission and records the
private result, but it never sends the private score or private test output to
the agent. The experiment with the highest public score is selected as the final
score and uses the private score for that same experiment as the final reward.

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

## Compute AUARC

Use the included script with a trial's `summary.json` file:

```bash
uv run python scripts/compute_auarc.py \
  jobs/<job>/<trial>/autoresearch/summary.json \
  --task-name cpu-llm-decode-throughput
```

Use the task's directory name for `--task-name`. New summaries save this name,
so the option can then be omitted. To compute final hidden-test AUARC for every
task under a jobs directory and their equal-weight mean:

```bash
uv run python scripts/compute_auarc.py jobs/<job> --all
```

The 29 final-panel difficulty mappings, canonical IDs, and reward names are saved
in `scripts/reward_maps.json`. We use these mappings to compute normalised rewards
before computing AUARC. Pass `--plain` only to reproduce AUARC from the
grader-reported rewards without applying those mappings.

The command prints JSON with hidden test AUARC at these blog time points:

* 15 minutes
* 1 hour
* 4 hours
* 12 hours
* 24 hours
