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

Install [uv](https://docs.astral.sh/uv/), start Docker, and configure your model
provider's credentials (`OPENAI_API_KEY` for the OpenAI example below).
Docker's default preflight checks for 32000 MiB of free space per trial for
retained artifacts, plus that task's storage requirement, summed across all
selected trials on the jobs host.
Clone this repository and create its Python 3.12 environment:

```bash
git clone https://github.com/bespokelabsai/AutoResearchExam-Terminus.git
cd AutoResearchExam-Terminus
uv sync --python 3.12 --extra modal
```

## Download the tasks

The dataset is public on Harbor Hub. Harbor downloads and caches each selected
task when a run starts. To download the full dataset into Harbor's cache first:

```bash
uv run harbor dataset download bespokelabs/autoresearch-exam@latest --cache
```

## Run one task

`harbor run` starts the full research and grading loop. `trial.py` is its
internal implementation.

```bash
uv run harbor run \
  -d bespokelabs/autoresearch-exam \
  -i bespokelabs/cpu-decoder-graph-executor \
  -a autoresearchexam-terminus \
  -m openai/gpt-5.6-sol \
  -e docker \
  --plugin autoresearch-exam \
  --pk max_iterations=1000 \
  --pk max_duration_seconds=86400 \
  --pk min_time_per_iteration=0 \
  --pk max_turns=10000 \
  --pk max_tokens=32000 \
  --pk reasoning_effort=max
```

Replace the task name and model with the ones you want to use.

## Run all tasks

Remove the `-i` task filter to run every task in the Harbor Hub dataset.
Check the [disk prerequisite](#install) for the full set of trials:

```bash
uv run harbor run \
  -d bespokelabs/autoresearch-exam \
  -a autoresearchexam-terminus \
  -m openai/gpt-5.6-sol \
  -e docker \
  --plugin autoresearch-exam \
  --pk max_iterations=1000 \
  --pk max_duration_seconds=86400 \
  --pk min_time_per_iteration=0 \
  --pk max_turns=10000 \
  --pk max_tokens=32000 \
  --pk reasoning_effort=max
```

The total research window (`max_duration_seconds`) includes agent work, public
validation, and private testing. The experiment or turn limit can end a run
earlier. These examples use the default benchmark settings: 1000 experiments,
10000 model turns, 24 hours, 32000 tokens per response, and `max` reasoning
effort, matching our Sol and Astra runs. There is no session-wide output token
budget. Task build and grader timeouts are separate limits.

After each submission, the agent receives:

* The public validation score.
* The public validation output or error.
* The previous iteration timing and the time remaining in the research window.

The harness runs the private test for every accepted submission and records the
private result, but it never sends the private score or private test output to
the agent. It selects the checkpoint with the highest public validation reward
and reports the private test reward for that same checkpoint.

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

Compute AUARC on final benchmark rewards. The default command converts saved
raw test metrics automatically, using the task name in the summary:

```bash
uv run python scripts/compute_auarc.py \
  jobs/<job>/<trial>/autoresearch/summary.json
```

For an older or custom summary without a saved task name,
pass `--task-name` with that run's task directory name. To compute final
hidden-test AUARC for every task under a jobs directory and their equal-weight
mean:

```bash
uv run python scripts/compute_auarc.py jobs/<job> --all
```

The 29 final-panel difficulty mappings, canonical IDs, and reward names are saved
in `scripts/reward_maps.json`. We use these mappings to compute normalised rewards
before computing AUARC. Some task graders use different reward maps.
`--plain` uses `test_score` unchanged; it is sufficient for benchmark scoring
when those values already contain the final benchmark rewards.

The command prints JSON with hidden test AUARC at these blog time points:

* 15 minutes
* 1 hour
* 4 hours
* 12 hours
* 24 hours

## Settings

The harness settings are:

| Setting | Meaning |
| --- | --- |
| `max_iterations` | Maximum number of submitted experiments. The default is 1000. The allowed range is 1 to 5000. |
| `max_duration_seconds` | Total research window in seconds. The default is 86400 (24 hours). |
| `min_time_per_iteration` | Minimum agent work time before each submission, in minutes. The default is 0, which permits an immediate submission. |
| `llm_backend` | LLM backend used by Terminus 2. The default is `litellm`; the alternative is `tinker`. |
| `max_turns` | Maximum model turns shared by the full agent session. The allowed range is 1 to 50000. The default is 10000. |
| `reasoning_effort` | Reasoning effort sent to the model provider. The default is `max`, as used for Sol and Astra in our benchmark runs. The provider must support the selected value. |
| `max_tokens` | Maximum tokens per model response. The default is 32000, as used in our benchmark runs. |
| `output_token_budget` | Maximum output tokens shared by the full agent session. The default is `None`, which means there is no limit. |
| `auto_summarization` | Whether to summarize the session between experiments. The default is `true`. |

An agent-level `--ak llm_backend=tinker` setting is also preserved when the
plugin setting is omitted.

Add `--extra tinker` when using the Tinker LLM backend.

If running on Modal, use `-e modal` and reduce `max_duration_seconds` to leave
room for grading before its
[24-hour sandbox timeout](https://modal.com/docs/guide/sandboxes#timeouts).
For this example, use `--pk max_duration_seconds=82200` (~23 hours).
The chosen duration also sets the AUARC scoring window.

The GPU tasks use one GPU, as set in `task.toml`. Docker runs use a temporary
task copy with NVIDIA reservations (`docker-compose.yaml` to support local GPUs)
for both the agent and verifier containers. Modal runs use the original task and
request the GPU from Modal.
