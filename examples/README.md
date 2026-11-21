# Minimal GPT-6 Coding Agent Harness (real Bash)

This is a small, **real** coding-agent harness: the host calls GPT-6 through the
[OpenAI Responses API](https://developers.openai.com/api/docs/guides/function-calling),
GPT-6 emits `bash` function calls, the host executes them inside a restricted Docker
container, returns observations, and repeats until the model answers.

The corresponding **Pyre exercise** is:
`/problems/coding_agent_harness?path=post-training-interview-essentials`.
Its online grading uses mock clients and mock Bash (no API key or shell execution).

## Run locally

On Linux, macOS or WSL with Docker running, install the OpenAI Python SDK:

```bash
python -m pip install openai
docker pull python:3.11-slim-bookworm
export OPENAI_API_KEY="YOUR_REAL_OPENAI_API_KEY"
python examples/gpt6_bash_agent.py \
  "Write a hello.py that defines add(a, b), write a unittest, and run it" \
  --workspace ./agent_workspace \
  --model gpt-6.2-sol --max-steps 8
```

Default: GPT-6.2 Sol. Override `--model` to another model your API project
can access, e.g. `gpt-6.2-astra`. **ChatGPT subscriptions and API billing
are separate.** This demo performs real billed API calls.

## The architecture

```text
Host process (Python):
  task -> client.responses.create(model="gpt-6.2-sol", tools=[bash])
       <- response.output: function_call(command, call_id)
       -> DockerBash(command)  [host controls Docker lifecycle]
         -> Docker container: bash -lc COMMAND
            - network disabled, no privileges, constrained CPU/RAM
            - /workspace bind-mounted to a dedicated host folder
         <- stdout, stderr, exit_code
       -> function_call_output(call_id, result), previous_response_id
       <- final response or another function_call
```

**Workspace vs. execution environment are different.**
- Workspace: `./agent_workspace` on the host, bind-mounted to `/workspace`
  inside each container. File edits persist across agent steps and after the run.
- Execution environment: a **new disposable Docker container per command**;
  shell variables, processes and installed dependencies do not persist.
- The API key is used only by the host OpenAI SDK, **never passed to Docker**.

## Scope / safety

- Container limits: `--network=none`, `--read-only` root filesystem,
  dropped Linux capabilities, no-new-privileges, unprivileged UID/GID,
  CPU/memory/PID limits, and a timeout with forced cleanup.
- Only mount a **dedicated, disposable folder with no secrets**, not your home
  directory, SSH keys, production source checkout, or Docker socket.
- Docker isolation is a useful defense but **not a hardened multi-tenant sandbox**.
  Review commands and permissions before using it with unknown input.
- The host loop bounds tool calls with `max_steps` and returns errors to GPT-6
  as observations, so it can fix code or explain a blocker.
- No web access or package download *inside* the container. The base image
  supports Python's standard library; use an audited prebuilt image for extras.
- This exercise is about the **harness**, not a full coding-agent product:
  no git patch protocol, context compaction, task RL, or cross-episode memory.

## Test without API credits or Docker

```bash
python -m pytest -q tests/test_interview_reference_solutions.py \
  tests/test_live_coding_agent_example.py
```

The Pyre judge task and these tests use scripted fake model replies and mock
Bash execution. They cover tool-call IDs, parallel calls, malformed inputs,
errors, follow-up iterations, and step limits. They do not claim to have
verified a paid, live GPT-6 run.

Official docs:
- https://developers.openai.com/api/docs/guides/function-calling
- https://developers.openai.com/api/docs/guides/tools-shell
- https://developers.openai.com/api/docs/models
