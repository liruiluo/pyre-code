"""Minimal real GPT-6 coding-agent harness with an isolated Docker Bash tool.

Usage (Linux / macOS / WSL with Docker):
    python -m pip install openai
    export OPENAI_API_KEY="your-api-key"
    python examples/gpt6_bash_agent.py \
      "Create hello.py, run it, and fix any errors" --workspace ./agent_workspace

The host process calls the OpenAI Responses API. Each Bash command runs in a
short-lived Docker container. Only /workspace is bind-mounted writable.
Container processes are ephemeral; files under --workspace persist between calls.
Never run against a folder containing credentials or private data.
"""

import argparse
import os
import shutil
import subprocess
import uuid
from pathlib import Path

import json

BASH_TOOL = {
    "type": "function",
    "name": "bash",
    "description": "Run a Bash command in the isolated coding workspace. Return stdout, stderr, and exit code.",
    "parameters": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
        "additionalProperties": False,
    },
    "strict": True,
}

AGENT_INSTRUCTIONS = (
    "You are a coding agent. Use bash to inspect and edit files, then run tests. "
    "Treat tool errors as observations and try to fix the code. "
    "Stay inside the isolated workspace and only claim success after verification."
)

def run_coding_agent(client, bash, task, model="gpt-6.2-sol", max_steps=8):
    if max_steps < 1:
        raise ValueError("max_steps must be positive")

    response = client.responses.create(
        model=model, instructions=AGENT_INSTRUCTIONS,
        input=[{"role": "user", "content": task}], tools=[BASH_TOOL],
    )
    for step in range(max_steps + 1):
        calls = [item for item in response.output if item.type == "function_call"]
        if not calls:
            return response.output_text
        if step == max_steps:
            raise RuntimeError("Coding agent exceeded max_steps")

        feedback = []
        for call in calls:
            try:
                if call.name != "bash":
                    raise ValueError("Unknown tool: " + str(call.name))
                arguments = json.loads(call.arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("Tool arguments must be a JSON object")
                command = arguments.get("command")
                if not isinstance(command, str) or not command.strip():
                    raise ValueError("bash requires a nonempty command")
                result = bash(command)
                output = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
            except Exception as exc:
                output = json.dumps({"error": f"{type(exc).__name__}: {exc}"})
            feedback.append({
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": output,
            })
        response = client.responses.create(
            model=model, instructions=AGENT_INSTRUCTIONS,
            previous_response_id=response.id, input=feedback, tools=[BASH_TOOL],
        )
    raise RuntimeError("Coding agent did not finish")


class DockerBash:
    """Execute a model-selected Bash command within a bounded Docker runtime.

    workspace: persistent HOST directory mounted at /workspace (read-write).
    runtime: disposable container created for each tool call; no network or secrets.
    The shell process and environment variables DO NOT persist between calls.
    """

    def __init__(self, workspace, timeout_s=20, image="python:3.11-slim-bookworm"):
        self.workspace = Path(workspace).expanduser().resolve()
        if self.workspace == Path("/"):
            raise ValueError("Never mount the host root as the agent workspace")
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.timeout_s = timeout_s
        self.image = image

    def __call__(self, command):
        if not isinstance(command, str) or not command.strip():
            raise ValueError("Bash command must be a nonempty string")
        if not shutil.which("docker"):
            raise RuntimeError("Docker CLI is required for the sandbox")
        if not hasattr(os, "getuid") or os.getuid() == 0:
            raise RuntimeError("Use a non-root Linux/macOS/WSL user for this example")
        name = "pyre-code-agent-" + uuid.uuid4().hex[:12]
        argv = [
            "docker", "run", "--rm", "--name", name,
            "--network=none",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--read-only",
            "--tmpfs=/tmp:rw,nosuid,size=64m",
            "--pids-limit=64",
            "--memory=512m",
            "--cpus=1",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "--mount", f"type=bind,src={self.workspace},dst=/workspace,rw",
            "--workdir", "/workspace",
            self.image, "bash", "-lc", command,
        ]
        try:
            run = subprocess.run(
                argv, capture_output=True, text=True, timeout=self.timeout_s, check=False
            )
            return {
                "stdout": run.stdout[:5000],
                "stderr": run.stderr[:5000],
                "exit_code": run.returncode,
            }
        except subprocess.TimeoutExpired:
            # CLI timeout may leave the container alive; force-clean by its unique name.
            subprocess.run(
                ["docker", "rm", "-f", name],
                capture_output=True, text=True, timeout=5, check=False,
            )
            return {
                "stdout": "",
                "stderr": f"Command exceeded {self.timeout_s}s timeout",
                "exit_code": 124,
            }


def main():
    parser = argparse.ArgumentParser(
        description="A minimal real GPT-6 Coding Agent with a Docker-isolated Bash tool."
    )
    parser.add_argument("task", help="Coding task for GPT-6 to complete")
    parser.add_argument("--workspace", default="./agent_workspace",
                        help="DEDICATED writable folder; never use a folder with secrets")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-6.2-sol"))
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=20)
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        parser.error("Set OPENAI_API_KEY locally; do not paste it into Pyre or GitHub")
    if not shutil.which("docker"):
        parser.error("Docker must be installed and available in PATH")
    if args.max_steps < 1 or args.timeout < 1:
        parser.error("--max-steps and --timeout must be positive")

    from openai import OpenAI  # optional dependency; only required for the live run

    client = OpenAI()
    bash = DockerBash(args.workspace, timeout_s=args.timeout)
    result = run_coding_agent(client, bash, args.task,
                              model=args.model, max_steps=args.max_steps)
    print("\nFinal answer:\n" + result)
    print("Workspace files:", bash.workspace)


if __name__ == "__main__":
    main()
