"""Real SDK HTTP serialization + real Docker smoke test (NO paid API calls).

The HTTP transport returns scripted OpenAI Responses objects in memory.
Actual Docker runs Bash in the workspace, writing and testing a Python file.
This tests the components together but does not contact the GPT-6 API.
"""

import json
import os
import shutil
import subprocess

import httpx
import pytest
from openai import OpenAI

from examples.gpt6_bash_agent import DockerBash, run_coding_agent


IMAGE = "python:3.11-slim-bookworm"


def test_real_docker_bash_persists_files(tmp_path):
    assert shutil.which("docker"), "Docker CLI is required for this smoke test"
    assert os.getuid() != 0, "Run smoke tests as a non-root user"
    workspace = tmp_path / "coding_workspace"
    bash = DockerBash(workspace, timeout_s=30, image=IMAGE)

    write = bash("printf 'print(40 + 2)\\n' > main.py")
    assert write["exit_code"] == 0, write
    run = bash("python -B main.py")
    assert run["exit_code"] == 0, run
    assert run["stdout"].strip() == "42"
    assert (workspace / "main.py").read_text().strip() == "print(40 + 2)"


def test_openai_sdk_http_and_real_docker_full_loop(tmp_path):
    """Use the *actual OpenAI SDK* to serialize /responses requests and parse replies."""
    assert shutil.which("docker"), "Docker CLI is required for this smoke test"
    assert os.getuid() != 0, "Run smoke tests as a non-root user"
    bash = DockerBash(tmp_path / "agent_workspace", timeout_s=30, image=IMAGE)
    seen = []
    cmd_write = "printf 'def add(a, b):\\n    return a + b\\n' > adder.py"
    cmd_test = (
        "python -B -c 'from adder import add; "
        "assert add(40, 2) == 42; print(add(40, 2))'"
    )

    def mkresponse(resp_id, output):
        return {
            "id": resp_id,
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "model": "gpt-6.2-sol",
            "output": output,
            "usage": {"input_tokens": 16, "output_tokens": 8, "total_tokens": 24},
        }

    def tool(call_id, cmd):
        return {
            "id": "fc_" + call_id,
            "type": "function_call",
            "status": "completed",
            "call_id": call_id,
            "name": "bash",
            "arguments": json.dumps({"command": cmd}),
        }

    def handler(request: httpx.Request):
        assert request.method == "POST"
        assert request.url.path.endswith("/responses")
        data = json.loads(request.content)
        seen.append(data)
        assert data["model"] == "gpt-6.2-sol"
        assert data["tools"][0]["type"] == "function"
        assert data["tools"][0]["name"] == "bash"
        assert data["tools"][0]["strict"] is True
        assert data["tools"][0]["parameters"]["additionalProperties"] is False
        step = len(seen)

        if step == 1:
            assert data["input"] == [{"role": "user", "content": "Write and verify adder.py"}]
            response = mkresponse("resp_one", [tool("call_write", cmd_write)])
        elif step == 2:
            assert data["previous_response_id"] == "resp_one"
            assert data["input"][0]["type"] == "function_call_output"
            assert data["input"][0]["call_id"] == "call_write"
            assert json.loads(data["input"][0]["output"])["exit_code"] == 0
            response = mkresponse("resp_two", [tool("call_test", cmd_test)])
        elif step == 3:
            assert data["previous_response_id"] == "resp_two"
            assert data["input"][0]["call_id"] == "call_test"
            obs = json.loads(data["input"][0]["output"])
            assert obs["exit_code"] == 0 and obs["stdout"].strip() == "42"
            response = mkresponse("resp_three", [{
                "id": "msg_final",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "Verified add(40, 2) == 42", "annotations": []}
                ],
            }])
        else:
            raise AssertionError(f"unexpected extra model call: {step}")
        return httpx.Response(200, json=response)

    http_client = httpx.Client(transport=httpx.MockTransport(handler), timeout=10)
    try:
        client = OpenAI(
            api_key="sk-fake-local-test-not-a-real-key",
            base_url="https://openai-api-mocked.invalid/v1",
            http_client=http_client,
        )
        final = run_coding_agent(client, bash, "Write and verify adder.py", max_steps=5)
    finally:
        http_client.close()

    assert final == "Verified add(40, 2) == 42"
    assert len(seen) == 3
    assert "return a + b" in (bash.workspace / "adder.py").read_text()
    assert all(item["instructions"] for item in seen)
