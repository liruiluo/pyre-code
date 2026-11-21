"""Offline tests of the *real-runner code path*, without API credits or Docker."""
import json
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest

from examples.gpt6_bash_agent import DockerBash, run_coding_agent


class FakeResponses:
    def __init__(self, *outputs):
        self.items = list(outputs)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.items.pop(0)


def response(response_id, *calls, text=""):
    return NS(id=response_id, output=list(calls), output_text=text)


def tool(command, call_id="c1"):
    return NS(type="function_call", name="bash",
              arguments=json.dumps({"command": command}), call_id=call_id)


def test_live_agent_loop_protocol_with_mock_openai_and_bash():
    api = FakeResponses(response("r1", tool("echo test > hello.py")),
                        response("r2", tool("python hello.py", "c2")),
                        response("r3", text="Completed."))
    client = NS(responses=api)
    commands = []
    result = run_coding_agent(client, lambda cmd: (commands.append(cmd) or
         {"exit_code": 0, "stdout": "ok", "stderr": ""}),
         "Write and test a Python script")
    assert result == "Completed."
    assert commands == ["echo test > hello.py", "python hello.py"]
    assert api.requests[1]["previous_response_id"] == "r1"
    assert api.requests[2]["previous_response_id"] == "r2"
    assert api.requests[1]["input"][0]["call_id"] == "c1"
    assert api.requests[2]["input"][0]["call_id"] == "c2"


def test_docker_bash_command_is_sandboxed_and_file_workspace_persists(tmp_path):
    result = NS(stdout="hello\n", stderr="", returncode=0)
    with patch("examples.gpt6_bash_agent.shutil.which", return_value="/usr/bin/docker"), \
         patch("examples.gpt6_bash_agent.os.getuid", return_value=1000), \
         patch("examples.gpt6_bash_agent.os.getgid", return_value=1000), \
         patch("examples.gpt6_bash_agent.subprocess.run", return_value=result) as proc:
        executor = DockerBash(tmp_path / "workspace")
        one = executor("python -V")
        two = executor("ls -la")
    assert one == two == {"stdout": "hello\n", "stderr": "", "exit_code": 0}
    first = proc.call_args_list[0].args[0]
    second = proc.call_args_list[1].args[0]
    assert "--network=none" in first and "--read-only" in first
    assert "--cap-drop=ALL" in first and "--security-opt=no-new-privileges" in first
    assert "--workdir" in first and "/workspace" in first
    assert "bash" in first and "-lc" in first
    mount = next(first[i + 1] for i, x in enumerate(first) if x == "--mount")
    assert f"src={executor.workspace},dst=/workspace" in mount
    assert next(second[i + 1] for i, x in enumerate(second) if x == "--mount") == mount
    assert all(x not in first for x in ["--privileged", "--net=host"])
    assert executor.workspace.is_dir()


def test_docker_bash_timeout_attempts_force_cleanup(tmp_path):
    from subprocess import TimeoutExpired
    with patch("examples.gpt6_bash_agent.shutil.which", return_value="/usr/bin/docker"), \
         patch("examples.gpt6_bash_agent.os.getuid", return_value=1000), \
         patch("examples.gpt6_bash_agent.os.getgid", return_value=1000), \
         patch("examples.gpt6_bash_agent.subprocess.run") as proc:
        proc.side_effect = [TimeoutExpired(cmd="docker", timeout=2), NS(returncode=0)]
        result = DockerBash(tmp_path / "workspace", timeout_s=2)("sleep 99")
    assert result["exit_code"] == 124
    cleanup = proc.call_args_list[1].args[0]
    assert cleanup[:3] == ["docker", "rm", "-f"]


def test_workspace_must_not_be_host_root():
    with pytest.raises(ValueError):
        DockerBash("/")

def test_incomplete_model_response_is_not_treated_as_success():
    bad = NS(id="resp_failed", output=[], output_text="", status="incomplete")
    client = NS(responses=FakeResponses(bad))
    with pytest.raises(RuntimeError, match="not completed"):
        run_coding_agent(client, lambda cmd: "", "Try coding")


def test_missing_final_answer_is_not_treated_as_success():
    empty = NS(id="resp_empty", output=[], output_text="", status="completed")
    client = NS(responses=FakeResponses(empty))
    with pytest.raises(RuntimeError, match="neither tool calls nor final text"):
        run_coding_agent(client, lambda cmd: "", "Try coding")
