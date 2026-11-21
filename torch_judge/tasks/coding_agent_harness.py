"""Practice the OpenAI Responses API coding-agent/Bash orchestration loop with mocks."""

TASK = {
    "title": "Minimal GPT-6 Coding Agent Loop (Bash)",
    "title_zh": "手写最小 GPT-6 Coding Agent Harness（Bash）",
    "difficulty": "Hard",
    "description_en": "Implement the real **Responses API function-tool loop** used by a minimal Coding Agent. The model may ask to run Bash, inspect or write files, run tests, react to errors, then produce a final answer. You implement the **orchestration**, not the LLM itself.\n\n**Signature:** \\`run_coding_agent(client, bash, task, model=\"gpt-6.2-sol\", max_steps=8) -> str\\`\n\n- \\`client\\` is an OpenAI-compatible client with \\`client.responses.create(...)\\`.\n- \\`bash(command)\\` is a supplied function that executes Bash in an **isolated sandbox** and returns a string or a dict containing stdout/stderr/exit_code.\n- \\`task\\` is the user's request.\n- Initial request: Responses API, user input, a single \\`bash\\` function tool with a JSON-schema \\`command\\` argument.\n- Model outputs may contain multiple \\`function_call\\` items; execute all in order and return one \\`function_call_output\\` per original \\`call_id\\`.\n- Continue with \\`previous_response_id=response.id\\` and ALL corresponding tool outputs; allow multiple iterations.\n- When no tool calls are returned, return \\`response.output_text\\`.\n- Tool errors, bad JSON and unsupported tool names should become error observations sent back to the model, not crash or execute unexpected commands.\n- Stop after \\`max_steps\\` **tool-execution rounds**; reject invalid limits.\n\n**Important:** In Pyre, clients and Bash are mocked so **no real API key is used and no user-supplied shell command executes on the judging host**. A separate, runnable example \\`examples/gpt6_bash_agent.py\\` uses the real GPT-6.2 Sol Responses API plus a Docker-isolated shell. Python package \\`openai\\` and API credentials are required only for the live example.\n\nThe **workspace** is the persistent folder of files; the **execution environment** is the sandbox/container running commands. They are not the same thing.",
    "description_zh": "实现真正的 **Responses API 函数工具循环（Coding Agent Harness）**：GPT-6 可要求运行 Bash、读取/修改代码文件、跑测试，并根据错误继续修改，最后输出结果。你要手写的是 **Agent 编排逻辑**，而不是模型。\n\n**函数签名:** \\`run_coding_agent(client, bash, task, model=\"gpt-6.2-sol\", max_steps=8) -> str\\`\n\n- \\`client\\`：具有 \\`client.responses.create(...)\\` 的 OpenAI 兼容客户端。\n- \\`bash(command)\\`：题目提供的执行器接口，代表**隔离容器中的 Bash**，返回字符串或包含 stdout/stderr/exit_code 的字典。\n- 第一次调用：传入用户任务以及 \\`bash\\` 工具定义（参数为 JSON Schema 的 \\`command\\`）。\n- 一个响应可能包含**多个** \\`function_call\\`；按顺序执行，并使用原始 \\`call_id\\` 返回对应的 \\`function_call_output\\`。\n- 使用 \\`previous_response_id=response.id\\` 和工具返回继续调用，直到 GPT-6 不再请求工具，返回 \\`response.output_text\\`。\n- 非法 JSON、未知工具、Bash 执行报错，必须变为 Observation 返回模型，不能误执行。\n- 最多执行 \\`max_steps\\` 轮工具，防止无限循环。\n\n**判题说明：** Pyre 用模拟 Client/Bash 测试，**不使用真实 API Key，也不在网站判题服务器执行任意 Bash**。仓库的 \\`examples/gpt6_bash_agent.py\\` 是真实调用 GPT-6.2 Sol + Docker 的可运行版本。\n\n**区分两个概念：** Workspace 是持久存储代码文件的目录；执行环境是运行 Bash 的隔离容器。二者不是同一件事。",
    "function_name": "run_coding_agent",
    "hint": "Define a bash function tool; call client.responses.create. Handle every response.output item with type=='function_call', validate JSON arguments, execute bash, append function_call_output with original call_id, and continue using previous_response_id; enforce max_steps.",
    "hint_zh": "定义 bash 函数工具；调用 client.responses.create。遍历 response.output 中的 function_call，解析参数并执行 bash，按原 call_id 构造 function_call_output，然后通过 previous_response_id 继续。限制 max_steps，并将错误作为 Observation 返回。",
    "tests": [
        {
            "name": "Direct final answer without any Bash command",
            "code": """
import json
from types import SimpleNamespace as NS

class FakeResponses:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.replies.pop(0)

def tool(command, call_id="call_1"):
    return NS(type="function_call", name="bash",
              arguments=json.dumps({"command": command}), call_id=call_id)

def other_tool(name, arguments, call_id="call_bad"):
    return NS(type="function_call", name=name, arguments=arguments, call_id=call_id)

def reply(response_id, *items, text=""):
    return NS(id=response_id, output=list(items), output_text=text)

def fake_client(*replies):
    return NS(responses=FakeResponses(replies))
client = fake_client(reply("r0", text="Already done."))
commands = []
result = {fn}(client, commands.append, "Check the code")
assert result == "Already done."
assert commands == []
request = client.responses.requests[0]
assert request["model"] == "gpt-6.2-sol"
assert request["input"] == [{"role": "user", "content": "Check the code"}]
assert request["tools"][0]["name"] == "bash"
assert len(client.responses.requests) == 1
""",
        },
        {
            "name": "One Bash call and final answer, correct call_id",
            "code": """
import json
from types import SimpleNamespace as NS

class FakeResponses:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.replies.pop(0)

def tool(command, call_id="call_1"):
    return NS(type="function_call", name="bash",
              arguments=json.dumps({"command": command}), call_id=call_id)

def other_tool(name, arguments, call_id="call_bad"):
    return NS(type="function_call", name=name, arguments=arguments, call_id=call_id)

def reply(response_id, *items, text=""):
    return NS(id=response_id, output=list(items), output_text=text)

def fake_client(*replies):
    return NS(responses=FakeResponses(replies))
client = fake_client(
    reply("r1", tool("printf 'print(1)\\n' > main.py", "c1")),
    reply("r2", text="Created main.py"),
)
commands = []
def bash(command):
    commands.append(command)
    return {"stdout": "", "stderr": "", "exit_code": 0}
answer = {fn}(client, bash, "Write main.py", model="gpt-6.2-sol")
assert answer == "Created main.py"
assert len(commands) == 1 and "main.py" in commands[0]
continuation = client.responses.requests[1]
assert continuation["previous_response_id"] == "r1"
assert len(continuation["input"]) == 1
output = continuation["input"][0]
assert output["type"] == "function_call_output" and output["call_id"] == "c1"
assert json.loads(output["output"])["exit_code"] == 0
""",
        },
        {
            "name": "Parallel tool calls are all executed and forwarded together",
            "code": """
import json
from types import SimpleNamespace as NS

class FakeResponses:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.replies.pop(0)

def tool(command, call_id="call_1"):
    return NS(type="function_call", name="bash",
              arguments=json.dumps({"command": command}), call_id=call_id)

def other_tool(name, arguments, call_id="call_bad"):
    return NS(type="function_call", name=name, arguments=arguments, call_id=call_id)

def reply(response_id, *items, text=""):
    return NS(id=response_id, output=list(items), output_text=text)

def fake_client(*replies):
    return NS(responses=FakeResponses(replies))
client = fake_client(
    reply("r1", tool("pwd", "c1"), tool("ls -la", "c2")),
    reply("r2", text="Inspected workspace."),
)
commands = []
def bash(command):
    commands.append(command)
    return {"stdout": command, "stderr": "", "exit_code": 0}
assert {fn}(client, bash, "Inspect") == "Inspected workspace."
assert commands == ["pwd", "ls -la"]
request = client.responses.requests[1]
assert [x["call_id"] for x in request["input"]] == ["c1", "c2"]
assert [json.loads(x["output"])["stdout"] for x in request["input"]] == commands
""",
        },
        {
            "name": "Tool result, failing test and second correction cycle",
            "code": """
import json
from types import SimpleNamespace as NS

class FakeResponses:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.replies.pop(0)

def tool(command, call_id="call_1"):
    return NS(type="function_call", name="bash",
              arguments=json.dumps({"command": command}), call_id=call_id)

def other_tool(name, arguments, call_id="call_bad"):
    return NS(type="function_call", name=name, arguments=arguments, call_id=call_id)

def reply(response_id, *items, text=""):
    return NS(id=response_id, output=list(items), output_text=text)

def fake_client(*replies):
    return NS(responses=FakeResponses(replies))
client = fake_client(
    reply("r1", tool("python -m pytest -q", "c1")),
    reply("r2", tool("sed -i 's/old/new/' main.py", "c2")),
    reply("r3", text="Fixed and tested."),
)
commands = []
def bash(command):
    commands.append(command)
    return {"stdout": "", "stderr": "AssertionError" if len(commands) == 1 else "",
            "exit_code": 1 if len(commands) == 1 else 0}
assert {fn}(client, bash, "Fix test") == "Fixed and tested."
assert len(commands) == 2
assert client.responses.requests[1]["previous_response_id"] == "r1"
assert client.responses.requests[2]["previous_response_id"] == "r2"
assert json.loads(client.responses.requests[1]["input"][0]["output"])["exit_code"] == 1
""",
        },
        {
            "name": "Malformed arguments and unknown tools never run Bash",
            "code": """
import json
from types import SimpleNamespace as NS

class FakeResponses:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.replies.pop(0)

def tool(command, call_id="call_1"):
    return NS(type="function_call", name="bash",
              arguments=json.dumps({"command": command}), call_id=call_id)

def other_tool(name, arguments, call_id="call_bad"):
    return NS(type="function_call", name=name, arguments=arguments, call_id=call_id)

def reply(response_id, *items, text=""):
    return NS(id=response_id, output=list(items), output_text=text)

def fake_client(*replies):
    return NS(responses=FakeResponses(replies))
client = fake_client(
    reply("r1", other_tool("bash", "{bad json", "c1"),
          other_tool("sudo_shell", '{"command": "rm -rf /"}', "c2"),
          other_tool("bash", '{"command": 1}', "c3")),
    reply("r2", text="Cannot execute these."),
)
executed = []
assert {fn}(client, executed.append, "Avoid invalid calls") == "Cannot execute these."
assert executed == []
results = client.responses.requests[1]["input"]
assert len(results) == 3
assert all("error" in json.loads(x["output"]) for x in results)
assert [x["call_id"] for x in results] == ["c1", "c2", "c3"]
""",
        },
        {
            "name": "Bash exceptions are returned to model, not raised",
            "code": """
import json
from types import SimpleNamespace as NS

class FakeResponses:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.replies.pop(0)

def tool(command, call_id="call_1"):
    return NS(type="function_call", name="bash",
              arguments=json.dumps({"command": command}), call_id=call_id)

def other_tool(name, arguments, call_id="call_bad"):
    return NS(type="function_call", name=name, arguments=arguments, call_id=call_id)

def reply(response_id, *items, text=""):
    return NS(id=response_id, output=list(items), output_text=text)

def fake_client(*replies):
    return NS(responses=FakeResponses(replies))
client = fake_client(reply("r1", tool("make test", "c1")), reply("r2", text="Unavailable."))
def bash(command):
    raise TimeoutError("shell timed out")
assert {fn}(client, bash, "Run tests") == "Unavailable."
out = client.responses.requests[1]["input"][0]
assert "TimeoutError" in json.loads(out["output"])["error"]
""",
        },
        {
            "name": "Max steps prevents unbounded tool loops",
            "code": """
import json
from types import SimpleNamespace as NS

class FakeResponses:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.replies.pop(0)

def tool(command, call_id="call_1"):
    return NS(type="function_call", name="bash",
              arguments=json.dumps({"command": command}), call_id=call_id)

def other_tool(name, arguments, call_id="call_bad"):
    return NS(type="function_call", name=name, arguments=arguments, call_id=call_id)

def reply(response_id, *items, text=""):
    return NS(id=response_id, output=list(items), output_text=text)

def fake_client(*replies):
    return NS(responses=FakeResponses(replies))
client = fake_client(reply("r1", tool("pwd", "c1")),
                     reply("r2", tool("pwd", "c2")),
                     reply("r3", tool("pwd", "c3")))
commands = []
try:
    {fn}(client, commands.append, "Loop", max_steps=2)
except RuntimeError as exc:
    assert "max_steps" in str(exc)
else:
    raise AssertionError("must stop after max_steps")
assert commands == ["pwd", "pwd"]
assert len(client.responses.requests) == 3
""",
        },
        {
            "name": "Invalid maximum steps rejected without model request",
            "code": """
import json
from types import SimpleNamespace as NS

class FakeResponses:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.replies.pop(0)

def tool(command, call_id="call_1"):
    return NS(type="function_call", name="bash",
              arguments=json.dumps({"command": command}), call_id=call_id)

def other_tool(name, arguments, call_id="call_bad"):
    return NS(type="function_call", name=name, arguments=arguments, call_id=call_id)

def reply(response_id, *items, text=""):
    return NS(id=response_id, output=list(items), output_text=text)

def fake_client(*replies):
    return NS(responses=FakeResponses(replies))
client = fake_client()
try:
    {fn}(client, lambda cmd: {}, "No steps", max_steps=0)
except ValueError:
    pass
else:
    raise AssertionError("invalid max_steps must fail")
assert client.responses.requests == []
""",
        },
    ],
    "solution": '''import json

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
    raise RuntimeError("Coding agent did not finish")''',
    "explanation": "A minimal **real** Coding Agent harness is a loop: call GPT-6 → inspect function_call items → dispatch Bash in a sandbox → send function_call_output with the same call_id → continue via previous_response_id → return the final response. Multiple tool calls in one response require matching outputs. Tests use a mock API and executor; the live example in examples/gpt6_bash_agent.py uses an actual OpenAI Responses client and a Docker sandbox. The persistent **workspace** holds code, whereas the Docker runtime executes Bash. Never put API keys in the web grader or run an unrestricted model-command subprocess on the Pyre host.",
    "demo": """
# This demo uses a mock client, not the OpenAI API.
from types import SimpleNamespace as NS
class FakeResponses:
    def __init__(self): self.n = 0
    def create(self, **kwargs):
        self.n += 1
        if self.n == 1:
            return NS(id="r1", output=[NS(type="function_call", name="bash",
                call_id="c1", arguments='{"command":"python -V"}')], output_text="")
        return NS(id="r2", output=[], output_text="Python is available.")
client = NS(responses=FakeResponses())
print(run_coding_agent(client, lambda cmd: {"stdout":"Python 3.11","exit_code":0}, "Check Python"))
""",
}
