"""ReAct trace parsing task."""

TASK = {
    "title": "Parse a ReAct Agent Trace",
    "title_zh": "解析 ReAct Agent 轨迹",
    "difficulty": "Easy",
    "description_en": "Parse the text output of a ReAct-style agent into structured steps.\n\nReAct interleaves `Thought:`, `Action:`, `Action Input:` and `Observation:` blocks. Agent frameworks need to parse these traces to extract which tools were called and what the environment returned. The final step may be incomplete — the agent emitted an action but the observation has not arrived yet.\n\n**Signature:** `parse_react_trace(text) -> list[dict]`\n\n**Parameters:**\n- `text` — full ReAct trace, one directive per line\n\n**Returns:** list of steps, each `{'thought': str, 'action': str, 'action_input': str, 'observation': str | None}` in order. A trailing step without an observation gets `observation=None`. Fields never seen default to `None`.\n\n**Constraints:**\n- Lines are matched on the exact prefixes `Thought:`, `Action:`, `Action Input:`, `Observation:` (leading/trailing whitespace of each line tolerated)\n- A new `Thought:` line starts a new step\n- Blank lines and unmarked lines are ignored",
    "description_zh": "把 ReAct 风格 agent 的文本输出解析成结构化步骤。\n\nReAct 交替输出 `Thought:`、`Action:`、`Action Input:` 和 `Observation:` 块。agent 框架需要解析这种轨迹来提取调用了哪些工具、环境返回了什么。最后一步可能不完整——agent 已发出动作但观察还没到。\n\n**签名:** `parse_react_trace(text) -> list[dict]`\n\n**参数:**\n- `text` — 完整 ReAct 轨迹，每行一个指令\n\n**返回:** 步骤列表，每步 `{'thought': str, 'action': str, 'action_input': str, 'observation': str | None}`，按顺序。末尾缺观察的步骤 `observation=None`。从未出现的字段默认 `None`。\n\n**约束:**\n- 按精确前缀 `Thought:`、`Action:`、`Action Input:`、`Observation:` 匹配（容忍行首尾空白）\n- 新的 `Thought:` 行开启新步骤\n- 空行和无标记行忽略",
    "function_name": "parse_react_trace",
    "hint": "Walk line by line keeping a `current` dict; open a new step on each `Thought:` (appending the previous one), and flush the last step at the end.",
    "hint_zh": "逐行扫描维护一个 `current` 字典；每个 `Thought:` 开新步骤（提交上一条），结尾提交最后一条。",
    "tests": [
        {
            "name": "Two complete steps",
            "code": """
text = '''Thought: I need to find the weather.
Action: get_weather
Action Input: Beijing

Observation: 22C, sunny

Thought: Now I can answer.
Action: finish
Action Input: It is 22C and sunny.

Observation: done'''
steps = {fn}(text)
assert isinstance(steps, list) and len(steps) == 2, f'Expected 2 steps, got {len(steps)}'
assert steps[0]['thought'] == 'I need to find the weather.'
assert steps[0]['action'] == 'get_weather'
assert steps[0]['action_input'] == 'Beijing'
assert steps[0]['observation'] == '22C, sunny'
assert steps[1]['action'] == 'finish'
assert steps[1]['observation'] == 'done'
""",
        },
        {
            "name": "Trailing incomplete step",
            "code": """
text = '''Thought: Look up the price.
Action: search
Action Input: gpu price'''
steps = {fn}(text)
assert len(steps) == 1, f'Expected 1 step, got {len(steps)}'
assert steps[0]['action'] == 'search'
assert steps[0]['observation'] is None, 'Missing observation should be None'
""",
        },
        {
            "name": "Intermediate step without observation",
            "code": """
text = '''Thought: one
Action: a
Action Input: x

Thought: two
Action: b
Action Input: y

Observation: ok'''
steps = {fn}(text)
assert len(steps) == 2, f'Expected 2 steps, got {len(steps)}'
assert steps[0]['observation'] is None, 'First step has no observation'
assert steps[1]['observation'] == 'ok'
""",
        },
        {
            "name": "Whitespace tolerance and blank lines",
            "code": """
text = 'Thought:   spaced out   \\n\\nAction:  tool_a\\n\\n   Action Input: arg1\\n\\nObservation:   result   '
steps = {fn}(text)
assert steps[0]['thought'] == 'spaced out'
assert steps[0]['action'] == 'tool_a'
assert steps[0]['action_input'] == 'arg1'
assert steps[0]['observation'] == 'result'
""",
        },
        {
            "name": "Every step dict has all four keys",
            "code": """
text = '''Action: only_action
Action Input: only_input'''
steps = {fn}(text)
assert len(steps) == 1
for key in ('thought', 'action', 'action_input', 'observation'):
    assert key in steps[0], f'Missing key: {key}'
assert steps[0]['thought'] is None
assert steps[0]['observation'] is None
""",
        },
    ],
    "solution": '''def parse_react_trace(text):
    keys = ("thought", "action", "action_input", "observation")
    template = {k: None for k in keys}
    steps = []
    current = dict(template)

    def flush():
        nonlocal current
        if any(v is not None for v in current.values()):
            steps.append(current)
        current = dict(template)

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Thought:"):
            flush()
            current = dict(template)
            current["thought"] = stripped[len("Thought:"):].strip()
        elif stripped.startswith("Action Input:"):
            current["action_input"] = stripped[len("Action Input:"):].strip()
        elif stripped.startswith("Action:"):
            current["action"] = stripped[len("Action:"):].strip()
        elif stripped.startswith("Observation:"):
            current["observation"] = stripped[len("Observation:"):].strip()
        # blank / unmarked lines are ignored

    flush()
    return steps''',
    "demo": """text = '''Thought: I should check the weather.
Action: get_weather
Action Input: Beijing

Observation: 22C, sunny

Thought: Now I can answer.
Action: finish
Action Input: It is 22C and sunny.'''
for i, step in enumerate(parse_react_trace(text), 1):
    print(f"Step {i}: {step}")""",
}
