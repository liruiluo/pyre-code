"""Tool-call parsing task (Qwen Hermes format)."""

TASK = {
    "title": "Parse Qwen-Style Tool Calls",
    "title_zh": "解析 Qwen 风格的工具调用",
    "difficulty": "Medium",
    "description_en": "Extract structured tool calls from raw model output, using the format emitted by Qwen2.5 / Qwen3 chat templates.\n\nQwen's Hermes-style template wraps each function call between special tags, with a JSON body: `{\"name\": ..., \"arguments\": {...}}`. A serving loop must turn that text into an executable call — and survive malformed JSON when the model samples it.\n\n**Signature:** `parse_tool_calls(text) -> list[dict]`\n\n**Parameters:**\n- `text` — raw model output; may contain narrative text around the tag blocks\n\n**Returns:** a list of `{'name': str, 'arguments': dict}` for every valid block, in order of appearance\n\n**Constraints:**\n- Blocks are delimited by the special open/close tags\n- The body between the tags must be valid JSON; a block whose body fails to parse, whose `name` is not a string, or whose `arguments` is not a dict is skipped\n- A block missing the `arguments` key is treated as empty arguments `{}`\n- Text outside the tags is ignored",
    "description_zh": "从模型原始输出中提取结构化工具调用，使用 Qwen2.5 / Qwen3 对话模板的格式。\n\nQwen 的 Hermes 风格模板把每个函数调用放在特殊标签之间，正文是 JSON：`{\"name\": ..., \"arguments\": {...}}`。服务循环必须把这段文本变成可执行的调用——并且要在模型采样坏 JSON 时存活。\n\n**签名:** `parse_tool_calls(text) -> list[dict]`\n\n**参数:**\n- `text` — 模型原始输出，标签块周围可能有叙述文字\n\n**返回:** 所有合法块的 `{'name': str, 'arguments': dict}` 列表，按出现顺序\n\n**约束:**\n- 块由特殊开/闭标签界定\n- 标签之间必须是合法 JSON；解析失败、`name` 不是字符串、或 `arguments` 不是 dict 的块直接跳过\n- 缺 `arguments` 键的块按空参数 `{}` 处理\n- 标签之外的文本忽略",
    "function_name": "parse_tool_calls",
    "hint": "`re.findall` over the open-tag + body + close-tag pattern with `re.DOTALL`, then `json.loads` each body inside try/except and validate the shape.",
    "hint_zh": "用 `re.findall` 按 开标签+正文+闭标签 的模式（`re.DOTALL`）取块，逐块 try/except `json.loads` 并校验结构。",
    "tests": [
        {
            "name": "Single call",
            "code": """
text = '<tool_call>\\n{"name": "get_weather", "arguments": {"city": "Beijing", "unit": "celsius"}}\\n</tool_call>'
calls = {fn}(text)
assert calls == [{'name': 'get_weather', 'arguments': {'city': 'Beijing', 'unit': 'celsius'}}], f'Got: {calls}'
""",
        },
        {
            "name": "Parallel calls in order",
            "code": """
text = 'I will check both cities.\\n\\n<tool_call>\\n{"name": "search", "arguments": {"q": "a"}}\\n</tool_call>\\n<tool_call>\\n{"name": "search", "arguments": {"q": "b"}}\\n</tool_call>'
calls = {fn}(text)
assert len(calls) == 2, f'Expected 2 calls, got {len(calls)}'
assert calls[0]['arguments']['q'] == 'a'
assert calls[1]['arguments']['q'] == 'b'
""",
        },
        {
            "name": "Empty arguments and missing arguments key",
            "code": """
text = '<tool_call>\\n{"name": "list_files", "arguments": {}}\\n</tool_call>\\n<tool_call>\\n{"name": "ping"}\\n</tool_call>'
calls = {fn}(text)
assert calls[0] == {'name': 'list_files', 'arguments': {}}
assert calls[1] == {'name': 'ping', 'arguments': {}}, 'Missing arguments key defaults to empty dict'
""",
        },
        {
            "name": "Malformed JSON is skipped",
            "code": """
text = '<tool_call>\\n{"name": "oops", "arguments": \\n</tool_call>\\n<tool_call>\\n{"name": "fine", "arguments": {"k": 1}}\\n</tool_call>'
calls = {fn}(text)
assert calls == [{'name': 'fine', 'arguments': {'k': 1}}], f'Malformed block must be skipped, got: {calls}'
""",
        },
        {
            "name": "Bad shapes are skipped",
            "code": """
text = ('<tool_call>\\n[1, 2, 3]\\n</tool_call>'
        '<tool_call>\\n{"name": 42, "arguments": {}}\\n</tool_call>'
        '<tool_call>\\n{"name": "ok", "arguments": [1, 2]}\\n</tool_call>'
        '<tool_call>\\n{"name": "valid", "arguments": {"a": 1}}\\n</tool_call>')
calls = {fn}(text)
assert calls == [{'name': 'valid', 'arguments': {'a': 1}}], f'Only the valid block survives, got: {calls}'
""",
        },
        {
            "name": "Prose only and unicode arguments",
            "code": """
assert {fn}('The user asked about weather but I need more info.') == []
calls = {fn}('<tool_call>\\n{"name": "echo", "arguments": {"text": "hello world"}}\\n</tool_call>')
assert calls == [{'name': 'echo', 'arguments': {'text': 'hello world'}}]
""",
        },
    ],
    "solution": '''import json
import re


def parse_tool_calls(text):
    calls = []
    for body in re.findall(r"<tool_call>\\n(.*?)</tool_call>", text, re.DOTALL):
        try:
            obj = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("name")
        arguments = obj.get("arguments", {})
        if not isinstance(name, str) or not isinstance(arguments, dict):
            continue
        calls.append({"name": name, "arguments": arguments})
    return calls''',
    "demo": """text = ('Let me check the weather and the news.\\n\\n'
        '<tool_call>\\n{"name": "get_weather", "arguments": {"city": "Beijing"}}\\n</tool_call>\\n'
        '<tool_call>\\n{"name": "get_news", "arguments": {"topic": "AI"}}\\n</tool_call>')
for call in parse_tool_calls(text):
    print(call)""",
}
