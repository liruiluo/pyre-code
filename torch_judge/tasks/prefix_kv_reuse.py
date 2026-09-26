"""Multi-turn KV prefix reuse task."""

TASK = {
    "title": "Plan Multi-Turn KV Cache Reuse",
    "title_zh": "规划多轮 KV Cache 复用",
    "difficulty": "Medium",
    "description_en": "Compute the prefill plan for the next turn of an agent conversation, the way vLLM-style serving stacks reuse the KV cache across turns.\n\nIn an agent loop, every tool result appends messages. The KV cache of the previous turn can be reused for the longest common prefix of messages (compared at message granularity: same role and same content). Everything after the divergence point — plus the newly appended messages — must be prefilled again.\n\n**Signature:** `compute_prefill_plan(prev_messages, new_messages) -> dict`\n\n**Parameters:**\n- `prev_messages`, `new_messages` — lists of `{'role': str, 'content': str, 'tokens': int}` in conversation order\n\n**Returns:** `{'cached_tokens': int, 'prefill_tokens': int}`\n- `cached_tokens` — token count of the shared prefix that can be reused\n- `prefill_tokens` — token count that must be computed fresh for the new turn (`sum of new tokens - cached_tokens`)\n\n**Constraints:**\n- Messages are compared pairwise in order; the prefix ends at the first mismatch (or when either list runs out)\n- An empty `prev_messages` gives zero reuse\n- Reuse is measured in tokens of the *new* message list (identical messages have identical token counts)",
    "description_zh": "为 agent 对话的下一轮计算预填充计划，即 vLLM 式服务栈的跨轮 KV 复用。\n\nagent 循环里每个工具结果都会追加消息。上一轮的 KV cache 可以在最长公共消息前缀上复用（消息粒度比较：role 和 content 都相同）。分叉点之后的消息——加上新追加的——必须重新预填充。\n\n**签名:** `compute_prefill_plan(prev_messages, new_messages) -> dict`\n\n**参数:**\n- `prev_messages`、`new_messages` — 按对话顺序的 `{'role': str, 'content': str, 'tokens': int}` 列表\n\n**返回:** `{'cached_tokens': int, 'prefill_tokens': int}`\n- `cached_tokens` — 可复用的公共前缀 token 数\n- `prefill_tokens` — 新一轮必须现算的 token 数（`新消息总 token - cached_tokens`）\n\n**约束:**\n- 消息按序两两比较；前缀在首个不匹配处（或任一列表耗尽处）终止\n- `prev_messages` 为空则零复用\n- 复用按新消息列表的 token 计（相同消息 token 数相同）",
    "function_name": "compute_prefill_plan",
    "hint": "Walk both lists in lockstep while role and content match, summing the new-side token counts; `prefill = sum(all new tokens) - cached`.",
    "hint_zh": "两列表同步前进，role 和 content 都相同就累加新侧 token；`prefill = 新消息总 token - cached`。",
    "tests": [
        {
            "name": "Pure append reuses the entire previous turn",
            "code": """
prev = [
    {'role': 'system', 'content': 'You are helpful.', 'tokens': 10},
    {'role': 'user', 'content': 'What is 2+2?', 'tokens': 5},
    {'role': 'assistant', 'content': 'Let me compute.', 'tokens': 6},
]
new = prev + [
    {'role': 'tool', 'content': '4', 'tokens': 2},
    {'role': 'assistant', 'content': 'The answer is 4.', 'tokens': 7},
]
plan = {fn}(prev, new)
assert plan == {'cached_tokens': 21, 'prefill_tokens': 9}, f'Got: {plan}'
""",
        },
        {
            "name": "Divergence cuts the reusable prefix",
            "code": """
prev = [
    {'role': 'system', 'content': 'sys', 'tokens': 4},
    {'role': 'user', 'content': 'old question', 'tokens': 8},
    {'role': 'assistant', 'content': 'old answer', 'tokens': 9},
]
new = [
    {'role': 'system', 'content': 'sys', 'tokens': 4},
    {'role': 'user', 'content': 'NEW question', 'tokens': 8},
    {'role': 'assistant', 'content': 'new answer', 'tokens': 7},
]
plan = {fn}(prev, new)
assert plan == {'cached_tokens': 4, 'prefill_tokens': 15}, f'Got: {plan}'
""",
        },
        {
            "name": "Same content but different role is a mismatch",
            "code": """
prev = [
    {'role': 'system', 'content': 'note', 'tokens': 3},
    {'role': 'user', 'content': 'hi', 'tokens': 2},
]
new = [
    {'role': 'user', 'content': 'note', 'tokens': 3},
    {'role': 'user', 'content': 'hi', 'tokens': 2},
]
plan = {fn}(prev, new)
assert plan == {'cached_tokens': 0, 'prefill_tokens': 5}, f'Role mismatch must stop reuse, got: {plan}'
""",
        },
        {
            "name": "Empty previous turn means zero reuse",
            "code": """
new = [
    {'role': 'system', 'content': 'sys', 'tokens': 10},
    {'role': 'user', 'content': 'hello', 'tokens': 3},
]
plan = {fn}([], new)
assert plan == {'cached_tokens': 0, 'prefill_tokens': 13}, f'Got: {plan}'
""",
        },
        {
            "name": "Longer previous turn reuses up to the new length",
            "code": """
prev = [
    {'role': 'system', 'content': 'sys', 'tokens': 5},
    {'role': 'user', 'content': 'q', 'tokens': 2},
    {'role': 'assistant', 'content': 'a', 'tokens': 6},
    {'role': 'user', 'content': 'later', 'tokens': 4},
]
new = [
    {'role': 'system', 'content': 'sys', 'tokens': 5},
    {'role': 'user', 'content': 'q', 'tokens': 2},
]
plan = {fn}(prev, new)
assert plan == {'cached_tokens': 7, 'prefill_tokens': 0}, f'New turn fully cached, got: {plan}'
""",
        },
    ],
    "solution": '''def compute_prefill_plan(prev_messages, new_messages):
    cached = 0
    for old, cur in zip(prev_messages, new_messages):
        if old["role"] != cur["role"] or old["content"] != cur["content"]:
            break
        cached += cur["tokens"]
    total = sum(m["tokens"] for m in new_messages)
    return {"cached_tokens": cached, "prefill_tokens": total - cached}''',
    "demo": """prev = [
    {'role': 'system', 'content': 'You are helpful.', 'tokens': 10},
    {'role': 'user', 'content': 'Book a flight.', 'tokens': 6},
]
new = prev + [
    {'role': 'tool', 'content': 'flight list', 'tokens': 20},
    {'role': 'assistant', 'content': 'Which one?', 'tokens': 5},
]
print(compute_prefill_plan(prev, new))""",
}
