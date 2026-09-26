"""Agent context truncation task."""

TASK = {
    "title": "Truncate Agent Context Under a Token Budget",
    "title_zh": "在 token 预算内截断 Agent 上下文",
    "difficulty": "Medium",
    "description_en": "Fit a conversation into a token budget the way agent serving stacks do before every model call.\n\nRules of the game: `pinned` messages (system prompt, safety notices, task-critical tool results) are always kept. The remaining budget is filled with the most recent non-pinned messages, newest first. Return the indices of the kept messages in their original order — the conversation must not be reordered.\n\nThis is the core of context-window management: agent memory systems, Claude Code style compaction, and serving-side history trimming all reduce to a recency-priority packing under a budget.\n\n**Signature:** `truncate_context(messages, max_tokens) -> list[int]`\n\n**Parameters:**\n- `messages` — list of `{'role': str, 'tokens': int, 'pinned': bool}` (`pinned` optional, default `False`)\n- `max_tokens` — total token budget\n\n**Returns:** sorted list of kept message indices (ascending, original order)\n\n**Constraints:**\n- Pinned messages are always kept, even if their sum exceeds `max_tokens`\n- Non-pinned messages are added from the end of the list backwards, each taken whole or not at all, while the running total stays within `max_tokens`\n- The output preserves original indices; it is a selection, not a reorder",
    "description_zh": "像 agent 服务栈在每次模型调用前那样，把对话装进 token 预算。\n\n规则：`pinned` 消息（系统提示、安全声明、任务关键的工具结果）永远保留。剩余预算从最新往旧填充非固定消息。返回被保留消息的原始顺序索引——对话不能重排。\n\n这是上下文窗口管理的核心：agent 记忆系统、Claude Code 式压缩、服务侧历史裁剪，最终都归结为预算下的近因优先装包。\n\n**签名:** `truncate_context(messages, max_tokens) -> list[int]`\n\n**参数:**\n- `messages` — `{'role': str, 'tokens': int, 'pinned': bool}` 列表（`pinned` 可选，默认 `False`）\n- `max_tokens` — 总 token 预算\n\n**返回:** 保留消息的索引列表（升序，即原始顺序）\n\n**约束:**\n- 固定消息永远保留，即使总和超过 `max_tokens`\n- 非固定消息从列表尾部往前加，每条要么整条保留要么不要，只要累计总量不超 `max_tokens`\n- 输出保持原始索引；这是选择，不是重排",
    "function_name": "truncate_context",
    "hint": "Sum the pinned tokens first, then walk the non-pinned indices from the end while the running total + next message fits; finally sort the kept indices.",
    "hint_zh": "先累加固定消息的 token，然后从尾部遍历非固定索引，只要累计+下一条仍放得下就收；最后对保留索引排序。",
    "tests": [
        {
            "name": "Everything fits",
            "code": """
messages = [
    {'role': 'system', 'tokens': 10, 'pinned': True},
    {'role': 'user', 'tokens': 20},
    {'role': 'assistant', 'tokens': 30},
    {'role': 'user', 'tokens': 20},
]
kept = {fn}(messages, max_tokens=100)
assert kept == [0, 1, 2, 3], f'Expected all indices, got {kept}'
""",
        },
        {
            "name": "Oldest non-pinned messages are dropped",
            "code": """
messages = [
    {'role': 'system', 'tokens': 10, 'pinned': True},
    {'role': 'user', 'tokens': 40},
    {'role': 'assistant', 'tokens': 40},
    {'role': 'user', 'tokens': 30},
    {'role': 'assistant', 'tokens': 30},
]
kept = {fn}(messages, max_tokens=100)
# 10 (pinned) + 30 + 30 = 70 <= 100; adding the 40-token assistant would hit 110 > 100
assert kept == [0, 3, 4], f'Expected [0, 3, 4], got {kept}'
""",
        },
        {
            "name": "Pinned messages survive even when old and over budget",
            "code": """
messages = [
    {'role': 'system', 'tokens': 30, 'pinned': True},
    {'role': 'user', 'tokens': 50},
    {'role': 'tool', 'tokens': 40, 'pinned': True},
    {'role': 'user', 'tokens': 20},
]
kept = {fn}(messages, max_tokens=50)
assert kept == [0, 2], f'Pinned must survive, got {kept}'
""",
        },
        {
            "name": "Whole-message granularity, no partial keeps",
            "code": """
messages = [
    {'role': 'system', 'tokens': 5, 'pinned': True},
    {'role': 'user', 'tokens': 8},
    {'role': 'assistant', 'tokens': 8},
]
kept = {fn}(messages, max_tokens=12)
# 5 + 8 = 13 > 12, so the 8-token assistant does not fit either
assert kept == [0], f'No partial keeps allowed, got {kept}'
""",
        },
        {
            "name": "pinned defaults to False",
            "code": """
messages = [
    {'role': 'system', 'tokens': 30},
    {'role': 'user', 'tokens': 30},
    {'role': 'assistant', 'tokens': 30},
]
kept = {fn}(messages, max_tokens=65)
assert kept == [1, 2], f'Without an explicit pinned flag nothing is pinned, got {kept}'
""",
        },
        {
            "name": "Budget exactly reached is allowed",
            "code": """
messages = [
    {'role': 'system', 'tokens': 10, 'pinned': True},
    {'role': 'user', 'tokens': 45},
    {'role': 'assistant', 'tokens': 45},
]
kept = {fn}(messages, max_tokens=100)
assert kept == [0, 1, 2], f'10 + 45 + 45 = 100 should fit exactly, got {kept}'
""",
        },
    ],
    "solution": '''def truncate_context(messages, max_tokens):
    pinned = [i for i, m in enumerate(messages) if m.get("pinned", False)]
    non_pinned = [i for i, m in enumerate(messages) if not m.get("pinned", False)]

    total = sum(messages[i]["tokens"] for i in pinned)
    kept = set(pinned)
    for i in reversed(non_pinned):
        cost = messages[i]["tokens"]
        if total + cost <= max_tokens:
            kept.add(i)
            total += cost
    return sorted(kept)''',
    "demo": """messages = [
    {'role': 'system', 'tokens': 10, 'pinned': True},
    {'role': 'user', 'tokens': 40},
    {'role': 'assistant', 'tokens': 40},
    {'role': 'tool', 'tokens': 15, 'pinned': True},
    {'role': 'user', 'tokens': 30},
]
print('kept indices:', truncate_context(messages, max_tokens=100))""",
}
