"""Temperature sampling task."""

TASK = {
    "title": "Temperature Sampling",
    "title_zh": "温度采样",
    "difficulty": "Easy",
    "description_en": "Implement temperature sampling for language-model decoding.\n\nTemperature rescales logits before softmax: lower values make the distribution sharper, higher values make it flatter.\n\n**Signature:** `temperature_sampling(logits, temperature=1.0) -> int`\n\n**Parameters:**\n- `logits` — raw logits over vocabulary, shape `(V,)`\n- `temperature` — sampling temperature; `0` means greedy argmax\n\n**Returns:** sampled token index as a Python int\n\n**Constraints:**\n- `temperature=0` should return `argmax`\n- For positive temperature, sample from `softmax(logits / temperature)`",
    "description_zh": "实现语言模型解码中的温度采样。\n\nTemperature 在 softmax 前缩放 logits：温度越低分布越尖锐，温度越高分布越平坦。\n\n**签名:** `temperature_sampling(logits, temperature=1.0) -> int`\n\n**参数:**\n- `logits` — 词表上的原始 logits，形状 `(V,)`\n- `temperature` — 采样温度；`0` 表示贪心 argmax\n\n**返回:** Python int 类型的采样 token index\n\n**约束:**\n- `temperature=0` 应返回 `argmax`\n- 正温度下从 `softmax(logits / temperature)` 中采样",
    "function_name": "temperature_sampling",
    "hint": "1. If `temperature == 0`, return `torch.argmax(logits).item()`\n2. Otherwise compute `probs = softmax(logits / temperature)`\n3. Return `torch.multinomial(probs, 1).item()`",
    "hint_zh": "1. 如果 `temperature == 0`，返回 `torch.argmax(logits).item()`\n2. 否则计算 `probs = softmax(logits / temperature)`\n3. 返回 `torch.multinomial(probs, 1).item()`",
    "tests": [
        {
            "name": "temperature=0 returns argmax",
            "code": """
import torch
logits = torch.tensor([1.0, 3.0, 2.0])
for _ in range(10):
    assert {fn}(logits, temperature=0.0) == 1
""",
        },
        {
            "name": "Low temperature concentrates on argmax",
            "code": """
import torch
logits = torch.tensor([1.0, 4.0, 2.0])
counts = [0, 0, 0]
for seed in range(80):
    torch.manual_seed(seed)
    counts[{fn}(logits, temperature=0.05)] += 1
assert counts[1] >= 75, f'Low temperature should mostly choose argmax, got {counts}'
""",
        },
        {
            "name": "High temperature allows multiple tokens",
            "code": """
import torch
logits = torch.tensor([0.0, 1.0, 2.0, 3.0])
seen = set()
for seed in range(200):
    torch.manual_seed(seed)
    seen.add({fn}(logits, temperature=5.0))
assert len(seen) >= 3, f'High temperature should sample diverse tokens, saw {seen}'
""",
        },
        {
            "name": "Returns Python int in range",
            "code": """
import torch
logits = torch.randn(17)
torch.manual_seed(0)
token = {fn}(logits, temperature=1.0)
assert isinstance(token, int), f'Return value should be int, got {type(token)}'
assert 0 <= token < 17
""",
        },
    ],
    "solution": '''def temperature_sampling(logits, temperature=1.0):
    if temperature == 0:
        return torch.argmax(logits).item()
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, 1).item()''',
    "demo": """logits = torch.tensor([1.0, 3.0, 2.0])
print('greedy:', temperature_sampling(logits, temperature=0.0))
print('sample:', temperature_sampling(logits, temperature=1.0))""",
}
