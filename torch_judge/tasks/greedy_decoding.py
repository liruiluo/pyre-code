"""Greedy decoding task."""

TASK = {
    "title": "Greedy Decoding",
    "title_zh": "贪心解码",
    "difficulty": "Easy",
    "description_en": "Implement greedy autoregressive decoding.\n\nAt each generation step, greedy decoding selects the token with the highest logit/probability and appends it to the sequence.\n\n**Signature:** `greedy_decode(logits_fn, start_token, max_len, eos_token=None) -> list[int]`\n\n**Parameters:**\n- `logits_fn` — callable that takes a 1-D token tensor and returns logits over the vocabulary\n- `start_token` — first token id\n- `max_len` — maximum returned sequence length including the start token\n- `eos_token` — optional stop token\n\n**Returns:** list of token ids\n\n**Constraints:**\n- Use `argmax` at each step\n- Stop early when the generated token equals `eos_token`\n- Never return more than `max_len` tokens",
    "description_zh": "实现自回归贪心解码。\n\n每个生成步选择 logits/概率最大的 token，并追加到序列末尾。\n\n**签名:** `greedy_decode(logits_fn, start_token, max_len, eos_token=None) -> list[int]`\n\n**参数:**\n- `logits_fn` — 接受一维 token 张量并返回词表 logits 的可调用对象\n- `start_token` — 起始 token id\n- `max_len` — 返回序列最大长度，包含起始 token\n- `eos_token` — 可选终止 token\n\n**返回:** token id 列表\n\n**约束:**\n- 每一步使用 `argmax`\n- 生成 token 等于 `eos_token` 时提前停止\n- 返回长度不得超过 `max_len`",
    "function_name": "greedy_decode",
    "hint": "Initialize `seq = [start_token]`. While `len(seq) < max_len`, call `logits_fn(torch.tensor(seq))`, append `argmax`, and break on EOS.",
    "hint_zh": "初始化 `seq = [start_token]`。当 `len(seq) < max_len` 时调用 `logits_fn(torch.tensor(seq))`，追加 `argmax`，遇到 EOS 退出。",
    "tests": [
        {
            "name": "Follows argmax path",
            "code": """
import torch
def logits_fn(tokens):
    logits = torch.full((6,), -10.0)
    logits[min(len(tokens), 5)] = 1.0
    return logits
seq = {fn}(logits_fn, start_token=0, max_len=6, eos_token=5)
assert seq == [0, 1, 2, 3, 4, 5], f'Got {seq}'
""",
        },
        {
            "name": "Stops at eos",
            "code": """
import torch
def logits_fn(tokens):
    logits = torch.zeros(4)
    logits[3] = 99.0
    return logits
seq = {fn}(logits_fn, start_token=0, max_len=10, eos_token=3)
assert seq == [0, 3], f'Should stop after EOS, got {seq}'
""",
        },
        {
            "name": "Respects max_len without eos",
            "code": """
import torch
def logits_fn(tokens):
    logits = torch.zeros(3)
    logits[1] = 1.0
    return logits
seq = {fn}(logits_fn, start_token=2, max_len=4, eos_token=None)
assert seq == [2, 1, 1, 1], f'Got {seq}'
""",
        },
        {
            "name": "Returns list for max_len one",
            "code": """
import torch
def logits_fn(tokens):
    return torch.randn(5)
seq = {fn}(logits_fn, start_token=7, max_len=1)
assert seq == [7]
""",
        },
    ],
    "solution": '''def greedy_decode(logits_fn, start_token, max_len, eos_token=None):
    seq = [int(start_token)]
    while len(seq) < max_len:
        logits = logits_fn(torch.tensor(seq))
        next_token = int(torch.argmax(logits).item())
        seq.append(next_token)
        if eos_token is not None and next_token == eos_token:
            break
    return seq''',
    "demo": """def next_is_length(tokens):
    logits = torch.full((6,), -10.0)
    logits[min(len(tokens), 5)] = 1.0
    return logits
print(greedy_decode(next_is_length, start_token=0, max_len=6, eos_token=5))""",
}
