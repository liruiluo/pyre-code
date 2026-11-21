"""Top-k / Top-p Sampling task."""

TASK = {
    "title": "Top-k / Top-p Sampling",
    "title_zh": "Top-k / Top-p 采样",
    "difficulty": "Medium",
    "description_en": "Implement top-k / top-p (nucleus) sampling for language model decoding.\n\nThese sampling strategies filter the vocabulary to high-probability tokens before sampling, balancing diversity and quality in text generation.\n\n**Signature:** `sample_top_k_top_p(logits, top_k=0, top_p=1.0, temperature=1.0) -> int`\n\n**Parameters:**\n- `logits` — raw logits over vocabulary (V,)\n- `top_k` — keep only top-k tokens (0 = disabled)\n- `top_p` — keep tokens with cumulative prob <= p (1.0 = disabled)\n- `temperature` — temperature scaling\n\n**Returns:** sampled token index (int)\n\n**Constraints:**\n- Apply temperature first, then top-k, then top-p\n- `top_k=1` must always return argmax\n- Do not mutate the caller's `logits`: work on a copy\n- top-k keeps exactly `top_k` indices (take them from `topk`; a value threshold would keep tied maxima too)\n- top-p keeps the shortest prefix whose cumulative mass reaches `p`\n- Reject invalid options with `ValueError`: `temperature <= 0`, `top_p` outside `(0, 1]`, `top_k < 0`, or `top_k > V`; never silently clamp",
    "description_zh": "实现语言模型解码的 top-k / top-p（核）采样。\n\n这些采样策略在采样前将词表过滤为高概率 token，在文本生成中平衡多样性和质量。\n\n**签名:** `sample_top_k_top_p(logits, top_k=0, top_p=1.0, temperature=1.0) -> int`\n\n**参数:**\n- `logits` — 词表上的原始 logits (V,)\n- `top_k` — 仅保留 top-k 个 token（0 = 禁用）\n- `top_p` — 保留累积概率 <= p 的 token（1.0 = 禁用）\n- `temperature` — 温度缩放\n\n**返回:** 采样的 token 索引（整数）\n\n**约束:**\n- 先应用温度，再 top-k，再 top-p\n- `top_k=1` 必须始终返回 argmax\n- 不得修改调用方传入的 `logits`（在副本上操作）\n- top-k 恰好保留 `top_k` 个索引（用 `topk` 取索引；用值阈值会把并列的最大值一起留下）\n- top-p 保留累计概率达到 `p` 的最短前缀\n- 非法参数抛 `ValueError`：`temperature <= 0`、`top_p` 不在 `(0, 1]`、`top_k < 0`、`top_k > V`；绝不静默 clamp",
    "function_name": "sample_top_k_top_p",
    "hint": "0. validate first: raise ValueError for temperature <= 0, top_p outside (0, 1], top_k < 0 or top_k > V\n1. scores = logits / temperature — a NEW tensor; never write into the caller's logits\n2. top-k: keep only the indices from scores.topk(top_k) (a value threshold keeps tied maxima too)\n3. top-p: sort desc → softmax → cumsum → set to -inf every position whose preceding cumulative mass already reaches p\n4. sample from softmax(scores)",
    "hint_zh": "0. 先校验：temperature <= 0、top_p 不在 (0, 1]、top_k < 0 或 top_k > V 一律抛 ValueError\n1. scores = logits / temperature —— 新建张量，绝不写回调用方的 logits\n2. top-k：只保留 scores.topk(top_k) 返回的索引（用值阈值会把并列最大值一起留下）\n3. top-p：降序 → softmax → cumsum → 把「前面的累计质量已达到 p」的位置设为 -inf\n4. 从 softmax(scores) 中采样",
    "tests": [
        {
            "name": "top_k=1 always returns argmax",
            "code": "\nimport torch\ntorch.manual_seed(0)\nlogits = torch.tensor([1.0, 5.0, 2.0, 0.5])\nfor _ in range(10):\n    assert {fn}(logits.clone(), top_k=1) == 1, 'top_k=1 should return argmax'\n"
        },
        {
            "name": "Low temperature concentrates",
            "code": "\nimport torch\ntorch.manual_seed(42)\nlogits = torch.tensor([1.0, 3.0, 2.0])\ncounts = [0, 0, 0]\nfor _ in range(100):\n    counts[{fn}(logits.clone(), temperature=0.01)] += 1\nassert counts[1] > 90, f'Low temp should pick argmax, got {counts}'\n"
        },
        {
            "name": "All tokens reachable (no filtering)",
            "code": "\nimport torch\nlogits = torch.zeros(5)\nseen = set()\nfor i in range(200):\n    torch.manual_seed(i)\n    seen.add({fn}(logits.clone()))\nassert len(seen) == 5, f'Only saw {seen}'\n"
        },
        {
            "name": "Returns valid index",
            "code": "\nimport torch\ntorch.manual_seed(0)\nV = 100\nlogits = torch.randn(V)\nfor _ in range(20):\n    t = {fn}(logits.clone(), top_k=10, top_p=0.9)\n    assert 0 <= t < V, f'Token {t} out of range'\n"
        },
        {
            "name": "top_p excludes low-probability tokens",
            "code": "\nimport torch\n# Token 0 has prob ~1.0 after softmax; top_p=0.5 should only allow token 0\nlogits = torch.zeros(10)\nlogits[0] = 10.0\ntorch.manual_seed(0)\nresults = set()\nfor _ in range(50):\n    results.add({fn}(logits.clone(), top_k=0, top_p=0.5, temperature=1.0))\nexpected = {0}\nassert results == expected, f'With top_p=0.5 and dominant token 0, only token 0 should be sampled, got ' + str(results)\n"
        },
        {
            "name": "Does not modify the caller's logits",
            "code": """
import torch
logits = torch.tensor([0.5, 1.0, -1.0, 2.0])
original = logits.clone()
for _ in range(5):
    {fn}(logits, top_k=2, top_p=0.8)
assert torch.equal(logits, original)
""",
        },
        {
            "name": "Top-k ties keep at most k candidates",
            "code": """
import torch
logits = torch.tensor([2.0, 2.0, 2.0, 0.0])
torch.manual_seed(3)
samples = { {fn}(logits, top_k=1) for _ in range(40) }
assert len(samples) == 1, f"top_k=1 should choose one tied index, saw {samples}"
""",
        },
        {
            "name": "Invalid sampling options fail clearly",
            "code": """
import torch
for kwargs in ({"temperature": 0}, {"top_p": 0}, {"top_p": 1.1}, {"top_k": -1}, {"top_k": 8}):
    try:
        {fn}(torch.randn(3), **kwargs)
    except ValueError:
        pass
    else:
        raise AssertionError(f"Must reject {kwargs}")
""",
        },
    ],
    "solution": '''def sample_top_k_top_p(logits, top_k=0, top_p=1.0, temperature=1.0):
    if logits.ndim != 1 or logits.numel() == 0:
        raise ValueError("logits must be a nonempty 1-D tensor")
    if temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError("temperature must be positive and top_p in (0, 1]")
    if top_k < 0 or top_k > logits.numel():
        raise ValueError("top_k must be between 0 and vocabulary size")
    # Work on a new tensor: callers may reuse logits for scoring or gradients.
    scores = logits / temperature
    if top_k:
        values, indices = scores.topk(top_k)
        scores = torch.full_like(scores, float("-inf")).scatter(0, indices, values)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = probs.cumsum(dim=-1)
        remove = (cumulative - probs) >= top_p
        remove[0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        scores = torch.empty_like(scores).scatter(0, sorted_indices, sorted_logits)
    return int(torch.multinomial(torch.softmax(scores, dim=-1), 1).item())''',
    "demo": """logits = torch.tensor([1.0, 5.0, 2.0, 0.5])
print('top_k=1:', sample_top_k_top_p(logits.clone(), top_k=1))
print('top_p=0.5:', sample_top_k_top_p(logits.clone(), top_p=0.5))""",

}