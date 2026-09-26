"""DeepSeek-V3 sigmoid MoE routing task."""

TASK = {
    "title": "Sigmoid MoE Routing (DeepSeek-V3)",
    "title_zh": "Sigmoid MoE 路由（DeepSeek-V3）",
    "difficulty": "Medium",
    "description_en": "Route tokens through a sparse MoE layer with sigmoid gating — the DeepSeek-V3 detail.\n\nClassic MoE (Mixtral, Qwen2-MoE) scores experts with a softmax over all experts, which couples every score to every other. DeepSeek-V3 replaces it with element-wise sigmoid scores: each expert is scored independently in `(0, 1)`. The top-k experts are selected per token, and the selected scores are renormalized to sum to one — selection is decoupled from normalization.\n\n**Signature:** `sigmoid_moe_routing(hidden_states, router_weight, top_k) -> (weights, indices)`\n\n**Parameters:**\n- `hidden_states` — shape `(T, d)`\n- `router_weight` — shape `(E, d)`, one router vector per expert\n- `top_k` — number of experts to activate per token, `1 <= k <= E`\n\n**Returns:**\n- `weights` — `(T, k)`, the renormalized selected scores, each row summing to 1\n- `indices` — `(T, k)`, the selected expert ids, sorted by score descending\n\n**Constraints:**\n- Scores are `sigmoid(hidden @ router_weight.T)`, never a softmax\n- Renormalization divides each selected score by the sum of the selected scores",
    "description_zh": "用 sigmoid 门控把 token 路由进稀疏 MoE 层——DeepSeek-V3 的细节。\n\n经典 MoE（Mixtral、Qwen2-MoE）对所有专家做 softmax 打分，每个分数与其他所有分数耦合。DeepSeek-V3 换成逐元素 sigmoid 分数：每个专家独立打 `(0, 1)` 区间的分。每个 token 选 top-k 专家，再把选中分数重归一到和为 1——选择与归一化解耦。\n\n**签名:** `sigmoid_moe_routing(hidden_states, router_weight, top_k) -> (weights, indices)`\n\n**参数:**\n- `hidden_states` — 形状 `(T, d)`\n- `router_weight` — 形状 `(E, d)`，每个专家一个路由向量\n- `top_k` — 每个 token 激活的专家数，`1 <= k <= E`\n\n**返回:**\n- `weights` — `(T, k)`，重归一后的选中分数，每行和为 1\n- `indices` — `(T, k)`，选中的专家 id，按分数降序\n\n**约束:**\n- 分数是 `sigmoid(hidden @ router_weight.T)`，绝不用 softmax\n- 重归一化是把每个选中分数除以选中分数之和",
    "function_name": "sigmoid_moe_routing",
    "hint": "`scores = torch.sigmoid(x @ W.T)`; `vals, idx = scores.topk(k, dim=-1)`; `weights = vals / vals.sum(-1, keepdim=True)`.",
    "hint_zh": "`scores = torch.sigmoid(x @ W.T)`；`vals, idx = scores.topk(k, dim=-1)`；`weights = vals / vals.sum(-1, keepdim=True)`。",
    "tests": [
        {
            "name": "Selected weights sum to one",
            "code": """
import torch
torch.manual_seed(0)
x = torch.randn(7, 16)
W = torch.randn(8, 16)
weights, indices = {fn}(x, W, top_k=3)
assert weights.shape == (7, 3) and indices.shape == (7, 3)
assert torch.allclose(weights.sum(dim=-1), torch.ones(7), atol=1e-5), 'Each row must renormalize to 1'
""",
        },
        {
            "name": "Top-k indices are the sigmoid-score leaders",
            "code": """
import torch
torch.manual_seed(1)
x = torch.randn(5, 8)
W = torch.randn(6, 8)
weights, indices = {fn}(x, W, top_k=2)
scores = torch.sigmoid(x @ W.T)
manual_idx = scores.argsort(dim=-1, descending=True)[:, :2]
assert torch.equal(indices, manual_idx), f'Expected argmax top-k, got {indices} vs {manual_idx}'
""",
        },
        {
            "name": "Weights are the renormalized sigmoid scores",
            "code": """
import torch
torch.manual_seed(2)
x = torch.randn(4, 12)
W = torch.randn(5, 12)
weights, indices = {fn}(x, W, top_k=4)
scores = torch.gather(torch.sigmoid(x @ W.T), 1, indices)
expected = scores / scores.sum(dim=-1, keepdim=True)
assert torch.allclose(weights, expected, atol=1e-6)
""",
        },
        {
            "name": "Sigmoid, not softmax",
            "code": """
import torch
x = torch.tensor([[0.0, 0.0, 0.0]])
W = torch.eye(3)
weights, indices = {fn}(x, W, top_k=3)
# softmax would give 1/3 each; sigmoid gives 0.5 each, renormalized to 1/3 —
# so distinguish with asymmetric logits instead:
x2 = torch.tensor([[2.0, -2.0, 0.0]])
w2, _ = {fn}(x2, W, top_k=3)
sig = torch.sigmoid(x2[0])
manual = sig.sort(descending=True).values / sig.sum()
assert torch.allclose(w2[0], manual, atol=1e-6), 'Scores must come from sigmoid renormalization'
""",
        },
        {
            "name": "k=1 gives a deterministic one-hot",
            "code": """
import torch
torch.manual_seed(3)
x = torch.randn(6, 10)
W = torch.randn(4, 10)
weights, indices = {fn}(x, W, top_k=1)
assert torch.allclose(weights, torch.ones(6, 1), atol=1e-6), 'A single selected expert takes all the weight'
best = torch.sigmoid(x @ W.T).argmax(dim=-1, keepdim=True)
assert torch.equal(indices, best)
""",
        },
        {
            "name": "Differentiable",
            "code": """
import torch
torch.manual_seed(4)
x = torch.randn(3, 6, requires_grad=True)
W = torch.randn(4, 6)
weights, _ = {fn}(x, W, top_k=2)
weights.pow(2).sum().backward()
assert x.grad is not None and x.grad.abs().sum() > 0, 'Routing must be trainable end-to-end'
""",
        },
    ],
    "solution": '''import torch


def sigmoid_moe_routing(hidden_states, router_weight, top_k):
    scores = torch.sigmoid(hidden_states @ router_weight.T)
    vals, idx = scores.topk(top_k, dim=-1)
    weights = vals / vals.sum(dim=-1, keepdim=True)
    return weights, idx''',
    "demo": """import torch
torch.manual_seed(0)
x = torch.randn(4, 8)
W = torch.randn(6, 8)
weights, indices = sigmoid_moe_routing(x, W, top_k=2)
print('indices:\\n', indices)
print('weights:\\n', weights)""",
}
