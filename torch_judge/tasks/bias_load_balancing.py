"""Auxiliary-loss-free bias routing task (DeepSeek-V3 style)."""

TASK = {
    "title": "Auxiliary-Loss-Free MoE Routing (DeepSeek-V3)",
    "title_zh": "免辅助损失的 MoE 路由（DeepSeek-V3）",
    "difficulty": "Medium",
    "description_en": "Balance a sparse MoE layer with per-expert routing biases instead of an auxiliary loss — the auxiliary-loss-free strategy DeepSeek-V3 introduced and Kimi K2 adopted.\n\nClassic load balancing adds an auxiliary loss that fights the language objective and destabilizes training at scale. DeepSeek-V3's fix: give every expert a bias `b_i` that never enters gradients nor gate weights, and use it ONLY to steer selection. Selection picks the top-k experts by `s_i + b_i`; the gate weight of a selected expert is its raw score `s_i` renormalized over the selection. Outside training, the biases are stepped down for overloaded experts and up for underloaded ones — control by construction, not by gradient.\n\n**Signature:** `bias_adjusted_routing(hidden_states, router_weight, expert_biases, top_k) -> (weights, indices)`\n\n**Parameters:**\n- `hidden_states` — `(T, d)`\n- `router_weight` — `(E, d)`\n- `expert_biases` — `(E,)`, the load-balancing biases (not part of the graph)\n- `top_k` — experts selected per token\n\n**Returns:**\n- `weights` — `(T, k)`: the renormalized raw scores of the selected experts, rows summing to 1\n- `indices` — `(T, k)`: experts chosen by highest `s_i + b_i`, sorted by that combined score descending\n\n**Constraints:**\n- Selection ranks by `s + b`; gate weights use `s` only — biases must never leak into the weights\n- Zero biases reduce the function to plain sigmoid routing",
    "description_zh": "用每专家路由偏置而非辅助损失来平衡稀疏 MoE 层——DeepSeek-V3 引入、Kimi K2 沿用的免辅助损失策略。\n\n经典负载均衡加辅助损失，会与语言目标打架、大尺度训练失稳。DeepSeek-V3 的修法：给每个专家一个偏置 `b_i`——永不进梯度、永不进门控权重，只用于引导选择。选择按 `s_i + b_i` 取 top-k；被选专家的门控权重是原始分数 `s_i` 在选集上重归一。训练之外，偏置按负载步进——过载调低、欠载调高——靠构造控制而非梯度。\n\n**签名:** `bias_adjusted_routing(hidden_states, router_weight, expert_biases, top_k) -> (weights, indices)`\n\n**参数:**\n- `hidden_states` — `(T, d)`\n- `router_weight` — `(E, d)`\n- `expert_biases` — `(E,)`，负载均衡偏置（不在计算图里）\n- `top_k` — 每个 token 选的专家数\n\n**返回:**\n- `weights` — `(T, k)`：被选专家的原始分数重归一，每行和为 1\n- `indices` — `(T, k)`：按 `s_i + b_i` 最高选出的专家，按组合分数降序\n\n**约束:**\n- 选择按 `s + b` 排序；门控权重只用 `s`——偏置绝不能漏进权重\n- 偏置全零时退化为普通 sigmoid 路由",
    "function_name": "bias_adjusted_routing",
    "hint": "`s = sigmoid(x @ W.T)`; rank by `s + b` with topk; gather the selected `s` values and renormalize those (never the biased scores).",
    "hint_zh": "`s = sigmoid(x @ W.T)`；用 `s + b` topk 排序；gather 出被选的 `s` 值并只对它们重归一（绝不用带偏置的分数）。",
    "tests": [
        {
            "name": "Zero biases reduce to plain sigmoid routing",
            "code": """
import torch
torch.manual_seed(0)
x = torch.randn(6, 12)
W = torch.randn(8, 12)
w_a, i_a = {fn}(x, W, torch.zeros(8), 3)
s = torch.sigmoid(x @ W.T)
vals, idx = s.topk(3, dim=-1)
w_e = vals / vals.sum(dim=-1, keepdim=True)
assert torch.equal(i_a, idx) and torch.allclose(w_a, w_e, atol=1e-6), 'No bias: plain sigmoid top-k routing'
""",
        },
        {
            "name": "Biases steer selection",
            "code": """
import torch
torch.manual_seed(1)
x = torch.randn(4, 10)
W = torch.randn(5, 10)
s = torch.sigmoid(x @ W.T)
# push the globally-weakest expert to the top of every token's selection
weakest = s.mean(dim=0).argmin().item()
biases = torch.zeros(5)
biases[weakest] = 10.0
_, idx = {fn}(x, W, biases, 1)
assert (idx[:, 0] == weakest).all(), 'A large bias must win the selection'
""",
        },
        {
            "name": "Gate weights exclude the bias",
            "code": """
import torch
torch.manual_seed(2)
x = torch.randn(3, 16)
W = torch.randn(6, 16)
biases = torch.tensor([0.0, 2.0, -1.0, 0.5, 0.0, 0.0])
weights, indices = {fn}(x, W, biases, 3)
s = torch.sigmoid(x @ W.T)
sel = s.gather(1, indices)
expected = sel / sel.sum(dim=-1, keepdim=True)
assert torch.allclose(weights, expected, atol=1e-6), 'Weights must be raw-score renormalized, bias-free'
assert torch.allclose(weights.sum(-1), torch.ones(3), atol=1e-6)
""",
        },
        {
            "name": "Indices are ordered by the biased score",
            "code": """
import torch
torch.manual_seed(3)
x = torch.randn(5, 8)
W = torch.randn(7, 8)
biases = torch.randn(7)
_, indices = {fn}(x, W, biases, 4)
combined = torch.sigmoid(x @ W.T) + biases
manual = combined.argsort(dim=-1, descending=True)[:, :4]
assert torch.equal(indices, manual), 'Selection order follows s + b'
""",
        },
        {
            "name": "Gradients flow through scores but not biases",
            "code": """
import torch
torch.manual_seed(4)
x = torch.randn(3, 6, requires_grad=True)
W = torch.randn(4, 6, requires_grad=True)
b = torch.randn(4, requires_grad=True)
weights, _ = {fn}(x, W, b, 2)
weights.pow(2).sum().backward()
assert x.grad is not None and x.grad.abs().sum() > 0, 'Scores must be trainable'
assert W.grad is not None and W.grad.abs().sum() > 0
assert b.grad is None, 'Biases must stay outside the computation graph'
""",
        },
        {
            "name": "A biased winner can carry near-zero weight",
            "code": """
import torch
x = torch.tensor([[1.0, 0.0]])
W = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # expert 0 scores high on x
biases = torch.tensor([0.0, 8.0])            # expert 1 wins selection via bias only
weights, indices = {fn}(x, W, biases, 2)
assert indices[0, 0].item() == 1, 'Bias lifts expert 1 to the top of the selection'
assert weights[0, 0] < 0.5, 'Expert 1 has the smaller raw score, so the smaller weight'
""",
        },
    ],
    "solution": '''import torch


def bias_adjusted_routing(hidden_states, router_weight, expert_biases, top_k):
    scores = torch.sigmoid(hidden_states @ router_weight.T)
    combined = scores + expert_biases
    _, idx = combined.topk(top_k, dim=-1)
    sel = scores.gather(1, idx)
    weights = sel / sel.sum(dim=-1, keepdim=True)
    return weights, idx''',
    "demo": """import torch
torch.manual_seed(0)
x = torch.randn(4, 8)
W = torch.randn(6, 8)
b = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 2.0])  # expert 5 needs traffic
weights, indices = bias_adjusted_routing(x, W, b, 2)
print('selected experts:\\n', indices)
print('bias-free gate weights:\\n', weights)""",
}
