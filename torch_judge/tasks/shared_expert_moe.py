"""Shared-expert sparse MoE task (Qwen3-MoE style)."""

TASK = {
    "title": "Sparse MoE with Shared Expert (Qwen3-MoE)",
    "title_zh": "带共享专家的稀疏 MoE（Qwen3-MoE）",
    "difficulty": "Hard",
    "description_en": "Combine sparse routed experts with one always-on shared expert — the Qwen3-MoE / DeepSeek-V2 architecture detail; Kimi K3 carries two shared experts on top of 896 routed ones.\n\nPure sparse routing lets every token see only `k` of `E` experts, which multiplies parameter count cheaply but fragments knowledge: facts every token needs get smeared across experts. The fix is a shared expert, activated for every token, carrying the common computation; routed experts add specialized knowledge on top.\n\nRouting (Qwen3-MoE convention): softmax over all experts, select top-k, renormalize the selected scores to sum to one. The output is `shared(x) + sum_i w_i * E_i(x)`.\n\n**Signature:** `shared_expert_moe(hidden_states, router_weight, expert_weights, shared_expert_weight, top_k) -> Tensor`\n\n**Parameters:**\n- `hidden_states` — `(T, d)`\n- `router_weight` — `(E, d)`, one router vector per expert\n- `expert_weights` — `(E, d, d_out)`, expert `i` computes `x @ expert_weights[i]` (no bias)\n- `shared_expert_weight` — `(d, d_out)`, the shared expert computes `x @ shared_expert_weight`\n- `top_k` — routed experts activated per token\n\n**Returns:** `(T, d_out)` output tensor\n\n**Constraints:**\n- The shared expert is added exactly once per token, outside the routing\n- Selected routed weights are renormalized over the top-k scores",
    "description_zh": "把稀疏路由专家与一个常开的共享专家组合——Qwen3-MoE / DeepSeek-V2 的架构细节。\n\n纯稀疏路由让每个 token 只见 `E` 个专家中的 `k` 个，参数量便宜地翻倍，但知识碎片化：每个 token 都需要的事实被抹平到各专家。解法是共享专家——每个 token 都激活，承载公共计算；路由专家在其上叠加专门知识。\n\n路由（Qwen3-MoE 约定）：对所有专家 softmax，选 top-k，选中分数重归一到和为 1。输出为 `shared(x) + Σ_i w_i · E_i(x)`。\n\n**签名:** `shared_expert_moe(hidden_states, router_weight, expert_weights, shared_expert_weight, top_k) -> Tensor`\n\n**参数:**\n- `hidden_states` — `(T, d)`\n- `router_weight` — `(E, d)`，每个专家一个路由向量\n- `expert_weights` — `(E, d, d_out)`，专家 `i` 计算 `x @ expert_weights[i]`（无偏置）\n- `shared_expert_weight` — `(d, d_out)`，共享专家计算 `x @ shared_expert_weight`\n- `top_k` — 每个 token 激活的路由专家数\n\n**返回:** `(T, d_out)` 输出张量\n\n**约束:**\n- 共享专家在路由之外，每个 token 恰好加一次\n- 选中的路由权重在 top-k 分数上重归一化",
    "function_name": "shared_expert_moe",
    "hint": "`expert_out = torch.einsum('td,edo->teo', x, expert_weights)` gives every expert output at once; gather the top-k with `idx.unsqueeze(-1).expand(...)`; renormalize like the sigmoid task; add the shared expert last.",
    "hint_zh": "`expert_out = torch.einsum('td,edo->teo', x, expert_weights)` 一次算出所有专家输出；用 `idx.unsqueeze(-1).expand(...)` gather 出 top-k；照 sigmoid 题那样重归一化；最后加共享专家。",
    "tests": [
        {
            "name": "Matches the manual loop",
            "code": """
import torch
torch.manual_seed(0)
T, d, E, d_out, k = 5, 6, 4, 3, 2
x = torch.randn(T, d)
router = torch.randn(E, d)
experts = torch.randn(E, d, d_out)
shared = torch.randn(d, d_out)
scores = torch.softmax(x @ router.T, dim=-1)
vals, idx = scores.topk(k, dim=-1)
w = vals / vals.sum(dim=-1, keepdim=True)
expected = x @ shared
for t in range(T):
    for j in range(k):
        expected[t] += w[t, j] * (x[t] @ experts[idx[t, j]])
out = {fn}(x, router, experts, shared, k)
assert out.shape == (T, d_out), f'Expected (T, d_out), got {out.shape}'
assert torch.allclose(out, expected, atol=1e-5), f'Manual mismatch: {(out - expected).abs().max()}'
""",
        },
        {
            "name": "k=E reduces to the dense softmax mixture plus shared",
            "code": """
import torch
torch.manual_seed(1)
T, d, E, d_out = 3, 4, 5, 2
x = torch.randn(T, d)
router = torch.randn(E, d)
experts = torch.randn(E, d, d_out)
shared = torch.randn(d, d_out)
out = {fn}(x, router, experts, shared, E)
gates = torch.softmax(x @ router.T, dim=-1)
expected = x @ shared
for e in range(E):
    expected += gates[:, e:e+1] * (x @ experts[e])
assert torch.allclose(out, expected, atol=1e-5), 'k=E: renormalized top-k equals the full softmax'
""",
        },
        {
            "name": "The shared expert always contributes",
            "code": """
import torch
torch.manual_seed(2)
T, d, E, d_out = 4, 3, 3, 3
x = torch.randn(T, d)
router = torch.randn(E, d)
experts = torch.zeros(E, d, d_out)          # routed experts contribute nothing
shared = torch.randn(d, d_out)
out = {fn}(x, router, experts, shared, 2)
assert torch.allclose(out, x @ shared, atol=1e-6), 'Shared expert must carry the output alone'
""",
        },
        {
            "name": "k=1 ignores the score magnitude",
            "code": """
import torch
torch.manual_seed(3)
T, d, E, d_out = 2, 4, 4, 2
x = torch.randn(T, d)
router1 = torch.randn(E, d)
router2 = router1 * 3.0                    # same argmax, sharper distribution
experts = torch.randn(E, d, d_out)
shared = torch.randn(d, d_out)
out1 = {fn}(x, router1, experts, shared, 1)
out2 = {fn}(x, router2, experts, shared, 1)
assert torch.allclose(out1, out2, atol=1e-5), 'k=1 renormalizes the winner to 1.0 regardless of its score'
""",
        },
        {
            "name": "Gradients reach experts, router and hidden states",
            "code": """
import torch
torch.manual_seed(4)
T, d, E, d_out = 3, 5, 4, 2
x = torch.randn(T, d, requires_grad=True)
router = torch.randn(E, d, requires_grad=True)
experts = torch.randn(E, d, d_out, requires_grad=True)
shared = torch.randn(d, d_out, requires_grad=True)
out = {fn}(x, router, experts, shared, 2)
out.pow(2).sum().backward()
for name, p in (('x', x), ('router', router), ('experts', experts), ('shared', shared)):
    assert p.grad is not None and p.grad.abs().sum() > 0, f'No gradient reached {name}'
""",
        },
        {
            "name": "Shared expert added exactly once",
            "code": """
import torch
T, d, E, d_out = 2, 3, 2, 3
x = torch.randn(T, d)
router = torch.zeros(E, d)                  # uniform scores
experts = torch.zeros(E, d, d_out)
shared = torch.ones(d, d_out)
out = {fn}(x, router, experts, shared, 2)
# routed experts are zero, shared is all-ones: output = x.sum(-1, keepdim) * ones
expected = x.sum(dim=-1, keepdim=True).expand(T, d_out)
assert torch.allclose(out, expected, atol=1e-6), 'Shared must appear once, not k times'
""",
        },
    ],
    "solution": '''import torch


def shared_expert_moe(hidden_states, router_weight, expert_weights, shared_expert_weight, top_k):
    scores = torch.softmax(hidden_states @ router_weight.T, dim=-1)
    vals, idx = scores.topk(top_k, dim=-1)
    weights = vals / vals.sum(dim=-1, keepdim=True)

    expert_out = torch.einsum("td,edo->teo", hidden_states, expert_weights)
    d_out = expert_out.shape[-1]
    sel = expert_out.gather(1, idx.unsqueeze(-1).expand(-1, -1, d_out))
    routed = (weights.unsqueeze(-1) * sel).sum(dim=1)
    shared = hidden_states @ shared_expert_weight
    return routed + shared''',
    "demo": """import torch
torch.manual_seed(0)
x = torch.randn(4, 8)
router = torch.randn(6, 8)
experts = torch.randn(6, 8, 8)
shared = torch.randn(8, 8)
out = shared_expert_moe(x, router, experts, shared, top_k=2)
print('output shape:', tuple(out.shape))
print('activated params: 2/6 routed experts + 1 shared expert per token')""",
}
