"""Delta-rule linear attention scan task (Kimi Delta Attention family)."""

TASK = {
    "title": "Delta-Rule State Scan (Kimi K3 KDA)",
    "title_zh": "Delta 规则状态扫描（Kimi K3 KDA）",
    "difficulty": "Hard",
    "description_en": "Implement the recurrent state update behind Kimi K3's KDA (Kimi Delta Attention) layers — 69 of K3's 93 layers — and the gated-DeltaNet family of linear attentions also used by Qwen3-Next and MiniMax.\n\nLinear attention keeps a fixed-size state `S (d_k, d_v)` instead of a growing KV cache. The delta rule updates it associatively and self-correcting: before writing the new association `(k_t, v_t)`, the state first removes whatever it currently predicts for `k_t`, then writes the residual. With decay strength `beta_t`:\n\n`S_t = S_{t-1} (I - beta_t * k_t k_t^T) + beta_t * v_t k_t^T`\n\nBecause the write is the prediction error (delta), repeating the same association drives `k_t S` toward `v_t` — the test-time-training property that lets these layers recall associations exactly without attention over history.\n\n**Signature:** `delta_rule_scan(k, v, beta, S0) -> Tensor`\n\n**Parameters:**\n- `k` — `(T, d_k)` keys, unit-normalized rows\n- `v` — `(T, d_v)` values\n- `beta` — `(T,)` gate strengths in `[0, 1]`\n- `S0` — `(d_k, d_v)` initial state\n\n**Returns:** `(T, d_k, d_v)`, the state AFTER each step's update\n\n**Constraints:**\n- Step order is sequential over `T`; each output row is the post-update state\n- `beta = 0` leaves the state untouched for that step",
    "description_zh": "实现 Kimi K3 的 KDA（Kimi Delta Attention）层背后的递归状态更新——K3 共 93 层里占 69 层——以及 Qwen3-Next、MiniMax 使用的 gated-DeltaNet 系线性注意力。\n\n线性注意力维护固定尺寸状态 `S (d_k, d_v)` 而非增长的 KV cache。delta 规则的更新是联想且自纠错的：写入新关联 `(k_t, v_t)` 前，状态先减去它当前对 `k_t` 的预测，再写入残差。衰减强度 `beta_t`：\n\n`S_t = S_{t-1} (I − beta_t · k_t k_t^T) + beta_t · v_t k_t^T`\n\n因为写入的是预测误差（delta），重复同一关联会把 `k_t S` 推向 `v_t`——test-time-training 性质让这些层无需对历史做注意力就能精确召回关联。\n\n**签名:** `delta_rule_scan(k, v, beta, S0) -> Tensor`\n\n**参数:**\n- `k` — `(T, d_k)` 键，行单位归一\n- `v` — `(T, d_v)` 值\n- `beta` — `(T,)` 门控强度，`[0, 1]`\n- `S0` — `(d_k, d_v)` 初始状态\n\n**返回:** `(T, d_k, d_v)`，每步更新后的状态\n\n**约束:**\n- 按序对 `T` 逐步执行；每行输出是更新后状态\n- `beta = 0` 的步骤状态不变",
    "function_name": "delta_rule_scan",
    "hint": "Expand the update: `S = S - beta * outer(k, k @ S - v)` — remove the current prediction along k, then add the target. Loop over T recording states.",
    "hint_zh": "展开更新式：`S = S − beta · outer(k, k @ S − v)`——先沿 k 减去当前预测，再加上目标。对 T 循环记录状态。",
    "tests": [
        {
            "name": "Matches the manual recurrence",
            "code": """
import torch
torch.manual_seed(0)
T, dk, dv = 6, 4, 3
k = torch.nn.functional.normalize(torch.randn(T, dk), dim=-1)
v = torch.randn(T, dv)
beta = torch.rand(T)
S0 = torch.randn(dk, dv)
out = {fn}(k, v, beta, S0)
S = S0.clone()
states = []
for t in range(T):
    S = (torch.eye(dk) - beta[t] * torch.outer(k[t], k[t])) @ S + beta[t] * torch.outer(k[t], v[t])
    states.append(S.clone())
expected = torch.stack(states)
assert out.shape == (T, dk, dv), f'Expected (T, dk, dv), got {out.shape}'
assert torch.allclose(out, expected, atol=1e-5), 'State recurrence mismatch'
""",
        },
        {
            "name": "Zero beta freezes the state",
            "code": """
import torch
torch.manual_seed(1)
T, dk, dv = 4, 3, 2
k = torch.nn.functional.normalize(torch.randn(T, dk), dim=-1)
v = torch.randn(T, dv)
beta = torch.zeros(T)
S0 = torch.randn(dk, dv)
out = {fn}(k, v, beta, S0)
assert torch.allclose(out, S0.unsqueeze(0).expand(T, dk, dv), atol=1e-6), 'beta=0 must not touch the state'
""",
        },
        {
            "name": "Unit beta with v=k projects onto the key direction",
            "code": """
import torch
torch.manual_seed(2)
dk, dv = 3, 3
k1 = torch.nn.functional.normalize(torch.randn(dk), dim=0)
v1 = k1  # write the key as its own value
S0 = torch.eye(dk)
out = {fn}(k1.view(1, dk), v1.view(1, dv), torch.ones(1), S0)
S1 = out[0]
expected = S0 - torch.outer(k1, k1 @ S0 - v1)  # removes the current prediction along k1, writes the residual
assert torch.allclose(S1, expected, atol=1e-5), 'Delta write removes the prediction error'
assert torch.allclose(k1 @ S1, k1, atol=1e-5), 'After writing (k, k), the state maps k to k'
""",
        },
        {
            "name": "Repeated writes converge to exact recall",
            "code": """
import torch
torch.manual_seed(3)
dk, dv = 8, 6
k1 = torch.nn.functional.normalize(torch.randn(dk), dim=0)
v1 = torch.randn(dv)
T = 200
k = k1.unsqueeze(0).expand(T, dk).contiguous()
v = v1.unsqueeze(0).expand(T, dv).contiguous()
beta = torch.ones(T)
out = {fn}(k, v, beta, torch.zeros(dk, dv))
final = out[-1]
assert torch.allclose(k1 @ final, v1, atol=1e-4), 'The delta rule converges: k^T S -> v^T'
""",
        },
        {
            "name": "Zero state stays zero until written",
            "code": """
import torch
torch.manual_seed(4)
T, dk, dv = 5, 4, 4
k = torch.nn.functional.normalize(torch.randn(T, dk), dim=-1)
v = torch.randn(T, dv)
beta = torch.tensor([0.0, 0.0, 1.0, 0.5, 1.0])
out = {fn}(k, v, beta, torch.zeros(dk, dv))
assert torch.allclose(out[0], torch.zeros(dk, dv), atol=1e-7)
assert torch.allclose(out[1], torch.zeros(dk, dv), atol=1e-7), 'No gate, no state'
assert not torch.allclose(out[2], torch.zeros(dk, dv), atol=1e-6), 'Gate open must write'
""",
        },
        {
            "name": "Differentiable through the scan",
            "code": """
import torch
torch.manual_seed(5)
T, dk, dv = 4, 3, 2
k = torch.nn.functional.normalize(torch.randn(T, dk), dim=-1).requires_grad_(True)
v = torch.randn(T, dv, requires_grad=True)
beta = torch.rand(T)
S0 = torch.randn(dk, dv)
out = {fn}(k, v, beta, S0)
out.pow(2).sum().backward()
assert k.grad is not None and k.grad.abs().sum() > 0, 'Keys must receive gradient'
assert v.grad is not None and v.grad.abs().sum() > 0, 'Values must receive gradient'
""",
        },
    ],
    "solution": '''import torch


def delta_rule_scan(k, v, beta, S0):
    T, dk = k.shape
    dv = v.shape[-1]
    states = []
    S = S0
    for t in range(T):
        kt = k[t]
        vt = v[t]
        # remove the current prediction along k, write the residual
        S = S - beta[t] * torch.outer(kt, kt @ S - vt)
        states.append(S)
    return torch.stack(states)''',
    "demo": """import torch
torch.manual_seed(0)
dk, dv = 8, 6
k = torch.nn.functional.normalize(torch.randn(5, dk), dim=-1)
v = torch.randn(5, dv)
beta = torch.ones(5)
states = delta_rule_scan(k, v, beta, torch.zeros(dk, dv))
print('state sequence:', tuple(states.shape))
print('first association recalled:', torch.allclose(k[0] @ states[0], v[0], atol=1e-5))""",
}
