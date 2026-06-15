"""REINFORCE policy gradient loss task."""

TASK = {
    "title": "REINFORCE Policy Gradient Loss",
    "title_zh": "REINFORCE 策略梯度损失",
    "difficulty": "Medium",
    "description_en": "Implement the REINFORCE policy-gradient loss.\n\nGiven sampled action log-probabilities and returns, REINFORCE minimizes `-log_prob * advantage`. An optional baseline reduces variance.\n\n**Signature:** `reinforce_loss(logps, returns, baseline=None) -> Tensor`\n\n**Parameters:**\n- `logps` — log-probabilities of sampled actions, shape `(B,)` or `(B, T)`\n- `returns` — Monte-Carlo returns with the same shape\n- `baseline` — optional scalar or tensor broadcastable to returns\n\n**Returns:** scalar policy loss\n\n**Constraints:**\n- Advantages should be detached so gradients flow only through `logps`\n- With no baseline, advantage equals returns",
    "description_zh": "实现 REINFORCE 策略梯度损失。\n\n给定采样动作的对数概率和回报，REINFORCE 最小化 `-log_prob * advantage`。可选 baseline 用于降低方差。\n\n**签名:** `reinforce_loss(logps, returns, baseline=None) -> Tensor`\n\n**参数:**\n- `logps` — 采样动作的对数概率，形状 `(B,)` 或 `(B, T)`\n- `returns` — Monte-Carlo 回报，形状相同\n- `baseline` — 可广播到 returns 的可选标量或张量\n\n**返回:** 标量策略损失\n\n**约束:**\n- advantage 要 detach，使梯度只流经 `logps`\n- 没有 baseline 时 advantage 等于 returns",
    "function_name": "reinforce_loss",
    "hint": "1. `advantage = returns - baseline` if baseline is provided, otherwise `returns`\n2. `advantage = advantage.detach()`\n3. `loss = -(logps * advantage).mean()`",
    "hint_zh": "1. 如果有 baseline，`advantage = returns - baseline`，否则为 `returns`\n2. `advantage = advantage.detach()`\n3. `loss = -(logps * advantage).mean()`",
    "tests": [
        {
            "name": "Matches manual no-baseline formula",
            "code": """
import torch
logps = torch.tensor([-0.2, -1.0, -0.5], requires_grad=True)
returns = torch.tensor([1.0, 0.5, -0.2])
loss = {fn}(logps, returns)
expected = -(logps * returns).mean()
assert torch.allclose(loss, expected, atol=1e-6), f'{loss.item():.6f} vs {expected.item():.6f}'
""",
        },
        {
            "name": "Baseline subtracts from returns",
            "code": """
import torch
logps = torch.tensor([-0.2, -1.0, -0.5])
returns = torch.tensor([1.0, 0.5, -0.2])
baseline = torch.tensor([0.2, 0.2, 0.2])
loss = {fn}(logps, returns, baseline=baseline)
expected = -(logps * (returns - baseline)).mean()
assert torch.allclose(loss, expected, atol=1e-6)
""",
        },
        {
            "name": "Gradient flows to logps only",
            "code": """
import torch
logps = torch.randn(4, requires_grad=True)
returns = torch.randn(4, requires_grad=True)
loss = {fn}(logps, returns)
loss.backward()
assert logps.grad is not None, 'logps should receive gradient'
assert returns.grad is None, 'returns/advantages should be detached'
""",
        },
        {
            "name": "Supports sequence-shaped tensors",
            "code": """
import torch
logps = torch.randn(2, 3, requires_grad=True)
returns = torch.randn(2, 3)
loss = {fn}(logps, returns, baseline=returns.mean(dim=1, keepdim=True))
assert loss.dim() == 0
loss.backward()
assert logps.grad.shape == logps.shape
""",
        },
    ],
    "solution": '''def reinforce_loss(logps, returns, baseline=None):
    if baseline is None:
        advantages = returns
    else:
        advantages = returns - baseline
    advantages = advantages.detach()
    return -(logps * advantages).mean()''',
    "demo": """logps = torch.tensor([-0.2, -1.0, -0.5])
returns = torch.tensor([1.0, 0.5, -0.2])
print('REINFORCE loss:', reinforce_loss(logps, returns).item())""",
}
