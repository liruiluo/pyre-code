"""RLHF sampled-token KL penalty task."""

TASK = {
    "title": "RLHF KL Penalty",
    "title_zh": "RLHF KL 惩罚",
    "difficulty": "Medium",
    "description_en": "Implement the sampled-token KL penalty commonly used in RLHF/PPO training.\n\nFor generated tokens, a simple per-token KL estimate is `policy_logp - reference_logp`; PPO-style RLHF adds a sampled log-ratio as a KL estimator to discourage drift. Individual samples or finite minibatch means can be negative; only the expectation under the policy equals the nonnegative `KL(policy || reference)`.\n\n**Signature:** `kl_penalty(policy_logps, ref_logps, mask=None, beta=0.1) -> Tensor`\n\n**Parameters:**\n- `policy_logps`, `ref_logps` — sampled-token log-probs with the same shape\n- `mask` — optional 0/1 mask for valid response tokens\n- `beta` — KL coefficient\n\n**Returns:** scalar penalty over valid tokens\n\n**Constraints:**\n- If `mask` is provided, average only over masked positions\n- `ref_logps` and `mask` should not receive gradients",
    "description_zh": "实现 RLHF/PPO 训练中常见的采样 token KL 惩罚。\n\n对生成 token，一个简单的逐 token KL 估计是 `policy_logp - reference_logp`；PPO 风格 RLHF 用采样 log-ratio 估计 KL 以限制策略偏离参考模型。单个采样值或有限批次均值可以为负；在当前策略分布下求期望才等于非负的 `KL(policy || reference)`。\n\n**签名:** `kl_penalty(policy_logps, ref_logps, mask=None, beta=0.1) -> Tensor`\n\n**参数:**\n- `policy_logps`, `ref_logps` — 形状相同的采样 token 对数概率\n- `mask` — 可选 0/1 有效回复 token mask\n- `beta` — KL 系数\n\n**返回:** 有效 token 上的标量惩罚\n\n**约束:**\n- 若提供 `mask`，只在 mask 位置求平均\n- `ref_logps` 和 `mask` 不应接收梯度",
    "function_name": "kl_penalty",
    "hint": "1. `kl = policy_logps - ref_logps.detach()`\n2. If `mask` is provided: `penalty = (kl * mask).sum() / mask.sum().clamp_min(1)`\n3. Return `beta * penalty`",
    "hint_zh": "1. `kl = policy_logps - ref_logps.detach()`\n2. 若提供 mask：`penalty = (kl * mask).sum() / mask.sum().clamp_min(1)`\n3. 返回 `beta * penalty`",
    "tests": [
        {
            "name": "Unmasked mean penalty",
            "code": """
import torch
policy = torch.tensor([-0.2, -0.4, -0.8])
ref = torch.tensor([-0.3, -0.5, -0.6])
out = {fn}(policy, ref, beta=0.2)
expected = 0.2 * (policy - ref).mean()
assert torch.allclose(out, expected, atol=1e-6), f'{out.item():.6f} vs {expected.item():.6f}'
""",
        },
        {
            "name": "Masked positions only",
            "code": """
import torch
policy = torch.tensor([[-0.2, -0.4, -0.8], [-1.0, -0.1, -0.3]])
ref = torch.tensor([[-0.3, -0.5, -0.6], [-1.2, -0.4, -0.1]])
mask = torch.tensor([[1, 1, 0], [0, 1, 0]], dtype=torch.float32)
out = {fn}(policy, ref, mask=mask, beta=0.5)
expected = 0.5 * (((policy - ref) * mask).sum() / mask.sum())
assert torch.allclose(out, expected, atol=1e-6)
""",
        },
        {
            "name": "Gradient flows only to policy logps",
            "code": """
import torch
policy = torch.randn(4, requires_grad=True)
ref = torch.randn(4, requires_grad=True)
loss = {fn}(policy, ref, beta=0.1)
loss.backward()
assert policy.grad is not None, 'policy_logps should receive gradient'
assert ref.grad is None, 'ref_logps should be detached'
""",
        },
        {
            "name": "Zero when policy equals reference",
            "code": """
import torch
logps = torch.randn(3, 5)
out = {fn}(logps, logps, mask=torch.ones_like(logps), beta=0.3)
assert torch.allclose(out, torch.tensor(0.0), atol=1e-6)
""",
        },
        {
            "name": "All-zero mask gives connected zero loss",
            "code": """
import torch
policy = torch.randn(2, 4, requires_grad=True)
reference = torch.randn(2, 4, requires_grad=True)
loss = {fn}(policy, reference, mask=torch.zeros(2, 4))
assert loss.item() == 0.0 and torch.isfinite(loss)
loss.backward()
assert policy.grad is not None and torch.all(policy.grad == 0)
assert reference.grad is None
""",
        },
    ],
    "solution": '''def kl_penalty(policy_logps, ref_logps, mask=None, beta=0.1):
    # A sampled log-ratio can be negative; its policy expectation is KL.
    log_ratio = policy_logps - ref_logps.detach()
    if mask is None:
        return beta * log_ratio.mean()
    valid = mask.to(device=log_ratio.device, dtype=log_ratio.dtype).detach()
    return beta * (log_ratio * valid).sum() / valid.sum().clamp_min(1)''',
    "demo": """policy = torch.tensor([-0.2, -0.4, -0.8])
ref = torch.tensor([-0.3, -0.5, -0.6])
print('KL penalty:', kl_penalty(policy, ref, beta=0.2).item())""",
}
