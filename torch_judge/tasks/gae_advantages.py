"""Generalized Advantage Estimation task for PPO-style policy gradients."""

TASK = {
    "title": "Generalized Advantage Estimation (GAE)",
    "title_zh": "广义优势估计（GAE）",
    "difficulty": "Medium",
    "description_en": """Implement Generalized Advantage Estimation, the advantage estimator commonly used before PPO's clipped policy loss.

The existing PPO loss task receives `advantages` as an input. In a full PPO training loop, those advantages are usually computed from rewards, value estimates, and episode-end flags using GAE.

**Signature:** `gae_advantages(rewards, values, dones, gamma=0.99, lam=0.95) -> Tensor`

**Parameters:**
- `rewards` — rewards over rollout time, shape `(T,)` or batch-first `(B, T)`
- `values` — value estimates plus one bootstrap step, shape `(T+1,)` or batch-first `(B, T+1)`
- `dones` — episode-end flags, shape `(T,)` or batch-first `(B, T)`
- `gamma` — discount factor
- `lam` — GAE lambda

**Returns:** advantage estimates with the same shape as `rewards`

**Shape convention:** the last dimension is time. For batched rollouts, batch is dimension 0 and time is dimension 1: `(B, T)`.

**Constraints:**
- TD residual: `delta_t = r_t + gamma * V_{t+1} * (1-done_t) - V_t`
- Backward recursion over time: `A_t = delta_t + gamma * lam * (1-done_t) * A_{t+1}`
- Return detached advantages; gradients should not flow through rewards or values""",
    "description_zh": """实现广义优势估计（GAE），它通常在 PPO 的裁剪策略损失之前用来计算优势值。

现有 PPO loss 题把 `advantages` 当成输入。完整 PPO 训练里，这些优势值通常由奖励、价值函数估计和 episode 结束标记通过 GAE 得到。

**签名:** `gae_advantages(rewards, values, dones, gamma=0.99, lam=0.95) -> Tensor`

**参数:**
- `rewards` — rollout 时间维上的奖励，形状 `(T,)` 或 batch-first `(B, T)`
- `values` — value 估计加一个 bootstrap step，形状 `(T+1,)` 或 batch-first `(B, T+1)`
- `dones` — episode 结束标记，形状 `(T,)` 或 batch-first `(B, T)`
- `gamma` — 折扣因子
- `lam` — GAE lambda

**返回:** 与 `rewards` 同形状的优势估计

**形状约定:** 最后一维是时间维。批量 rollout 使用 batch-first：batch 是第 0 维，时间是第 1 维，即 `(B, T)`。

**约束:**
- TD 残差：`delta_t = r_t + gamma * V_{t+1} * (1-done_t) - V_t`
- 沿时间维反向递推：`A_t = delta_t + gamma * lam * (1-done_t) * A_{t+1}`
- 返回 detached 的优势值；梯度不应流过 rewards 或 values""",
    "function_name": "gae_advantages",
    "hint": (
        "1. detach rewards and values\n"
        "2. T = rewards.shape[-1]; batch dimensions, if any, are leading dimensions\n"
        "3. initialize gae = torch.zeros_like(rewards[..., 0])\n"
        "4. iterate with for t in reversed(range(T))\n"
        "5. use rewards[..., t], values[..., t], and values[..., t+1]"
    ),
    "hint_zh": "1. detach rewards 和 values\n2. T = rewards.shape[-1]；如果有 batch 维，它们在前面\n3. 用 gae = torch.zeros_like(rewards[..., 0]) 初始化\n4. 用 for t in reversed(range(T)) 反向遍历\n5. 使用 rewards[..., t]、values[..., t] 和 values[..., t+1]",
    "tests": [
        {
            "name": "Basic shape & type",
            "code": "\n"
            "import torch\n"
            "from torch import Tensor\n"
            "rewards = torch.randn(5)\n"
            "values = torch.randn(6)\n"
            "dones = torch.zeros(5, dtype=torch.bool)\n"
            "advantages = {fn}(rewards, values, dones)\n"
            "assert isinstance(advantages, Tensor), 'Advantages must be a Tensor'\n"
            "assert advantages.shape == rewards.shape, 'Advantages must have the same shape as rewards'\n"
        },
        {
            "name": "Numeric check with episode boundary",
            "code": "\n"
            "import torch\n"
            "rewards = torch.tensor([1.0, 0.5, -0.25, 2.0])\n"
            "values = torch.tensor([0.2, 0.1, -0.1, 0.3, 0.7])\n"
            "dones = torch.tensor([False, True, False, False])\n"
            "advantages = {fn}(rewards, values, dones, gamma=0.9, lam=0.8)\n"
            "expected = torch.tensor([1.1780, 0.4000, 1.7976, 2.3300])\n"
            "assert torch.allclose(advantages, expected, atol=1e-4, rtol=0), f'Expected {expected}, got {advantages}'\n"
        },
        {
            "name": "Batch-first rollout over parallel examples",
            "code": "\n"
            "import torch\n"
            "rewards = torch.tensor([[1.0, 0.5, -0.25], [0.0, 1.0, 0.2]])\n"
            "values = torch.tensor([[0.2, 0.1, -0.1, 0.0], [0.1, 0.3, 0.4, 0.5]])\n"
            "dones = torch.tensor([[False, True, False], [False, False, True]])\n"
            "advantages = {fn}(rewards, values, dones, gamma=0.9, lam=0.8)\n"
            "expected = torch.tensor([[1.1780, 0.4000, -0.1500], [0.8295, 0.9160, -0.2000]])\n"
            "assert advantages.shape == rewards.shape, 'Batched GAE should preserve (B, T) shape'\n"
            "assert torch.allclose(advantages, expected, atol=1e-4, rtol=0), f'Expected {expected}, got {advantages}'\n"
        },
        {
            "name": "Stops gradient through rewards and values",
            "code": "\n"
            "import torch\n"
            "rewards = torch.randn(4, 3, requires_grad=True)\n"
            "values = torch.randn(4, 4, requires_grad=True)\n"
            "dones = torch.zeros(4, 3)\n"
            "advantages = {fn}(rewards, values, dones)\n"
            "assert not advantages.requires_grad, 'Advantages should be detached constants for PPO loss'\n"
        },
    ],
    "solution": '''def gae_advantages(rewards: Tensor, values: Tensor, dones: Tensor,
                   gamma: float = 0.99, lam: float = 0.95) -> Tensor:
    """Generalized Advantage Estimation.

    rewards: (T,) or batch-first (B, T) rewards
    values: (T+1,) or batch-first (B, T+1) value estimates including bootstrap value
    dones: (T,) or batch-first (B, T) episode-end flags
    returns: detached advantages with the same shape as rewards
    """
    rewards = rewards.detach()
    values = values.detach()
    dones = dones.to(dtype=rewards.dtype)

    advantages = torch.empty_like(rewards)
    gae = torch.zeros_like(rewards[..., 0])
    T = rewards.shape[-1]
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[..., t]
        delta = rewards[..., t] + gamma * values[..., t + 1] * nonterminal - values[..., t]
        gae = delta + gamma * lam * nonterminal * gae
        advantages[..., t] = gae
    return advantages.detach()''',
    "demo": """rewards = torch.tensor([[1.0, 0.5, -0.25], [0.0, 1.0, 0.2]])
values = torch.tensor([[0.2, 0.1, -0.1, 0.0], [0.1, 0.3, 0.4, 0.5]])
dones = torch.tensor([[False, True, False], [False, False, True]])
print('Advantages:', gae_advantages(rewards, values, dones, gamma=0.9, lam=0.8))""",
}
