"""PPO clipped policy loss task with internal GAE."""

TASK = {
    "title": "PPO Clipped Policy Loss with GAE",
    "title_zh": "PPO 裁剪策略损失（内置 GAE）",
    "difficulty": "Hard",
    "description_en": """Implement PPO's clipped policy loss while computing GAE internally.

The simpler `ppo_loss` task receives `advantages` directly. This task combines the two policy-update pieces students usually need together: first compute Generalized Advantage Estimation (GAE) from rewards, rollout value estimates, and done flags; then use those advantages in PPO's clipped surrogate policy loss.

This task intentionally excludes value loss and entropy bonus. Those are important in many actor-critic training loops, but they are separate terms from the clipped PPO policy objective and would make this exercise less focused.

**Signature:** `ppo_full_loss(new_logps, old_logps, rewards, values, dones, gamma=0.99, lam=0.95, clip_ratio=0.2) -> Tensor`

**Parameters:**
- `new_logps` — current policy log-probabilities, shape `(T,)` or batch-first `(B, T)`
- `old_logps` — rollout/old policy log-probabilities, shape `(T,)` or `(B, T)`, treated as constant
- `rewards` — rollout rewards, shape `(T,)` or `(B, T)`, treated as constant
- `values` — rollout value estimates plus one bootstrap step, shape `(T+1,)` or `(B, T+1)`, treated as constant
- `dones` — episode-end flags, shape `(T,)` or `(B, T)`
- `gamma` — discount factor for GAE
- `lam` — GAE lambda
- `clip_ratio` — policy ratio clipping range

**Returns:** scalar clipped PPO policy loss

**Shape convention:** the last dimension is time. For batched rollouts, batch is dimension 0 and time is dimension 1: `(B, T)`.

**Constraints:**
- GAE TD residual: `delta_t = reward_t + gamma * value_{t+1} * (1-done_t) - value_t`
- GAE recursion over the last dimension: `A_t = delta_t + gamma * lam * (1-done_t) * A_{t+1}`
- Policy ratio: `ratio = exp(new_logps - old_logps.detach())`
- Policy loss: `-mean(min(ratio*A, clamp(ratio, 1±clip_ratio)*A))`
- Gradients should flow through `new_logps` only""",
    "description_zh": """实现一个内部计算 GAE 的 PPO 裁剪策略损失。

较简单的 `ppo_loss` 题直接把 `advantages` 作为输入。本题把策略更新里最常放在一起的两步合并起来：先从 rewards、rollout value estimates 和 done flags 计算广义优势估计（GAE），再把这些优势值放进 PPO clipped surrogate policy loss。

本题有意不包含 value loss 和 entropy bonus。它们在很多 actor-critic 训练循环里很重要，但属于裁剪 PPO 策略目标之外的附加项，放进来会让这道题不够聚焦。

**签名:** `ppo_full_loss(new_logps, old_logps, rewards, values, dones, gamma=0.99, lam=0.95, clip_ratio=0.2) -> Tensor`

**参数:**
- `new_logps` — 当前策略对数概率，形状 `(T,)` 或 batch-first `(B, T)`
- `old_logps` — rollout/旧策略对数概率，形状 `(T,)` 或 `(B, T)`，视为常量
- `rewards` — rollout 奖励，形状 `(T,)` 或 `(B, T)`，视为常量
- `values` — rollout value 估计加一个 bootstrap step，形状 `(T+1,)` 或 `(B, T+1)`，视为常量
- `dones` — episode 结束标记，形状 `(T,)` 或 `(B, T)`
- `gamma` — GAE 折扣因子
- `lam` — GAE lambda
- `clip_ratio` — 策略 ratio 裁剪范围

**返回:** 标量 PPO 裁剪策略损失

**形状约定:** 最后一维是时间维。批量 rollout 使用 batch-first：batch 是第 0 维，时间是第 1 维，即 `(B, T)`。

**约束:**
- GAE TD 残差：`delta_t = reward_t + gamma * value_{t+1} * (1-done_t) - value_t`
- 沿最后一维反向递推：`A_t = delta_t + gamma * lam * (1-done_t) * A_{t+1}`
- 策略比率：`ratio = exp(new_logps - old_logps.detach())`
- 策略损失：`-mean(min(ratio*A, clamp(ratio, 1±clip_ratio)*A))`
- 梯度只应流过 `new_logps`""",
    "function_name": "ppo_full_loss",
    "hint": (
        "1. detach old_logps, rewards, values, and dones before GAE\n"
        "2. T = rewards.shape[-1]; batch dimensions are leading dimensions\n"
        "3. compute GAE by iterating with for t in reversed(range(T))\n"
        "4. ratio = exp(new_logps - old_logps.detach())\n"
        "5. return -mean(min(ratio*A, clamp(ratio, 1±clip_ratio)*A))"
    ),
    "hint_zh": "1. GAE 前先 detach old_logps、rewards、values 和 dones\n2. T = rewards.shape[-1]；batch 维在前面\n3. 用 for t in reversed(range(T)) 反向遍历计算 GAE\n4. ratio = exp(new_logps - old_logps.detach())\n5. 返回 -mean(min(ratio*A, clamp(ratio, 1±clip_ratio)*A))",
    "tests": [
        {
            "name": "Basic shape & type with batch-first rollout",
            "code": "\n"
            "import torch\n"
            "from torch import Tensor\n"
            "new_logps = torch.randn(4, 8, requires_grad=True)\n"
            "old_logps = torch.randn(4, 8)\n"
            "rewards = torch.randn(4, 8)\n"
            "values = torch.randn(4, 9)\n"
            "dones = torch.zeros(4, 8, dtype=torch.bool)\n"
            "loss = {fn}(new_logps, old_logps, rewards, values, dones)\n"
            "assert isinstance(loss, Tensor) and loss.dim() == 0, 'Loss must be a scalar Tensor'\n"
        },
        {
            "name": "Numeric check vs reference on one trajectory",
            "code": "\n"
            "import torch\n"
            "from torch import Tensor\n"
            "\n"
            "def reference(new_logps: Tensor, old_logps: Tensor, rewards: Tensor, values: Tensor, dones: Tensor, gamma: float = 0.9, lam: float = 0.8, clip_ratio: float = 0.2) -> Tensor:\n"
            "    old_logps = old_logps.detach()\n"
            "    rewards = rewards.detach()\n"
            "    values = values.detach()\n"
            "    dones = dones.to(dtype=rewards.dtype)\n"
            "    advantages = torch.empty_like(rewards)\n"
            "    gae = torch.zeros_like(rewards[..., 0])\n"
            "    T = rewards.shape[-1]\n"
            "    for t in reversed(range(T)):\n"
            "        nonterminal = 1.0 - dones[..., t]\n"
            "        delta = rewards[..., t] + gamma * values[..., t + 1] * nonterminal - values[..., t]\n"
            "        gae = delta + gamma * lam * nonterminal * gae\n"
            "        advantages[..., t] = gae\n"
            "    advantages = advantages.detach()\n"
            "    ratios = torch.exp(new_logps - old_logps)\n"
            "    policy_unclipped = ratios * advantages\n"
            "    policy_clipped = torch.clamp(ratios, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages\n"
            "    return -torch.min(policy_unclipped, policy_clipped).mean()\n"
            "\n"
            "new_logps = torch.tensor([0.2, -0.4, -0.9, -0.1])\n"
            "old_logps = torch.tensor([0.0, -0.5, -0.6, -0.2])\n"
            "rewards = torch.tensor([1.0, 0.5, -0.25, 2.0])\n"
            "values = torch.tensor([0.2, 0.1, -0.1, 0.3, 0.7])\n"
            "dones = torch.tensor([False, True, False, False])\n"
            "loss_student = {fn}(new_logps, old_logps, rewards, values, dones, gamma=0.9, lam=0.8, clip_ratio=0.2)\n"
            "loss_ref = reference(new_logps, old_logps, rewards, values, dones)\n"
            "assert torch.allclose(loss_student, loss_ref, atol=1e-6, rtol=1e-6), f'Expected {loss_ref}, got {loss_student}'\n"
        },
        {
            "name": "Batch-first rollout matches reference",
            "code": "\n"
            "import torch\n"
            "from torch import Tensor\n"
            "\n"
            "def reference(new_logps: Tensor, old_logps: Tensor, rewards: Tensor, values: Tensor, dones: Tensor, gamma: float = 0.9, lam: float = 0.8, clip_ratio: float = 0.2) -> Tensor:\n"
            "    old_logps = old_logps.detach()\n"
            "    rewards = rewards.detach()\n"
            "    values = values.detach()\n"
            "    dones = dones.to(dtype=rewards.dtype)\n"
            "    advantages = torch.empty_like(rewards)\n"
            "    gae = torch.zeros_like(rewards[..., 0])\n"
            "    T = rewards.shape[-1]\n"
            "    for t in reversed(range(T)):\n"
            "        nonterminal = 1.0 - dones[..., t]\n"
            "        delta = rewards[..., t] + gamma * values[..., t + 1] * nonterminal - values[..., t]\n"
            "        gae = delta + gamma * lam * nonterminal * gae\n"
            "        advantages[..., t] = gae\n"
            "    advantages = advantages.detach()\n"
            "    ratios = torch.exp(new_logps - old_logps)\n"
            "    policy_unclipped = ratios * advantages\n"
            "    policy_clipped = torch.clamp(ratios, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages\n"
            "    return -torch.min(policy_unclipped, policy_clipped).mean()\n"
            "\n"
            "new_logps = torch.tensor([[0.2, -0.4, -0.9], [-0.1, 0.0, -0.3]])\n"
            "old_logps = torch.tensor([[0.0, -0.5, -0.6], [-0.2, -0.1, -0.2]])\n"
            "rewards = torch.tensor([[1.0, 0.5, -0.25], [0.0, 1.0, 0.2]])\n"
            "values = torch.tensor([[0.2, 0.1, -0.1, 0.0], [0.1, 0.3, 0.4, 0.5]])\n"
            "dones = torch.tensor([[False, True, False], [False, False, True]])\n"
            "loss_student = {fn}(new_logps, old_logps, rewards, values, dones, gamma=0.9, lam=0.8)\n"
            "loss_ref = reference(new_logps, old_logps, rewards, values, dones)\n"
            "assert torch.allclose(loss_student, loss_ref, atol=1e-6, rtol=1e-6), f'Expected {loss_ref}, got {loss_student}'\n"
        },
        {
            "name": "Clipping is conservative for too-large positive-ratio updates",
            "code": "\n"
            "import torch\n"
            "new_logps = torch.tensor([1.0, -1.0])\n"
            "old_logps = torch.tensor([0.0, 0.0])\n"
            "rewards = torch.tensor([1.0, 0.0])\n"
            "values = torch.tensor([0.0, 0.0, 0.0])\n"
            "dones = torch.tensor([False, True])\n"
            "loss = {fn}(new_logps, old_logps, rewards, values, dones, gamma=1.0, lam=1.0, clip_ratio=0.2)\n"
            "advantages = torch.tensor([1.0, 0.0])\n"
            "ratio = torch.exp(new_logps - old_logps)\n"
            "unclipped_no_clip_loss = -(ratio * advantages).mean()\n"
            "assert loss > unclipped_no_clip_loss, 'Positive-advantage ratio above clip range should be clipped to a more conservative loss'\n"
        },
        {
            "name": "Gradient flows through new_logps only",
            "code": "\n"
            "import torch\n"
            "new_logps = torch.randn(4, 3, requires_grad=True)\n"
            "old_logps = torch.randn(4, 3, requires_grad=True)\n"
            "rewards = torch.randn(4, 3, requires_grad=True)\n"
            "values = torch.randn(4, 4, requires_grad=True)\n"
            "dones = torch.zeros(4, 3)\n"
            "loss = {fn}(new_logps, old_logps, rewards, values, dones)\n"
            "loss.backward()\n"
            "assert new_logps.grad is not None, 'Actor gradients should flow through new_logps'\n"
            "assert old_logps.grad is None, 'old_logps should be treated as rollout constants'\n"
            "assert rewards.grad is None, 'rewards should be treated as rollout constants'\n"
            "assert values.grad is None, 'values should be treated as rollout constants for GAE'\n"
        },
    ],
    "solution": '''def ppo_full_loss(new_logps: Tensor, old_logps: Tensor, rewards: Tensor,
                  values: Tensor, dones: Tensor, gamma: float = 0.99,
                  lam: float = 0.95, clip_ratio: float = 0.2) -> Tensor:
    """PPO clipped policy loss with internal GAE.

    Batch-first convention; time is the last dimension.
    new_logps: (T,) or (B, T) current policy log-probabilities
    old_logps: (T,) or (B, T) rollout/old policy log-probabilities, treated as constant
    rewards: (T,) or (B, T) rollout rewards, treated as constant
    values: (T+1,) or (B, T+1) rollout values, including bootstrap value
    dones: (T,) or (B, T) episode-end flags
    returns: scalar clipped PPO policy loss Tensor
    """
    old_logps = old_logps.detach()
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

    advantages = advantages.detach()
    ratios = torch.exp(new_logps - old_logps)
    policy_unclipped = ratios * advantages
    policy_clipped = torch.clamp(ratios, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    return -torch.min(policy_unclipped, policy_clipped).mean()''',
    "demo": """new_logps = torch.tensor([[0.2, -0.4, -0.9], [-0.1, 0.0, -0.3]])
old_logps = torch.tensor([[0.0, -0.5, -0.6], [-0.2, -0.1, -0.2]])
rewards = torch.tensor([[1.0, 0.5, -0.25], [0.0, 1.0, 0.2]])
values = torch.tensor([[0.2, 0.1, -0.1, 0.0], [0.1, 0.3, 0.4, 0.5]])
dones = torch.tensor([[False, True, False], [False, False, True]])
print('PPO clipped policy loss with GAE:', ppo_full_loss(new_logps, old_logps, rewards, values, dones, gamma=0.9, lam=0.8))""",
}
