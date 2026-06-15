"""Clipped GRPO loss task."""

TASK = {
    "title": "GRPO Clipped Surrogate Loss",
    "title_zh": "GRPO 裁剪代理损失",
    "difficulty": "Hard",
    "description_en": "Implement a PPO-style clipped surrogate for GRPO.\n\nThe simpler GRPO task in this site teaches group-relative reward normalization with a policy-gradient style loss. This advanced version keeps that group-relative advantage computation, then adds the old-policy importance ratio and clipping used in practical GRPO-style objectives.\n\n**Signature:** `grpo_clipped_loss(new_logps, old_logps, rewards, group_ids, clip_ratio=0.2, eps=1e-5) -> Tensor`\n\n**Parameters:**\n- `new_logps` — current policy log-probabilities, shape (B,)\n- `old_logps` — old policy log-probabilities, shape (B,), treated as constant\n- `rewards` — scalar rewards, shape (B,)\n- `group_ids` — integer prompt/group identifiers, shape (B,)\n- `clip_ratio` — clipping range epsilon\n- `eps` — epsilon for reward-standard-deviation stability\n\n**Returns:** scalar loss\n\n**Constraints:**\n- Per-group normalized advantage: `A_i = (reward_i - mean_group) / (std_group + eps)`\n- Importance ratio: `ratio_i = exp(new_logps_i - old_logps_i.detach())`\n- Loss: `-mean(min(ratio_i * A_i, clamp(ratio_i, 1-clip_ratio, 1+clip_ratio) * A_i))`\n- Gradients flow only through `new_logps`",
    "description_zh": "实现带 PPO-style ratio clipping 的 GRPO 裁剪代理损失。\n\n本站较简化的 GRPO 题用于练习组内奖励归一化和 policy-gradient 风格损失。本进阶版保留组相对优势计算，再加入旧策略重要性采样比率和裁剪，这更接近实际 GRPO 类目标。\n\n**签名:** `grpo_clipped_loss(new_logps, old_logps, rewards, group_ids, clip_ratio=0.2, eps=1e-5) -> Tensor`\n\n**参数:**\n- `new_logps` — 当前策略对数概率，形状 (B,)\n- `old_logps` — 旧策略对数概率，形状 (B,)，视为常量\n- `rewards` — 标量奖励，形状 (B,)\n- `group_ids` — 整数提示/组标识，形状 (B,)\n- `clip_ratio` — 裁剪范围 epsilon\n- `eps` — 奖励标准差的数值稳定项\n\n**返回:** 标量损失\n\n**约束:**\n- 组内归一化优势：`A_i = (reward_i - mean_group) / (std_group + eps)`\n- 重要性采样比率：`ratio_i = exp(new_logps_i - old_logps_i.detach())`\n- 损失：`-mean(min(ratio_i * A_i, clamp(ratio_i, 1-clip_ratio, 1+clip_ratio) * A_i))`\n- 梯度仅通过 `new_logps` 流动",
    "function_name": "grpo_clipped_loss",
    "hint": (
        "1. compute group-normalized rewards as detached advantages\n"
        "2. ratio = exp(new_logps - old_logps.detach())\n"
        "3. unclipped = ratio * advantages\n"
        "4. clipped = clamp(ratio, 1-eps_clip, 1+eps_clip) * advantages\n"
        "5. return -min(unclipped, clipped).mean()"
    ),
    "hint_zh": "1. 先按 group 归一化 rewards，得到 detached advantages\n2. ratio = exp(new_logps - old_logps.detach())\n3. unclipped = ratio * advantages\n4. clipped = clamp(ratio, 1-eps_clip, 1+eps_clip) * advantages\n5. 返回 -min(unclipped, clipped).mean()",
    "tests": [
        {
            "name": "Basic shape & type",
            "code": "\n"
            "import torch\n"
            "from torch import Tensor\n"
            "new_logps = torch.randn(6, requires_grad=True)\n"
            "old_logps = torch.randn(6)\n"
            "rewards = torch.randn(6)\n"
            "group_ids = torch.tensor([0, 0, 0, 1, 1, 1])\n"
            "loss = {fn}(new_logps, old_logps, rewards, group_ids)\n"
            "assert isinstance(loss, Tensor) and loss.dim() == 0, 'Loss must be scalar Tensor'\n"
        },
        {
            "name": "Numeric check vs reference",
            "code": "\n"
            "import torch\n"
            "from torch import Tensor\n"
            "\n"
            "def reference(new_logps: Tensor, old_logps: Tensor, rewards: Tensor, group_ids: Tensor, clip_ratio: float = 0.2, eps: float = 1e-5) -> Tensor:\n"
            "    advantages = torch.empty_like(rewards)\n"
            "    for gid in group_ids.unique():\n"
            "        mask = group_ids == gid\n"
            "        r = rewards[mask]\n"
            "        advantages[mask] = (r - r.mean()) / (r.std(unbiased=False) + eps)\n"
            "    advantages = advantages.detach()\n"
            "    ratio = torch.exp(new_logps - old_logps.detach())\n"
            "    unclipped = ratio * advantages\n"
            "    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages\n"
            "    return -torch.min(unclipped, clipped).mean()\n"
            "\n"
            "new_logps = torch.tensor([0.3, -0.7, -0.6, -1.0])\n"
            "old_logps = torch.tensor([0.0, -0.5, -1.0, -1.5])\n"
            "rewards = torch.tensor([1.0, 0.8, 0.2, 0.0])\n"
            "group_ids = torch.tensor([0, 0, 1, 1])\n"
            "loss_student = {fn}(new_logps, old_logps, rewards, group_ids, clip_ratio=0.2)\n"
            "loss_ref = reference(new_logps, old_logps, rewards, group_ids, clip_ratio=0.2)\n"
            "assert torch.allclose(loss_student, loss_ref, atol=1e-5, rtol=1e-5), 'Loss should match the clipped GRPO reference'\n"
        },
        {
            "name": "Clipping changes an over-large update",
            "code": "\n"
            "import torch\n"
            "new_logps = torch.tensor([1.0, -1.0], requires_grad=True)\n"
            "old_logps = torch.tensor([0.0, 0.0])\n"
            "rewards = torch.tensor([1.0, 0.0])\n"
            "group_ids = torch.tensor([0, 0])\n"
            "loss = {fn}(new_logps, old_logps, rewards, group_ids, clip_ratio=0.2)\n"
            "advantages = torch.tensor([1.0, -1.0]) / (1.0 + 2e-5)\n"
            "ratio = torch.exp(new_logps - old_logps)\n"
            "unclipped_no_clip_loss = -(ratio * advantages).mean()\n"
            "assert loss > unclipped_no_clip_loss, 'Positive-advantage ratio above clip range should be clipped to a more conservative objective'\n"
        },
        {
            "name": "Gradient flows to new_logps only",
            "code": "\n"
            "import torch\n"
            "new_logps = torch.randn(4, requires_grad=True)\n"
            "old_logps = torch.randn(4, requires_grad=True)\n"
            "rewards = torch.randn(4, requires_grad=True)\n"
            "group_ids = torch.tensor([0, 0, 1, 1])\n"
            "loss = {fn}(new_logps, old_logps, rewards, group_ids)\n"
            "loss.backward()\n"
            "assert new_logps.grad is not None, 'Gradients should flow through new_logps'\n"
            "assert old_logps.grad is None, 'Gradients should not flow through old_logps'\n"
            "assert rewards.grad is None, 'Gradients should not flow through rewards'\n"
        },
    ],
    "solution": '''def grpo_clipped_loss(new_logps: Tensor, old_logps: Tensor, rewards: Tensor,
                      group_ids: Tensor, clip_ratio: float = 0.2,
                      eps: float = 1e-5) -> Tensor:
    """Clipped GRPO surrogate loss.

    new_logps: (B,) current policy log-probabilities
    old_logps: (B,) rollout/old policy log-probabilities, treated as constant
    rewards: (B,) scalar rewards for sampled responses
    group_ids: (B,) integer group ids; same id means same prompt/group
    returns: scalar clipped GRPO loss Tensor
    """
    advantages = torch.empty_like(rewards)
    for gid in group_ids.unique():
        mask = group_ids == gid
        group_rewards = rewards[mask]
        mean = group_rewards.mean()
        std = group_rewards.std(unbiased=False)
        advantages[mask] = (group_rewards - mean) / (std + eps)

    advantages = advantages.detach()
    ratios = torch.exp(new_logps - old_logps.detach())
    unclipped = ratios * advantages
    clipped = torch.clamp(ratios, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    return -torch.min(unclipped, clipped).mean()''',
    "demo": """new_logps = torch.tensor([0.3, -0.7, -0.6, -1.0])
old_logps = torch.tensor([0.0, -0.5, -1.0, -1.5])
rewards = torch.tensor([1.0, 0.8, 0.2, 0.0])
group_ids = torch.tensor([0, 0, 1, 1])
print('Loss:', grpo_clipped_loss(new_logps, old_logps, rewards, group_ids).item())""",
}
