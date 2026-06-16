"""Causal language-model loss task."""

TASK = {
    "title": "Causal Language-Model Loss",
    "title_zh": "因果语言模型损失",
    "difficulty": "Medium",
    "description_en": "Implement the standard next-token language-model loss.\n\nAutoregressive language models predict token `t+1` from the logits at position `t`, so the logits and labels must be shifted before cross-entropy.\n\n**Signature:** `causal_lm_loss(logits, labels, ignore_index=-100) -> Tensor`\n\n**Parameters:**\n- `logits` — model logits of shape `(B, T, V)`\n- `labels` — token ids of shape `(B, T)`, with ignored positions set to `ignore_index`\n- `ignore_index` — label value skipped by cross-entropy\n\n**Returns:** scalar mean cross-entropy over valid next-token labels\n\n**Constraints:**\n- Use logits at positions `0..T-2` to predict labels at positions `1..T-1`\n- Flatten batch/time before cross-entropy\n- Ignored shifted labels must not contribute to loss or gradients",
    "description_zh": "实现标准 next-token 语言模型损失。\n\n自回归语言模型用位置 `t` 的 logits 预测 token `t+1`，所以计算交叉熵前需要对 logits 和 labels 做 shift。\n\n**签名:** `causal_lm_loss(logits, labels, ignore_index=-100) -> Tensor`\n\n**参数:**\n- `logits` — 模型 logits，形状 `(B, T, V)`\n- `labels` — token id，形状 `(B, T)`，忽略位置设为 `ignore_index`\n- `ignore_index` — 交叉熵跳过的标签值\n\n**返回:** 有效 next-token 标签上的平均交叉熵标量\n\n**约束:**\n- 用位置 `0..T-2` 的 logits 预测位置 `1..T-1` 的 labels\n- 先展平 batch/time 再计算交叉熵\n- 被忽略的 shifted labels 不得贡献 loss 或梯度",
    "function_name": "causal_lm_loss",
    "hint": "Shift first: `shift_logits = logits[:, :-1]`, `shift_labels = labels[:, 1:]`. Then call `F.cross_entropy(shift_logits.reshape(-1, V), shift_labels.reshape(-1), ignore_index=ignore_index)`.",
    "hint_zh": "先 shift：`shift_logits = logits[:, :-1]`，`shift_labels = labels[:, 1:]`。然后调用 `F.cross_entropy(shift_logits.reshape(-1, V), shift_labels.reshape(-1), ignore_index=ignore_index)`。",
    "tests": [
        {
            "name": "Matches PyTorch shifted cross-entropy",
            "code": """
import torch
import torch.nn.functional as F
torch.manual_seed(0)
B, T, V = 3, 5, 11
logits = torch.randn(B, T, V)
labels = torch.randint(0, V, (B, T))
out = {fn}(logits, labels)
expected = F.cross_entropy(logits[:, :-1, :].reshape(-1, V), labels[:, 1:].reshape(-1))
assert torch.allclose(out, expected, atol=1e-6), f'{out} vs {expected}'
""",
        },
        {
            "name": "Uses next-token shift, not same-position labels",
            "code": """
import torch
import torch.nn.functional as F
V = 4
logits = torch.full((1, 4, V), -10.0)
labels = torch.tensor([[3, 1, 2, 0]])
# Correct shifted targets are [1, 2, 0]. Make those positions confident.
logits[0, 0, 1] = 10.0
logits[0, 1, 2] = 10.0
logits[0, 2, 0] = 10.0
# Last position should be ignored by the shift.
logits[0, 3, 3] = 10.0
loss = {fn}(logits, labels)
assert loss.item() < 1e-4, f'Expected near-zero shifted loss, got {loss.item():.6f}'
same_position = F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1))
assert same_position.item() > 5.0, 'This test should distinguish shifted LM loss from same-position CE'
""",
        },
        {
            "name": "Ignore index applies after shifting",
            "code": """
import torch
import torch.nn.functional as F
torch.manual_seed(1)
B, T, V = 2, 6, 7
logits = torch.randn(B, T, V)
labels = torch.randint(0, V, (B, T))
labels[0, 2] = -100
labels[1, 5] = -100
out = {fn}(logits, labels, ignore_index=-100)
expected = F.cross_entropy(
    logits[:, :-1, :].reshape(-1, V),
    labels[:, 1:].reshape(-1),
    ignore_index=-100,
)
assert torch.allclose(out, expected, atol=1e-6), f'{out} vs {expected}'
""",
        },
        {
            "name": "Gradient reaches used logits but not the final position",
            "code": """
import torch
torch.manual_seed(2)
logits = torch.randn(2, 4, 5, requires_grad=True)
labels = torch.randint(0, 5, (2, 4))
loss = {fn}(logits, labels)
loss.backward()
assert logits.grad is not None, 'Missing gradient on logits'
assert logits.grad[:, :-1, :].abs().sum() > 0, 'Used logits should receive gradient'
assert torch.allclose(logits.grad[:, -1, :], torch.zeros_like(logits.grad[:, -1, :])), 'Final-position logits should be unused'
""",
        },
    ],
    "solution": '''def causal_lm_loss(logits, labels, ignore_index=-100):
    vocab_size = logits.shape[-1]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )''',
    "demo": """torch.manual_seed(0)
logits = torch.randn(2, 5, 10)
labels = torch.randint(0, 10, (2, 5))
print('LM loss:', causal_lm_loss(logits, labels).item())""",
}
