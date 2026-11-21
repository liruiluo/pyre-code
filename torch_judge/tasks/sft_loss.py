"""SFT loss task."""

TASK = {
    "title": "SFT Masked Language-Model Loss",
    "title_zh": "SFT 掩码语言模型损失",
    "difficulty": "Medium",
    "description_en": "Implement the supervised fine-tuning (SFT) loss for language models.\n\nSFT usually trains only selected response tokens: prompt tokens are masked out with `ignore_index=-100`, and valid tokens use cross-entropy over the vocabulary. Logits and labels are ALREADY ALIGNED in this exercise. For next-token causal LM training, shift them before calling this function (see `causal_lm_loss`).\n\n**Signature:** `sft_loss(logits, labels, ignore_index=-100) -> Tensor`\n\n**Parameters:**\n- `logits` — model logits, shape `(B, T, V)`\n- `labels` — token labels, shape `(B, T)`, with ignored positions set to `ignore_index`\n\n**Returns:** scalar mean cross-entropy over non-ignored positions; exactly `0.0` (still attached to the autograd graph) when every position is ignored\n\n**Constraints:**\n- Flatten batch/time before cross-entropy\n- Ignored positions must not contribute to loss or gradients\n- If no position is valid, return a connected zero (`flat_logits.sum() * 0`); `F.cross_entropy` alone would give NaN",
    "description_zh": "实现语言模型监督微调（SFT）损失。\n\nSFT 通常只训练被选中的回复 token：prompt token 用 `ignore_index=-100` 遮掉，有效 token 在词表上做交叉熵。本题 logits 与 labels 已对齐；若直接使用原始 Causal LM logits，应先进行 next-token shift（见 `causal_lm_loss`）。\n\n**签名:** `sft_loss(logits, labels, ignore_index=-100) -> Tensor`\n\n**参数:**\n- `logits` — 模型 logits，形状 `(B, T, V)`\n- `labels` — token 标签，形状 `(B, T)`，忽略位置设为 `ignore_index`\n\n**返回:** 非忽略位置上的平均交叉熵标量；若所有位置都被忽略，返回恰为 `0.0` 且仍连在计算图上\n\n**约束:**\n- 先展平 batch/time 再计算交叉熵\n- 被忽略位置不得贡献 loss 或梯度\n- 没有有效位置时返回连在计算图上的 0（`flat_logits.sum() * 0`）；直接调用 `F.cross_entropy` 会得到 NaN",
    "function_name": "sft_loss",
    "hint": "Use `F.cross_entropy(logits.view(-1, V), labels.view(-1), ignore_index=ignore_index)`. Guard the all-ignored case: it returns NaN, so return `logits.view(-1, V).sum() * 0` instead.",
    "hint_zh": "使用 `F.cross_entropy(logits.view(-1, V), labels.view(-1), ignore_index=ignore_index)`。全忽略要守卫：它会返回 NaN，应改成返回 `logits.view(-1, V).sum() * 0`。",
    "tests": [
        {
            "name": "Matches F.cross_entropy with ignore_index",
            "code": """
import torch
torch.manual_seed(0)
B, T, V = 2, 4, 7
logits = torch.randn(B, T, V)
labels = torch.tensor([[1, 2, -100, 3], [-100, 4, 5, -100]])
out = {fn}(logits, labels)
ref = torch.nn.functional.cross_entropy(logits.view(-1, V), labels.view(-1), ignore_index=-100)
assert torch.allclose(out, ref, atol=1e-6), f'{out.item():.6f} vs {ref.item():.6f}'
""",
        },
        {
            "name": "Prompt-only positions are ignored",
            "code": """
import torch
logits = torch.zeros(1, 3, 4)
labels_a = torch.tensor([[-100, 1, 2]])
labels_b = torch.tensor([[3, 1, 2]])
loss_a = {fn}(logits, labels_a)
loss_b = {fn}(logits[:, 1:], labels_a[:, 1:])
loss_c = {fn}(logits, labels_b)
assert torch.allclose(loss_a, loss_b, atol=1e-6), 'Ignored prompt token should not affect loss'
assert torch.allclose(loss_a, torch.tensor(1.3862944), atol=1e-5)
assert torch.allclose(loss_c, loss_a, atol=1e-6), 'Uniform logits have same CE, but valid count changes safely'
""",
        },
        {
            "name": "Gradient only on valid positions",
            "code": """
import torch
logits = torch.randn(1, 3, 5, requires_grad=True)
labels = torch.tensor([[-100, 2, -100]])
loss = {fn}(logits, labels)
loss.backward()
assert logits.grad[0, 0].abs().sum() == 0, 'Ignored position 0 should have zero gradient'
assert logits.grad[0, 1].abs().sum() > 0, 'Valid position should have gradient'
assert logits.grad[0, 2].abs().sum() == 0, 'Ignored position 2 should have zero gradient'
""",
        },
        {
            "name": "Scalar output",
            "code": """
import torch
out = {fn}(torch.randn(3, 5, 11), torch.randint(0, 11, (3, 5)))
assert out.dim() == 0, 'Loss must be scalar'
""",
        },
        {
            "name": "All ignored positions have connected zero loss",
            "code": """
import torch
logits = torch.randn(2, 5, 11, requires_grad=True)
labels = torch.full((2, 5), -100, dtype=torch.long)
loss = {fn}(logits, labels)
assert loss.item() == 0.0
loss.backward()
assert logits.grad is not None and torch.all(logits.grad == 0)
""",
        },
    ],
    "solution": '''def sft_loss(logits, labels, ignore_index=-100):
    v = logits.shape[-1]
    flat_logits, flat_labels = logits.reshape(-1, v), labels.reshape(-1)
    if not (flat_labels != ignore_index).any():
        return flat_logits.sum() * 0.0
    return F.cross_entropy(flat_logits, flat_labels, ignore_index=ignore_index)''',
    "demo": """logits = torch.randn(2, 4, 10)
labels = torch.tensor([[1, 2, -100, 3], [-100, 4, 5, -100]])
print('SFT loss:', sft_loss(logits, labels).item())""",
}
