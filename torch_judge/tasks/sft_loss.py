"""SFT loss task."""

TASK = {
    "title": "SFT Masked Language-Model Loss",
    "title_zh": "SFT 掩码语言模型损失",
    "difficulty": "Medium",
    "description_en": "Implement the supervised fine-tuning (SFT) loss for language models.\n\nSFT usually trains only selected response tokens: prompt tokens are masked out with `ignore_index=-100`, and valid tokens use cross-entropy over the vocabulary.\n\n**Signature:** `sft_loss(logits, labels, ignore_index=-100) -> Tensor`\n\n**Parameters:**\n- `logits` — model logits, shape `(B, T, V)`\n- `labels` — token labels, shape `(B, T)`, with ignored positions set to `ignore_index`\n\n**Returns:** scalar mean cross-entropy over non-ignored positions\n\n**Constraints:**\n- Flatten batch/time before cross-entropy\n- Ignored positions must not contribute to loss or gradients",
    "description_zh": "实现语言模型监督微调（SFT）损失。\n\nSFT 通常只训练被选中的回复 token：prompt token 用 `ignore_index=-100` 遮掉，有效 token 在词表上做交叉熵。\n\n**签名:** `sft_loss(logits, labels, ignore_index=-100) -> Tensor`\n\n**参数:**\n- `logits` — 模型 logits，形状 `(B, T, V)`\n- `labels` — token 标签，形状 `(B, T)`，忽略位置设为 `ignore_index`\n\n**返回:** 非忽略位置上的平均交叉熵标量\n\n**约束:**\n- 先展平 batch/time 再计算交叉熵\n- 被忽略位置不得贡献 loss 或梯度",
    "function_name": "sft_loss",
    "hint": "Use `F.cross_entropy(logits.view(-1, V), labels.view(-1), ignore_index=ignore_index)`.",
    "hint_zh": "使用 `F.cross_entropy(logits.view(-1, V), labels.view(-1), ignore_index=ignore_index)`。",
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
    ],
    "solution": '''def sft_loss(logits, labels, ignore_index=-100):
    vocab_size = logits.shape[-1]
    return F.cross_entropy(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1),
        ignore_index=ignore_index,
    )''',
    "demo": """logits = torch.randn(2, 4, 10)
labels = torch.tensor([[1, 2, -100, 3], [-100, 4, 5, -100]])
print('SFT loss:', sft_loss(logits, labels).item())""",
}
