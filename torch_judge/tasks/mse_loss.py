"""Mean squared error loss task."""

TASK = {
    "title": "MSE Loss",
    "title_zh": "均方误差损失",
    "difficulty": "Easy",
    "description_en": "Implement mean squared error loss from scratch.\n\nMSE is the standard regression loss: average the squared difference between prediction and target.\n\n**Signature:** `mse_loss(pred, target, reduction='mean') -> Tensor`\n\n**Parameters:**\n- `pred` — prediction tensor of any shape\n- `target` — target tensor, same shape as `pred`\n- `reduction` — `'mean'`, `'sum'`, or `'none'`\n\n**Returns:** reduced loss tensor\n\n**Constraints:**\n- Match `torch.nn.functional.mse_loss`\n- Preserve gradients with respect to `pred`",
    "description_zh": "从零实现均方误差损失。\n\nMSE 是回归任务的标准损失：对预测值和目标值之差的平方取平均。\n\n**签名:** `mse_loss(pred, target, reduction='mean') -> Tensor`\n\n**参数:**\n- `pred` — 任意形状的预测张量\n- `target` — 与 `pred` 同形状的目标张量\n- `reduction` — `'mean'`、`'sum'` 或 `'none'`\n\n**返回:** 归约后的损失张量\n\n**约束:**\n- 与 `torch.nn.functional.mse_loss` 一致\n- 保留对 `pred` 的梯度",
    "function_name": "mse_loss",
    "hint": "Compute `(pred - target) ** 2`, then apply the requested reduction: mean, sum, or none.",
    "hint_zh": "先计算 `(pred - target) ** 2`，再按 `mean`、`sum` 或 `none` 做归约。",
    "tests": [
        {
            "name": "Matches torch mean reduction",
            "code": """
import torch
import torch.nn.functional as F
torch.manual_seed(0)
pred = torch.randn(4, 5)
target = torch.randn(4, 5)
out = {fn}(pred, target)
ref = F.mse_loss(pred, target)
assert torch.allclose(out, ref, atol=1e-6), f'{out.item():.6f} vs {ref.item():.6f}'
""",
        },
        {
            "name": "Sum reduction",
            "code": """
import torch
pred = torch.tensor([1.0, 2.0, 3.0])
target = torch.tensor([0.0, 2.0, 5.0])
out = {fn}(pred, target, reduction='sum')
assert torch.allclose(out, torch.tensor(5.0)), f'Expected 5.0, got {out}'
""",
        },
        {
            "name": "No reduction keeps shape",
            "code": """
import torch
pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
target = torch.tensor([[1.5, 1.0], [1.0, 6.0]])
out = {fn}(pred, target, reduction='none')
expected = (pred - target) ** 2
assert out.shape == pred.shape
assert torch.allclose(out, expected)
""",
        },
        {
            "name": "Gradient flow",
            "code": """
import torch
pred = torch.randn(8, requires_grad=True)
target = torch.randn(8)
loss = {fn}(pred, target)
loss.backward()
assert pred.grad is not None, 'Missing gradient'
expected_grad = 2 * (pred.detach() - target) / pred.numel()
assert torch.allclose(pred.grad, expected_grad, atol=1e-6)
""",
        },
        {
            "name": "Invalid reduction fails clearly",
            "code": """
import torch
try:
    {fn}(torch.zeros(2), torch.ones(2), reduction='median')
except ValueError:
    pass
else:
    raise AssertionError('Invalid reduction should raise ValueError')
""",
        },
    ],
    "solution": '''def mse_loss(pred, target, reduction='mean'):
    loss = (pred - target) ** 2
    if reduction == 'mean':
        return loss.mean()
    if reduction == 'sum':
        return loss.sum()
    if reduction == 'none':
        return loss
    raise ValueError("reduction must be 'mean', 'sum', or 'none'")''',
    "demo": """pred = torch.tensor([1.0, 2.0, 3.0])
target = torch.tensor([0.0, 2.0, 5.0])
print('mean:', mse_loss(pred, target).item())
print('sum: ', mse_loss(pred, target, reduction='sum').item())""",
}
