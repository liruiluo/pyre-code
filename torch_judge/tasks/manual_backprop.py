"""Manual backpropagation for a tiny MLP task."""

TASK = {
    "title": "Manual Backpropagation for a Tiny MLP",
    "title_zh": "手写 Tiny MLP 反向传播",
    "difficulty": "Medium",
    "description_en": "Implement the forward and backward pass for a two-layer ReLU MLP without calling `.backward()`.\n\nThe network is `pred = relu(x @ w1 + b1) @ w2 + b2`, trained with mean squared error against `target`. Return both the scalar loss and manual parameter gradients.\n\n**Signature:** `tiny_mlp_backward(x, w1, b1, w2, b2, target) -> (loss, grads)`\n\n**Parameters:**\n- `x` — input tensor `(N, D)`\n- `w1`, `b1` — first-layer weight `(D, H)` and bias `(H,)`\n- `w2`, `b2` — second-layer weight `(H, C)` and bias `(C,)`\n- `target` — regression target `(N, C)`\n\n**Returns:** `(loss, (grad_w1, grad_b1, grad_w2, grad_b2))`\n\n**Constraints:**\n- Use MSE with mean reduction over all elements\n- Derive gradients manually with the chain rule\n- Do not call `loss.backward()` inside the function",
    "description_zh": "不调用 `.backward()`，手写两层 ReLU MLP 的前向和反向传播。\n\n网络为 `pred = relu(x @ w1 + b1) @ w2 + b2`，用均方误差拟合 `target`。返回标量 loss 和手写参数梯度。\n\n**签名:** `tiny_mlp_backward(x, w1, b1, w2, b2, target) -> (loss, grads)`\n\n**参数:**\n- `x` — 输入张量 `(N, D)`\n- `w1`, `b1` — 第一层权重 `(D, H)` 和偏置 `(H,)`\n- `w2`, `b2` — 第二层权重 `(H, C)` 和偏置 `(C,)`\n- `target` — 回归目标 `(N, C)`\n\n**返回:** `(loss, (grad_w1, grad_b1, grad_w2, grad_b2))`\n\n**约束:**\n- MSE 对所有元素取平均\n- 用链式法则手动推导梯度\n- 函数内部不要调用 `loss.backward()`",
    "function_name": "tiny_mlp_backward",
    "hint": "Forward: `h_pre = x @ w1 + b1`, `h = relu(h_pre)`, `pred = h @ w2 + b2`. For mean MSE, `d_pred = 2 * (pred - target) / pred.numel()`. Then chain through `w2`, ReLU, and `w1`.",
    "hint_zh": "前向：`h_pre = x @ w1 + b1`，`h = relu(h_pre)`，`pred = h @ w2 + b2`。mean MSE 下 `d_pred = 2 * (pred - target) / pred.numel()`，再依次链式传过 `w2`、ReLU 和 `w1`。",
    "tests": [
        {
            "name": "Returns scalar loss and four gradients",
            "code": """
import torch
torch.manual_seed(0)
x = torch.randn(4, 3)
w1 = torch.randn(3, 5)
b1 = torch.randn(5)
w2 = torch.randn(5, 2)
b2 = torch.randn(2)
target = torch.randn(4, 2)
loss, grads = {fn}(x, w1, b1, w2, b2, target)
assert isinstance(loss, torch.Tensor) and loss.shape == (), f'Loss should be scalar tensor, got {loss}'
assert isinstance(grads, (tuple, list)) and len(grads) == 4, 'Return grads as (grad_w1, grad_b1, grad_w2, grad_b2)'
for grad, ref in zip(grads, (w1, b1, w2, b2)):
    assert grad.shape == ref.shape, f'Gradient shape {grad.shape} should match {ref.shape}'
""",
        },
        {
            "name": "Loss matches manual MSE",
            "code": """
import torch
torch.manual_seed(1)
x = torch.randn(3, 4)
w1 = torch.randn(4, 6)
b1 = torch.randn(6)
w2 = torch.randn(6, 3)
b2 = torch.randn(3)
target = torch.randn(3, 3)
loss, _ = {fn}(x, w1, b1, w2, b2, target)
expected = ((torch.relu(x @ w1 + b1) @ w2 + b2 - target) ** 2).mean()
assert torch.allclose(loss, expected, atol=1e-6), f'{loss} vs {expected}'
""",
        },
        {
            "name": "Gradients match autograd",
            "code": """
import torch
torch.manual_seed(2)
x = torch.randn(5, 3)
w1 = torch.randn(3, 4, requires_grad=True)
b1 = torch.randn(4, requires_grad=True)
w2 = torch.randn(4, 2, requires_grad=True)
b2 = torch.randn(2, requires_grad=True)
target = torch.randn(5, 2)
loss, grads = {fn}(x, w1, b1, w2, b2, target)
ref_pred = torch.relu(x @ w1 + b1) @ w2 + b2
ref_loss = ((ref_pred - target) ** 2).mean()
ref_loss.backward()
for name, grad, ref in zip(['w1', 'b1', 'w2', 'b2'], grads, [w1.grad, b1.grad, w2.grad, b2.grad]):
    assert torch.allclose(grad, ref, atol=1e-5), f'{name} grad mismatch: max diff {(grad - ref).abs().max()}'
""",
        },
        {
            "name": "Handles inactive ReLU units",
            "code": """
import torch
x = torch.ones(2, 3)
w1 = -torch.ones(3, 4)
b1 = -torch.ones(4)
w2 = torch.randn(4, 2)
b2 = torch.zeros(2)
target = torch.ones(2, 2)
_, (grad_w1, grad_b1, grad_w2, grad_b2) = {fn}(x, w1, b1, w2, b2, target)
assert torch.allclose(grad_w1, torch.zeros_like(grad_w1)), 'Inactive ReLU should block w1 gradient'
assert torch.allclose(grad_b1, torch.zeros_like(grad_b1)), 'Inactive ReLU should block b1 gradient'
assert torch.allclose(grad_w2, torch.zeros_like(grad_w2)), 'Hidden activation is zero, so w2 gradient should be zero'
assert torch.allclose(grad_b2, torch.full_like(grad_b2, -1.0)), f'b2 grad should be mean MSE gradient, got {grad_b2}'
""",
        },
        {
            "name": "Does not populate .grad by side effect",
            "code": """
import torch
torch.manual_seed(3)
x = torch.randn(2, 3)
w1 = torch.randn(3, 4, requires_grad=True)
b1 = torch.randn(4, requires_grad=True)
w2 = torch.randn(4, 2, requires_grad=True)
b2 = torch.randn(2, requires_grad=True)
target = torch.randn(2, 2)
{fn}(x, w1, b1, w2, b2, target)
assert w1.grad is None and b1.grad is None and w2.grad is None and b2.grad is None, 'Do not call backward() or write parameter .grad inside the function'
""",
        },
    ],
    "solution": '''def tiny_mlp_backward(x, w1, b1, w2, b2, target):
    h_pre = x @ w1 + b1
    h = torch.relu(h_pre)
    pred = h @ w2 + b2
    diff = pred - target
    loss = (diff ** 2).mean()

    d_pred = 2.0 * diff / diff.numel()
    grad_w2 = h.T @ d_pred
    grad_b2 = d_pred.sum(dim=0)

    d_h = d_pred @ w2.T
    d_h_pre = d_h * (h_pre > 0).to(h_pre.dtype)
    grad_w1 = x.T @ d_h_pre
    grad_b1 = d_h_pre.sum(dim=0)

    return loss, (grad_w1, grad_b1, grad_w2, grad_b2)''',
    "demo": """torch.manual_seed(0)
x = torch.randn(4, 3)
w1, b1 = torch.randn(3, 5), torch.randn(5)
w2, b2 = torch.randn(5, 2), torch.randn(2)
target = torch.randn(4, 2)
loss, grads = tiny_mlp_backward(x, w1, b1, w2, b2, target)
print('loss:', loss.item())
print('grad shapes:', [g.shape for g in grads])""",
}
