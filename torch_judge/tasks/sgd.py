"""Plain SGD optimizer task."""

TASK = {
    "title": "SGD Optimizer",
    "title_zh": "SGD 优化器",
    "difficulty": "Easy",
    "description_en": "Implement plain stochastic gradient descent as an optimizer class.\n\nSGD updates each parameter with `p = p - lr * grad`. This is the base optimizer that momentum and Adam build on.\n\n**Signature:** `MySGD(params, lr=1e-2)`\n\n**Methods:**\n- `step()` — apply one SGD update to every parameter with a gradient\n- `zero_grad()` — zero all parameter gradients\n\n**Constraints:**\n- Match `torch.optim.SGD` without momentum or weight decay\n- Skip parameters whose `grad` is `None`\n- Update parameters without tracking autograd operations",
    "description_zh": "实现最基础的随机梯度下降优化器。\n\nSGD 对每个参数执行 `p = p - lr * grad`。Momentum 和 Adam 等优化器都以它为基础。\n\n**签名:** `MySGD(params, lr=1e-2)`\n\n**方法:**\n- `step()` — 对每个有梯度的参数执行一次 SGD 更新\n- `zero_grad()` — 清零所有参数梯度\n\n**约束:**\n- 与无 momentum、无 weight decay 的 `torch.optim.SGD` 一致\n- 跳过 `grad is None` 的参数\n- 参数更新不能被 autograd 追踪",
    "function_name": "MySGD",
    "hint": "Store `list(params)`. In `step`, use `with torch.no_grad(): p -= lr * p.grad`. In `zero_grad`, call `p.grad.zero_()` if grad exists.",
    "hint_zh": "保存 `list(params)`。`step` 中使用 `with torch.no_grad(): p -= lr * p.grad`。`zero_grad` 中若梯度存在则 `p.grad.zero_()`。",
    "tests": [
        {
            "name": "Matches torch.optim.SGD",
            "code": """
import torch
torch.manual_seed(0)
w1 = torch.randn(4, requires_grad=True)
w2 = w1.detach().clone().requires_grad_(True)
opt1 = {fn}([w1], lr=0.05)
opt2 = torch.optim.SGD([w2], lr=0.05)
for _ in range(6):
    loss1 = (w1 ** 2).sum() + w1.sum()
    loss2 = (w2 ** 2).sum() + w2.sum()
    loss1.backward(); loss2.backward()
    opt1.step(); opt1.zero_grad()
    opt2.step(); opt2.zero_grad()
assert torch.allclose(w1.detach(), w2.detach(), atol=1e-6), f'Max diff {(w1 - w2).abs().max().item():.8f}'
""",
        },
        {
            "name": "Single manual update",
            "code": """
import torch
w = torch.tensor([1.0, -2.0], requires_grad=True)
w.grad = torch.tensor([0.5, -1.0])
opt = {fn}([w], lr=0.1)
opt.step()
expected = torch.tensor([0.95, -1.9])
assert torch.allclose(w.detach(), expected), f'{w.detach()} vs {expected}'
""",
        },
        {
            "name": "Skips missing gradients",
            "code": """
import torch
w = torch.tensor([3.0], requires_grad=True)
opt = {fn}([w], lr=0.1)
opt.step()
assert torch.allclose(w.detach(), torch.tensor([3.0]))
""",
        },
        {
            "name": "zero_grad works",
            "code": """
import torch
w = torch.randn(5, requires_grad=True)
opt = {fn}([w])
(w ** 2).sum().backward()
assert w.grad.abs().sum() > 0
opt.zero_grad()
assert w.grad.abs().sum() == 0
""",
        },
    ],
    "solution": '''class MySGD:
    def __init__(self, params, lr=1e-2):
        self.params = list(params)
        self.lr = lr

    def step(self):
        with torch.no_grad():
            for p in self.params:
                if p.grad is None:
                    continue
                p -= self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()''',
    "demo": """w = torch.tensor([1.0, -2.0], requires_grad=True)
opt = MySGD([w], lr=0.1)
(w ** 2).sum().backward()
opt.step()
print(w)""",
}
