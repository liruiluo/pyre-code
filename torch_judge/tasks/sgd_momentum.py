"""SGD with Momentum optimizer task."""

TASK = {
    "title": "SGD with Momentum",
    "title_zh": "带动量的 SGD",
    "difficulty": "Medium",
    "description_en": "Implement stochastic gradient descent with momentum from scratch.\n\nMomentum keeps a velocity buffer for each parameter so repeated gradients accumulate direction and noisy gradients are smoothed.\n\n**Signature:** `SGDMomentum(params, lr=0.01, momentum=0.9)`\n\n**Methods:**\n- `step()` — update parameters using the momentum buffer\n- `zero_grad()` — zero all parameter gradients\n\n**Constraints:**\n- Match `torch.optim.SGD(..., momentum=momentum)` for the default PyTorch momentum rule\n- Skip parameters whose `.grad` is `None`",
    "description_zh": "从零实现带动量的随机梯度下降。\n\nMomentum 为每个参数维护速度缓存，使连续梯度累积方向并平滑噪声梯度。\n\n**签名:** `SGDMomentum(params, lr=0.01, momentum=0.9)`\n\n**方法:**\n- `step()` — 使用动量缓存更新参数\n- `zero_grad()` — 将所有参数梯度清零\n\n**约束:**\n- 与 `torch.optim.SGD(..., momentum=momentum)` 的默认动量规则一致\n- 跳过 `.grad is None` 的参数",
    "function_name": "SGDMomentum",
    "hint": "1. Keep one velocity buffer per parameter\n2. Update `v = momentum * v + grad`\n3. Update parameter with `p -= lr * v` under `torch.no_grad()`",
    "hint_zh": "1. 为每个参数维护一个 velocity 缓存\n2. 更新 `v = momentum * v + grad`\n3. 在 `torch.no_grad()` 下执行 `p -= lr * v`",
    "tests": [
        {
            "name": "Parameters change after step",
            "code": """
import torch
torch.manual_seed(0)
w = torch.randn(4, 3, requires_grad=True)
opt = {fn}([w], lr=0.1, momentum=0.9)
(w ** 2).sum().backward()
w_before = w.detach().clone()
opt.step()
assert not torch.equal(w.detach(), w_before), 'Parameters should change after step'
""",
        },
        {
            "name": "Matches torch.optim.SGD with momentum",
            "code": """
import torch
torch.manual_seed(0)
w1 = torch.randn(8, 4, requires_grad=True)
w2 = w1.detach().clone().requires_grad_(True)
opt1 = {fn}([w1], lr=0.05, momentum=0.8)
opt2 = torch.optim.SGD([w2], lr=0.05, momentum=0.8)
for _ in range(6):
    (w1.pow(2).sum() + w1.sum() * 0.1).backward()
    (w2.pow(2).sum() + w2.sum() * 0.1).backward()
    opt1.step(); opt1.zero_grad()
    opt2.step(); opt2.zero_grad()
assert torch.allclose(w1.detach(), w2.detach(), atol=1e-6), f'Max diff: {(w1 - w2).abs().max().item():.8f}'
""",
        },
        {
            "name": "Momentum accumulates across steps",
            "code": """
import torch
w = torch.tensor([1.0], requires_grad=True)
opt = {fn}([w], lr=1.0, momentum=0.5)
w.grad = torch.tensor([2.0])
opt.step()
assert torch.allclose(w.detach(), torch.tensor([-1.0]))
w.grad = torch.tensor([2.0])
opt.step()
# velocity = 0.5 * 2 + 2 = 3, so w = -1 - 3 = -4
assert torch.allclose(w.detach(), torch.tensor([-4.0]))
""",
        },
        {
            "name": "zero_grad works",
            "code": """
import torch
w = torch.randn(4, requires_grad=True)
opt = {fn}([w])
(w ** 2).sum().backward()
assert w.grad.abs().sum() > 0
opt.zero_grad()
assert w.grad.abs().sum() == 0, 'zero_grad should clear gradients'
""",
        },
    ],
    "solution": '''class SGDMomentum:
    def __init__(self, params, lr=0.01, momentum=0.9):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.velocity = [torch.zeros_like(p) for p in self.params]

    def step(self):
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue
                self.velocity[i] = self.momentum * self.velocity[i] + p.grad
                p -= self.lr * self.velocity[i]

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()''',
    "demo": """w = torch.tensor([1.0], requires_grad=True)
opt = SGDMomentum([w], lr=0.1, momentum=0.9)
for step in range(3):
    (w ** 2).sum().backward()
    opt.step()
    opt.zero_grad()
    print(step, w.item())""",
}
