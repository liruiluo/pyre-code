"""AdamW optimizer task."""

TASK = {
    "title": "AdamW Optimizer",
    "title_zh": "AdamW 优化器",
    "difficulty": "Medium",
    "description_en": "Implement AdamW from scratch.\n\nAdamW is Adam with decoupled weight decay: the parameter decay is applied directly to weights, instead of being mixed into the adaptive gradient moments.\n\n**Signature:** `MyAdamW(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)`\n\n**Methods:**\n- `step()` — apply decoupled weight decay and Adam update\n- `zero_grad()` — zero all parameter gradients\n\n**Constraints:**\n- Match `torch.optim.AdamW` numerically\n- Apply weight decay as `p *= (1 - lr * weight_decay)`",
    "description_zh": "从零实现 AdamW。\n\nAdamW 是带解耦权重衰减的 Adam：weight decay 直接作用在参数上，而不是混入自适应梯度矩。\n\n**签名:** `MyAdamW(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)`\n\n**方法:**\n- `step()` — 执行解耦权重衰减和 Adam 更新\n- `zero_grad()` — 将所有参数梯度清零\n\n**约束:**\n- 与 `torch.optim.AdamW` 数值一致\n- weight decay 按 `p *= (1 - lr * weight_decay)` 直接作用于参数",
    "function_name": "MyAdamW",
    "hint": "1. Under `torch.no_grad()`, first apply `p *= 1 - lr * weight_decay`\n2. Update Adam moments and bias-correct them\n3. Apply `p -= lr * m_hat / (sqrt(v_hat) + eps)`",
    "hint_zh": "1. 在 `torch.no_grad()` 下先执行 `p *= 1 - lr * weight_decay`\n2. 更新 Adam 一阶/二阶矩并做偏差校正\n3. 执行 `p -= lr * m_hat / (sqrt(v_hat) + eps)`",
    "tests": [
        {
            "name": "Matches torch.optim.AdamW",
            "code": """
import torch
torch.manual_seed(0)
w1 = torch.randn(5, 3, requires_grad=True)
w2 = w1.detach().clone().requires_grad_(True)
opt1 = {fn}([w1], lr=0.002, betas=(0.8, 0.95), eps=1e-8, weight_decay=0.05)
opt2 = torch.optim.AdamW([w2], lr=0.002, betas=(0.8, 0.95), eps=1e-8, weight_decay=0.05)
for _ in range(8):
    loss1 = (w1 ** 2).sum() + 0.2 * w1.sum()
    loss2 = (w2 ** 2).sum() + 0.2 * w2.sum()
    loss1.backward(); loss2.backward()
    opt1.step(); opt1.zero_grad()
    opt2.step(); opt2.zero_grad()
assert torch.allclose(w1.detach(), w2.detach(), atol=1e-6), f'Max diff: {(w1 - w2).abs().max().item():.8f}'
""",
        },
        {
            "name": "Decoupled decay happens even with zero gradient",
            "code": """
import torch
w = torch.tensor([10.0], requires_grad=True)
opt = {fn}([w], lr=0.1, weight_decay=0.2)
w.grad = torch.zeros_like(w)
opt.step()
assert torch.allclose(w.detach(), torch.tensor([9.8]), atol=1e-6), f'Expected decoupled decay to 9.8, got {w.item():.6f}'
""",
        },
        {
            "name": "No decay when grad is missing",
            "code": """
import torch
w = torch.tensor([10.0], requires_grad=True)
opt = {fn}([w], lr=0.1, weight_decay=0.2)
opt.step()
assert torch.allclose(w.detach(), torch.tensor([10.0])), 'Parameters with grad=None should be skipped'
""",
        },
        {
            "name": "zero_grad works",
            "code": """
import torch
w = torch.randn(3, requires_grad=True)
opt = {fn}([w])
(w ** 2).sum().backward()
assert w.grad.abs().sum() > 0
opt.zero_grad()
assert w.grad.abs().sum() == 0
""",
        },
    ],
    "solution": '''class MyAdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = [0 for _ in self.params]
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def step(self):
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue
                self.t[i] += 1
                if self.weight_decay != 0:
                    p *= 1 - self.lr * self.weight_decay
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * p.grad ** 2
                m_hat = self.m[i] / (1 - self.beta1 ** self.t[i])
                v_hat = self.v[i] / (1 - self.beta2 ** self.t[i])
                p -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()''',
    "demo": """w = torch.randn(4, requires_grad=True)
opt = MyAdamW([w], lr=0.01, weight_decay=0.1)
for step in range(3):
    (w ** 2).sum().backward()
    opt.step()
    opt.zero_grad()
    print(step, w.norm().item())""",
}
