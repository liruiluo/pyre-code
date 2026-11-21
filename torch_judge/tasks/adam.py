"""Adam Optimizer task."""

TASK = {
    "title": "Adam Optimizer",
    "title_zh": "Adam 优化器",
    "difficulty": "Medium",
    "description_en": "Implement the Adam optimizer from scratch.\n\nAdam combines momentum (1st moment) and RMSProp (2nd moment) with bias correction for adaptive per-parameter learning rates.\n\n**Signature:** `MyAdam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8)`\n\n**Methods:**\n- `step()` — update parameters using bias-corrected moments\n- `zero_grad()` — zero all parameter gradients\n\n**Constraints:**\n- Must match `torch.optim.Adam` numerically\n- Bias correction: `m_hat = m / (1 - beta1^t)`\n- `t` is per-parameter: only increment a parameter's counter on steps where it has a gradient (this is what `torch.optim.Adam` does)",
    "description_zh": "从零实现 Adam 优化器。\n\nAdam 结合了动量（一阶矩）和 RMSProp（二阶矩），并通过偏差校正实现自适应的逐参数学习率。\n\n**签名:** `MyAdam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8)`\n\n**方法:**\n- `step()` — 使用偏差校正后的矩更新参数\n- `zero_grad()` — 将所有参数梯度清零\n\n**约束:**\n- 必须与 `torch.optim.Adam` 数值一致\n- 偏差校正: `m_hat = m / (1 - beta1^t)`\n- `t` 是**逐参数**计数：某参数该步 `grad is None` 就不给它 +1（与 `torch.optim.Adam` 一致）",
    "function_name": "MyAdam",
    "hint": "1. m = β1·m + (1-β1)·g,  v = β2·v + (1-β2)·g²\n2. Bias-correct per parameter: m̂ = m/(1-β1^tᵢ),  v̂ = v/(1-β2^tᵢ)  — tᵢ 只数该参数有梯度的步数\n3. p -= lr · m̂ / (√v̂ + ε)",
    "hint_zh": "1. m = β1·m + (1-β1)·g,  v = β2·v + (1-β2)·g²\n2. 偏差校正（逐参数）：m̂ = m/(1-β1^tᵢ),  v̂ = v/(1-β2^tᵢ) —— tᵢ 只数该参数有梯度的步数\n3. p -= lr · m̂ / (√v̂ + ε)",
    "tests": [
        {
            "name": "Parameters change after step",
            "code": "\nimport torch\ntorch.manual_seed(0)\nw = torch.randn(4, 3, requires_grad=True)\nopt = {fn}([w], lr=0.01)\n(w ** 2).sum().backward()\nw_before = w.data.clone()\nopt.step()\nassert not torch.equal(w.data, w_before), 'Should change after step'\n"
        },
        {
            "name": "Matches torch.optim.Adam",
            "code": "\nimport torch\ntorch.manual_seed(0)\nw1 = torch.randn(8, 4, requires_grad=True)\nw2 = w1.data.clone().requires_grad_(True)\nopt1 = {fn}([w1], lr=0.001, betas=(0.9, 0.999), eps=1e-8)\nopt2 = torch.optim.Adam([w2], lr=0.001, betas=(0.9, 0.999), eps=1e-8)\nfor _ in range(5):\n    (w1 ** 2).sum().backward()\n    opt1.step(); opt1.zero_grad()\n    (w2 ** 2).sum().backward()\n    opt2.step(); opt2.zero_grad()\nassert torch.allclose(w1.data, w2.data, atol=1e-5), f'Max diff: {(w1.data-w2.data).abs().max():.6f}'\n"
        },
        {
            "name": "zero_grad works",
            "code": "\nimport torch\nw = torch.randn(4, requires_grad=True)\nopt = {fn}([w], lr=0.01)\n(w ** 2).sum().backward()\nassert w.grad.abs().sum() > 0\nopt.zero_grad()\nassert w.grad.abs().sum() == 0, 'zero_grad should zero all gradients'\n"
        },
        {
            "name": "Per-parameter step counts when gradients are missing",
            "code": """
import torch
w1 = torch.tensor([1.0], requires_grad=True)
w2 = torch.tensor([-1.0], requires_grad=True)
r1 = w1.detach().clone().requires_grad_(True)
r2 = w2.detach().clone().requires_grad_(True)
opt = {fn}([w1, w2], lr=0.01)
ref = torch.optim.Adam([r1, r2], lr=0.01)
for i in range(5):
    w1.grad = torch.tensor([0.5])
    r1.grad = torch.tensor([0.5])
    w2.grad = torch.tensor([0.3]) if i in (0, 2, 4) else None
    r2.grad = torch.tensor([0.3]) if i in (0, 2, 4) else None
    opt.step()
    ref.step()
assert torch.allclose(w1, r1, atol=1e-6)
assert torch.allclose(w2, r2, atol=1e-6)
""",
        },
    ],
    "solution": '''class MyAdam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = [0 for _ in self.params]
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def step(self):
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue
                self.t[i] += 1
                grad = p.grad
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad.square()
                m_hat = self.m[i] / (1 - self.beta1 ** self.t[i])
                v_hat = self.v[i] / (1 - self.beta2 ** self.t[i])
                p -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()''',
    "demo": """torch.manual_seed(0)
w = torch.randn(4, 3, requires_grad=True)
opt = MyAdam([w], lr=0.01)
for i in range(5):
    loss = (w ** 2).sum()
    loss.backward()
    opt.step()
    opt.zero_grad()
    print(f'Step {i}: loss={loss.item():.4f}')""",

}