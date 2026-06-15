"""Logistic regression task."""

TASK = {
    "title": "Logistic Regression",
    "title_zh": "逻辑回归",
    "difficulty": "Medium",
    "description_en": "Implement binary logistic regression with PyTorch tensors.\n\nLogistic regression is a linear classifier followed by a sigmoid, trained with binary cross-entropy.\n\n**Signature:** `LogisticRegression(in_features)`\n\n**Methods:**\n- `forward(x) -> Tensor` — return probabilities of shape `(N,)`\n- `loss(x, y) -> Tensor` — binary cross-entropy against labels `(N,)`\n- `predict(x, threshold=0.5) -> Tensor` — return 0/1 class predictions\n\n**Constraints:**\n- The module should inherit from `nn.Module`\n- Use an `nn.Linear(in_features, 1)` layer",
    "description_zh": "用 PyTorch Tensor 实现二分类逻辑回归。\n\n逻辑回归是线性分类器加 sigmoid，用二元交叉熵训练。\n\n**签名:** `LogisticRegression(in_features)`\n\n**方法:**\n- `forward(x) -> Tensor` — 返回形状 `(N,)` 的正类概率\n- `loss(x, y) -> Tensor` — 对标签 `(N,)` 计算二元交叉熵\n- `predict(x, threshold=0.5) -> Tensor` — 返回 0/1 分类结果\n\n**约束:**\n- 模块应继承 `nn.Module`\n- 使用 `nn.Linear(in_features, 1)` 层",
    "function_name": "LogisticRegression",
    "hint": "1. In `__init__`, create `self.linear = nn.Linear(in_features, 1)`\n2. `forward`: `sigmoid(self.linear(x)).squeeze(-1)`\n3. `loss`: binary cross-entropy on probabilities and float labels",
    "hint_zh": "1. 在 `__init__` 中创建 `self.linear = nn.Linear(in_features, 1)`\n2. `forward`: `sigmoid(self.linear(x)).squeeze(-1)`\n3. `loss`: 对概率和 float 标签计算二元交叉熵",
    "tests": [
        {
            "name": "Is nn.Module and returns probability shape",
            "code": """
import torch, torch.nn as nn
model = {fn}(3)
assert isinstance(model, nn.Module), 'LogisticRegression should inherit nn.Module'
out = model.forward(torch.randn(5, 3))
assert out.shape == (5,), f'Expected (5,), got {out.shape}'
assert torch.all((out >= 0) & (out <= 1)), 'Outputs should be probabilities in [0, 1]'
""",
        },
        {
            "name": "Known weights give known probabilities",
            "code": """
import torch
model = {fn}(2)
with torch.no_grad():
    model.linear.weight.copy_(torch.tensor([[1.0, -1.0]]))
    model.linear.bias.copy_(torch.tensor([0.5]))
x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
expected = torch.sigmoid(torch.tensor([1.5, -0.5]))
out = model.forward(x)
assert torch.allclose(out, expected, atol=1e-6), f'{out} vs {expected}'
""",
        },
        {
            "name": "Loss matches BCE",
            "code": """
import torch
torch.manual_seed(0)
model = {fn}(4)
x = torch.randn(6, 4)
y = torch.tensor([0, 1, 1, 0, 1, 0], dtype=torch.float32)
loss = model.loss(x, y)
ref = torch.nn.functional.binary_cross_entropy(model.forward(x), y)
assert loss.dim() == 0
assert torch.allclose(loss, ref, atol=1e-6), f'{loss.item():.6f} vs {ref.item():.6f}'
""",
        },
        {
            "name": "Predict threshold",
            "code": """
import torch
model = {fn}(1)
with torch.no_grad():
    model.linear.weight.fill_(10.0)
    model.linear.bias.zero_()
pred = model.predict(torch.tensor([[-1.0], [0.0], [1.0]]), threshold=0.5)
assert torch.equal(pred, torch.tensor([0, 1, 1])), f'Unexpected predictions: {pred}'
""",
        },
        {
            "name": "Gradient flow",
            "code": """
import torch
model = {fn}(3)
x = torch.randn(8, 3)
y = torch.randint(0, 2, (8,), dtype=torch.float32)
loss = model.loss(x, y)
loss.backward()
assert model.linear.weight.grad is not None and model.linear.bias.grad is not None
""",
        },
    ],
    "solution": '''class LogisticRegression(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x)).squeeze(-1)

    def loss(self, x, y):
        return F.binary_cross_entropy(self.forward(x), y.float())

    def predict(self, x, threshold=0.5):
        return (self.forward(x) >= threshold).long()''',
    "demo": """model = LogisticRegression(2)
x = torch.randn(4, 2)
y = torch.tensor([0, 1, 0, 1], dtype=torch.float32)
print('prob:', model.forward(x))
print('loss:', model.loss(x, y).item())""",
}
