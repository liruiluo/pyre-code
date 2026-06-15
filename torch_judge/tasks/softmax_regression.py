"""Softmax regression task."""

TASK = {
    "title": "Softmax Regression",
    "title_zh": "Softmax 回归",
    "difficulty": "Medium",
    "description_en": "Implement multinomial logistic regression, also called softmax regression.\n\nThis is the simplest multiclass classifier: a linear layer produces class logits and cross-entropy trains the weights.\n\n**Signature:** `SoftmaxRegression(in_features, num_classes)` (nn.Module)\n\n**Methods:**\n- `forward(x) -> Tensor` — return logits of shape `(N, C)`\n- `loss(x, y) -> Tensor` — mean cross-entropy loss\n- `predict(x) -> Tensor` — class indices from `argmax(logits)`\n\n**Constraints:**\n- Must inherit from `nn.Module`\n- Use a single linear mapping from features to classes\n- `loss` must match `F.cross_entropy`",
    "description_zh": "实现多分类逻辑回归，也叫 Softmax 回归。\n\n这是最简单的多分类器：线性层输出类别 logits，用交叉熵训练权重。\n\n**签名:** `SoftmaxRegression(in_features, num_classes)`（nn.Module）\n\n**方法:**\n- `forward(x) -> Tensor` — 返回形状 `(N, C)` 的 logits\n- `loss(x, y) -> Tensor` — 平均交叉熵损失\n- `predict(x) -> Tensor` — `argmax(logits)` 得到类别索引\n\n**约束:**\n- 必须继承 `nn.Module`\n- 只使用一个从特征到类别的线性映射\n- `loss` 必须与 `F.cross_entropy` 一致",
    "function_name": "SoftmaxRegression",
    "hint": "Use `self.linear = nn.Linear(in_features, num_classes)`. Forward returns logits, loss calls `F.cross_entropy`, predict uses `argmax(dim=-1)`.",
    "hint_zh": "使用 `self.linear = nn.Linear(in_features, num_classes)`。前向返回 logits，loss 调 `F.cross_entropy`，predict 用 `argmax(dim=-1)`。",
    "tests": [
        {
            "name": "Module and parameter shapes",
            "code": """
import torch
import torch.nn as nn
model = {fn}(in_features=4, num_classes=3)
assert isinstance(model, nn.Module), 'Must inherit from nn.Module'
assert hasattr(model, 'linear'), 'Need self.linear'
assert model.linear.weight.shape == (3, 4)
assert model.linear.bias.shape == (3,)
""",
        },
        {
            "name": "Forward returns logits",
            "code": """
import torch
model = {fn}(5, 7)
x = torch.randn(11, 5)
logits = model(x)
assert logits.shape == (11, 7), f'Expected (11, 7), got {logits.shape}'
""",
        },
        {
            "name": "Loss matches cross entropy",
            "code": """
import torch
import torch.nn.functional as F
torch.manual_seed(0)
model = {fn}(4, 3)
x = torch.randn(8, 4)
y = torch.randint(0, 3, (8,))
out = model.loss(x, y)
ref = F.cross_entropy(model(x), y)
assert torch.allclose(out, ref, atol=1e-6), f'{out.item():.6f} vs {ref.item():.6f}'
""",
        },
        {
            "name": "Predict uses argmax",
            "code": """
import torch
torch.manual_seed(0)
model = {fn}(3, 2)
with torch.no_grad():
    model.linear.weight.copy_(torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]))
    model.linear.bias.zero_()
x = torch.tensor([[2.0, 0.0, 0.0], [-3.0, 0.0, 0.0]])
pred = model.predict(x)
assert pred.tolist() == [0, 1], f'Got {pred.tolist()}'
""",
        },
        {
            "name": "Train step reduces loss",
            "code": """
import torch
torch.manual_seed(0)
x = torch.randn(60, 2)
y = (x[:, 0] - x[:, 1] > 0).long()
model = {fn}(2, 2)
opt = torch.optim.SGD(model.parameters(), lr=0.2)
loss0 = model.loss(x, y).item()
for _ in range(80):
    opt.zero_grad()
    loss = model.loss(x, y)
    loss.backward()
    opt.step()
loss1 = model.loss(x, y).item()
assert loss1 < loss0 * 0.7, f'Loss did not decrease enough: {loss0:.4f} -> {loss1:.4f}'
""",
        },
    ],
    "solution": '''class SoftmaxRegression(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.linear(x)

    def loss(self, x, y):
        return F.cross_entropy(self(x), y)

    def predict(self, x):
        return torch.argmax(self(x), dim=-1)''',
    "demo": """model = SoftmaxRegression(4, 3)
x = torch.randn(5, 4)
y = torch.randint(0, 3, (5,))
print('logits:', model(x).shape)
print('loss:', model.loss(x, y).item())""",
}
