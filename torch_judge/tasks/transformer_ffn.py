"""Transformer feed-forward network task."""

TASK = {
    "title": "Transformer Feed-Forward Network",
    "title_zh": "Transformer 前馈网络",
    "difficulty": "Medium",
    "description_en": "Implement the standard Transformer feed-forward network (FFN).\n\nThe classic Transformer FFN applies two position-wise linear layers with a nonlinearity and optional dropout between them.\n\n**Signature:** `TransformerFFN(d_model, d_ff, dropout=0.0)` (nn.Module)\n\n**Forward:** `forward(x) -> Tensor`\n- `x` — input tensor of shape `(..., d_model)`\n\n**Returns:** tensor with the same leading shape and final dimension `d_model`\n\n**Constraints:**\n- Layers: `Linear(d_model, d_ff)`, GELU, Dropout, `Linear(d_ff, d_model)`\n- Apply the same module to every position independently\n- Preserve batch/sequence dimensions",
    "description_zh": "实现标准 Transformer 前馈网络（FFN）。\n\n经典 Transformer FFN 是逐位置的两层线性网络，中间接非线性激活和可选 dropout。\n\n**签名:** `TransformerFFN(d_model, d_ff, dropout=0.0)`（nn.Module）\n\n**前向传播:** `forward(x) -> Tensor`\n- `x` — 形状为 `(..., d_model)` 的输入张量\n\n**返回:** 前置维度不变、最后一维为 `d_model` 的张量\n\n**约束:**\n- 层结构：`Linear(d_model, d_ff)`、GELU、Dropout、`Linear(d_ff, d_model)`\n- 对每个位置独立应用同一组参数\n- 保持 batch/sequence 维度",
    "function_name": "TransformerFFN",
    "hint": "Create `fc1`, `dropout`, and `fc2`. Forward is `fc2(dropout(gelu(fc1(x))))`.",
    "hint_zh": "创建 `fc1`、`dropout` 和 `fc2`。前向为 `fc2(dropout(gelu(fc1(x))))`。",
    "tests": [
        {
            "name": "Module and parameter shapes",
            "code": """
import torch
import torch.nn as nn
ffn = {fn}(d_model=32, d_ff=128, dropout=0.1)
assert isinstance(ffn, nn.Module)
assert ffn.fc1.weight.shape == (128, 32)
assert ffn.fc2.weight.shape == (32, 128)
assert hasattr(ffn, 'dropout')
""",
        },
        {
            "name": "3-D output shape",
            "code": """
import torch
ffn = {fn}(d_model=16, d_ff=64)
x = torch.randn(2, 5, 16)
out = ffn(x)
assert out.shape == (2, 5, 16), f'Got {out.shape}'
""",
        },
        {
            "name": "2-D output shape",
            "code": """
import torch
ffn = {fn}(d_model=8, d_ff=24)
x = torch.randn(7, 8)
out = ffn(x)
assert out.shape == (7, 8), f'Got {out.shape}'
""",
        },
        {
            "name": "Numerical composition in eval mode",
            "code": """
import torch
import torch.nn.functional as F
torch.manual_seed(0)
ffn = {fn}(d_model=8, d_ff=16, dropout=0.5)
ffn.eval()
x = torch.randn(3, 4, 8)
out = ffn(x)
expected = ffn.fc2(F.gelu(ffn.fc1(x)))
assert torch.allclose(out, expected, atol=1e-6), 'Forward should be fc2(gelu(fc1(x))) in eval mode'
""",
        },
        {
            "name": "Gradient flow",
            "code": """
import torch
ffn = {fn}(d_model=12, d_ff=48)
x = torch.randn(2, 3, 12, requires_grad=True)
ffn(x).sum().backward()
assert x.grad is not None
assert all(p.grad is not None for p in ffn.parameters()), 'All parameters should receive gradients'
""",
        },
    ],
    "solution": '''class TransformerFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))''',
    "demo": """ffn = TransformerFFN(d_model=64, d_ff=256)
x = torch.randn(2, 8, 64)
print(ffn(x).shape)""",
}
