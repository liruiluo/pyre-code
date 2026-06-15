"""Pre-norm vs post-norm residual block task."""

TASK = {
    "title": "Pre-Norm vs Post-Norm Residual Block",
    "title_zh": "Pre-Norm 与 Post-Norm 残差块",
    "difficulty": "Medium",
    "description_en": "Implement a residual wrapper that supports both Transformer normalization orders.\n\nPre-norm applies LayerNorm before the sublayer: `x + sublayer(norm(x))`. Post-norm applies LayerNorm after the residual addition: `norm(x + sublayer(x))`.\n\n**Signature:** `ResidualNormBlock(d_model, sublayer, mode='pre')` (nn.Module)\n\n**Forward:** `forward(x) -> Tensor`\n\n**Constraints:**\n- `mode='pre'`: return `x + sublayer(norm(x))`\n- `mode='post'`: return `norm(x + sublayer(x))`\n- Use one `nn.LayerNorm(d_model)`\n- Raise `ValueError` for unknown modes",
    "description_zh": "实现同时支持 Transformer 两种归一化顺序的残差包装层。\n\nPre-norm 在子层前做 LayerNorm：`x + sublayer(norm(x))`。Post-norm 在残差相加后做 LayerNorm：`norm(x + sublayer(x))`。\n\n**签名:** `ResidualNormBlock(d_model, sublayer, mode='pre')`（nn.Module）\n\n**前向传播:** `forward(x) -> Tensor`\n\n**约束:**\n- `mode='pre'`：返回 `x + sublayer(norm(x))`\n- `mode='post'`：返回 `norm(x + sublayer(x))`\n- 使用一个 `nn.LayerNorm(d_model)`\n- 未知 mode 要抛出 `ValueError`",
    "function_name": "ResidualNormBlock",
    "hint": "Store the sublayer, a LayerNorm, and mode. Branch in `forward`: pre uses normalized input inside the sublayer; post normalizes after residual addition.",
    "hint_zh": "保存 sublayer、LayerNorm 和 mode。`forward` 中分支：pre 把归一化后的输入送入子层；post 先残差相加再归一化。",
    "tests": [
        {
            "name": "Pre-norm formula",
            "code": """
import torch
import torch.nn as nn
torch.manual_seed(0)
sublayer = nn.Linear(4, 4, bias=False)
block = {fn}(4, sublayer, mode='pre')
x = torch.randn(2, 3, 4)
out = block(x)
expected = x + sublayer(block.norm(x))
assert torch.allclose(out, expected, atol=1e-6), 'pre mode should be x + sublayer(norm(x))'
""",
        },
        {
            "name": "Post-norm formula",
            "code": """
import torch
import torch.nn as nn
torch.manual_seed(0)
sublayer = nn.Linear(4, 4, bias=False)
block = {fn}(4, sublayer, mode='post')
x = torch.randn(2, 3, 4)
out = block(x)
expected = block.norm(x + sublayer(x))
assert torch.allclose(out, expected, atol=1e-6), 'post mode should be norm(x + sublayer(x))'
""",
        },
        {
            "name": "Preserves shape and module type",
            "code": """
import torch
import torch.nn as nn
block = {fn}(8, nn.Linear(8, 8), mode='pre')
assert isinstance(block, nn.Module)
x = torch.randn(5, 7, 8)
assert block(x).shape == x.shape
assert isinstance(block.norm, nn.LayerNorm)
""",
        },
        {
            "name": "Invalid mode raises",
            "code": """
import torch.nn as nn
try:
    {fn}(4, nn.Linear(4, 4), mode='middle')
except ValueError:
    pass
else:
    raise AssertionError('Unknown mode should raise ValueError')
""",
        },
        {
            "name": "Gradient flow through sublayer and norm",
            "code": """
import torch
import torch.nn as nn
block = {fn}(6, nn.Linear(6, 6), mode='pre')
x = torch.randn(2, 4, 6, requires_grad=True)
block(x).sum().backward()
assert x.grad is not None
assert all(p.grad is not None for p in block.parameters()), 'All parameters should receive gradients'
""",
        },
    ],
    "solution": '''class ResidualNormBlock(nn.Module):
    def __init__(self, d_model, sublayer, mode='pre'):
        super().__init__()
        if mode not in ('pre', 'post'):
            raise ValueError("mode must be 'pre' or 'post'")
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(d_model)
        self.mode = mode

    def forward(self, x):
        if self.mode == 'pre':
            return x + self.sublayer(self.norm(x))
        return self.norm(x + self.sublayer(x))''',
    "demo": """sublayer = nn.Linear(16, 16)
pre = ResidualNormBlock(16, sublayer, mode='pre')
x = torch.randn(2, 4, 16)
print(pre(x).shape)""",
}
