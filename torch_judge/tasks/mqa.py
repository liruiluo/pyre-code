"""Multi-Query Attention task."""

TASK = {
    "title": "Multi-Query Attention",
    "title_zh": "多查询注意力（MQA）",
    "difficulty": "Hard",
    "description_en": "Implement Multi-Query Attention (MQA).\n\nMQA keeps many query heads but shares a single key head and a single value head across all query heads, reducing KV-cache memory during decoding.\n\n**Signature:** `MultiQueryAttention(d_model, num_heads)`\n\n**Forward:** `forward(x) -> Tensor`\n- `x` — input tensor `(B, T, d_model)`\n\n**Returns:** attention output `(B, T, d_model)`\n\n**Constraints:**\n- `W_q` projects to `d_model`\n- `W_k` and `W_v` each project to one head dimension `d_model // num_heads`\n- Expand K/V to all query heads before scaled dot-product attention",
    "description_zh": "实现多查询注意力（MQA）。\n\nMQA 保留多个 query head，但所有 query head 共享一个 key head 和一个 value head，从而降低解码阶段 KV cache 内存。\n\n**签名:** `MultiQueryAttention(d_model, num_heads)`\n\n**前向传播:** `forward(x) -> Tensor`\n- `x` — 输入张量 `(B, T, d_model)`\n\n**返回:** 注意力输出 `(B, T, d_model)`\n\n**约束:**\n- `W_q` 投影到 `d_model`\n- `W_k` 和 `W_v` 各投影到一个 head 维度 `d_model // num_heads`\n- 在缩放点积注意力前将 K/V 扩展到所有 query head",
    "function_name": "MultiQueryAttention",
    "hint": "1. `q`: `(B, H, T, d_k)`; `k/v`: `(B, 1, T, d_k)`\n2. Expand `k/v` with `.expand(B, H, T, d_k)`\n3. Compute scaled dot-product attention and project back with `W_o`",
    "hint_zh": "1. `q`: `(B, H, T, d_k)`；`k/v`: `(B, 1, T, d_k)`\n2. 用 `.expand(B, H, T, d_k)` 扩展 `k/v`\n3. 计算缩放点积注意力并用 `W_o` 投影回去",
    "tests": [
        {
            "name": "Is nn.Module and output shape",
            "code": """
import torch, torch.nn as nn
torch.manual_seed(0)
mqa = {fn}(d_model=32, num_heads=8)
assert isinstance(mqa, nn.Module)
out = mqa.forward(torch.randn(2, 5, 32))
assert out.shape == (2, 5, 32), f'Shape mismatch: {out.shape}'
""",
        },
        {
            "name": "Projection shapes use one KV head",
            "code": """
import torch, torch.nn as nn
mqa = {fn}(d_model=32, num_heads=8)
d_k = 32 // 8
assert mqa.W_q.weight.shape == (32, 32)
assert mqa.W_k.weight.shape == (d_k, 32), f'W_k shape {mqa.W_k.weight.shape}'
assert mqa.W_v.weight.shape == (d_k, 32), f'W_v shape {mqa.W_v.weight.shape}'
assert isinstance(mqa.W_o, nn.Linear)
""",
        },
        {
            "name": "All heads share the same K values",
            "code": """
import torch
torch.manual_seed(0)
D, H = 16, 4
mqa = {fn}(D, H)
x = torch.randn(1, 3, D)
d_k = D // H
k = mqa.W_k(x).view(1, 1, 3, d_k)
k_exp = k.expand(1, H, 3, d_k)
assert torch.equal(k_exp[:, 0], k_exp[:, 1])
assert torch.equal(k_exp[:, 1], k_exp[:, 2])
""",
        },
        {
            "name": "Matches manual computation for one example",
            "code": """
import torch, math
torch.manual_seed(0)
D, H = 8, 2
mqa = {fn}(D, H)
x = torch.randn(1, 4, D)
out = mqa.forward(x)
d_k = D // H
q = mqa.W_q(x).view(1, 4, H, d_k).transpose(1, 2)
k = mqa.W_k(x).view(1, 1, 4, d_k).expand(1, H, 4, d_k)
v = mqa.W_v(x).view(1, 1, 4, d_k).expand(1, H, 4, d_k)
weights = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k), dim=-1)
manual = torch.matmul(weights, v).transpose(1, 2).contiguous().view(1, 4, D)
manual = mqa.W_o(manual)
assert torch.allclose(out, manual, atol=1e-6)
""",
        },
        {
            "name": "Gradient flow",
            "code": """
import torch
mqa = {fn}(16, 4)
x = torch.randn(2, 3, 16, requires_grad=True)
mqa.forward(x).sum().backward()
assert x.grad is not None
assert mqa.W_q.weight.grad is not None and mqa.W_k.weight.grad is not None and mqa.W_v.weight.grad is not None
""",
        },
    ],
    "solution": '''class MultiQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, self.d_k)
        self.W_v = nn.Linear(d_model, self.d_k)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.W_q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, 1, T, self.d_k).expand(B, self.num_heads, T, self.d_k)
        v = self.W_v(x).view(B, 1, T, self.d_k).expand(B, self.num_heads, T, self.d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        weights = torch.softmax(scores, dim=-1)
        attn = torch.matmul(weights, v)
        out = attn.transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_o(out)''',
    "demo": """mqa = MultiQueryAttention(32, 8)
x = torch.randn(2, 5, 32)
print(mqa.forward(x).shape)""",
}
