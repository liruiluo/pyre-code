"""Transformer encoder layer task."""

TASK = {
    "title": "Transformer Encoder Layer",
    "title_zh": "Transformer 编码器层",
    "difficulty": "Hard",
    "description_en": "Implement a pre-norm Transformer encoder layer, BERT-style.\n\nAn encoder layer uses bidirectional self-attention followed by a position-wise feed-forward network, each wrapped in residual connections. Unlike GPT decoding, it must not apply a causal mask.\n\n**Signature:** `TransformerEncoderLayer(d_model, num_heads, d_ff=None, dropout=0.0)` (nn.Module)\n\n**Forward:** `forward(x) -> Tensor`\n- `x` — input tensor `(B, S, d_model)`\n\n**Returns:** encoded tensor `(B, S, d_model)`\n\n**Constraints:**\n- Pre-norm residuals: `x = x + self_attn(norm1(x))`, then `x = x + ffn(norm2(x))`\n- Self-attention is bidirectional: every position may attend to every other position\n- Use explicit Q/K/V/O linear projections and an FFN with `d_ff or 4 * d_model` hidden size",
    "description_zh": "实现 BERT 风格的 pre-norm Transformer 编码器层。\n\n编码器层由双向自注意力和逐位置前馈网络组成，两者都包在残差连接里。不同于 GPT 解码，它不能使用因果遮蔽。\n\n**签名:** `TransformerEncoderLayer(d_model, num_heads, d_ff=None, dropout=0.0)`（nn.Module）\n\n**前向传播:** `forward(x) -> Tensor`\n- `x` — 输入张量 `(B, S, d_model)`\n\n**返回:** 编码后张量 `(B, S, d_model)`\n\n**约束:**\n- Pre-norm 残差：`x = x + self_attn(norm1(x))`，再 `x = x + ffn(norm2(x))`\n- 自注意力是双向的：每个位置都可以看见其他位置\n- 使用显式 Q/K/V/O 线性投影，FFN 隐藏维为 `d_ff or 4 * d_model`",
    "function_name": "TransformerEncoderLayer",
    "hint": "Project normalized `x` into Q/K/V heads, compute unmasked scaled dot-product attention, project with `W_o`, add residual, then apply `fc1 -> GELU -> dropout -> fc2` to the second normalized residual.",
    "hint_zh": "把归一化后的 `x` 投影成 Q/K/V 多头，计算无遮挡缩放点积注意力，用 `W_o` 投影后做残差相加；再对第二个归一化残差应用 `fc1 -> GELU -> dropout -> fc2`。",
    "tests": [
        {
            "name": "Module, output shape, and core attributes",
            "code": """
import torch
import torch.nn as nn
block = {fn}(d_model=32, num_heads=4)
assert isinstance(block, nn.Module)
out = block(torch.randn(2, 5, 32))
assert out.shape == (2, 5, 32), f'Output shape: {out.shape}'
for name in ['ln1', 'ln2', 'W_q', 'W_k', 'W_v', 'W_o', 'fc1', 'fc2']:
    assert hasattr(block, name), f'Missing {name}'
""",
        },
        {
            "name": "FFN uses requested hidden size",
            "code": """
import torch
block = {fn}(d_model=16, num_heads=4, d_ff=40)
assert block.fc1.weight.shape == (40, 16), f'fc1 shape: {block.fc1.weight.shape}'
assert block.fc2.weight.shape == (16, 40), f'fc2 shape: {block.fc2.weight.shape}'
""",
        },
        {
            "name": "Bidirectional attention lets future tokens affect past",
            "code": """
import torch
torch.manual_seed(0)
block = {fn}(d_model=32, num_heads=4, dropout=0.0)
block.eval()
x = torch.randn(1, 6, 32)
out1 = block(x)
x2 = x.clone()
x2[:, 4:] = torch.randn(1, 2, 32)
out2 = block(x2)
assert not torch.allclose(out1[:, :2], out2[:, :2], atol=1e-5), 'Encoder self-attention should be bidirectional, not causal'
""",
        },
        {
            "name": "Numerical composition matches reference",
            "code": """
import torch
import torch.nn.functional as F
torch.manual_seed(1)
B, S, D, H = 2, 4, 16, 4
block = {fn}(d_model=D, num_heads=H, d_ff=32, dropout=0.0)
block.eval()
x = torch.randn(B, S, D)
out = block(x)
with torch.no_grad():
    d_k = D // H
    y = block.ln1(x)
    q = block.W_q(y).view(B, S, H, d_k).transpose(1, 2)
    k = block.W_k(y).view(B, S, H, d_k).transpose(1, 2)
    v = block.W_v(y).view(B, S, H, d_k).transpose(1, 2)
    weights = torch.softmax((q @ k.transpose(-2, -1)) / (d_k ** 0.5), dim=-1)
    attn = (weights @ v).transpose(1, 2).contiguous().view(B, S, D)
    x1 = x + block.W_o(attn)
    expected = x1 + block.fc2(block.dropout(F.gelu(block.fc1(block.ln2(x1)))))
assert torch.allclose(out, expected, atol=1e-5), f'Max diff {(out - expected).abs().max()}'
""",
        },
        {
            "name": "Gradient flow",
            "code": """
import torch
torch.manual_seed(2)
block = {fn}(d_model=16, num_heads=4)
x = torch.randn(2, 3, 16, requires_grad=True)
block(x).sum().backward()
assert x.grad is not None, 'Missing input gradient'
assert all(p.grad is not None for p in block.parameters()), 'All parameters should receive gradients'
""",
        },
    ],
    "solution": '''class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.0):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_ff, d_model)

    def self_attn(self, x):
        B, S, _ = x.shape
        q = self.W_q(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        attn = torch.matmul(weights, v)
        attn = attn.transpose(1, 2).contiguous().view(B, S, -1)
        return self.W_o(attn)

    def forward(self, x):
        x = x + self.self_attn(self.ln1(x))
        x = x + self.fc2(self.dropout(F.gelu(self.fc1(self.ln2(x)))))
        return x''',
    "demo": """block = TransformerEncoderLayer(d_model=64, num_heads=4)
x = torch.randn(2, 8, 64)
print(block(x).shape)""",
}
