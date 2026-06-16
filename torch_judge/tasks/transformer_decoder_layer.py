"""Transformer decoder layer task."""

TASK = {
    "title": "Transformer Decoder Layer",
    "title_zh": "Transformer 解码器层",
    "difficulty": "Hard",
    "description_en": "Implement a pre-norm encoder-decoder Transformer decoder layer.\n\nA decoder layer first applies causal self-attention over the target sequence, then cross-attention from target queries to encoder keys/values, then a position-wise feed-forward network.\n\n**Signature:** `TransformerDecoderLayer(d_model, num_heads, d_ff=None, dropout=0.0)` (nn.Module)\n\n**Forward:** `forward(x, encoder_output) -> Tensor`\n- `x` — target-side input `(B, T, d_model)`\n- `encoder_output` — source-side encoded states `(B, S, d_model)`\n\n**Returns:** decoder output `(B, T, d_model)`\n\n**Constraints:**\n- Causal self-attention: target position `t` must not see future target positions\n- Cross-attention is not causal: every target position may attend to all encoder positions\n- Use three pre-norm residual blocks: self-attn, cross-attn, then FFN",
    "description_zh": "实现 pre-norm 的 encoder-decoder Transformer 解码器层。\n\n解码器层先对目标序列做因果自注意力，再用目标 query 对 encoder key/value 做交叉注意力，最后接逐位置前馈网络。\n\n**签名:** `TransformerDecoderLayer(d_model, num_heads, d_ff=None, dropout=0.0)`（nn.Module）\n\n**前向传播:** `forward(x, encoder_output) -> Tensor`\n- `x` — 目标侧输入 `(B, T, d_model)`\n- `encoder_output` — 源侧 encoder 状态 `(B, S, d_model)`\n\n**返回:** decoder 输出 `(B, T, d_model)`\n\n**约束:**\n- 因果自注意力：目标位置 `t` 不能看到未来目标位置\n- 交叉注意力不因果：每个目标位置都可以关注所有 encoder 位置\n- 使用三段 pre-norm 残差：self-attn、cross-attn、FFN",
    "function_name": "TransformerDecoderLayer",
    "hint": "Use one masked self-attention block for `x`, one cross-attention block where Q comes from normalized decoder states and K/V come from `encoder_output`, then an FFN residual. The self-attention scores need an upper-triangular `-inf` mask.",
    "hint_zh": "先对 `x` 做带上三角 `-inf` 遮蔽的 self-attention；再做 cross-attention，其中 Q 来自归一化后的 decoder 状态，K/V 来自 `encoder_output`；最后接 FFN 残差。",
    "tests": [
        {
            "name": "Module, output shape, and core attributes",
            "code": """
import torch
import torch.nn as nn
block = {fn}(d_model=32, num_heads=4)
assert isinstance(block, nn.Module)
out = block(torch.randn(2, 5, 32), torch.randn(2, 7, 32))
assert out.shape == (2, 5, 32), f'Output shape: {out.shape}'
for name in ['ln1', 'ln2', 'ln3', 'self_q', 'self_k', 'self_v', 'self_o', 'cross_q', 'cross_k', 'cross_v', 'cross_o', 'fc1', 'fc2']:
    assert hasattr(block, name), f'Missing {name}'
""",
        },
        {
            "name": "Causal target self-attention protects past positions",
            "code": """
import torch
torch.manual_seed(0)
block = {fn}(d_model=32, num_heads=4, dropout=0.0)
block.eval()
x = torch.randn(1, 6, 32)
enc = torch.randn(1, 5, 32)
out1 = block(x, enc)
x2 = x.clone()
x2[:, 3:] = torch.randn(1, 3, 32)
out2 = block(x2, enc)
assert torch.allclose(out1[:, :3], out2[:, :3], atol=1e-5), 'Future target tokens affected past decoder outputs'
""",
        },
        {
            "name": "Cross-attention lets encoder states affect every target position",
            "code": """
import torch
torch.manual_seed(1)
block = {fn}(d_model=32, num_heads=4, dropout=0.0)
block.eval()
x = torch.randn(1, 4, 32)
enc = torch.randn(1, 6, 32)
out1 = block(x, enc)
enc2 = enc.clone()
enc2[:, -1] = torch.randn(1, 32)
out2 = block(x, enc2)
assert not torch.allclose(out1[:, 0], out2[:, 0], atol=1e-5), 'Changing encoder states should affect target position 0 through cross-attention'
""",
        },
        {
            "name": "FFN uses requested hidden size",
            "code": """
block = {fn}(d_model=16, num_heads=4, d_ff=48)
assert block.fc1.weight.shape == (48, 16), f'fc1 shape: {block.fc1.weight.shape}'
assert block.fc2.weight.shape == (16, 48), f'fc2 shape: {block.fc2.weight.shape}'
""",
        },
        {
            "name": "Numerical composition matches reference",
            "code": """
import torch
import torch.nn.functional as F
torch.manual_seed(2)
B, T, S, D, H = 2, 4, 5, 16, 4
block = {fn}(d_model=D, num_heads=H, d_ff=32, dropout=0.0)
block.eval()
x = torch.randn(B, T, D)
enc = torch.randn(B, S, D)
out = block(x, enc)

def split_heads(t, heads, d_k):
    return t.view(t.shape[0], t.shape[1], heads, d_k).transpose(1, 2)

with torch.no_grad():
    d_k = D // H
    y = block.ln1(x)
    q = split_heads(block.self_q(y), H, d_k)
    k = split_heads(block.self_k(y), H, d_k)
    v = split_heads(block.self_v(y), H, d_k)
    scores = q @ k.transpose(-2, -1) / (d_k ** 0.5)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float('-inf'))
    self_ctx = (torch.softmax(scores, dim=-1) @ v).transpose(1, 2).contiguous().view(B, T, D)
    x1 = x + block.self_o(self_ctx)

    q2 = split_heads(block.cross_q(block.ln2(x1)), H, d_k)
    k2 = split_heads(block.cross_k(enc), H, d_k)
    v2 = split_heads(block.cross_v(enc), H, d_k)
    cross_ctx = (torch.softmax(q2 @ k2.transpose(-2, -1) / (d_k ** 0.5), dim=-1) @ v2).transpose(1, 2).contiguous().view(B, T, D)
    x2 = x1 + block.cross_o(cross_ctx)
    expected = x2 + block.fc2(block.dropout(F.gelu(block.fc1(block.ln3(x2)))))
assert torch.allclose(out, expected, atol=1e-5), f'Max diff {(out - expected).abs().max()}'
""",
        },
        {
            "name": "Gradient flow",
            "code": """
import torch
torch.manual_seed(3)
block = {fn}(d_model=16, num_heads=4)
x = torch.randn(2, 3, 16, requires_grad=True)
enc = torch.randn(2, 5, 16, requires_grad=True)
block(x, enc).sum().backward()
assert x.grad is not None and enc.grad is not None, 'Missing input gradients'
assert all(p.grad is not None for p in block.parameters()), 'All parameters should receive gradients'
""",
        },
    ],
    "solution": '''class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.0):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

        self.self_q = nn.Linear(d_model, d_model)
        self.self_k = nn.Linear(d_model, d_model)
        self.self_v = nn.Linear(d_model, d_model)
        self.self_o = nn.Linear(d_model, d_model)

        self.cross_q = nn.Linear(d_model, d_model)
        self.cross_k = nn.Linear(d_model, d_model)
        self.cross_v = nn.Linear(d_model, d_model)
        self.cross_o = nn.Linear(d_model, d_model)

        self.fc1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_ff, d_model)

    def split_heads(self, x):
        B, S, _ = x.shape
        return x.view(B, S, self.num_heads, self.d_k).transpose(1, 2)

    def merge_heads(self, x):
        B, _, S, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, S, -1)

    def causal_self_attn(self, x):
        B, T, _ = x.shape
        q = self.split_heads(self.self_q(x))
        k = self.split_heads(self.self_k(x))
        v = self.split_heads(self.self_v(x))
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        return self.self_o(self.merge_heads(torch.matmul(weights, v)))

    def cross_attn(self, x, encoder_output):
        q = self.split_heads(self.cross_q(x))
        k = self.split_heads(self.cross_k(encoder_output))
        v = self.split_heads(self.cross_v(encoder_output))
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        return self.cross_o(self.merge_heads(torch.matmul(weights, v)))

    def forward(self, x, encoder_output):
        x = x + self.causal_self_attn(self.ln1(x))
        x = x + self.cross_attn(self.ln2(x), encoder_output)
        x = x + self.fc2(self.dropout(F.gelu(self.fc1(self.ln3(x)))))
        return x''',
    "demo": """block = TransformerDecoderLayer(d_model=64, num_heads=4)
x = torch.randn(2, 6, 64)
encoder_output = torch.randn(2, 9, 64)
print(block(x, encoder_output).shape)""",
}
