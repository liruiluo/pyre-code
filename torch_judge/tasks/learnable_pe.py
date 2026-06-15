"""Learnable position encoding task."""

TASK = {
    "title": "Learnable Position Encoding",
    "title_zh": "可学习位置编码",
    "difficulty": "Easy",
    "description_en": "Implement learnable absolute position encoding.\n\nUnlike sinusoidal position encoding, learnable PE stores one trainable vector per position and adds it to token embeddings.\n\n**Signature:** `LearnablePositionEncoding(max_len, d_model)`\n\n**Forward:** `forward(x) -> Tensor`\n- `x` — token embeddings of shape `(B, T, d_model)`\n\n**Returns:** `x + position_embedding[0:T]` with the same shape\n\n**Constraints:**\n- Use `nn.Embedding(max_len, d_model)`\n- Position ids should start at 0 for each sequence",
    "description_zh": "实现可学习绝对位置编码。\n\n不同于正弦位置编码，可学习 PE 为每个位置存储一个可训练向量，并加到 token embedding 上。\n\n**签名:** `LearnablePositionEncoding(max_len, d_model)`\n\n**前向传播:** `forward(x) -> Tensor`\n- `x` — token embedding，形状 `(B, T, d_model)`\n\n**返回:** 形状相同的 `x + position_embedding[0:T]`\n\n**约束:**\n- 使用 `nn.Embedding(max_len, d_model)`\n- 每个序列的位置 id 从 0 开始",
    "function_name": "LearnablePositionEncoding",
    "hint": "1. Create `self.position_embedding = nn.Embedding(max_len, d_model)`\n2. For `x` with length T, make positions `torch.arange(T, device=x.device)`\n3. Add the position vectors with broadcasting over the batch dimension",
    "hint_zh": "1. 创建 `self.position_embedding = nn.Embedding(max_len, d_model)`\n2. 对长度 T 的 `x`，构造 `torch.arange(T, device=x.device)`\n3. 通过 batch 维广播加上位置向量",
    "tests": [
        {
            "name": "Is nn.Module and keeps shape",
            "code": """
import torch, torch.nn as nn
pe = {fn}(max_len=16, d_model=8)
assert isinstance(pe, nn.Module)
x = torch.zeros(2, 5, 8)
out = pe.forward(x)
assert out.shape == x.shape, f'Shape mismatch: {out.shape}'
""",
        },
        {
            "name": "Uses nn.Embedding with expected shape",
            "code": """
import torch, torch.nn as nn
pe = {fn}(max_len=10, d_model=6)
assert isinstance(pe.position_embedding, nn.Embedding)
assert pe.position_embedding.weight.shape == (10, 6)
""",
        },
        {
            "name": "Adds positions in order",
            "code": """
import torch
pe = {fn}(max_len=6, d_model=3)
with torch.no_grad():
    pe.position_embedding.weight.copy_(torch.arange(18, dtype=torch.float32).view(6, 3))
x = torch.ones(2, 4, 3)
out = pe.forward(x)
expected = x + torch.arange(12, dtype=torch.float32).view(4, 3).unsqueeze(0)
assert torch.allclose(out, expected), f'{out} vs {expected}'
""",
        },
        {
            "name": "Gradients reach position embeddings",
            "code": """
import torch
pe = {fn}(max_len=8, d_model=4)
x = torch.randn(2, 5, 4, requires_grad=True)
pe.forward(x).sum().backward()
assert x.grad is not None
assert pe.position_embedding.weight.grad is not None
assert pe.position_embedding.weight.grad[:5].abs().sum() > 0
assert pe.position_embedding.weight.grad[5:].abs().sum() == 0, 'Unused positions should have zero gradient'
""",
        },
    ],
    "solution": '''class LearnablePositionEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.position_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device)
        pos = self.position_embedding(positions)
        return x + pos.unsqueeze(0)''',
    "demo": """pe = LearnablePositionEncoding(max_len=16, d_model=8)
x = torch.zeros(2, 5, 8)
print(pe.forward(x).shape)""",
}
