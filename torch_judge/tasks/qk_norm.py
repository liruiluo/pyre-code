"""QK-Norm task (Qwen3-style)."""

TASK = {
    "title": "QK-Norm (Qwen3 Style)",
    "title_zh": "QK-Norm（Qwen3 式）",
    "difficulty": "Medium",
    "description_en": "Apply RMSNorm to the query and key head vectors before RoPE — the QK-Norm detail introduced by ViT-22B and shipped in every Qwen3 model.\n\nDeep stacks can blow up the attention logits; normalizing each query and key head to unit RMS right after the projections (before RoPE) keeps the attention scale stable during training. Qwen3, Chameleon and Seed-OSS all place this QK-Norm with learnable per-channel weights and biases.\n\n**Signature:** `qk_norm(q, k, weight_q, bias_q, weight_k, bias_k) -> (q_out, k_out)`\n\n**Parameters:**\n- `q`, `k` — shape `(B, H, T, D)`: batch, heads, sequence, head dim\n- `weight_q`, `bias_q`, `weight_k`, `bias_k` — shape `(D,)`, affine parameters over the head dim\n\n**Returns:** the normalized `(q_out, k_out)`, same shapes: `x / sqrt(mean(x^2, dim=-1) + eps) * weight + bias` with `eps = 1e-6`\n\n**Constraints:**\n- Normalization is per position and per head, over the head dimension only\n- An all-zero head vector normalizes to zero (the eps guard), so the output is just the bias",
    "description_zh": "在 RoPE 之前对 query/key 的头向量施加 RMSNorm——ViT-22B 引入、Qwen3 全系搭载的 QK-Norm 细节。\n\n深层堆叠会放大注意力 logits；在投影之后（RoPE 之前）把每个 query/key 头归一到单位 RMS，能让训练期注意力尺度保持稳定。Qwen3、Chameleon、Seed-OSS 的 QK-Norm 都带可学习的逐通道权重和偏置。\n\n**签名:** `qk_norm(q, k, weight_q, bias_q, weight_k, bias_k) -> (q_out, k_out)`\n\n**参数:**\n- `q`、`k` — 形状 `(B, H, T, D)`：batch、头数、序列、头维\n- `weight_q`、`bias_q`、`weight_k`、`bias_k` — 形状 `(D,)`，头维上的仿射参数\n\n**返回:** 归一化后的 `(q_out, k_out)`，形状不变：`x / sqrt(mean(x^2, dim=-1) + eps) * weight + bias`，`eps = 1e-6`\n\n**约束:**\n- 归一化按位置按头，只在头维上\n- 全零头向量归一后为零（eps 守卫），输出只剩偏置",
    "function_name": "qk_norm",
    "hint": "RMS over the last dim: `rms = x.pow(2).mean(-1, keepdim=True).sqrt()` — but add eps inside the sqrt; then scale and shift.",
    "hint_zh": "最后一维的 RMS：`rms = x.pow(2).mean(-1, keepdim=True).sqrt()`——eps 加在 sqrt 里；然后缩放和平移。",
    "tests": [
        {
            "name": "Output RMS equals the weight magnitude",
            "code": """
import torch
torch.manual_seed(0)
q = torch.randn(2, 3, 4, 8)
k = torch.randn(2, 3, 4, 8)
w_q = torch.full((8,), 2.5); b_q = torch.zeros(8)
w_k = torch.full((8,), 0.5); b_k = torch.zeros(8)
q_out, k_out = {fn}(q, k, w_q, b_q, w_k, b_k)
rms_q = q_out.pow(2).mean(dim=-1).sqrt()
rms_k = k_out.pow(2).mean(dim=-1).sqrt()
assert torch.allclose(rms_q, torch.full_like(rms_q, 2.5), atol=1e-3), 'Unit-RMS input scaled by |w| keeps output RMS at |w|'
assert torch.allclose(rms_k, torch.full_like(rms_k, 0.5), atol=1e-3)
""",
        },
        {
            "name": "Bias shifts without changing the normalized direction",
            "code": """
import torch
torch.manual_seed(1)
q = torch.randn(1, 2, 3, 6)
k = torch.randn(1, 2, 3, 6)
w = torch.ones(6)
b_zero = torch.zeros(6)
b_shift = torch.tensor([1.0, -2.0, 0.5, 0.0, 3.0, -1.0])
q0, k0 = {fn}(q, k, w, b_zero, w, b_zero)
q1, k1 = {fn}(q, k, w, b_shift, w, b_zero)
assert torch.allclose(q1 - q0, b_shift.expand_as(q1), atol=1e-5), 'Bias must add element-wise'
assert torch.allclose(k1, k0, atol=1e-6), 'Key side untouched when its params do not change'
""",
        },
        {
            "name": "Shapes preserved",
            "code": """
import torch
q = torch.randn(2, 4, 5, 16)
k = torch.randn(2, 4, 5, 16)
w = torch.ones(16); b = torch.zeros(16)
q_out, k_out = {fn}(q, k, w, b, w, b)
assert q_out.shape == q.shape and k_out.shape == k.shape
""",
        },
        {
            "name": "Zero head does not NaN and yields the bias",
            "code": """
import torch
q = torch.zeros(1, 1, 1, 4)
k = torch.zeros(1, 1, 1, 4)
w = torch.tensor([1.0, 2.0, 3.0, 4.0])
b = torch.tensor([0.1, 0.2, 0.3, 0.4])
q_out, k_out = {fn}(q, k, w, b, w, b)
assert not torch.isnan(q_out).any() and not torch.isnan(k_out).any()
assert torch.allclose(q_out, b.view(1, 1, 1, 4), atol=1e-6), 'Zero input normalizes to zero, leaving the bias'
""",
        },
        {
            "name": "Matches the reference formula",
            "code": """
import torch
torch.manual_seed(2)
q = torch.randn(2, 3, 5, 8) * 3.0
k = torch.randn(2, 3, 5, 8) * 0.1
w_q = torch.randn(8); b_q = torch.randn(8)
w_k = torch.randn(8); b_k = torch.randn(8)
def ref(x, w, b, eps=1e-6):
    return x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * w + b
q_out, k_out = {fn}(q, k, w_q, b_q, w_k, b_k)
assert torch.allclose(q_out, ref(q, w_q, b_q), atol=1e-5)
assert torch.allclose(k_out, ref(k, w_k, b_k), atol=1e-5)
""",
        },
        {
            "name": "Gradient flows through q and k",
            "code": """
import torch
q = torch.randn(1, 2, 2, 4, requires_grad=True)
k = torch.randn(1, 2, 2, 4, requires_grad=True)
w = torch.ones(4); b = torch.zeros(4)
q_out, k_out = {fn}(q, k, w, b, w, b)
(q_out.sum() + k_out.sum()).backward()
assert q.grad is not None and q.grad.abs().sum() > 0
assert k.grad is not None and k.grad.abs().sum() > 0
""",
        },
    ],
    "solution": '''import torch


def _rms_norm(x, weight, bias, eps=1e-6):
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x / rms * weight + bias


def qk_norm(q, k, weight_q, bias_q, weight_k, bias_k):
    return _rms_norm(q, weight_q, bias_q), _rms_norm(k, weight_k, bias_k)''',
    "demo": """import torch
torch.manual_seed(0)
q = torch.randn(1, 2, 3, 8) * 5.0   # badly scaled queries
k = torch.randn(1, 2, 3, 8)
w = torch.ones(8); b = torch.zeros(8)
q_out, k_out = qk_norm(q, k, w, b, w, b)
print('before RMS:', q.pow(2).mean(-1).sqrt()[0, 0].tolist())
print('after  RMS:', q_out.pow(2).mean(-1).sqrt()[0, 0].tolist())""",
}
