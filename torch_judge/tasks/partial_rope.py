"""Partial RoPE task (GLM-style)."""

TASK = {
    "title": "Partial RoPE (GLM)",
    "title_zh": "部分旋转位置编码（GLM）",
    "difficulty": "Medium",
    "description_en": "Rotate only part of each head dimension — the partial RoPE used across the GLM line since ChatGLM2 (25% rotated there, GLM-4.5 rotates 75%).\n\nFull RoPE rotates every dimension of the query/key head. GLM keeps a slice of the head dim rotation-free: those channels stay position-agnostic and act as content registers, which stabilizes long-context behavior. The mechanics: the first `R` dimensions of the head go through the standard rotary pairing, the remaining `D - R` pass through untouched.\n\n**Signature:** `partial_rope(x, cos, sin) -> Tensor`\n\n**Parameters:**\n- `x` — `(B, H, T, D)` query/key tensor\n- `cos`, `sin` — `(T, R)` rotary tables for the rotated channels, in the doubled form `[f, f]` (each half identical, `R` even, `R <= D`)\n\n**Returns:** `(B, H, T, D)`: channels `[0, R)` rotated, channels `[R, D)` passed through\n\n**Constraints:**\n- Rotation pairs dimension `i` with `i + R/2` within the rotated block, exactly like full RoPE\n- With `R == D` the result must equal full RoPE",
    "description_zh": "只旋转头维度的一部分——GLM 系列自 ChatGLM2 起使用的部分旋转 RoPE（ChatGLM2 旋转 25%，GLM-4.5 旋转 75%）。\n\n完整 RoPE 旋转 query/key 头的每一个维度。GLM 保留一段不旋转：这些通道与位置无关，充当内容寄存器，长上下文更稳。机制：头的前 `R` 维走标准旋转配对，其余 `D − R` 维直通。\n\n**签名:** `partial_rope(x, cos, sin) -> Tensor`\n\n**参数:**\n- `x` — `(B, H, T, D)` 的 query/key 张量\n- `cos`、`sin` — `(T, R)` 旋转通道的表，双拼形式 `[f, f]`（两半相同，`R` 偶数，`R <= D`）\n\n**返回:** `(B, H, T, D)`：通道 `[0, R)` 被旋转，通道 `[R, D)` 直通\n\n**约束:**\n- 旋转把旋转块内的维度 `i` 与 `i + R/2` 配对，与完整 RoPE 完全一致\n- `R == D` 时结果必须等于完整 RoPE",
    "function_name": "partial_rope",
    "hint": "Split off the pass-through tail first: `x_rot, x_pass = x[..., :R], x[..., R:]`; take `f_c = cos[..., :R//2]`, `f_s = sin[..., :R//2]` (the doubled table repeats it) and rotate the block by pairing `x_rot[..., :R//2]` with `x_rot[..., R//2:]`.",
    "hint_zh": "先切出直通尾部：`x_rot, x_pass = x[..., :R], x[..., R:]`；旋转块内把 `x_rot[..., :R//2]` 与 `x_rot[..., R//2:]` 配对，`(T, R)` 的表广播到 `(B, H, T, R)`。",
    "tests": [
        {
            "name": "Pass-through channels are untouched",
            "code": """
import torch
torch.manual_seed(0)
B, H, T, D, R = 2, 3, 5, 8, 6
x = torch.randn(B, H, T, D)
cos = torch.randn(T, R)
sin = torch.randn(T, R)
out = {fn}(x, cos, sin)
assert out.shape == x.shape
assert torch.equal(out[..., R:], x[..., R:]), 'Channels [R, D) must pass through unchanged'
""",
        },
        {
            "name": "Rotated channels follow the pairing rule",
            "code": """
import torch
torch.manual_seed(1)
B, H, T, D, R = 1, 2, 4, 6, 4
x = torch.randn(B, H, T, D)
cos = torch.randn(T, R)
sin = torch.randn(T, R)
out = {fn}(x, cos, sin)
x1 = x[..., : R // 2]
x2 = x[..., R // 2 : R]
c = cos[:, : R // 2].unsqueeze(0).unsqueeze(0)
s = sin[:, : R // 2].unsqueeze(0).unsqueeze(0)
expected_rot = torch.cat((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1)
assert torch.allclose(out[..., :R], expected_rot, atol=1e-5), 'Rotated block must follow the RoPE pairing'
""",
        },
        {
            "name": "R == D reduces to full RoPE",
            "code": """
import torch
torch.manual_seed(2)
B, H, T, D = 2, 2, 6, 8
x = torch.randn(B, H, T, D)
half = torch.randn(T, D // 2)
cos = torch.cat((half, half), dim=-1)
sin = torch.cat((half * 0.5, half * 0.5), dim=-1)
out = {fn}(x, cos, sin)
x1, x2 = x[..., : D // 2], x[..., D // 2 :]
c = half.unsqueeze(0).unsqueeze(0)
s = (half * 0.5).unsqueeze(0).unsqueeze(0)
full = torch.cat((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1)
assert torch.allclose(out, full, atol=1e-5), 'Full rotation must match full RoPE'
assert not torch.allclose(out, x, atol=1e-3), 'Something must actually rotate'
""",
        },
        {
            "name": "Rotation preserves per-position norms",
            "code": """
import torch
torch.manual_seed(3)
B, H, T, D, R = 2, 4, 5, 8, 6
x = torch.randn(B, H, T, D)
theta = torch.rand(T, R // 2) * 6.28
ones = torch.ones(T, R // 2)
cos = torch.cat((theta.cos(), theta.cos()), dim=-1)
sin = torch.cat((theta.sin(), theta.sin()), dim=-1)
out = {fn}(x, cos, sin)
assert torch.allclose(out.norm(dim=-1), x.norm(dim=-1), atol=1e-4), 'A rotary pair is an orthogonal 2x2 map'
""",
        },
        {
            "name": "GLM-4.5-style 75 percent split",
            "code": """
import torch
torch.manual_seed(4)
B, H, T, D = 1, 1, 3, 16
R = 12  # GLM-4.5: partial_rotary_factor 0.75
x = torch.randn(B, H, T, D)
half = torch.randn(T, R // 2)
cos = torch.cat((half, half), dim=-1)
sin = torch.cat((half, half), dim=-1)
out = {fn}(x, cos, sin)
assert torch.equal(out[..., R:], x[..., 12:]), 'The unrotated 25% acts as content registers'
assert not torch.allclose(out[..., :R], x[..., :R], atol=1e-4), 'The rotated 75% must change'
""",
        },
        {
            "name": "Gradients flow",
            "code": """
import torch
B, H, T, D, R = 1, 2, 4, 8, 4
x = torch.randn(B, H, T, D, requires_grad=True)
cos = torch.randn(T, R, requires_grad=True)
sin = torch.randn(T, R, requires_grad=True)
out = {fn}(x, cos, sin)
out.pow(2).sum().backward()
assert x.grad is not None and x.grad.abs().sum() > 0
assert cos.grad is not None and cos.grad.abs().sum() > 0
assert sin.grad is not None and sin.grad.abs().sum() > 0
""",
        },
    ],
    "solution": '''import torch


def partial_rope(x, cos, sin):
    B, H, T, D = x.shape
    R = cos.shape[-1]
    half = R // 2
    x_rot = x[..., :R]
    x_pass = x[..., R:]
    x1 = x_rot[..., :half]
    x2 = x_rot[..., half:]
    # the (T, R) tables are in doubled form [f, f]; the first half carries f
    c = cos[..., :half].unsqueeze(0).unsqueeze(0)  # (1, 1, T, half)
    s = sin[..., :half].unsqueeze(0).unsqueeze(0)
    rot = torch.cat((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1)
    return torch.cat((rot, x_pass), dim=-1)''',
    "demo": """import torch
torch.manual_seed(0)
x = torch.randn(1, 2, 4, 8)            # head dim 8
half = torch.randn(4, 3)               # rotate 6 of 8 channels (GLM-4.5 style 75%)
cos = torch.cat((half, half), dim=-1)
sin = torch.cat((half, half), dim=-1)
out = partial_rope(x, cos, sin)
print('rotated channels changed :', not torch.allclose(out[..., :6], x[..., :6]))
print('register channels intact :', torch.equal(out[..., 6:], x[..., 6:]))""",
}
