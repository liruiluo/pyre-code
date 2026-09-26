"""Gemma-2 logit soft-capping task."""

TASK = {
    "title": "Logit Soft-Capping (Gemma-2)",
    "title_zh": "Logit 软顶（Gemma-2）",
    "difficulty": "Easy",
    "description_en": "Cap logits smoothly — the Gemma-2 trick that replaced hard clipping.\n\nAttention logits and final logits can explode during training; hard clamping kills gradients at the cap. Gemma-2 instead soft-caps at `cap = 30.0`: `out = cap * tanh(x / cap)`. The map is identity-near-zero, smoothly saturating toward `+-cap`, with a non-zero slope everywhere — gradients survive even at the extremes.\n\n**Signature:** `logit_softcap(logits, cap) -> Tensor`\n\n**Parameters:**\n- `logits` — tensor of any shape\n- `cap` — positive soft-cap value\n\n**Returns:** `cap * tanh(logits / cap)`, same shape\n\n**Constraints:**\n- Output magnitude stays strictly below `cap`\n- The slope at zero is exactly 1 (identity near the origin)\n- Gradients never vanish entirely: `d(out)/dx = sech^2(x / cap) > 0`",
    "description_zh": "平滑地封顶 logits——Gemma-2 取代硬截断的技巧。\n\n注意力 logits 和最终 logits 在训练中可能爆炸；硬截断会把顶端的梯度杀光。Gemma-2 改用 `cap = 30.0` 的软顶：`out = cap * tanh(x / cap)`。该映射在零附近是恒等，平滑饱和到 `+-cap`，处处斜率非零——即使在极端处梯度也存活。\n\n**签名:** `logit_softcap(logits, cap) -> Tensor`\n\n**参数:**\n- `logits` — 任意形状张量\n- `cap` — 正的软顶值\n\n**返回:** `cap * tanh(logits / cap)`，形状不变\n\n**约束:**\n- 输出幅值严格小于 `cap`\n- 零点斜率恰为 1（原点附近近似恒等）\n- 梯度从不完全消失：`d(out)/dx = sech^2(x / cap) > 0`",
    "function_name": "logit_softcap",
    "hint": "One line: `cap * torch.tanh(logits / cap)`.",
    "hint_zh": "一行：`cap * torch.tanh(logits / cap)`。",
    "tests": [
        {
            "name": "Near-identity for small logits",
            "code": """
import torch
x = torch.tensor([0.001, -0.002, 0.01, -0.05])
out = {fn}(x, 30.0)
assert torch.allclose(out, x, atol=1e-4), f'Small logits must pass through, got {out}'
""",
        },
        {
            "name": "Saturates below the cap",
            "code": """
import torch
x = torch.tensor([0.5, -3.0, 240.0, -240.0])
out = {fn}(x, 30.0)
assert (out.abs() < 30.0).all(), 'Output must stay strictly below the cap'
assert out[2] > 29.0 and out[3] < -29.0, 'Saturated logits must be near the cap'
""",
        },
        {
            "name": "Slope at zero is one",
            "code": """
import torch
x = torch.tensor([0.0], requires_grad=True)
out = {fn}(x, 50.0)
out.backward()
assert abs(x.grad.item() - 1.0) < 1e-6, f'd/dx at 0 must be 1, got {x.grad.item()}'
""",
        },
        {
            "name": "Gradient survives at saturation",
            "code": """
import torch
x = torch.tensor([240.0], requires_grad=True)  # ratio 8: tanh still below 1 in float32
out = {fn}(x, 30.0)
out.backward()
assert x.grad.item() > 0, 'Gradient must survive deep into saturation'
""",
        },
        {
            "name": "Odd function on any shape",
            "code": """
import torch
torch.manual_seed(0)
x = torch.randn(3, 4, 5)
a = {fn}(x, 10.0)
b = {fn}(-x, 10.0)
assert a.shape == x.shape
assert torch.allclose(a, -b, atol=1e-6), 'Soft-capping must be odd-symmetric'
""",
        },
        {
            "name": "Cap parameter controls the ceiling",
            "code": """
import torch
assert {fn}(torch.tensor([40.0]), 5.0).item() < 5.0
assert {fn}(torch.tensor([400.0]), 50.0).item() > 49.0
""",
        },
    ],
    "solution": '''import torch


def logit_softcap(logits, cap):
    return cap * torch.tanh(logits / cap)''',
    "demo": """import torch
x = torch.tensor([0.5, 5.0, 30.0, 60.0, 200.0])
print('in :', x.tolist())
print('out:', logit_softcap(x, 30.0).tolist())""",
}
