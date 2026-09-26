"""Newton-Schulz orthogonalization task (Muon / Kimi K2 MuonClip)."""

TASK = {
    "title": "Newton-Schulz Orthogonalization (Muon)",
    "title_zh": "Newton-Schulz 正交化（Muon）",
    "difficulty": "Hard",
    "description_en": "Implement the matrix iteration at the heart of the Muon optimizer — the optimizer Kimi K2 scaled to 1 trillion parameters with the MuonClip recipe (15.5T training tokens, zero loss spikes), beating AdamW on token efficiency.\n\nMuon updates every 2-D weight matrix with a momentum step that is first equalized: large singular values are pushed down, small ones pushed up, all toward the unit spectrum. The equalizer is a quintic Newton-Schulz iteration — matrix multiplications only, no SVD, which is what makes it cheap enough to sit inside an optimizer step. Given the Frobenius-normalized matrix `X = G / (||G||_F + 1e-7)`:\n\n`A = X X^T;  B = b * A + c * (A @ A);  X = a * X + B @ X`  (repeated `steps` times)\n\nwith the standard quintic coefficients `a = 3.4445, b = -4.7750, c = 2.0315`. Five sweeps take a typical matrix's singular-value spread from an order of magnitude down to about 1.6, with every value inside a narrow band around 1.\n\n**Signature:** `newton_schulz(G, steps=5) -> Tensor`\n\n**Parameters:**\n- `G` — 2-D matrix, any shape (tall, wide, or square)\n- `steps` — iteration count\n\n**Returns:** the equalized matrix, same shape and dtype as the input\n\n**Constraints:**\n- Normalize by the Frobenius norm first: `X = G / (G.norm() + 1e-7)`\n- Use exactly the coefficients above; no SVD calls\n- The result is scale-invariant: `newton_schulz(c * G)` equals `newton_schulz(G)` for any `c > 0`",
    "description_zh": "实现 Muon 优化器核心的矩阵迭代——Kimi K2 用 MuonClip 配方把它推到 1 万亿参数（15.5T 训练 token、零 loss spike）的优化器，token 效率压过 AdamW。\n\nMuon 对每个 2-D 权重矩阵的动量步先做谱均衡：大奇异值压下去、小奇异值托起来，全部推向单位谱。均衡器是五次 Newton-Schulz 迭代——只用矩阵乘、不用 SVD，这才便宜到能塞进优化器步。给定 Frobenius 归一后的矩阵 `X = G / (||G||_F + 1e-7)`：\n\n`A = X X^T;  B = b·A + c·(A @ A);  X = a·X + B @ X`（重复 `steps` 次）\n\n标准五次系数 `a = 3.4445, b = −4.7750, c = 2.0315`。五轮把典型矩阵的奇异值谱距从一个数量级压到约 1.6，所有值落在 1 附近的窄带里。\n\n**签名:** `newton_schulz(G, steps=5) -> Tensor`\n\n**参数:**\n- `G` — 任意形状（高、宽、方阵）的 2-D 矩阵\n- `steps` — 迭代次数\n\n**返回:** 谱均衡后的矩阵，形状与 dtype 同输入\n\n**约束:**\n- 先做 Frobenius 归一：`X = G / (G.norm() + 1e-7)`\n- 严格用上述系数；不许调 SVD\n- 结果尺度不变：任意 `c > 0` 有 `newton_schulz(c·G) == newton_schulz(G)`",
    "function_name": "newton_schulz",
    "hint": "Iterate `A = X @ X.T; B = b*A + c*(A @ A); X = a*X + B @ X`. Keep the input dtype; only the initial Frobenius normalization uses `G.norm()`.",
    "hint_zh": "迭代 `A = X @ X.T; B = b*A + c*(A@A); X = a*X + B @ X`。保持输入 dtype；只有初始 Frobenius 归一用 `G.norm()`。",
    "tests": [
        {
            "name": "Matches the reference iteration",
            "code": """
import torch
torch.manual_seed(1)
G = torch.randn(6, 9)
a, b, c = 3.4445, -4.7750, 2.0315
X = G / (G.norm() + 1e-7)
for _ in range(5):
    A = X @ X.T
    B = b * A + c * (A @ A)
    X = a * X + B @ X
out = {fn}(G, steps=5)
assert torch.allclose(out, X, atol=1e-6), 'Must reproduce the quintic Newton-Schulz iteration'
""",
        },
        {
            "name": "Works for tall, wide and square inputs",
            "code": """
import torch
torch.manual_seed(0)
for shape in [(16, 8), (8, 16), (8, 8)]:
    G = torch.randn(*shape)
    out = {fn}(G, steps=5)
    assert out.shape == shape, f'{shape}: shape changed to {out.shape}'
    assert out.dtype == G.dtype
    assert torch.isfinite(out).all(), f'{shape}: non-finite output'
""",
        },
        {
            "name": "Singular values collapse into a band around one",
            "code": """
import torch
torch.manual_seed(2)
G = torch.randn(8, 16)
normalized = G / (G.norm() + 1e-7)
before = torch.linalg.svdvals(normalized)
after = torch.linalg.svdvals({fn}(G, steps=5))
assert (after > 0.5).all() and (after < 1.3).all(), f'Singular values must land in [0.5, 1.3], got {after}'
assert after.max() / after.min() < 2.0, f'Spread must collapse below 2, got {after.max() / after.min()}'
assert after.max() / after.min() < before.max() / before.min(), 'Equalization must strictly shrink the spread'
""",
        },
        {
            "name": "Extra steps stay bounded",
            "code": """
import torch
torch.manual_seed(4)
G = torch.randn(12, 5)
x5 = {fn}(G, steps=5)
x10 = {fn}(G, steps=10)
assert torch.isfinite(x10).all(), 'The iteration must not diverge'
assert (x5 - x10).abs().max() < 0.5, f'More steps should stay near the fixed band, moved {(x5 - x10).abs().max()}'
sv10 = torch.linalg.svdvals(x10)
assert sv10.max() / sv10.min() < 2.5, 'Ten steps keep the equalized spectrum'
""",
        },
        {
            "name": "Scale invariance from the Frobenius normalization",
            "code": """
import torch
torch.manual_seed(5)
G = torch.randn(6, 6)
out1 = {fn}(G, steps=5)
out100 = {fn}(G * 100.0, steps=5)
assert torch.allclose(out1, out100, atol=1e-4), 'Scaling the input must not change the output'
""",
        },
        {
            "name": "Gradient flows through the iteration",
            "code": """
import torch
torch.manual_seed(3)
G = torch.randn(7, 4, requires_grad=True)
out = {fn}(G, steps=3)
out.sum().backward()
assert G.grad is not None and G.grad.abs().sum() > 0, 'Muon needs gradients through the orthogonalizer'
""",
        },
    ],
    "solution": '''import torch


def newton_schulz(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X''',
    "demo": """import torch
torch.manual_seed(0)
G = torch.randn(8, 16) * 10.0
before = torch.linalg.svdvals(G / G.norm())
after = torch.linalg.svdvals(newton_schulz(G, steps=5))
print(f'spread before: {before.max() / before.min():.1f}')
print(f'spread after : {after.max() / after.min():.2f}  (band [{after.min():.3f}, {after.max():.3f}])')""",
}
