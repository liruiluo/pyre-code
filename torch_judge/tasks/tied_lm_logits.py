"""Tied LM head task."""

TASK = {
    "title": "Tied Embedding LM Head",
    "title_zh": "共享权重的 LM 头",
    "difficulty": "Easy",
    "description_en": "Reuse the input embedding matrix as the output language-model head — the weight tying used by GPT-2, every Gemma, and the smaller Qwen models (Qwen3-0.6B/1.7B tie; 4B and above untie).\n\nThe input embedding is a `(V, d)` matrix mapping token ids to vectors; the LM head is a `(V, d)` projection back to vocabulary logits. Tying them shares one parameter block: `logits = hidden @ E.T`, optionally scaled (Gemma multiplies embeddings by a normalizer and compensates on the output side). Tying removes `V * d` parameters — a 30% saving at small vocabulary-embedding ratios and a classic regularization.\n\n**Signature:** `tied_lm_logits(hidden, embedding_weight, scale=1.0) -> Tensor`\n\n**Parameters:**\n- `hidden` — final hidden states, shape `(B, T, d)` (or `(T, d)`)\n- `embedding_weight` — the shared embedding matrix `(V, d)`\n- `scale` — output scaling applied to the logits\n\n**Returns:** logits of shape `(..., V)`: `scale * (hidden @ embedding_weight.T)`\n\n**Constraints:**\n- The embedding matrix is used as-is — no copy, no clone\n- Gradients from the LM head reach the embedding weight tensor itself (shared parameter)",
    "description_zh": "把输入嵌入矩阵复用为输出语言模型头——GPT-2、全部 Gemma、以及较小的 Qwen 模型（Qwen3-0.6B/1.7B 共享；4B 起不共享）使用的权重共享。\n\n输入嵌入是 `(V, d)` 矩阵，把 token id 映射成向量；LM 头是 `(V, d)` 投影，把向量映回词表 logits。共享即合用一个参数块：`logits = hidden @ E.T`，可选缩放（Gemma 给嵌入乘归一化因子并在输出侧补偿）。共享省掉 `V * d` 个参数——小模型上能省 30%，还是经典正则。\n\n**签名:** `tied_lm_logits(hidden, embedding_weight, scale=1.0) -> Tensor`\n\n**参数:**\n- `hidden` — 末层隐状态，形状 `(B, T, d)`（或 `(T, d)`）\n- `embedding_weight` — 共享嵌入矩阵 `(V, d)`\n- `scale` — 应用在 logits 上的输出缩放\n\n**返回:** 形状 `(..., V)` 的 logits：`scale * (hidden @ embedding_weight.T)`\n\n**约束:**\n- 嵌入矩阵原样使用——不 copy、不 clone\n- LM 头的梯度必须抵达嵌入权重张量本身（共享参数）",
    "function_name": "tied_lm_logits",
    "hint": "`F.linear(hidden, embedding_weight) * scale` — or `hidden @ embedding_weight.T`.",
    "hint_zh": "`F.linear(hidden, embedding_weight) * scale`——或 `hidden @ embedding_weight.T`。",
    "tests": [
        {
            "name": "Matches a plain linear projection",
            "code": """
import torch
import torch.nn.functional as F
torch.manual_seed(0)
hidden = torch.randn(2, 5, 16)
E = torch.randn(100, 16)
out = {fn}(hidden, E)
expected = F.linear(hidden, E)
assert out.shape == (2, 5, 100), f'Expected (2, 5, 100), got {out.shape}'
assert torch.allclose(out, expected, atol=1e-6)
""",
        },
        {
            "name": "2-D hidden also works",
            "code": """
import torch
hidden = torch.randn(7, 4)
E = torch.randn(9, 4)
out = {fn}(hidden, E)
assert out.shape == (7, 9), f'Expected (7, 9), got {out.shape}'
""",
        },
        {
            "name": "Scale is applied to the logits",
            "code": """
import torch
hidden = torch.randn(3, 8)
E = torch.randn(6, 8)
base = {fn}(hidden, E)
scaled = {fn}(hidden, E, scale=2.5)
assert torch.allclose(scaled, 2.5 * base, atol=1e-6)
""",
        },
        {
            "name": "Gradients reach the shared embedding weight",
            "code": """
import torch
hidden = torch.randn(2, 3, 8)
E = torch.randn(10, 8, requires_grad=True)
out = {fn}(hidden, E)
out.sum().backward()
assert E.grad is not None and E.grad.abs().sum() > 0, 'The LM head must train the embedding weight itself'
""",
        },
        {
            "name": "One row per vocabulary entry",
            "code": """
import torch
hidden = torch.eye(4)          # 4 one-hot states
E = torch.randn(4, 4)
out = {fn}(hidden, E)
assert torch.allclose(out, E.t(), atol=1e-6), 'One-hot state i must surface embedding column i'
""",
        },
    ],
    "solution": '''import torch
import torch.nn.functional as F


def tied_lm_logits(hidden, embedding_weight, scale=1.0):
    return F.linear(hidden, embedding_weight) * scale''',
    "demo": """import torch
torch.manual_seed(0)
E = torch.randn(50, 8)              # shared embedding (V=50, d=8)
hidden = torch.randn(1, 6, 8)       # final layer states
logits = tied_lm_logits(hidden, E, scale=1.0 / 8 ** 0.5)
print('logits shape:', tuple(logits.shape))
print('param blocks: 1 (embedding serves input AND output)')""",
}
