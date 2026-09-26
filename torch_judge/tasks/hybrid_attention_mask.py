"""Hybrid local/global attention mask task (Gemma-2 style)."""

TASK = {
    "title": "Hybrid Local-Global Attention Masks (Gemma-2)",
    "title_zh": "局部-全局混合注意力掩码（Gemma-2）",
    "difficulty": "Hard",
    "description_en": "Build the per-layer attention masks of a hybrid-attention stack — the Gemma-2/Gemma-3 recipe (and the same idea as Qwen3-Next, which replaces 3 of every 4 layers with linear attention).\n\nGlobal self-attention over a long context is quadratic; sliding-window attention only sees the last `window_size` tokens and misses distant context. Gemma-2 alternates them 1:1; Xiaomi MiMo-V2.6 runs a 6:1 stack — 60 sliding-window layers to 10 global ones — information still reaches layer 0 from far away, it just needs `seq_len / window_size` local hops to get there, while every global layer pays full price only once per pair.\n\n**Signature:** `hybrid_attention_mask(seq_len, layer_types, window_size) -> list[Tensor]`\n\n**Parameters:**\n- `seq_len` — sequence length `L`\n- `layer_types` — list like `['local', 'global', 'local', 'global']`, one entry per layer\n- `window_size` — receptive window `W` for local layers, `W >= 1`\n\n**Returns:** one `(L, L)` boolean mask per layer, in order. `True` at `[i, j]` means query `i` may attend to key `j`.\n\n**Constraints:**\n- Both layer types are causal: no query attends to a later position\n- Global layers attend to all earlier positions\n- Local layers attend only to positions `j` with `i - W + 1 <= j <= i` (a window of `W` including self)\n- `window_size >= seq_len` makes local identical to global",
    "description_zh": "构建混合注意力栈的逐层掩码——Gemma-2/Gemma-3 的配方（与 Qwen3-Next 同思路：每 4 层里 3 层换成线性注意力）。\n\n长上下文上的全局自注意力是二次方的；滑窗注意力只看最近 `window_size` 个 token，错过远距上下文。Gemma-2 按 1:1 交替；小米 MiMo-V2.6 用 6:1 配比——60 层滑窗对 10 层全局——信息仍能从远处到达第 0 层，只是需要 `seq_len / window_size` 跳局部接力，而每个全局层只为每对位置付一次全价。\n\n**签名:** `hybrid_attention_mask(seq_len, layer_types, window_size) -> list[Tensor]`\n\n**参数:**\n- `seq_len` — 序列长度 `L`\n- `layer_types` — 如 `['local', 'global', 'local', 'global']`，每层一项\n- `window_size` — 局部层的感受窗 `W`，`W >= 1`\n\n**返回:** 每层一个 `(L, L)` 布尔掩码，按层序。`[i, j]` 为 `True` 表示查询 `i` 可以关注键 `j`。\n\n**约束:**\n- 两类层都是因果的：任何查询不关注后面的位置\n- 全局层关注所有更早的位置\n- 局部层只关注满足 `i - W + 1 <= j <= i` 的位置（含自身的 `W` 窗口）\n- `window_size >= seq_len` 时局部层与全局层相同",
    "function_name": "hybrid_attention_mask",
    "hint": "Causal base: `torch.tril(torch.ones(L, L, dtype=torch.bool))`; the local mask additionally requires `i - j < window_size`, i.e. a band — build it from broadcast index grids `rows` and `cols`.",
    "hint_zh": "因果基底：`torch.tril(torch.ones(L, L, dtype=torch.bool))`；局部掩码额外要求 `i - j < window_size`（条带）——用广播索引网格 `rows`、`cols` 构造。",
    "tests": [
        {
            "name": "One mask per layer, correct shapes",
            "code": """
import torch
masks = {fn}(6, ['local', 'global', 'local', 'local'], 3)
assert isinstance(masks, list) and len(masks) == 4
for m in masks:
    assert m.shape == (6, 6) and m.dtype == torch.bool
""",
        },
        {
            "name": "Global layers are fully causal",
            "code": """
import torch
masks = {fn}(5, ['global'], 2)
expected = torch.tril(torch.ones(5, 5, dtype=torch.bool))
assert torch.equal(masks[0], expected), 'Global mask must be lower-triangular'
""",
        },
        {
            "name": "Local layers see only the window",
            "code": """
import torch
L, W = 8, 3
masks = {fn}(L, ['local'], W)
m = masks[0]
i = torch.arange(L)
for qi in range(L):
    for kj in range(L):
        allowed = kj <= qi and (qi - kj) < W
        assert m[qi, kj].item() == allowed, f'[{qi},{kj}] should be {allowed}'
""",
        },
        {
            "name": "Every layer is causal",
            "code": """
import torch
L, W = 7, 2
masks = {fn}(L, ['local', 'global', 'local', 'global', 'local'], W)
for li, m in enumerate(masks):
    upper = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
    assert not (m & upper).any(), f'Layer {li} attends to the future'
""",
        },
        {
            "name": "Large window makes local equal global",
            "code": """
import torch
L = 6
masks = {fn}(L, ['local', 'global'], 100)
assert torch.equal(masks[0], masks[1]), 'window >= seq_len must reduce local to causal-global'
""",
        },
        {
            "name": "Window of one is strictly diagonal",
            "code": """
import torch
masks = {fn}(5, ['local'], 1)
assert torch.equal(masks[0], torch.eye(5, dtype=torch.bool)), 'W=1 attends to self only'
""",
        },
        {
            "name": "Alternating pattern matches expectations",
            "code": """
import torch
L, W = 10, 4
local, global_m = {fn}(L, ['local', 'global'], W)
# query 8 in the local layer must not see position 3 (distance 5 >= W)
assert not local[8, 3].item() and local[8, 5].item()
assert global_m[8, 3].item(), 'Global layer must see far history'
# far-position reachability: local layer needs hops, global does not
assert local[:, 0].sum().item() == W, 'Only the first W queries see position 0 locally'
assert global_m[:, 0].all(), 'Everyone sees position 0 globally'
""",
        },
    ],
    "solution": '''import torch


def hybrid_attention_mask(seq_len, layer_types, window_size):
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    rows = torch.arange(seq_len).unsqueeze(1)   # query index i
    cols = torch.arange(seq_len).unsqueeze(0)   # key index j
    band = (cols <= rows) & (rows - cols < window_size)
    masks = []
    for layer in layer_types:
        if layer == "local":
            masks.append(band.clone())
        elif layer == "global":
            masks.append(causal.clone())
        else:
            raise ValueError(f"unknown layer type: {layer!r}")
    return masks''',
    "demo": """import torch
masks = hybrid_attention_mask(12, ['local', 'global', 'local', 'global'], 4)
for i, m in enumerate(masks):
    kind = ['local', 'global', 'local', 'global'][i]
    print(f'layer {i} ({kind:6s}) — connections: {int(m.sum())} / {12 * 12}')""",
}
