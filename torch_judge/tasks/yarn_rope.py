"""YaRN rope cache task."""

TASK = {
    "title": "YaRN Long-Context RoPE Cache",
    "title_zh": "YaRN 长上下文 RoPE 缓存",
    "difficulty": "Hard",
    "description_en": "Build the rotary embedding cache with YaRN scaling — the long-context recipe behind Qwen2.5-1M-class and DeepSeek-style context extensions.\n\nNaive RoPE interpolation (divide all frequencies by the scale) blurs high-frequency positional detail; pure NTK extrapolation loses coherence at range. YaRN (Yet another RoPE extensioN) interpolates only the low-frequency dimensions and extrapolates the rest, with a linear ramp between the two regimes, plus an attention temperature `attn_factor = 1 + 0.1 * ln(scale)` that compensates the averaged attention drop.\n\nPer dimension `i` of the `head_dim // 2` frequency pairs, with `inv_freq_i = base^(-2i/head_dim)`:\n- correction dim: `corr(n) = head_dim * ln(original_max / (n * 2*pi)) / (2 * ln(base))`\n- `low = max(floor(corr(beta_fast)), 0)`, `high = min(ceil(corr(beta_slow)), head_dim - 1)`; if `low == high`, `high += 1`\n- ramp: `ramp_i = clamp((i - low) / (high - low), 0, 1)`; `mask_i = 1 - ramp_i`\n- `inv_freq_i = interpolation_i * (1 - mask_i) + extrapolation_i * mask_i`, where `interpolation = inv_freq / scale_factor` and `extrapolation = inv_freq`\n- `freqs = outer(positions, inv_freq)`; cos/sin are the concatenated doubled halves, multiplied by `attn_factor`\n\n**Signature:** `yarn_rope_cache(positions, head_dim, base=10000.0, original_max_position=2048, scale_factor=4.0, beta_fast=32.0, beta_slow=1.0) -> (cos, sin)`\n\n**Parameters:**\n- `positions` — 1-D integer/float tensor of sequence positions\n- `head_dim` — even\n\n**Returns:** `(cos, sin)`, each `(T, head_dim)` with the two halves duplicated",
    "description_zh": "构建带 YaRN 缩放的旋转位置编码缓存——Qwen2.5-1M 级、DeepSeek 式上下文扩展背后的长上下文配方。\n\n朴素 RoPE 插值（所有频率除以缩放倍数）会模糊高频位置细节；纯 NTK 外推又会在远处失焦。YaRN（Yet another RoPE extensioN）只对低频维度插值、其余外推，两区之间线性渐变，并用注意力温度 `attn_factor = 1 + 0.1 * ln(scale)` 补偿平均注意力下降。\n\n对 `head_dim // 2` 组频率的第 `i` 维，`inv_freq_i = base^(-2i/head_dim)`：\n- 修正维：`corr(n) = head_dim * ln(original_max / (n * 2*pi)) / (2 * ln(base))`\n- `low = max(floor(corr(beta_fast)), 0)`，`high = min(ceil(corr(beta_slow)), head_dim - 1)`；若 `low == high` 则 `high += 1`\n- 渐变：`ramp_i = clamp((i - low) / (high - low), 0, 1)`；`mask_i = 1 - ramp_i`\n- `inv_freq_i = interpolation_i * (1 - mask_i) + extrapolation_i * mask_i`，其中 `interpolation = inv_freq / scale_factor`，`extrapolation = inv_freq`\n- `freqs = outer(positions, inv_freq)`；cos/sin 为拼接翻倍的两半，再乘 `attn_factor`\n\n**签名:** `yarn_rope_cache(positions, head_dim, base=10000.0, original_max_position=2048, scale_factor=4.0, beta_fast=32.0, beta_slow=1.0) -> (cos, sin)`\n\n**参数:**\n- `positions` — 一维整数/浮点位置张量\n- `head_dim` — 偶数\n\n**返回:** `(cos, sin)`，各 `(T, head_dim)`，两半翻倍拼接",
    "function_name": "yarn_rope_cache",
    "hint": "Implement the correction-dim formulas with math.log, build the ramp with torch.arange clamped, blend the two inv_freq variants, then `torch.outer`; remember the attn_factor on both cos and sin.",
    "hint_zh": "用 math.log 实现修正维公式，torch.arange 加 clamp 构建渐变，混合两种 inv_freq 后 `torch.outer`；别忘了 cos 和 sin 都要乘 attn_factor。",
    "tests": [
        {
            "name": "Scale 1 reduces to plain RoPE",
            "code": """
import torch
positions = torch.arange(0, 128)
cos, sin = {fn}(positions, 16, base=10000.0, original_max_position=2048, scale_factor=1.0)
inv = 10000.0 ** (torch.arange(0, 16, 2).float() / 16)
freqs = torch.outer(positions.float(), inv)
expected_cos = torch.cat((freqs.cos(), freqs.cos()), dim=-1)
expected_sin = torch.cat((freqs.sin(), freqs.sin()), dim=-1)
assert torch.allclose(cos, expected_cos, atol=1e-5), 'scale=1 must be plain RoPE'
assert torch.allclose(sin, expected_sin, atol=1e-5)
""",
        },
        {
            "name": "Output shapes",
            "code": """
import torch
positions = torch.arange(0, 64)
cos, sin = {fn}(positions, 32)
assert cos.shape == (64, 32) and sin.shape == (64, 32), f'Got {cos.shape}, {sin.shape}'
assert not torch.isnan(cos).any() and not torch.isnan(sin).any()
""",
        },
        {
            "name": "Matches the YaRN reference computation",
            "code": """
import math
import torch
def reference(positions, head_dim, base, orig_max, scale, beta_fast=32.0, beta_slow=1.0):
    half = head_dim // 2
    inv = base ** (torch.arange(0, head_dim, 2).float() / head_dim)
    corr = lambda n: head_dim * math.log(orig_max / (n * 2 * math.pi)) / (2 * math.log(base))
    low = max(math.floor(corr(beta_fast)), 0)
    high = min(math.ceil(corr(beta_slow)), head_dim - 1)
    if low == high:
        high += 1
    idx = torch.arange(half, dtype=torch.float32)
    ramp = ((idx - low) / (high - low)).clamp(0, 1)
    mask = 1 - ramp
    inv_freq = (inv / scale) * (1 - mask) + inv * mask
    attn = 1.0 + 0.1 * math.log(scale)
    freqs = torch.outer(positions.float(), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos() * attn, emb.sin() * attn
positions = torch.arange(0, 512)
cos, sin = {fn}(positions, 64, base=10000.0, original_max_position=4096, scale_factor=8.0)
ref_cos, ref_sin = reference(positions, 64, 10000.0, 4096, 8.0)
assert torch.allclose(cos, ref_cos, atol=1e-5), 'cos mismatch vs reference'
assert torch.allclose(sin, ref_sin, atol=1e-5), 'sin mismatch vs reference'
""",
        },
        {
            "name": "Extrapolation and interpolation regimes are correct",
            "code": """
import math
import torch
head_dim, base, orig_max, scale = 32, 10000.0, 2048, 4.0
positions = torch.arange(0, 256)
corr = lambda n: head_dim * math.log(orig_max / (n * 2 * math.pi)) / (2 * math.log(base))
low = max(math.floor(corr(32.0)), 0)
high = min(math.ceil(corr(1.0)), head_dim - 1)
assert low < high
cos, sin = {fn}(positions, head_dim, base=base, original_max_position=orig_max, scale_factor=scale)
attn = 1.0 + 0.1 * math.log(scale)
half = head_dim // 2
inv = base ** (torch.arange(0, head_dim, 2).float() / head_dim)
plain = torch.outer(positions.float(), inv)
cos_first = cos[:, :half]  # the two halves are duplicated; compare the first
# dimensions i <= low stay extrapolated (unscaled), i >= high are interpolated (positions scaled)
assert torch.allclose(cos_first[:, :low + 1], plain[:, :low + 1].cos() * attn, atol=1e-4), 'low-dim indices must extrapolate'
scaled = torch.outer(positions.float() / scale, inv)
assert torch.allclose(cos_first[:, high:], scaled[:, high:].cos() * attn, atol=1e-4), 'high-dim indices must interpolate'
""",
        },
        {
            "name": "Attention factor is applied",
            "code": """
import math
import torch
head_dim, base, orig_max, scale = 16, 10000.0, 2048, 16.0
positions = torch.arange(0, 32)
cos, sin = {fn}(positions, head_dim, base=base, original_max_position=orig_max, scale_factor=scale)
corr = lambda n: head_dim * math.log(orig_max / (n * 2 * math.pi)) / (2 * math.log(base))
low = max(math.floor(corr(32.0)), 0)
attn = 1.0 + 0.1 * math.log(scale)
inv = base ** (torch.arange(0, head_dim, 2).float() / head_dim)
plain = torch.outer(positions.float(), inv)
ratio = (cos[:, 0] / plain[:, 0].cos()).median()
assert torch.allclose(ratio, torch.tensor(attn), atol=1e-3), f'Column 0 carries the attention factor, ratio {ratio}'
""",
        },
    ],
    "solution": '''import math
import torch


def _correction_dim(num_rotations, head_dim, base, max_position):
    return (head_dim * math.log(max_position / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def yarn_rope_cache(positions, head_dim, base=10000.0, original_max_position=2048,
                    scale_factor=4.0, beta_fast=32.0, beta_slow=1.0):
    half = head_dim // 2
    inv_freq = base ** (torch.arange(0, head_dim, 2).float() / head_dim)
    interpolation = inv_freq / scale_factor
    extrapolation = inv_freq

    low = max(math.floor(_correction_dim(beta_fast, head_dim, base, original_max_position)), 0)
    high = min(math.ceil(_correction_dim(beta_slow, head_dim, base, original_max_position)), head_dim - 1)
    if low == high:
        high += 1

    idx = torch.arange(half, dtype=torch.float32)
    ramp = ((idx - low) / (high - low)).clamp(0, 1)
    mask = 1 - ramp
    inv_freq = interpolation * (1 - mask) + extrapolation * mask

    attn_factor = 1.0 + 0.1 * math.log(scale_factor)
    freqs = torch.outer(positions.float(), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos() * attn_factor, emb.sin() * attn_factor''',
    "demo": """import torch
positions = torch.arange(0, 1024, 16)
cos, sin = yarn_rope_cache(positions, 32, original_max_position=4096, scale_factor=8.0)
print('cos shape:', tuple(cos.shape))
print('position 1000, dims 0/4/12:', cos[62, 0].item(), cos[62, 8].item(), cos[62, 24].item())""",
}
