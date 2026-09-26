"""Multi-turn agent loss masking task."""

TASK = {
    "title": "Build a Multi-Turn Agent Loss Mask",
    "title_zh": "构建多轮 Agent 损失掩码",
    "difficulty": "Medium",
    "description_en": "Build the token-level training mask for a multi-turn agent trajectory.\n\nWhen SFT-ing an agent model on conversations that mix `system`, `user`, `assistant` and `tool` segments, you only train on the tokens the model itself should emit. The trajectory arrives as segments with token counts; the loss mask marks which of the concatenated tokens contribute to the loss.\n\n**Signature:** `build_agent_loss_mask(segments, trainable_roles=None) -> Tensor`\n\n**Parameters:**\n- `segments` — list of `{'role': str, 'tokens': int}` in trajectory order\n- `trainable_roles` — roles to train on; defaults to `['assistant']`\n\n**Returns:** boolean tensor of length `sum(tokens)`, `True` where a token belongs to a trainable segment\n\n**Constraints:**\n- The mask must align exactly: segment boundaries concatenate without gaps or overlap\n- `tokens` may be zero for a segment — it contributes nothing\n- An empty segment list returns an empty boolean tensor",
    "description_zh": "为多轮 agent 轨迹构建 token 级训练掩码。\n\n在混合 `system`、`user`、`assistant`、`tool` 段的对话上对 agent 模型做 SFT 时，只训练模型自己应该产出的 token。轨迹以带 token 数的段的形式到达；损失掩码标记拼接后哪些 token 计入损失。\n\n**签名:** `build_agent_loss_mask(segments, trainable_roles=None) -> Tensor`\n\n**参数:**\n- `segments` — 按轨迹顺序的 `{'role': str, 'tokens': int}` 列表\n- `trainable_roles` — 参与训练的角色；默认 `['assistant']`\n\n**返回:** 长度为 `sum(tokens)` 的布尔张量，属于可训练段的 token 为 `True`\n\n**约束:**\n- 掩码必须精确对齐：段边界拼接无缝隙、无重叠\n- 某段 `tokens` 可以为 0——它不贡献任何 token\n- 空段列表返回空布尔张量",
    "function_name": "build_agent_loss_mask",
    "hint": "Emit `torch.ones(tokens, dtype=torch.bool)` for trainable segments and `torch.zeros(...)` otherwise, then `torch.cat`.",
    "hint_zh": "可训练段发 `torch.ones(tokens, dtype=torch.bool)`，其余发 `torch.zeros(...)`，最后 `torch.cat`。",
    "tests": [
        {
            "name": "Only assistant tokens are trainable by default",
            "code": """
import torch
segments = [
    {'role': 'system', 'tokens': 5},
    {'role': 'user', 'tokens': 4},
    {'role': 'assistant', 'tokens': 6},
    {'role': 'tool', 'tokens': 3},
    {'role': 'assistant', 'tokens': 2},
]
mask = {fn}(segments)
assert isinstance(mask, torch.Tensor)
assert mask.dtype == torch.bool, f'dtype should be bool, got {mask.dtype}'
assert mask.shape == (20,), f'Expected shape (20,), got {mask.shape}'
expected = torch.tensor([0]*9 + [1]*6 + [0]*3 + [1]*2, dtype=torch.bool)
assert torch.equal(mask, expected), 'Mask segments misaligned'
""",
        },
        {
            "name": "Custom trainable roles",
            "code": """
import torch
segments = [
    {'role': 'user', 'tokens': 2},
    {'role': 'assistant', 'tokens': 3},
    {'role': 'tool', 'tokens': 4},
]
mask = {fn}(segments, trainable_roles=['assistant', 'tool'])
expected = torch.tensor([0]*2 + [1]*7, dtype=torch.bool)
assert torch.equal(mask, expected), 'assistant+tool roles must both be trainable'
""",
        },
        {
            "name": "Zero-length segments contribute nothing",
            "code": """
import torch
segments = [
    {'role': 'system', 'tokens': 0},
    {'role': 'assistant', 'tokens': 3},
    {'role': 'tool', 'tokens': 0},
]
mask = {fn}(segments)
assert mask.shape == (3,), f'Expected shape (3,), got {mask.shape}'
assert mask.all(), 'The 3 assistant tokens must be trainable'
""",
        },
        {
            "name": "Empty trajectory",
            "code": """
import torch
mask = {fn}([])
assert isinstance(mask, torch.Tensor) and mask.dtype == torch.bool
assert mask.numel() == 0, f'Expected empty tensor, got shape {mask.shape}'
""",
        },
        {
            "name": "Boundary alignment is exact",
            "code": """
import torch
torch.manual_seed(0)
segments = [
    {'role': 'system', 'tokens': 7},
    {'role': 'user', 'tokens': 11},
    {'role': 'assistant', 'tokens': 9},
    {'role': 'assistant', 'tokens': 1},
    {'role': 'tool', 'tokens': 5},
]
mask = {fn}(segments)
assert int(mask.sum().item()) == 10, 'Two assistant segments: 9 + 1 trainable tokens'
assert int(mask[:7].sum().item()) == 0, 'system'
assert int(mask[7:18].sum().item()) == 0, 'user occupies indices 7..17'
assert int(mask[18:27].sum().item()) == 9, 'first assistant occupies 18..26'
assert int(mask[27:28].sum().item()) == 1 and int(mask[28:].sum().item()) == 0
""",
        },
    ],
    "solution": '''import torch


def build_agent_loss_mask(segments, trainable_roles=None):
    if trainable_roles is None:
        trainable_roles = ["assistant"]
    trainable = set(trainable_roles)
    pieces = []
    for seg in segments:
        n = int(seg["tokens"])
        if seg["role"] in trainable:
            pieces.append(torch.ones(n, dtype=torch.bool))
        else:
            pieces.append(torch.zeros(n, dtype=torch.bool))
    if not pieces:
        return torch.zeros(0, dtype=torch.bool)
    return torch.cat(pieces)''',
    "demo": """segments = [
    {'role': 'system', 'tokens': 5},
    {'role': 'user', 'tokens': 4},
    {'role': 'assistant', 'tokens': 6},
    {'role': 'tool', 'tokens': 3},
]
mask = build_agent_loss_mask(segments)
print(mask.int().tolist())""",
}
