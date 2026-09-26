"""On-policy self-distillation (OPSD) task — two views of one model."""

TASK = {
    "title": "On-Policy Self-Distillation (OPSD)",
    "title_zh": "在线策略自蒸馏（OPSD）",
    "difficulty": "Hard",
    "description_en": "One model plays both teacher and student — the idea behind On-Policy Self-Distillation, one of the loudest post-training recipes of 2026: no external teacher, no hand-crafted reward, and a dense signal far cheaper than GRPO-style sampling.\n\nThe trick is conditioning. Run the SAME model twice over one trajectory: the **privileged view** reads the question plus a reference answer or hint (a rationale, a gold trace), while the **deployable view** reads the question only. Train the deployable view, on its own freshly sampled tokens, toward the privileged view's distribution — a stop-gradient pull. At deployment the hint is gone, and whatever the pull taught stays in the weights. The family has already spread from LLM reasoning post-training to autoregressive video models (OPSD-V).\n\nThe mechanical core this problem drills is alignment: the two views see different contexts, so the shared rollout sits at **different token indices** in each. You must line them up before comparing distributions — and only the rollout tokens enter the loss; the hint positions of the privileged view are context, not targets.\n\n**Signature:** `opsd_loss(student_logits, privileged_logits, rollout_offsets, generated_mask, temperature=1.0) -> Tensor`\n\n**Parameters:**\n- `student_logits` — `(B, T, V)` logits of the deployable view over the rollout\n- `privileged_logits` — `(B, T', V)` logits of the privileged view over `[hint, rollout]`, `T' >= max(offset) + T`\n- `rollout_offsets` — `(B,)` longs; the rollout of batch item `b` starts at `rollout_offsets[b]` in the privileged view\n- `generated_mask` — boolean `(B, T)`, `True` where the token was sampled by the student\n- `temperature` — softening temperature\n\n**Returns:** scalar = `T^2 * mean over masked positions of D_KL(student || privileged)`; exactly `0.0` when no position is masked\n\n**Constraints:**\n- Position `t` of the student view aligns with `rollout_offsets[b] + t` of the privileged view\n- The privileged side is detached: gradients flow only to the student view\n- Privileged logits at hint positions (and any positions beyond the rollout) must not influence the loss",
    "description_zh": "同一个模型既当教师又当学生——这是在线策略自蒸馏（On-Policy Self-Distillation）背后的思想，2026 年后训练最热的配方之一：不需要外部教师、不需要人工奖励，信号稠密且远比 GRPO 式采样便宜。\n\n诀窍在条件化。让**同一个模型**在同一条轨迹上跑两遍：**特权视图**看到题目加参考答案或提示（推理链、金标轨迹），**可部署视图**只看到题目。在可部署视图自己新采样的 token 上，把它拉向特权视图的分布——stop-gradient 的拉力。部署时提示消失，拉力教会的东西留在权重里。这一家族已经从 LLM 推理后训练蔓延到自回归视频模型（OPSD-V）。\n\n本题要练的机械核心是对齐：两个视图看到的上下文不同，共享的 rollout 在各自序列里占据**不同的 token 下标**。比较分布之前必须先对齐——而且只有 rollout token 进损失；特权视图里的提示位置是上下文，不是目标。\n\n**签名:** `opsd_loss(student_logits, privileged_logits, rollout_offsets, generated_mask, temperature=1.0) -> Tensor`\n\n**参数:**\n- `student_logits` — `(B, T, V)` 可部署视图在 rollout 上的 logits\n- `privileged_logits` — `(B, T', V)` 特权视图在 `[提示, rollout]` 上的 logits，`T' >= max(offset) + T`\n- `rollout_offsets` — `(B,)` long；批次 `b` 的 rollout 在特权视图里从 `rollout_offsets[b]` 开始\n- `generated_mask` — 布尔 `(B, T)`，学生采样的 token 位置为 `True`\n- `temperature` — 软化温度\n\n**返回:** 标量 = `T^2 * (掩码位置的 D_KL(student || privileged) 平均)`；无掩码位置时恰为 `0.0`\n\n**约束:**\n- 学生视图的位置 `t` 对齐特权视图的 `rollout_offsets[b] + t`\n- 特权侧 detach：梯度只流给学生视图\n- 特权视图在提示位置（以及 rollout 之外的任何位置）的 logits 不得影响损失",
    "function_name": "opsd_loss",
    "hint": "Build the aligned privileged logits with advanced indexing: `idx = rollout_offsets[:, None] + arange(T)` then `privileged_logits.detach()[arange(B)[:, None], idx]`. Then it is the familiar on-policy recipe: per-token KL, mask-select, `.mean()`, multiply by `T^2`. Guard the empty mask.",
    "hint_zh": "用高级索引构造对齐后的特权 logits：`idx = rollout_offsets[:, None] + arange(T)`，然后 `privileged_logits.detach()[arange(B)[:, None], idx]`。剩下就是熟悉的在线策略配方：逐 token KL、掩码选取、`.mean()`、乘 `T^2`。空掩码要守卫。",
    "tests": [
        {
            "name": "Zero when the two views agree on the rollout",
            "code": """
import torch
torch.manual_seed(0)
B, T, V, Tp = 3, 6, 11, 18
priv = torch.randn(B, Tp, V)
off = torch.tensor([4, 0, 10])
stud = torch.zeros(B, T, V)
for b in range(B):
    stud[b] = priv[b, off[b]:off[b] + T]
mask = torch.ones(B, T, dtype=torch.bool)
out = {fn}(stud, priv, off, mask, temperature=2.0)
assert torch.allclose(out, torch.tensor(0.0), atol=1e-5), f'Aligned views must give 0, got {out}'
""",
        },
        {
            "name": "Matches the manual per-sample reference",
            "code": """
import torch
import torch.nn.functional as F
torch.manual_seed(1)
B, T, V, Tp = 3, 5, 8, 17
stud = torch.randn(B, T, V)
priv = torch.randn(B, Tp, V)
off = torch.tensor([2, 6, 11])
mask = torch.tensor([[True, True, False, True, False], [True, False, True, True, True], [True, True, True, False, False]])
temp = 1.5
log_p = F.log_softmax(stud / temp, dim=-1)
p = log_p.exp()
terms = []
for b in range(B):
    log_t = F.log_softmax(priv[b, off[b]:off[b] + T] / temp, dim=-1)
    per_tok = (p[b] * (log_p[b] - log_t)).sum(dim=-1)
    for t in range(T):
        if mask[b, t]:
            terms.append(per_tok[t])
expected = torch.stack(terms).mean() * temp ** 2
out = {fn}(stud, priv, off, mask, temperature=temp)
assert torch.allclose(out, expected, atol=1e-6), f'{out.item()} vs {expected.item()}'
""",
        },
        {
            "name": "Hint positions of the privileged view are invisible",
            "code": """
import torch
torch.manual_seed(2)
B, T, V, Tp = 3, 4, 7, 15
stud = torch.randn(B, T, V)
priv = torch.randn(B, Tp, V)
off = torch.tensor([5, 3, 8])
mask = torch.ones(B, T, dtype=torch.bool)
base = {fn}(stud, priv, off, mask)
priv2 = priv.clone()
for b in range(B):
    priv2[b, :off[b]] += 100.0
    priv2[b, off[b] + T:] -= 100.0
out = {fn}(stud, priv2, off, mask)
assert torch.allclose(base, out, atol=1e-5), 'Context (hint) positions must not leak into the loss'
""",
        },
        {
            "name": "Unmasked student positions are invisible",
            "code": """
import torch
torch.manual_seed(3)
B, T, V, Tp = 2, 6, 9, 14
stud = torch.randn(B, T, V)
priv = torch.randn(B, Tp, V)
off = torch.tensor([4, 7])
mask = torch.tensor([[True, False, True, True, False, True], [False, True, True, False, True, True]])
base = {fn}(stud, priv, off, mask)
stud2 = stud.clone()
stud2[:, 1] += 100.0
stud2[0, 4] -= 100.0
out = {fn}(stud2, priv, off, mask)
assert torch.allclose(base, out, atol=1e-5), 'Prompt/observation positions must not affect the loss'
""",
        },
        {
            "name": "Gradient reaches the student view but not the privileged view",
            "code": """
import torch
torch.manual_seed(4)
B, T, V, Tp = 2, 4, 6, 12
stud = torch.randn(B, T, V, requires_grad=True)
priv = torch.randn(B, Tp, V, requires_grad=True)
off = torch.tensor([3, 6])
mask = torch.ones(B, T, dtype=torch.bool)
out = {fn}(stud, priv, off, mask)
out.backward()
assert stud.grad is not None and stud.grad.abs().sum() > 0, 'Student view must train'
assert priv.grad is None, 'Privileged view must stay frozen (stop-gradient)'
""",
        },
        {
            "name": "Empty mask returns exact zero",
            "code": """
import torch
stud = torch.randn(2, 4, 6)
priv = torch.randn(2, 12, 6)
off = torch.tensor([2, 5])
mask = torch.zeros(2, 4, dtype=torch.bool)
out = {fn}(stud, priv, off, mask)
assert out.item() == 0.0 and out.dim() == 0, f'Expected scalar 0.0, got {out}'
""",
        },
        {
            "name": "Zero offsets reduce to plain on-policy distillation",
            "code": """
import torch
import torch.nn.functional as F
torch.manual_seed(5)
B, T, V = 2, 5, 10
stud = torch.randn(B, T, V)
teacher_view = torch.randn(B, T, V)
off = torch.zeros(B, dtype=torch.long)
mask = torch.tensor([[True, True, True, True, False], [True, True, False, True, True]])
temp = 2.0
out = {fn}(stud, teacher_view, off, mask, temperature=temp)
log_p = F.log_softmax(stud / temp, dim=-1)
p = log_p.exp()
log_t = F.log_softmax(teacher_view / temp, dim=-1)
per_tok = (p * (log_p - log_t)).sum(dim=-1)
expected = per_tok[mask].mean() * temp ** 2
assert torch.allclose(out, expected, atol=1e-6), 'offsets=0 must equal the plain on-policy loss'
""",
        },
    ],
    "solution": '''import torch
import torch.nn.functional as F


def opsd_loss(student_logits, privileged_logits, rollout_offsets, generated_mask, temperature=1.0):
    if not generated_mask.any():
        return torch.zeros((), dtype=student_logits.dtype)
    B, T, _ = student_logits.shape
    log_p = F.log_softmax(student_logits / temperature, dim=-1)
    p = log_p.exp()
    idx = rollout_offsets[:, None] + torch.arange(T, device=student_logits.device)[None, :]
    rows = torch.arange(B, device=student_logits.device)[:, None]
    priv = privileged_logits.detach()[rows, idx]
    log_t = F.log_softmax(priv / temperature, dim=-1)
    per_tok = (p * (log_p - log_t)).sum(dim=-1)
    return per_tok[generated_mask].mean() * (temperature ** 2)''',
    "demo": """import torch
torch.manual_seed(0)
B, T, V = 2, 6, 10
student = torch.randn(B, T, V)                       # question-only view, rollout logits
privileged = torch.randn(B, 14, V)                   # [hint, rollout] view
privileged[:, 4:4 + T] = student + 0.3 * torch.randn(B, T, V)  # views roughly agree
offsets = torch.tensor([4, 4])
mask = torch.ones(B, T, dtype=torch.bool)
for temp in (1.0, 2.0):
    loss = opsd_loss(student, privileged, offsets, mask, temperature=temp)
    print(f'temperature {temp}: {loss.item():.4f}')""",
}
