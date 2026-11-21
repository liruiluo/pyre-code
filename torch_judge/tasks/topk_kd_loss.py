"""Top-k sparse distillation loss task."""

TASK = {
    "title": "Top-k Sparse Distillation Loss",
    "title_zh": "Top-k 稀疏蒸馏损失",
    "difficulty": "Medium",
    "description_en": "Practice a deliberately truncated-vocabulary teacher top-k KL. This is NOT MiniLM's representation-distillation method.\n\nHere we keep the teacher's k largest-logit positions and renormalize BOTH teacher and student over only those positions. This ignores student mass outside top-k: in particular k=1 always gives zero loss. It is a pedagogical truncated KL, not a faithful substitute for full-vocabulary KD or top-k-plus-tail approximation.\n\n**Signature:** `topk_kd_loss(student_logits, teacher_logits, k, temperature=1.0) -> Tensor`\n\n**Parameters:**\n- `student_logits`, `teacher_logits` — shape `(B, V)`\n- `k` — number of teacher logits to keep, `1 <= k <= V`\n- `temperature` — softening temperature `T`\n\n**Returns:** scalar loss = `T^2 * mean over batch of KL(teacher_topk_T || student_topk_T)`\n\n**Constraints:**\n- The kept positions come from the teacher's top-k (per row); all other positions are masked to `-inf` on BOTH distributions before the softmax\n- With `k = V` this reduces to the full-vocabulary KD loss\n- `teacher_logits` is a constant: detach it, so `teacher_logits.grad is None`\n- Invalid `k` (outside `1..V`) must raise `ValueError`; never silently clamp it",
    "description_zh": "练习在教师 top-k 词表支持集上计算截断 KL。本题**不是** MiniLM 的表征蒸馏方法。\n\n本题只保留教师最大的 k 个 logits，并让教师与学生都在这 k 个位置重归一化。这会忽略学生在其他词上的概率质量，尤其 k=1 时 Loss 恒为 0。它是教学用的截断 KL，不可直接视为完整词表 KD 或 Top-k+Tail 的替代。\n\n**签名:** `topk_kd_loss(student_logits, teacher_logits, k, temperature=1.0) -> Tensor`\n\n**参数:**\n- `student_logits`、`teacher_logits` — 形状 `(B, V)`\n- `k` — 保留的教师 logits 数，`1 <= k <= V`\n- `temperature` — 软化温度 `T`\n\n**返回:** 标量损失 = `T^2 * batch 平均的 KL(teacher_topk_T || student_topk_T)`\n\n**约束:**\n- 保留位置来自教师每行的 top-k；softmax 前两侧分布的其余位置都置 `-inf`\n- `k = V` 时退化为全词表 KD loss\n- `teacher_logits` 是常量：必须 detach，保证 `teacher_logits.grad is None`\n- `k` 超出 `1..V` 必须抛 `ValueError`；不能静默 clamp",
    "function_name": "topk_kd_loss",
    "hint": "Validate `1 <= k <= V` first (raise `ValueError` otherwise). `values, idx = teacher.detach().topk(k, dim=-1)`, then slice BOTH sides to those k columns — `student_logits.gather(-1, idx)` for the student and `values` for the teacher. Compute the KL over just the k kept columns (`log_softmax(student_topk / T)` vs `softmax(values / T)`); do NOT use `-inf` masks together with `p * log p`, because `0 * -inf` gives NaN.",
    "hint_zh": "先校验 `1 <= k <= V`（否则抛 `ValueError`）。`values, idx = teacher.detach().topk(k, dim=-1)`，然后两侧都按这 k 列切：学生 `student_logits.gather(-1, idx)`、教师直接用 `values`。只在这 k 列上算 KL（`log_softmax(student_topk / T)` 对 `softmax(values / T)`）；不要用 `-inf` 掩码再乘 `p * log p`，`0 * -inf` 会得到 NaN。",
    "tests": [
        {
            "name": "Reduces to full KD when k equals vocabulary",
            "code": """
import torch
torch.manual_seed(0)
s = torch.randn(3, 8)
t = torch.randn(3, 8)
full = {fn}(s, t, k=8, temperature=2.0)
log_s = torch.log_softmax(s / 2.0, dim=-1)
q = torch.softmax(t / 2.0, dim=-1)
expected = (q * (q.clamp_min(1e-12).log() - log_s)).sum(-1).mean() * 4.0
assert torch.allclose(full, expected, atol=1e-6), f'k=V must equal full KD: {full.item()} vs {expected.item()}'
""",
        },
        {
            "name": "k=1 makes the loss exactly zero",
            "code": """
import torch
s = torch.randn(4, 10)
t = torch.randn(4, 10)
out = {fn}(s, t, k=1)
assert torch.allclose(out, torch.tensor(0.0), atol=1e-6), f'Both sides become the same one-hot, got {out}'
""",
        },
        {
            "name": "Non-topk student logits are ignored",
            "code": """
import torch
torch.manual_seed(1)
s = torch.randn(2, 10)
t = torch.randn(2, 10)
base = {fn}(s, t, k=3)
perturbed = s.clone()
vals, idx = t.topk(3, dim=-1)
mask = torch.ones_like(t, dtype=torch.bool)
mask.scatter_(-1, idx, False)
perturbed[mask] += 100.0
out = {fn}(perturbed, t, k=3)
assert torch.allclose(base, out, atol=1e-5), 'Off-topk student logits must not affect the loss'
""",
        },
        {
            "name": "Matches the masked reference computation",
            "code": """
import torch
torch.manual_seed(2)
s = torch.randn(3, 9)
t = torch.randn(3, 9)
k, T = 4, 2.0
vals, idx = t.topk(k, dim=-1)
s_top = s.gather(-1, idx)
q = torch.softmax(vals / T, dim=-1)
log_p = torch.log_softmax(s_top / T, dim=-1)
expected = (q * (q.clamp_min(1e-12).log() - log_p)).sum(-1).mean() * T**2
out = {fn}(s, t, k, temperature=T)
assert torch.allclose(out, expected, atol=1e-5), f'{out.item()} vs {expected.item()}'
""",
        },
        {
            "name": "Teacher tail is dropped, only the top-k shape matters",
            "code": """
import torch
t = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]])
s_aligned = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]])
s_mass_elsewhere = torch.tensor([[-1.0, -2.0, 5.0, 4.0, 3.0]])
s_swapped = torch.tensor([[4.0, 5.0, 3.0, 2.0, 1.0]])
out_aligned = {fn}(s_aligned, t, k=2)
out_mass = {fn}(s_mass_elsewhere, t, k=2)
out_swapped = {fn}(s_swapped, t, k=2)
assert out_aligned.item() < 1e-6, 'Matching teacher top-2 should be near zero'
assert out_mass.item() < 1e-6, f'Student mass outside the top-k support is invisible, got {out_mass.item()}'
assert out_swapped.item() > 0.3, f'Swapped top-2 ordering must be penalized, got {out_swapped.item()}'
""",
        },
        {
            "name": "Teacher gradients are stopped; invalid k is rejected",
            "code": """
import torch
student = torch.randn(2, 8, requires_grad=True)
teacher = torch.randn(2, 8, requires_grad=True)
{fn}(student, teacher, 3).backward()
assert student.grad is not None and teacher.grad is None
for k in (0, 9):
    try:
        {fn}(student.detach(), teacher.detach(), k)
    except ValueError:
        pass
    else:
        raise AssertionError(f"invalid k={k} must raise")
""",
        },
    ],
    "solution": '''import torch
import torch.nn.functional as F

def topk_kd_loss(student_logits, teacher_logits, k, temperature=1.0):
    if k < 1 or k > teacher_logits.shape[-1]:
        raise ValueError("k must satisfy 1 <= k <= vocabulary size")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    # Teacher selects support, but is a fixed distribution for distillation.
    values, indices = teacher_logits.detach().topk(k, dim=-1)
    student_topk = student_logits.gather(-1, indices)
    log_student = F.log_softmax(student_topk / temperature, dim=-1)
    teacher_probs = F.softmax(values / temperature, dim=-1)
    return F.kl_div(log_student, teacher_probs, reduction="batchmean") * temperature ** 2''',
    "demo": """import torch
torch.manual_seed(0)
teacher = torch.randn(2, 12)
student = teacher + 0.5 * torch.randn(2, 12)
for k in (1, 3, 12):
    print(f'k={k:2d}:', topk_kd_loss(student, teacher, k).item())""",
}
