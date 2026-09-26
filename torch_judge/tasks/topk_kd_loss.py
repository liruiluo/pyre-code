"""Top-k sparse distillation loss task."""

TASK = {
    "title": "Top-k Sparse Distillation Loss",
    "title_zh": "Top-k 稀疏蒸馏损失",
    "difficulty": "Medium",
    "description_en": "Distill only the teacher's top-k logits — the MiniLM recipe for representation distillation at scale.\n\nFull-vocabulary KL wastes capacity on the teacher's long tail. MiniLM keeps only the teacher's k largest logits, masks every other position to minus infinity on both sides, and computes the KL between the renormalized truncated distributions. The student is forced to match the teacher where it is confident.\n\n**Signature:** `topk_kd_loss(student_logits, teacher_logits, k, temperature=1.0) -> Tensor`\n\n**Parameters:**\n- `student_logits`, `teacher_logits` — shape `(B, V)`\n- `k` — number of teacher logits to keep, `1 <= k <= V`\n- `temperature` — softening temperature `T`\n\n**Returns:** scalar loss = `T^2 * mean over batch of KL(teacher_topk_T || student_topk_T)`\n\n**Constraints:**\n- The kept positions come from the teacher's top-k (per row); all other positions are masked to `-inf` on BOTH distributions before the softmax\n- With `k = V` this reduces to the full-vocabulary KD loss",
    "description_zh": "只蒸馏教师 top-k 的 logits——大规模表征蒸馏的 MiniLM 配方。\n\n全词表 KL 把容量浪费在教师的长尾上。MiniLM 只保留教师 k 个最大 logits，其余位置在两侧都置为负无穷，再在重归一化的截断分布之间算 KL。学生被迫在教师自信的地方对齐。\n\n**签名:** `topk_kd_loss(student_logits, teacher_logits, k, temperature=1.0) -> Tensor`\n\n**参数:**\n- `student_logits`、`teacher_logits` — 形状 `(B, V)`\n- `k` — 保留的教师 logits 数，`1 <= k <= V`\n- `temperature` — 软化温度 `T`\n\n**返回:** 标量损失 = `T^2 * batch 平均的 KL(teacher_topk_T || student_topk_T)`\n\n**约束:**\n- 保留位置来自教师每行的 top-k；softmax 前两侧分布的其余位置都置 `-inf`\n- `k = V` 时退化为全词表 KD 损失",
    "function_name": "topk_kd_loss",
    "hint": "`vals, idx = teacher.topk(k, dim=-1)`; build a mask from `torch.full_like(logits, -inf).scatter_(-1, idx, vals)` and apply the same mask to both sides.",
    "hint_zh": "`vals, idx = teacher.topk(k, dim=-1)`；用 `torch.full_like(logits, -inf).scatter_(-1, idx, vals)` 构造掩码后的教师 logits，学生用同样的索引掩码。",
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
    ],
    "solution": '''import torch
import torch.nn.functional as F


def topk_kd_loss(student_logits, teacher_logits, k, temperature=1.0):
    b, v = teacher_logits.shape
    k = min(int(k), v)
    vals, idx = teacher_logits.topk(k, dim=-1)
    s_top = student_logits.gather(-1, idx)
    log_p = F.log_softmax(s_top / temperature, dim=-1)
    q = F.softmax(vals / temperature, dim=-1)
    kl = F.kl_div(log_p, q, reduction="batchmean")
    return kl * (temperature ** 2)''',
    "demo": """import torch
torch.manual_seed(0)
teacher = torch.randn(2, 12)
student = teacher + 0.5 * torch.randn(2, 12)
for k in (1, 3, 12):
    print(f'k={k:2d}:', topk_kd_loss(student, teacher, k).item())""",
}
