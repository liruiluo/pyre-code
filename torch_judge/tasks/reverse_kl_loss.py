"""Reverse KL distillation loss task."""

TASK = {
    "title": "Reverse KL Loss (Mode-Seeking)",
    "title_zh": "反向 KL 损失（模式寻求）",
    "difficulty": "Medium",
    "description_en": "Implement the reverse KL divergence used by on-policy distillation and RLHF objectives: `D_KL(student || teacher)`.\n\nDirection matters. Forward KL `D_KL(teacher || student)` is mass-covering — the student spreads probability over everything the teacher likes. Reverse KL `D_KL(student || teacher)` is mode-seeking — wherever the student puts mass, the teacher must agree, so the student concentrates on the teacher's best modes. Reverse KL is the divergence you get when the student's own samples drive the loss, which is exactly what on-policy distillation (GKD-style) optimizes.\n\n**Signature:** `reverse_kl_loss(student_logits, teacher_logits, temperature=1.0) -> Tensor`\n\n**Parameters:**\n- `student_logits`, `teacher_logits` — shape `(B, V)`\n- `temperature` — softening temperature `T`\n\n**Returns:** scalar loss = `T^2 * mean over batch of D_KL(student_T || teacher_T)`\n\n**Constraints:**\n- Use stable log-softmax\n- Student first — this is the opposite direction from the classic KD loss",
    "description_zh": "实现在线策略蒸馏与 RLHF 目标使用的反向 KL 散度：`D_KL(student || teacher)`。\n\n方向决定行为。前向 KL `D_KL(teacher || student)` 是覆盖型——学生把概率摊到教师喜欢的一切上。反向 KL `D_KL(student || teacher)` 是模式寻求型——学生放 mass 的地方教师必须同意，于是学生集中在教师最强的模式上。当损失由学生自己的采样驱动时，得到的就是反向 KL——这正是 GKD 式在线策略蒸馏所优化的。\n\n**签名:** `reverse_kl_loss(student_logits, teacher_logits, temperature=1.0) -> Tensor`\n\n**参数:**\n- `student_logits`、`teacher_logits` — 形状 `(B, V)`\n- `temperature` — 软化温度 `T`\n\n**返回:** 标量损失 = `T^2 * batch 平均的 D_KL(student_T || teacher_T)`\n\n**约束:**\n- 使用稳定的 log-softmax\n- 学生在前——与经典 KD 损失方向相反",
    "function_name": "reverse_kl_loss",
    "hint": "`log_p = F.log_softmax(student/T)`, `q = F.softmax(student/T)` — both from the student; then `(q * (log_p - log_t)).sum(-1).mean() * T**2` where `log_t = F.log_softmax(teacher/T)`.",
    "hint_zh": "`log_p = F.log_softmax(student/T)`、`q = F.softmax(student/T)`——都来自学生；然后 `(q * (log_p - log_t)).sum(-1).mean() * T**2`，其中 `log_t = F.log_softmax(teacher/T)`。",
    "tests": [
        {
            "name": "Zero when distributions match",
            "code": """
import torch
logits = torch.randn(4, 9)
out = {fn}(logits, logits, temperature=2.0)
assert torch.allclose(out, torch.tensor(0.0), atol=1e-6), f'Identical logits must give 0, got {out}'
""",
        },
        {
            "name": "Matches the manual reverse KL formula",
            "code": """
import torch
torch.manual_seed(0)
s = torch.randn(3, 6)
t = torch.randn(3, 6)
T = 3.0
log_s = torch.log_softmax(s / T, dim=-1)
p = torch.softmax(s / T, dim=-1)
log_t = torch.log_softmax(t / T, dim=-1)
expected = (p * (log_s - log_t)).sum(-1).mean() * T**2
out = {fn}(s, t, temperature=T)
assert torch.allclose(out, expected, atol=1e-6), f'{out.item()} vs {expected.item()}'
""",
        },
        {
            "name": "Direction differs from forward KL",
            "code": """
import torch
import torch.nn.functional as F
s = torch.tensor([[3.0, 0.0, -1.0]])
t = torch.tensor([[0.0, 2.0, 0.0]])
def classic_kd(teacher, student):  # D_KL(teacher || student), the classic KD direction
    log_q = F.log_softmax(teacher, dim=-1)
    q = log_q.exp()
    log_p = F.log_softmax(student, dim=-1)
    return (q * (log_q - log_p)).sum()
rev = {fn}(s, t, temperature=1.0)   # D_KL(student || teacher)
fwd = classic_kd(t, s)             # D_KL(teacher || student)
assert not torch.allclose(rev, fwd, atol=1e-3), 'Reverse and forward KL must differ on this pair'
""",
        },
        {
            "name": "Student mass on zero-teacher positions is punished hard",
            "code": """
import torch
t = torch.tensor([[5.0, 4.0, 0.0]])
s_confident_wrong = torch.tensor([[2.0, 0.0, 5.0]])   # student loves the teacher's dead position
s_following = torch.tensor([[4.6, 3.8, 0.0]])
wrong = {fn}(s_confident_wrong, t, temperature=1.0)
following = {fn}(s_following, t, temperature=1.0)
assert wrong > following, 'Reverse KL punishes student mass where the teacher has none'
assert following.item() < 0.1, f'Close-following student should be cheap, got {following.item()}'
""",
        },
        {
            "name": "Non-negative and gradient flows to the student",
            "code": """
import torch
torch.manual_seed(1)
s = torch.randn(3, 7, requires_grad=True)
t = torch.randn(3, 7)
out = {fn}(s, t)
assert out.item() >= -1e-6, 'KL must be non-negative'
out.backward()
assert s.grad is not None and s.grad.abs().sum() > 0
""",
        },
    ],
    "solution": '''import torch
import torch.nn.functional as F


def reverse_kl_loss(student_logits, teacher_logits, temperature=1.0):
    log_p = F.log_softmax(student_logits / temperature, dim=-1)
    p = log_p.exp()
    log_t = F.log_softmax(teacher_logits / temperature, dim=-1)
    kl = (p * (log_p - log_t)).sum(dim=-1).mean()
    return kl * (temperature ** 2)''',
    "demo": """import torch
torch.manual_seed(0)
teacher = torch.randn(2, 8)
student = teacher + 0.4 * torch.randn(2, 8)
print('reverse KL:', reverse_kl_loss(student, teacher, temperature=2.0).item())""",
}
