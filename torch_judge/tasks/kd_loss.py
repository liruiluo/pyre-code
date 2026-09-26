"""Classic knowledge distillation loss task."""

TASK = {
    "title": "Knowledge Distillation Loss",
    "title_zh": "知识蒸馏损失",
    "difficulty": "Easy",
    "description_en": "Implement Hinton-style knowledge distillation: match the student's temperature-softened distribution to the teacher's.\n\nDistillation transfers a teacher model's knowledge into a smaller student. With temperature `T`, both distributions are softened as `softmax(logits / T)`; the loss is the KL divergence between teacher and student, scaled by `T^2` to keep gradient magnitudes comparable to the label loss.\n\n**Signature:** `kd_loss(student_logits, teacher_logits, temperature=2.0) -> Tensor`\n\n**Parameters:**\n- `student_logits`, `teacher_logits` — shape `(B, V)`\n- `temperature` — softening temperature `T > 0`\n\n**Returns:** scalar loss = `T^2 * mean over batch of D_KL(teacher_T || student_T)`\n\n**Constraints:**\n- Use stable log-softmax\n- Teacher first: the KL places the teacher distribution in the first slot",
    "description_zh": "实现 Hinton 式知识蒸馏：让学生的温度软化分布逼近教师。\n\n蒸馏把教师模型的知识迁移进更小的学生。温度 `T` 下两个分布都软化为 `softmax(logits / T)`；损失是教师与学生之间的 KL 散度，乘 `T^2` 保持梯度量级与标签损失可比。\n\n**签名:** `kd_loss(student_logits, teacher_logits, temperature=2.0) -> Tensor`\n\n**参数:**\n- `student_logits`、`teacher_logits` — 形状 `(B, V)`\n- `temperature` — 软化温度 `T > 0`\n\n**返回:** 标量损失 = `T^2 * batch 平均的 D_KL(teacher_T || student_T)`\n\n**约束:**\n- 使用稳定的 log-softmax\n- 教师在前：KL 中教师分布放第一个槽位",
    "function_name": "kd_loss",
    "hint": "`log_p = F.log_softmax(student/T)`, `q = F.softmax(teacher/T)`, then `(q * (log q - log p)).sum(-1).mean() * T**2`. Use `q * (q.clamp_min(...).log() - log_p)` or `F.kl_div(log_p, q, reduction='batchmean') * T**2`.",
    "hint_zh": "`log_p = F.log_softmax(student/T)`，`q = F.softmax(teacher/T)`，然后 `(q * (log q - log p)).sum(-1).mean() * T**2`。可用 `F.kl_div(log_p, q, reduction='batchmean') * T**2`。",
    "tests": [
        {
            "name": "Zero when student matches teacher",
            "code": """
import torch
logits = torch.randn(4, 10)
out = {fn}(logits, logits, temperature=3.0)
assert torch.allclose(out, torch.tensor(0.0), atol=1e-6), f'Identical logits must give 0, got {out}'
""",
        },
        {
            "name": "Matches the manual formula",
            "code": """
import torch
torch.manual_seed(0)
s = torch.randn(3, 7)
t = torch.randn(3, 7)
T = 2.5
log_s = torch.log_softmax(s / T, dim=-1)
q = torch.softmax(t / T, dim=-1)
expected = (q * (q.clamp_min(1e-12).log() - log_s)).sum(dim=-1).mean() * T**2
out = {fn}(s, t, temperature=T)
assert torch.allclose(out, expected, atol=1e-6), f'{out.item():.6f} vs {expected.item():.6f}'
""",
        },
        {
            "name": "Temperature squared scaling",
            "code": """
import torch
torch.manual_seed(1)
s = torch.randn(2, 5)
t = torch.randn(2, 5)
base = {fn}(s, t, temperature=1.0)
high = {fn}(s * 4.0, t * 4.0, temperature=4.0)
assert torch.allclose(high, 16.0 * base, atol=1e-5), 'Scaling logits by T with temperature T keeps distributions and multiplies loss by T^2'
""",
        },
        {
            "name": "Non-negative and gradient flows to student",
            "code": """
import torch
torch.manual_seed(2)
s = torch.randn(3, 6, requires_grad=True)
t = torch.randn(3, 6)
out = {fn}(s, t)
assert out.item() >= -1e-6, 'KL must be non-negative'
out.backward()
assert s.grad is not None and s.grad.abs().sum() > 0, 'Gradient must reach student logits'
""",
        },
        {
            "name": "Direction is teacher-to-student",
            "code": """
import torch
s = torch.tensor([[2.0, 0.0, -1.0]])
t = torch.tensor([[0.0, 1.0, 0.0]])
# D_KL(teacher || student) is large when the student is confident where the teacher is not
big = {fn}(s, t, temperature=1.0)
small = {fn}(t * 1.5 - s * 0.5, t, temperature=1.0)
assert big > small, 'Loss should shrink as student moves toward the teacher'
""",
        },
    ],
    "solution": '''import torch
import torch.nn.functional as F


def kd_loss(student_logits, teacher_logits, temperature=2.0):
    log_p = F.log_softmax(student_logits / temperature, dim=-1)
    q = F.softmax(teacher_logits / temperature, dim=-1)
    kl = F.kl_div(log_p, q, reduction="batchmean")
    return kl * (temperature ** 2)''',
    "demo": """import torch
torch.manual_seed(0)
teacher = torch.randn(2, 8)
student = teacher + 0.3 * torch.randn(2, 8)
print('T=1 :', kd_loss(student, teacher, temperature=1.0).item())
print('T=4 :', kd_loss(student, teacher, temperature=4.0).item())""",
}
