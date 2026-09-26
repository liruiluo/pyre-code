"""Hidden-state distillation task."""

TASK = {
    "title": "Hidden-State Distillation (FitNets)",
    "title_zh": "隐状态蒸馏（FitNets）",
    "difficulty": "Medium",
    "description_en": "Distill intermediate representations — the FitNets / TinyBERT recipe.\n\nLogit distillation only supervises the output layer. Matching teacher and student hidden states layer by layer transfers richer structural knowledge: the student's `d_s`-dimensional states are projected by a learnable `nn.Linear(d_s, d_t)` onto the teacher's `d_t`-dimensional space, and the MSE is minimized over every token position.\n\n**Signature:** `hidden_distillation(student_hidden, teacher_hidden, projection) -> Tensor`\n\n**Parameters:**\n- `student_hidden` — shape `(B, T, d_s)`\n- `teacher_hidden` — shape `(B, T, d_t)`\n- `projection` — `nn.Linear(d_s, d_t)` mapping student to teacher space\n\n**Returns:** scalar MSE between `projection(student_hidden)` and `teacher_hidden`, averaged over all elements\n\n**Constraints:**\n- The teacher is detached inside: gradients flow only through the student states and the projection weights\n- The loss is the plain element-wise mean, no reduction over the token axis only",
    "description_zh": "蒸馏中间表征——FitNets / TinyBERT 配方。\n\nlogit 蒸馏只监督输出层。逐层对齐教师与学生的隐状态能迁移更丰富的结构知识：学生的 `d_s` 维状态经可学习 `nn.Linear(d_s, d_t)` 投到教师的 `d_t` 维空间，再对所有 token 位置最小化 MSE。\n\n**签名:** `hidden_distillation(student_hidden, teacher_hidden, projection) -> Tensor`\n\n**参数:**\n- `student_hidden` — 形状 `(B, T, d_s)`\n- `teacher_hidden` — 形状 `(B, T, d_t)`\n- `projection` — 把学生映射到教师空间的 `nn.Linear(d_s, d_t)`\n\n**返回:** `projection(student_hidden)` 与 `teacher_hidden` 的标量 MSE，对所有元素平均\n\n**约束:**\n- 教师在函数内部被 detach：梯度只流过学生状态和投影权重\n- 损失是纯逐元素平均，不做只对 token 轴的归约",
    "function_name": "hidden_distillation",
    "hint": "`diff = projection(student_hidden) - teacher_hidden` then `diff.pow(2).mean()`.",
    "hint_zh": "`diff = projection(student_hidden) - teacher_hidden`，然后 `diff.pow(2).mean()`。",
    "tests": [
        {
            "name": "Zero when the projection reproduces the teacher",
            "code": """
import torch
import torch.nn as nn
torch.manual_seed(0)
student = torch.randn(2, 3, 4)
teacher = torch.randn(2, 3, 6)
proj = nn.Linear(4, 6)
with torch.no_grad():
    teacher.copy_(proj(student))
out = {fn}(student, teacher, proj)
assert torch.allclose(out, torch.tensor(0.0), atol=1e-6), f'Perfect match must give 0, got {out}'
""",
        },
        {
            "name": "Matches the manual MSE",
            "code": """
import torch
import torch.nn as nn
torch.manual_seed(1)
student = torch.randn(2, 5, 8)
teacher = torch.randn(2, 5, 12)
proj = nn.Linear(8, 12)
expected = ((proj(student) - teacher) ** 2).mean()
out = {fn}(student, teacher, proj)
assert torch.allclose(out, expected, atol=1e-7), f'{out.item()} vs {expected.item()}'
""",
        },
        {
            "name": "Gradient reaches the student and the projection",
            "code": """
import torch
import torch.nn as nn
torch.manual_seed(2)
student = torch.randn(2, 4, 8, requires_grad=True)
teacher = torch.randn(2, 4, 12)
proj = nn.Linear(8, 12)
out = {fn}(student, teacher, proj)
out.backward()
assert student.grad is not None and student.grad.abs().sum() > 0, 'student grad missing'
assert proj.weight.grad is not None and proj.weight.grad.abs().sum() > 0, 'projection grad missing'
""",
        },
        {
            "name": "Teacher is not a gradient target",
            "code": """
import torch
import torch.nn as nn
torch.manual_seed(3)
student = torch.randn(2, 4, 8)
teacher = torch.randn(2, 4, 12, requires_grad=True)
proj = nn.Linear(8, 12)
out = {fn}(student, teacher, proj)
out.backward()
assert teacher.grad is None, 'The loss must not backprop into teacher states'
""",
        },
        {
            "name": "Element-wise mean, not per-token sum",
            "code": """
import torch
import torch.nn as nn
torch.manual_seed(4)
student = torch.zeros(1, 2, 3)
teacher = torch.zeros(1, 2, 3)
proj = nn.Linear(3, 3)
with torch.no_grad():
    proj.weight.zero_()
    proj.bias.fill_(1.0)
# diff is all ones: element mean = 1, per-token sum = 2, feature sum = 3
out = {fn}(student, teacher, proj)
assert torch.allclose(out, torch.tensor(1.0), atol=1e-6), f'Expected element-wise mean 1.0, got {out.item()}'
""",
        },
    ],
    "solution": '''import torch


def hidden_distillation(student_hidden, teacher_hidden, projection):
    diff = projection(student_hidden) - teacher_hidden.detach()
    return diff.pow(2).mean()''',
    "demo": """import torch
import torch.nn as nn
torch.manual_seed(0)
student = torch.randn(2, 6, 16)
teacher = torch.randn(2, 6, 32)
proj = nn.Linear(16, 32)
print('loss:', hidden_distillation(student, teacher, proj).item())""",
}
