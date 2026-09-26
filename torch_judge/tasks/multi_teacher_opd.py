"""Multi-teacher on-policy distillation task (Xiaomi MiMo MOPD style)."""

TASK = {
    "title": "Multi-Teacher On-Policy Distillation (MiMo MOPD)",
    "title_zh": "多教师在线策略蒸馏（MiMo MOPD）",
    "difficulty": "Hard",
    "description_en": "Blend several teachers into one on-policy distillation loss — the objective behind Xiaomi MiMo's MOPD (Multi-Teacher On-Policy Distillation), used in MiMo-V2.6 post-training to merge distinct teachers' capabilities into a single student.\n\nOne teacher is rarely best at everything: a math teacher, a coding teacher, a generalist. MOPD-style training keeps the student's own rollouts as the data (the on-policy property) and scores them against every teacher, combining the per-teacher signals with capability-mix weights. Each teacher contributes `w_m` of its masked reverse-KL; the weights are the knobs that decide which capability dominates this stage of training. (MiMo's exact weighting scheme is not public — this is the generic multi-teacher objective it instantiates.)\n\n**Signature:** `multi_teacher_opd_loss(student_logits, teacher_logits_stack, teacher_weights, generated_mask, temperature=1.0) -> Tensor`\n\n**Parameters:**\n- `student_logits` — `(B, T, V)`\n- `teacher_logits_stack` — `(M, B, T, V)`, one teacher per leading slot\n- `teacher_weights` — `(M,)`, non-negative capability-mix weights\n- `generated_mask` — boolean `(B, T)`, `True` where the token was sampled from the student\n- `temperature` — softening temperature\n\n**Returns:** scalar = `T^2 * sum_m w_m * (mean over masked positions of D_KL(student || teacher_m))`; exactly `0.0` when no position is masked\n\n**Constraints:**\n- The same masked positions and the same denominator are shared by every teacher's term\n- Teachers are detached: gradients flow only to the student\n- Logits at unmasked positions must not influence the loss",
    "description_zh": "把多个教师融合进一个在线策略蒸馏损失——小米 MiMo 的 MOPD（Multi-Teacher On-Policy Distillation）背后的目标，用于 MiMo-V2.6 后训练，把不同教师的能力合并进一个学生。\n\n单个教师很少样样最强：数学教师、代码教师、通才各有所长。MOPD 式训练保持学生自己的 rollout 作为数据（在线策略性质），对每个教师分别打分，再用能力组合权重融合各路信号。每个教师贡献 `w_m` 份其掩码反向 KL；权重就是决定这个训练阶段哪种能力主导的旋钮。（MiMo 的精确加权方案未公开——本题实现的是它实例化的通用多教师目标。）\n\n**签名:** `multi_teacher_opd_loss(student_logits, teacher_logits_stack, teacher_weights, generated_mask, temperature=1.0) -> Tensor`\n\n**参数:**\n- `student_logits` — `(B, T, V)`\n- `teacher_logits_stack` — `(M, B, T, V)`，逐个教师堆叠\n- `teacher_weights` — `(M,)`，非负能力组合权重\n- `generated_mask` — 布尔 `(B, T)`，学生采样的 token 位置为 `True`\n- `temperature` — 软化温度\n\n**返回:** 标量 = `T^2 * Σ_m w_m * (掩码位置的 D_KL(student || teacher_m) 平均)`；无掩码位置时恰为 `0.0`\n\n**约束:**\n- 每个教师项共享同一掩码位置和同一分母\n- 教师 detach：梯度只流给学生\n- 未掩码位置的 logits 不得影响损失",
    "function_name": "multi_teacher_opd_loss",
    "hint": "Loop over teachers: `log_t = log_softmax(stack[m].detach() / T)`; per-token reverse KL, select with the mask, `.mean()`; accumulate `w_m * kl_m`; multiply the total by `T^2` once. Guard the empty mask.",
    "hint_zh": "逐教师循环：`log_t = log_softmax(stack[m].detach() / T)`；逐 token 反向 KL，掩码选取后 `.mean()`；累加 `w_m * kl_m`；最后整体乘一次 `T^2`。空掩码要守卫。",
    "tests": [
        {
            "name": "Zero when the student matches the only teacher",
            "code": """
import torch
torch.manual_seed(0)
logits = torch.randn(2, 5, 9)
teachers = logits.clone().unsqueeze(0)          # single teacher, identical
weights = torch.ones(1)
mask = torch.ones(2, 5, dtype=torch.bool)
out = {fn}(logits, teachers, weights, mask, temperature=2.0)
assert torch.allclose(out, torch.tensor(0.0), atol=1e-6), f'Identical logits must give 0, got {out}'
""",
        },
        {
            "name": "Weighted sum of per-teacher losses",
            "code": """
import torch
torch.manual_seed(1)
s = torch.randn(2, 4, 7)
teachers = torch.randn(3, 2, 4, 7)
w = torch.tensor([0.5, 0.3, 0.2])
mask = torch.tensor([[True, True, False, False], [True, False, True, True]])
T = 1.5
log_p = torch.log_softmax(s / T, dim=-1)
p = log_p.exp()
per_teacher = []
for m in range(3):
    log_t = torch.log_softmax(teachers[m] / T, dim=-1)
    per_tok = (p * (log_p - log_t)).sum(-1)
    per_teacher.append(per_tok[mask].mean())
expected = (w[0] * per_teacher[0] + w[1] * per_teacher[1] + w[2] * per_teacher[2]) * T**2
out = {fn}(s, teachers, w, mask, temperature=T)
assert torch.allclose(out, expected, atol=1e-6), f'{out.item()} vs {expected.item()}'
""",
        },
        {
            "name": "Zero weight mutes a teacher",
            "code": """
import torch
torch.manual_seed(2)
s = torch.randn(2, 3, 6)
good = s + 0.1 * torch.randn(2, 3, 6)
wild = torch.randn(2, 3, 6) * 50.0             # a wildly different teacher
mask = torch.ones(2, 3, dtype=torch.bool)
l_one = {fn}(s, good.unsqueeze(0), torch.ones(1), mask)
l_two = {fn}(s, torch.stack((good, wild)), torch.tensor([1.0, 0.0]), mask)
assert torch.allclose(l_one, l_two, atol=1e-6), 'A teacher with weight 0 must contribute nothing'
""",
        },
        {
            "name": "Unmasked positions are invisible",
            "code": """
import torch
torch.manual_seed(3)
s = torch.randn(2, 4, 6)
teachers = torch.randn(2, 2, 4, 6)
w = torch.tensor([0.6, 0.4])
mask = torch.tensor([[True, False, True, False], [False, True, True, True]])
base = {fn}(s, teachers, w, mask)
s_perturbed = s.clone()
s_perturbed[:, 1] += 100.0
s_perturbed[0, 3] -= 100.0
out = {fn}(s_perturbed, teachers, w, mask)
assert torch.allclose(base, out, atol=1e-5), 'Prompt/observation positions must not affect the loss'
""",
        },
        {
            "name": "Empty mask returns exact zero",
            "code": """
import torch
s = torch.randn(2, 3, 5)
teachers = torch.randn(2, 2, 3, 5)
mask = torch.zeros(2, 3, dtype=torch.bool)
out = {fn}(s, teachers, torch.tensor([0.5, 0.5]), mask)
assert out.item() == 0.0 and out.dim() == 0, f'Expected scalar 0.0, got {out}'
""",
        },
        {
            "name": "Gradient reaches the student but not the teachers",
            "code": """
import torch
torch.manual_seed(4)
s = torch.randn(2, 3, 6, requires_grad=True)
teachers = torch.randn(2, 2, 3, 6, requires_grad=True)
mask = torch.ones(2, 3, dtype=torch.bool)
out = {fn}(s, teachers, torch.tensor([0.7, 0.3]), mask)
out.backward()
assert s.grad is not None and s.grad.abs().sum() > 0, 'Student must train'
assert teachers.grad is None, 'Teachers must stay frozen'
""",
        },
        {
            "name": "Single teacher reduces to the on-policy loss",
            "code": """
import torch
torch.manual_seed(5)
s = torch.randn(2, 4, 8)
t = torch.randn(2, 4, 8)
mask = torch.tensor([[True, True, True, False], [True, True, True, True]])
T = 2.0
out = {fn}(s, t.unsqueeze(0), torch.ones(1), mask, temperature=T)
log_p = torch.log_softmax(s / T, dim=-1)
p = log_p.exp()
log_t = torch.log_softmax(t / T, dim=-1)
per_tok = (p * (log_p - log_t)).sum(-1)
expected = per_tok[mask].mean() * T**2
assert torch.allclose(out, expected, atol=1e-6), 'M=1 with weight 1 is the plain on-policy loss'
""",
        },
    ],
    "solution": '''import torch
import torch.nn.functional as F


def multi_teacher_opd_loss(student_logits, teacher_logits_stack, teacher_weights, generated_mask, temperature=1.0):
    if generated_mask.sum() == 0:
        return torch.zeros((), dtype=student_logits.dtype)
    log_p = F.log_softmax(student_logits / temperature, dim=-1)
    p = log_p.exp()
    total = torch.zeros((), dtype=student_logits.dtype)
    for m in range(teacher_logits_stack.shape[0]):
        log_t = F.log_softmax(teacher_logits_stack[m].detach() / temperature, dim=-1)
        per_tok = (p * (log_p - log_t)).sum(dim=-1)
        kl_m = per_tok[generated_mask].mean()
        total = total + teacher_weights[m] * kl_m
    return total * (temperature ** 2)''',
    "demo": """import torch
torch.manual_seed(0)
student = torch.randn(2, 5, 10)
math_teacher = student + 0.5 * torch.randn(2, 5, 10)
code_teacher = student + 0.8 * torch.randn(2, 5, 10)
teachers = torch.stack((math_teacher, code_teacher))
mask = torch.ones(2, 5, dtype=torch.bool)
for w in ([1.0, 0.0], [0.5, 0.5], [0.0, 1.0]):
    loss = multi_teacher_opd_loss(student, teachers, torch.tensor(w), mask, temperature=2.0)
    print(f'weights {w}: {loss.item():.4f}')""",
}
