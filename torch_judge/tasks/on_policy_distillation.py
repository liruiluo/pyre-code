"""On-policy distillation loss task (GKD-style)."""

TASK = {
    "title": "On-Policy Distillation Loss (GKD)",
    "title_zh": "在线策略蒸馏损失（GKD）",
    "difficulty": "Hard",
    "description_en": "Implement a GKD-style on-policy distillation example with a chosen reverse-KL objective. GKD-style methods may instead use forward KL, JSD or other divergences.\n\nThe tokens the student generated itself (from its own sampling distribution) carry the distribution mismatch the training must fix. The trajectory contains both student-generated tokens and foreign tokens — the prompt, environment observations, tool outputs. Per-token reverse KL `D_KL(student_t || teacher_t)` is computed ONLY at positions the student produced, averaged over those positions, scaled by `T^2`.\n\nTraining on the student's own rollout contexts is the on-policy aspect. This exercise assumes those contexts and a generated-token mask are already provided: it does NOT generate rollouts, and reverse KL is an independent design choice.\n\n**Signature:** `on_policy_distillation_loss(student_logits, teacher_logits, generated_mask, temperature=1.0) -> Tensor`\n\n**Parameters:**\n- `student_logits`, `teacher_logits` — shape `(B, T, V)`\n- `generated_mask` — boolean `(B, T)`, `True` where the token was sampled from the student\n- `temperature` — softening temperature\n\n**Returns:** scalar = `T^2 * mean over masked positions of D_KL(student_T || teacher_T)`; exactly `0.0` when no position is masked\n\n**Constraints:**\n- The denominator is the number of masked positions, not `B*T`\n- Logits at unmasked positions must not influence the loss",
    "description_zh": "实现一个以 Reverse KL 为选择的 GKD 类 On-Policy 蒸馏练习。GKD 类方法也可以采用 Forward KL、JSD 或其他散度。\n\n学生自己生成（从它自己的采样分布）的 token 承载着训练要修正的分布失配。轨迹里既有学生生成的 token，也有外来 token——提示词、环境观察、工具输出。逐 token 的反向 KL `D_KL(student_t || teacher_t)` 只在学生产出的位置计算，对这些位置求平均，再乘 `T^2`。\n\nOn-Policy 指使用学生自己的 rollout 上下文进行训练。本题假定 rollout 与 generated-token mask 已提供，本身不执行采样；Reverse KL 是独立于采样方式的另一项设计选择。\n\n**签名:** `on_policy_distillation_loss(student_logits, teacher_logits, generated_mask, temperature=1.0) -> Tensor`\n\n**参数:**\n- `student_logits`、`teacher_logits` — 形状 `(B, T, V)`\n- `generated_mask` — 布尔 `(B, T)`，学生采样的 token 位置为 `True`\n- `temperature` — 软化温度\n\n**返回:** 标量 = `T^2 * 被掩码位置的 D_KL(student_T || teacher_T) 平均`；无掩码位置时恰为 `0.0`\n\n**约束:**\n- 分母是掩码位置数，不是 `B*T`\n- 未掩码位置的 logits 不得影响损失",
    "function_name": "on_policy_distillation_loss",
    "hint": "Per-token reverse KL gives `(B, T)`; select with `kl[generated_mask]`, `.mean()` over the selection; guard an empty mask with `student_logits.sum() * 0` to preserve device and gradient graph; detach teacher logits.",
    "hint_zh": "逐 token 反向 KL 得 `(B, T)`；用 `kl[generated_mask]` 选取后 `.mean()`；空 mask 时返回 `student_logits.sum() * 0` 保持 device 与梯度图；teacher logits 需要 detach。",
    "tests": [
        {
            "name": "Zero when student matches teacher on generated tokens",
            "code": """
import torch
torch.manual_seed(0)
logits = torch.randn(2, 6, 10)
mask = torch.zeros(2, 6, dtype=torch.bool)
mask[:, 2:] = True
out = {fn}(logits, logits.clone(), mask, temperature=2.0)
assert torch.allclose(out, torch.tensor(0.0), atol=1e-6), f'Identical logits must give 0, got {out}'
""",
        },
        {
            "name": "Averages over masked positions only",
            "code": """
import torch
torch.manual_seed(1)
s = torch.randn(2, 5, 8)
t = torch.randn(2, 5, 8)
mask = torch.tensor([[False, True, True, False, False], [True, True, False, False, False]])
T = 1.5
log_p = torch.log_softmax(s / T, dim=-1)
p = log_p.exp()
log_t = torch.log_softmax(t / T, dim=-1)
per_tok = (p * (log_p - log_t)).sum(-1)
expected = per_tok[mask].mean() * T**2
out = {fn}(s, t, mask, temperature=T)
assert torch.allclose(out, expected, atol=1e-6), f'{out.item()} vs {expected.item()}'
""",
        },
        {
            "name": "Unmasked logits are invisible",
            "code": """
import torch
torch.manual_seed(2)
s = torch.randn(2, 4, 6)
t = torch.randn(2, 4, 6)
mask = torch.tensor([[True, False, True, False], [True, True, True, False]])
base = {fn}(s, t, mask)
s_perturbed = s.clone()
s_perturbed[~mask] = s_perturbed[~mask] + torch.randn_like(s_perturbed[~mask]) * 20.0
out = {fn}(s_perturbed, t, mask)
assert torch.allclose(base, out, atol=1e-5), 'Prompt/observation logits must not affect the loss'
""",
        },
        {
            "name": "Empty mask returns exact zero",
            "code": """
import torch
s = torch.randn(3, 4, 5)
t = torch.randn(3, 4, 5)
mask = torch.zeros(3, 4, dtype=torch.bool)
out = {fn}(s, t, mask)
assert out.item() == 0.0, f'No generated tokens must give exact 0.0, got {out.item()}'
assert out.dim() == 0, 'Return a scalar tensor'
""",
        },
        {
            "name": "Temperature squared scaling",
            "code": """
import torch
torch.manual_seed(3)
s = torch.randn(2, 3, 7)
t = torch.randn(2, 3, 7)
mask = torch.ones(2, 3, dtype=torch.bool)
low = {fn}(s, t, mask, temperature=1.0)
high = {fn}(s * 3.0, t * 3.0, mask, temperature=3.0)
assert torch.allclose(high, 9.0 * low, atol=1e-4), 'Logits scaled by T with temperature T multiplies the loss by T^2'
""",
        },
        {
            "name": "Gradually drifting student sees growing loss",
            "code": """
import torch
torch.manual_seed(4)
t = torch.randn(2, 4, 6)
mask = torch.ones(2, 4, dtype=torch.bool)
losses = []
direction = torch.randn(2, 4, 6)
for eps in (0.0, 0.5, 2.0):
    s = t + eps * direction
    losses.append({fn}(s, t, mask).item())
assert losses[0] < losses[1] < losses[2], f'Loss must grow with student drift: {losses}'
""",
        },
        {
            "name": "Teacher stop-grad and connected empty-mask zero",
            "code": """
import torch
student = torch.randn(2, 4, 6, requires_grad=True)
teacher = torch.randn(2, 4, 6, requires_grad=True)
mask = torch.ones(2, 4, dtype=torch.bool)
{fn}(student, teacher, mask).backward()
assert student.grad is not None and student.grad.abs().sum() > 0
assert teacher.grad is None
student.grad = None
zero = {fn}(student, teacher, torch.zeros_like(mask))
assert zero.item() == 0.0 and zero.device == student.device
zero.backward()
assert student.grad is not None and torch.all(student.grad == 0)
""",
        },
    ],
    "solution": '''import torch
import torch.nn.functional as F

def on_policy_distillation_loss(student_logits, teacher_logits, generated_mask, temperature=1.0):
    # On-policy selects training contexts; reverse KL is the chosen divergence.
    if not generated_mask.any():
        return student_logits.sum() * 0.0
    log_student = F.log_softmax(student_logits / temperature, dim=-1)
    student_probs = log_student.exp()
    log_teacher = F.log_softmax(teacher_logits.detach() / temperature, dim=-1)
    per_token_kl = (student_probs * (log_student - log_teacher)).sum(dim=-1)
    return per_token_kl[generated_mask].mean() * temperature ** 2''',
    "demo": """import torch
torch.manual_seed(0)
teacher = torch.randn(2, 5, 9)
student = teacher + 0.3 * torch.randn(2, 5, 9)
mask = torch.tensor([[True, True, False, False, True], [True, False, True, True, False]])
print(on_policy_distillation_loss(student, teacher, mask, temperature=2.0).item())""",
}
