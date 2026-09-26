"""On-policy distillation loss task (GKD-style)."""

TASK = {
    "title": "On-Policy Distillation Loss (GKD)",
    "title_zh": "在线策略蒸馏损失（GKD）",
    "difficulty": "Hard",
    "description_en": "Implement the core objective of on-policy knowledge distillation — the GKD recipe.\n\nThe tokens the student generated itself (from its own sampling distribution) carry the distribution mismatch the training must fix. The trajectory contains both student-generated tokens and foreign tokens — the prompt, environment observations, tool outputs. Per-token reverse KL `D_KL(student_t || teacher_t)` is computed ONLY at positions the student produced, averaged over those positions, scaled by `T^2`.\n\nTraining on the student's own rollout data is what makes this on-policy: the student never drifts onto inputs it would not itself visit, and the objective is the mode-seeking reverse KL at exactly those states.\n\n**Signature:** `on_policy_distillation_loss(student_logits, teacher_logits, generated_mask, temperature=1.0) -> Tensor`\n\n**Parameters:**\n- `student_logits`, `teacher_logits` — shape `(B, T, V)`\n- `generated_mask` — boolean `(B, T)`, `True` where the token was sampled from the student\n- `temperature` — softening temperature\n\n**Returns:** scalar = `T^2 * mean over masked positions of D_KL(student_T || teacher_T)`; exactly `0.0` when no position is masked\n\n**Constraints:**\n- The denominator is the number of masked positions, not `B*T`\n- Logits at unmasked positions must not influence the loss",
    "description_zh": "实现在线策略知识蒸馏的核心目标——GKD 配方。\n\n学生自己生成（从它自己的采样分布）的 token 承载着训练要修正的分布失配。轨迹里既有学生生成的 token，也有外来 token——提示词、环境观察、工具输出。逐 token 的反向 KL `D_KL(student_t || teacher_t)` 只在学生产出的位置计算，对这些位置求平均，再乘 `T^2`。\n\n在学生自己的 rollout 数据上训练，这才是“在线策略”：学生永远不会漂到它自己不会到达的输入上，目标函数也恰是在这些状态上的模式寻求型反向 KL。\n\n**签名:** `on_policy_distillation_loss(student_logits, teacher_logits, generated_mask, temperature=1.0) -> Tensor`\n\n**参数:**\n- `student_logits`、`teacher_logits` — 形状 `(B, T, V)`\n- `generated_mask` — 布尔 `(B, T)`，学生采样的 token 位置为 `True`\n- `temperature` — 软化温度\n\n**返回:** 标量 = `T^2 * 被掩码位置的 D_KL(student_T || teacher_T) 平均`；无掩码位置时恰为 `0.0`\n\n**约束:**\n- 分母是掩码位置数，不是 `B*T`\n- 未掩码位置的 logits 不得影响损失",
    "function_name": "on_policy_distillation_loss",
    "hint": "Per-token reverse KL gives `(B, T)`; select with `kl[generated_mask]`, `.mean()` over the selection; guard `numel() == 0` and return `torch.zeros(())`.",
    "hint_zh": "逐 token 反向 KL 得 `(B, T)`；用 `kl[generated_mask]` 选取后 `.mean()`；`numel() == 0` 时守卫返回 `torch.zeros(())`。",
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
s_perturbed[:, [1, 3]] += 50.0   # torch advanced indexing
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
for eps in (0.0, 0.5, 2.0):
    s = t + eps * torch.randn(2, 4, 6)
    losses.append({fn}(s, t, mask).item())
assert losses[0] < losses[1] < losses[2], f'Loss must grow with student drift: {losses}'
""",
        },
    ],
    "solution": '''import torch
import torch.nn.functional as F


def on_policy_distillation_loss(student_logits, teacher_logits, generated_mask, temperature=1.0):
    log_p = F.log_softmax(student_logits / temperature, dim=-1)
    p = log_p.exp()
    log_t = F.log_softmax(teacher_logits / temperature, dim=-1)
    per_tok = (p * (log_p - log_t)).sum(dim=-1)
    sel = per_tok[generated_mask]
    if sel.numel() == 0:
        return torch.zeros((), dtype=per_tok.dtype)
    return sel.mean() * (temperature ** 2)''',
    "demo": """import torch
torch.manual_seed(0)
teacher = torch.randn(2, 5, 9)
student = teacher + 0.3 * torch.randn(2, 5, 9)
mask = torch.tensor([[True, True, False, False, True], [True, False, True, True, False]])
print(on_policy_distillation_loss(student, teacher, mask, temperature=2.0).item())""",
}
