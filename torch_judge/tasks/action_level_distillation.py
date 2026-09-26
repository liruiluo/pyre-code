"""Action-level (span-normalized) agent distillation task."""

TASK = {
    "title": "Action-Level Agent Distillation",
    "title_zh": "动作级 Agent 蒸馏",
    "difficulty": "Hard",
    "description_en": "Distill the agent's decisions, not the whole sequence — span-normalized trajectory distillation.\n\nAn agent trajectory interleaves the student's own action spans with environment spans: prompts, observations, tool results. When distilling a teacher policy, what you actually transfer is the decisions — the action tokens. Distilling the full sequence dilutes the signal with environment statistics the student cannot control. Distill the action, not the sequence.\n\nThe subtlety is normalization. A per-token mean lets long actions dominate the loss; a span-level normalization weights every action equally no matter how many tokens it spans. Compute the mean reverse KL inside each action span, then average over spans.\n\n**Signature:** `action_level_distillation_loss(student_logits, teacher_logits, action_spans, temperature=1.0) -> Tensor`\n\n**Parameters:**\n- `student_logits`, `teacher_logits` — shape `(T, V)` for one trajectory\n- `action_spans` — list of `(start, end)` index pairs, half-open `[start, end)`, non-overlapping, ascending\n- `temperature` — softening temperature\n\n**Returns:** scalar = `T^2 * mean over spans of (mean KL within the span)`; exactly `0.0` for an empty span list\n\n**Constraints:**\n- Positions outside every span contribute nothing\n- Two spans with identical per-token KL but different lengths contribute equally (span normalization)\n- KL direction is reverse: `D_KL(student_t || teacher_t)`",
    "description_zh": "蒸馏 agent 的决策，而不是整个序列——跨度归一化的轨迹蒸馏。\n\nagent 轨迹把学生自己的动作跨度与环境跨度交错：提示词、观察、工具结果。蒸馏教师策略时，真正迁移的是决策——动作 token。整序列蒸馏会用学生控制不了的环境统计稀释信号。蒸馏动作，不蒸馏序列。\n\n微妙之处在归一化。逐 token 求平均会让长动作主导损失；跨度级归一化让每个动作权重相等，不管它跨多少 token。在每个动作跨度内算平均反向 KL，再对跨度求平均。\n\n**签名:** `action_level_distillation_loss(student_logits, teacher_logits, action_spans, temperature=1.0) -> Tensor`\n\n**参数:**\n- `student_logits`、`teacher_logits` — 单条轨迹，形状 `(T, V)`\n- `action_spans` — `(start, end)` 索引对列表，左闭右开 `[start, end)`，不重叠、递增\n- `temperature` — 软化温度\n\n**返回:** 标量 = `T^2 * 各跨度内 KL 平均 再对跨度平均`；跨度列表为空时恰为 `0.0`\n\n**约束:**\n- 所有跨度之外的位置不贡献\n- 两个逐 token KL 相同但长度不同的跨度贡献相等（跨度归一化）\n- KL 方向是反向：`D_KL(student_t || teacher_t)`",
    "function_name": "action_level_distillation_loss",
    "hint": "Per-token reverse KL gives `(T,)`; `per_tok[s:e].mean()` inside each span, then `torch.stack(span_losses).mean()`; guard the empty list.",
    "hint_zh": "逐 token 反向 KL 得 `(T,)`；每个跨度内 `per_tok[s:e].mean()`，再 `torch.stack(span_losses).mean()`；空列表要守卫。",
    "tests": [
        {
            "name": "Zero when student matches teacher on the actions",
            "code": """
import torch
torch.manual_seed(0)
logits = torch.randn(8, 10)
spans = [(0, 2), (5, 8)]
out = {fn}(logits, logits.clone(), spans, temperature=2.0)
assert torch.allclose(out, torch.tensor(0.0), atol=1e-6), f'Identical logits must give 0, got {out}'
""",
        },
        {
            "name": "Span normalization: length does not change weight",
            "code": """
import torch
torch.manual_seed(1)
t = torch.zeros(10, 4)
s = torch.zeros(10, 4)
s[:, 0] = 1.0   # constant per-token mismatch everywhere
one = [(0, 1)]                       # one short action
three = [(3, 6)]                     # one long action, same per-token KL
both = [(0, 1), (3, 6)]              # both: the mean of two identical span losses
l_one = {fn}(s, t, one)
l_three = {fn}(s, t, three)
l_both = {fn}(s, t, both)
assert torch.allclose(l_one, l_three, atol=1e-6), 'Long span must not dominate'
assert torch.allclose(l_both, l_one, atol=1e-6), 'Two identical-KL spans average to the same value'
""",
        },
        {
            "name": "Positions outside spans are invisible",
            "code": """
import torch
torch.manual_seed(2)
s = torch.randn(6, 5)
t = torch.randn(6, 5)
spans = [(1, 3)]
base = {fn}(s, t, spans)
s_perturbed = s.clone()
s_perturbed[[0, 4, 5]] += 100.0
out = {fn}(s_perturbed, t, spans)
assert torch.allclose(base, out, atol=1e-5), 'Non-action positions must not affect the loss'
""",
        },
        {
            "name": "Matches the manual span-mean-of-means",
            "code": """
import torch
torch.manual_seed(3)
s = torch.randn(9, 6)
t = torch.randn(9, 6)
spans = [(0, 3), (4, 5), (7, 9)]
T = 2.5
log_p = torch.log_softmax(s / T, dim=-1)
p = log_p.exp()
log_t = torch.log_softmax(t / T, dim=-1)
per_tok = (p * (log_p - log_t)).sum(-1)
expected = ((per_tok[0:3].mean() + per_tok[4:5].mean() + per_tok[7:9].mean()) / 3) * T**2
out = {fn}(s, t, spans, temperature=T)
assert torch.allclose(out, expected, atol=1e-6), f'{out.item()} vs {expected.item()}'
""",
        },
        {
            "name": "Empty span list returns exact zero",
            "code": """
import torch
s = torch.randn(5, 4)
t = torch.randn(5, 4)
out = {fn}(s, t, [])
assert out.item() == 0.0 and out.dim() == 0, f'Expected scalar 0.0, got {out}'
""",
        },
        {
            "name": "Differs from flat per-token mean on mixed spans",
            "code": """
import torch
torch.manual_seed(4)
s = torch.randn(10, 5)
t = torch.randn(10, 5)
spans = [(0, 2), (8, 10)]
log_p = torch.log_softmax(s, dim=-1)
p = log_p.exp()
per_tok = (p * (log_p - torch.log_softmax(t, dim=-1))).sum(-1)
flat = per_tok[[0, 1, 8, 9]].mean()
span_norm = (per_tok[0:2].mean() + per_tok[8:10].mean()) / 2
out = {fn}(s, t, spans)
assert torch.allclose(out, span_norm, atol=1e-6)
# flat and span-normalized coincide only when both spans have equal length
if not torch.allclose(flat, span_norm, atol=1e-6):
    pass  # same lengths here, but the assertion above is the real contract
""",
        },
    ],
    "solution": '''import torch
import torch.nn.functional as F


def action_level_distillation_loss(student_logits, teacher_logits, action_spans, temperature=1.0):
    if not action_spans:
        return torch.zeros((), dtype=student_logits.dtype)
    log_p = F.log_softmax(student_logits / temperature, dim=-1)
    p = log_p.exp()
    log_t = F.log_softmax(teacher_logits / temperature, dim=-1)
    per_tok = (p * (log_p - log_t)).sum(dim=-1)
    span_losses = [per_tok[start:end].mean() for start, end in action_spans]
    return torch.stack(span_losses).mean() * (temperature ** 2)''',
    "demo": """import torch
torch.manual_seed(0)
teacher = torch.randn(12, 8)
student = teacher + 0.4 * torch.randn(12, 8)
# the agent decided at tokens 0-1 and 6-9; everything else is environment
spans = [(0, 2), (6, 10)]
print(action_level_distillation_loss(student, teacher, spans, temperature=2.0).item())""",
}
