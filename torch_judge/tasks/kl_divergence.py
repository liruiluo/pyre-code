"""KL divergence task."""

TASK = {
    "title": "KL Divergence from Logits",
    "title_zh": "从 Logits 计算 KL 散度",
    "difficulty": "Medium",
    "description_en": "Implement KL divergence between two categorical distributions represented by logits.\n\nKL divergence is asymmetric: `D_KL(P || Q) = sum_i P(i) * (log P(i) - log Q(i))`. It is widely used in distillation and RLHF-style regularization.\n\n**Signature:** `kl_divergence(p_logits, q_logits, reduction='batchmean') -> Tensor`\n\n**Parameters:**\n- `p_logits`, `q_logits` — logits with class dimension last, shape `(..., V)`\n- `reduction` — one of `'none'`, `'sum'`, `'mean'`, `'batchmean'`\n\n**Returns:** KL divergence tensor or scalar, depending on reduction\n\n**Constraints:**\n- Use stable log-softmax, not `softmax().log()`\n- Compute `D_KL(P || Q)`, not `D_KL(Q || P)`",
    "description_zh": "实现两个由 logits 表示的类别分布之间的 KL 散度。\n\nKL 散度不对称：`D_KL(P || Q) = sum_i P(i) * (log P(i) - log Q(i))`。它常用于知识蒸馏和 RLHF 风格正则。\n\n**签名:** `kl_divergence(p_logits, q_logits, reduction='batchmean') -> Tensor`\n\n**参数:**\n- `p_logits`, `q_logits` — 最后一维为类别维的 logits，形状 `(..., V)`\n- `reduction` — `'none'`, `'sum'`, `'mean'`, `'batchmean'` 之一\n\n**返回:** 根据 reduction 返回 KL 张量或标量\n\n**约束:**\n- 使用稳定的 log-softmax，不要 `softmax().log()`\n- 计算 `D_KL(P || Q)`，不是 `D_KL(Q || P)`",
    "function_name": "kl_divergence",
    "hint": "1. `log_p = log_softmax(p_logits, dim=-1)`, `log_q = log_softmax(q_logits, dim=-1)`\n2. `p = exp(log_p)`\n3. `kl = (p * (log_p - log_q)).sum(dim=-1)`\n4. Apply the requested reduction",
    "hint_zh": "1. `log_p = log_softmax(p_logits, dim=-1)`, `log_q = log_softmax(q_logits, dim=-1)`\n2. `p = exp(log_p)`\n3. `kl = (p * (log_p - log_q)).sum(dim=-1)`\n4. 按 reduction 汇总",
    "tests": [
        {
            "name": "Zero when distributions match",
            "code": """
import torch
logits = torch.randn(4, 7)
out = {fn}(logits, logits)
assert torch.allclose(out, torch.tensor(0.0), atol=1e-6), f'KL(P||P) should be zero, got {out}'
""",
        },
        {
            "name": "Matches manual formula",
            "code": """
import torch
torch.manual_seed(0)
p_logits = torch.randn(3, 5)
q_logits = torch.randn(3, 5)
log_p = torch.log_softmax(p_logits, dim=-1)
log_q = torch.log_softmax(q_logits, dim=-1)
expected = (log_p.exp() * (log_p - log_q)).sum(dim=-1).sum() / p_logits.shape[0]
out = {fn}(p_logits, q_logits, reduction='batchmean')
assert torch.allclose(out, expected, atol=1e-6), f'{out.item():.6f} vs {expected.item():.6f}'
""",
        },
        {
            "name": "Asymmetry is preserved",
            "code": """
import torch
p = torch.tensor([[3.0, 0.0, -1.0]])
q = torch.tensor([[0.5, 0.0, -0.5]])
pq = {fn}(p, q)
qp = {fn}(q, p)
assert not torch.allclose(pq, qp), 'KL divergence should be asymmetric'
""",
        },
        {
            "name": "none reduction returns per-row values",
            "code": """
import torch
p = torch.randn(2, 4)
q = torch.randn(2, 4)
out = {fn}(p, q, reduction='none')
assert out.shape == (2,), f'Expected per-row shape (2,), got {out.shape}'
assert torch.all(out >= -1e-6), 'KL should be non-negative up to numerical tolerance'
""",
        },
        {
            "name": "Numerical stability on large logits",
            "code": """
import torch
p = torch.tensor([[1000.0, 0.0, -1000.0]])
q = torch.tensor([[999.0, 1.0, -1000.0]])
out = {fn}(p, q)
assert not torch.isnan(out) and not torch.isinf(out), 'KL should remain finite for large logits'
""",
        },
    ],
    "solution": '''def kl_divergence(p_logits, q_logits, reduction='batchmean'):
    log_p = torch.log_softmax(p_logits, dim=-1)
    log_q = torch.log_softmax(q_logits, dim=-1)
    kl = (log_p.exp() * (log_p - log_q)).sum(dim=-1)
    if reduction == 'none':
        return kl
    if reduction == 'sum':
        return kl.sum()
    if reduction == 'mean':
        return kl.mean()
    if reduction == 'batchmean':
        return kl.sum() / p_logits.shape[0]
    raise ValueError("reduction must be one of 'none', 'sum', 'mean', 'batchmean'")''',
    "demo": """p_logits = torch.tensor([[2.0, 0.0, -1.0]])
q_logits = torch.tensor([[0.0, 2.0, -1.0]])
print('D_KL(P||Q):', kl_divergence(p_logits, q_logits).item())""",
}
