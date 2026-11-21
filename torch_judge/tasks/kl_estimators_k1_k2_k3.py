"""Schulman's k1, k2 and k3 KL estimators for RLHF/RLVR."""

TASK = {
    "title": "k1 / k2 / k3 KL Estimators",
    "title_zh": "k1 / k2 / k3 KL 估计器",
    "difficulty": "Medium",
    "description_en": "Implement Schulman's k1, k2, k3 estimators of KL(policy || reference) from log-probabilities of tokens sampled under the CURRENT policy.\n\nLet d = log pi(y|x) - log pi_ref(y|x) and r = exp(-d) = pi_ref / pi. For each sampled token:\n- k1 = d (unbiased in expectation under pi; can be negative on individual tokens)\n- k2 = 0.5 * d^2 (nonnegative but generally biased: a second-order approximation for nearby policies)\n- k3 = exp(-d) - 1 + d (unbiased in expectation under pi, nonnegative per token, often lower variance)\n\n**Signature:** `kl_estimators_k1_k2_k3(policy_logps, ref_logps, mask=None) -> (k1, k2, k3)`\n\n**Parameters:**\n- `policy_logps` — token log probabilities under current policy, shape (B, T) or any shared shape\n- `ref_logps` — log probabilities for the SAME sampled tokens under reference, same shape\n- `mask` — optional 0/1 tensor of valid generated positions, same shape\n\n**Returns:** three scalar tensors, each the mean over valid sampled positions; all-zero mask gives connected zeros.\n\n**Constraints:**\n- Detach ref_logps; gradients can flow to policy_logps, but these estimators are primarily **KL value estimators**. Differentiating the mean of on-policy sampled terms does NOT automatically yield the full KL gradient, because the sampling distribution and prefix distribution also depend on the policy.\n- Use torch.expm1(-d) + d for numerically stable k3 near d=0.\n- k1/k3 are unbiased only under the stated current-policy sampling and coverage assumptions. Old-policy/off-policy samples need correction.\n\nReference: John Schulman, \"Approximating KL Divergence\" (2020), http://joschu.net/blog/kl-approx.html.",
    "description_zh": "实现 John Schulman 提出的 k1、k2、k3 KL 估计器：目标是 KL(当前策略 || Reference)，Token 必须从**当前策略**采样。\n\n令 d = log pi(y|x) - log pi_ref(y|x)，r = exp(-d) = pi_ref / pi，则逐 Token：\n- k1 = d：当前策略采样下无偏，但单个样本可以为负、方差可能较大；\n- k2 = 0.5 * d^2：逐 Token 非负，但一般有偏，只在两分布接近时构成二阶近似；\n- k3 = exp(-d) - 1 + d：在当前策略采样下无偏、逐 Token 非负，通常方差较小。\n\n**签名:** `kl_estimators_k1_k2_k3(policy_logps, ref_logps, mask=None) -> (k1, k2, k3)`\n\n**参数:**\n- `policy_logps`：当前策略对已采样 Token 的对数概率，形状 (B, T) 或其他相同形状\n- `ref_logps`：Reference 对**同一批 Token**的对数概率，同形状\n- `mask`：可选 0/1 有效生成位置掩码，同形状\n\n**返回:** 3 个标量 Tensor，分别对有效位置求平均；全零 Mask 返回仍连接 student 梯度图的 0。\n\n**约束:**\n- Reference LogProbs 要 detach；虽然可求 policy_logps 的局部梯度，但这里首先考察 **KL 数值估计**，直接反传采样估计器不一定得到完整 KL 梯度，需考虑采样分布及自回归 Prefix 依赖。\n- k3 使用 `torch.expm1(-d) + d` 改善 d 接近 0 时的数值精度。\n- k1/k3 的无偏性依赖当前策略采样及分布支持条件；旧策略或异步 Rollout 需要分布修正。\n\n参考：John Schulman, \"Approximating KL Divergence\" (2020), http://joschu.net/blog/kl-approx.html。",
    "function_name": "kl_estimators_k1_k2_k3",
    "hint": "Let d = policy_logps - ref_logps.detach(). Return masked means of d, 0.5*d**2, and torch.expm1(-d)+d. Denominator: mask.sum().clamp_min(1). Remember the sampling distribution is policy.",
    "hint_zh": "令 d = policy_logps - ref_logps.detach()，分别计算 d、0.5*d**2 和 torch.expm1(-d)+d 的掩码平均；分母 mask.sum().clamp_min(1)。注意样本来自当前策略。",
    "tests": [
        {
            "name": "Known per-token mathematical definitions",
            "code": """
import torch
d = torch.tensor([0.0, 0.5, -0.8], dtype=torch.float64)
ref = torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float64)
k1, k2, k3 = {fn}(ref + d, ref)
assert torch.allclose(k1, d.mean(), atol=1e-12)
assert torch.allclose(k2, (0.5 * d.square()).mean(), atol=1e-12)
assert torch.allclose(k3, (d.exp().reciprocal() - 1 + d).mean(), atol=1e-12)
""",
        },
        {
            "name": "k1 may be negative, k2 and k3 are pointwise nonnegative",
            "code": """
import torch
policy = torch.tensor([-2.0, -1.0, -0.3], dtype=torch.float64)
ref = torch.tensor([-1.0, -0.5, -0.9], dtype=torch.float64)
d = policy - ref
for i in range(d.numel()):
    a,b,c = {fn}(policy[i:i+1], ref[i:i+1])
    assert torch.allclose(a, d[i], atol=1e-12)
    assert b.item() >= 0 and c.item() >= -1e-12
assert {fn}(policy[:1], ref[:1])[0].item() < 0
""",
        },
        {
            "name": "Weighted categorical samples: k1 and k3 match exact KL",
            "code": """
import torch
q = torch.tensor([0.65, 0.25, 0.10], dtype=torch.float64)
p = torch.tensor([0.30, 0.40, 0.30], dtype=torch.float64)
indices = torch.tensor([0]*65 + [1]*25 + [2]*10)
a,b,c = {fn}(q.log()[indices], p.log()[indices])
exact_kl = (q * (q.log() - p.log())).sum()
assert torch.allclose(a, exact_kl, atol=1e-12)
assert torch.allclose(c, exact_kl, atol=1e-12)
assert abs(b.item() - exact_kl.item()) > 0.001, "k2 is biased in general"
""",
        },
        {
            "name": "Mask averages valid generated positions only",
            "code": """
import torch
policy = torch.tensor([[-0.1, -0.7, -0.3], [-1.5, -0.4, -0.2]])
ref = torch.tensor([[-0.4, -0.9, -0.2], [-1.0, -0.1, -0.5]])
mask = torch.tensor([[1,0,1], [0,1,0]], dtype=torch.bool)
a,b,c = {fn}(policy, ref, mask)
d = (policy-ref)[mask]
assert torch.allclose(a, d.mean())
assert torch.allclose(b, (0.5 * d.square()).mean())
assert torch.allclose(c, (torch.expm1(-d) + d).mean())
""",
        },
        {
            "name": "All-masked returns connected zero; teacher is detached",
            "code": """
import torch
policy = torch.randn(2, 4, requires_grad=True)
ref = torch.randn(2, 4, requires_grad=True)
mask = torch.zeros(2,4, dtype=torch.bool)
out = {fn}(policy, ref, mask)
assert all(x.shape == () and x.item() == 0.0 and x.device == policy.device for x in out)
sum(out).backward()
assert policy.grad is not None and torch.all(policy.grad == 0)
assert ref.grad is None
""",
        },
        {
            "name": "Gradients reach policy but not reference",
            "code": """
import torch
policy = torch.randn(2, 5, requires_grad=True)
ref = torch.randn(2, 5, requires_grad=True)
a,b,c = {fn}(policy, ref)
(a+b+c).backward()
assert policy.grad is not None and policy.grad.abs().sum() > 0
assert ref.grad is None
""",
        },
        {
            "name": "k3 remains accurate for tiny log-ratios in float64",
            "code": """
import torch
d = torch.tensor([1e-5, -1e-5], dtype=torch.float64)
policy = torch.zeros_like(d) + d
ref = torch.zeros_like(d)
_, k2, k3 = {fn}(policy, ref)
assert torch.allclose(k3, k2, rtol=1e-4, atol=1e-12)
assert k3.item() > 0
""",
        },
        {
            "name": "Zero when current and reference log-probs coincide",
            "code": """
import torch
x = torch.randn(3, 5, dtype=torch.float64)
a,b,c = {fn}(x, x.clone())
for value in (a,b,c):
    assert torch.allclose(value, torch.tensor(0.0, dtype=x.dtype), atol=1e-14)
""",
        },
    ],
    "solution": '''import torch

def kl_estimators_k1_k2_k3(policy_logps, ref_logps, mask=None):
    # Samples are assumed drawn from policy, estimating KL(policy || ref).
    d = policy_logps - ref_logps.detach()
    k1 = d
    k2 = 0.5 * d.square()
    k3 = torch.expm1(-d) + d
    if mask is None:
        return k1.mean(), k2.mean(), k3.mean()
    valid = mask.to(device=d.device, dtype=d.dtype).detach()
    denom = valid.sum().clamp_min(1)
    return ((k1 * valid).sum() / denom,
            (k2 * valid).sum() / denom,
            (k3 * valid).sum() / denom)''',
    "demo": """import torch
p = torch.tensor([-0.2, -0.4, -0.8])
r = torch.tensor([-0.3, -0.5, -0.6])
print('k1, k2, k3:', kl_estimators_k1_k2_k3(p, r))""",
}
