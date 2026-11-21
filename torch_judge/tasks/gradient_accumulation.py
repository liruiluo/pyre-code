"""Gradient Accumulation task."""

TASK = {
    "title": "Gradient Accumulation",
    "title_zh": "梯度累积",
    "difficulty": "Easy",
    "description_en": "Implement gradient accumulation over micro-batches.\n\nGradient accumulation simulates a large batch by accumulating gradients from multiple smaller batches before a single optimizer step.\n\n**Signature:** `accumulated_step(model, optimizer, loss_fn, micro_batches) -> float`\n\n**Parameters:**\n- `model` — nn.Module\n- `optimizer` — torch optimizer\n- `loss_fn` — loss function\n- `micro_batches` — list of (x, y) tuples\n\n**Returns:** the accumulated loss as a float, numerically equal to the single full-batch (mean-reduced) loss\n\n**Constraints:**\n- For mean-reduced per-example losses, scale each micro-batch by `micro_batch_size / total_batch_size` before backward (equal-sized batches simplify to `1/n`)\n- Must match a single full-batch update numerically\n- An empty `micro_batches` list must raise `ValueError`",
    "description_zh": "实现微批次梯度累积。\n\n梯度累积通过在多个小批次上累积梯度后执行一次优化器步骤，模拟大批次训练。\n\n**签名:** `accumulated_step(model, optimizer, loss_fn, micro_batches) -> float`\n\n**参数:**\n- `model` — nn.Module\n- `optimizer` — torch 优化器\n- `loss_fn` — 损失函数\n- `micro_batches` — (x, y) 元组列表\n\n**返回:** 累计损失（浮点数），数值上等于单次全批次的 mean loss\n\n**约束:**\n- 采用按样本取 mean 的 loss 时，按 `micro_batch_size / total_batch_size` 缩放各微批次（仅等大小批次可简化为 `1/n`）\n- 必须与单次全批次更新数值一致\n- `micro_batches` 为空列表时必须抛 `ValueError`",
    "function_name": "accumulated_step",
    "hint": "0. raise ValueError first if micro_batches is empty\n1. optimizer.zero_grad() once\n2. for each micro-batch: forward → loss * (micro_batch_size / total_batch_size) → backward\n3. optimizer.step()\n   Dividing by n works only when micro-batches have equal sizes and loss uses mean reduction.",
    "hint_zh": "0. 先校验 `micro_batches` 非空，否则抛 `ValueError`\n1. optimizer.zero_grad() 一次\n2. 对每个微批次：前向 → loss * (micro_batch_size / total_batch_size) → 反向\n3. optimizer.step()\n   仅当微批次等大小且 loss 使用 mean reduction 时才能直接除以 n。",
    "tests": [
        {
            "name": "Matches full batch update",
            "code": "\nimport torch, torch.nn as nn\ntorch.manual_seed(0)\nmodel = nn.Linear(4, 2, bias=False)\nmodel_ref = nn.Linear(4, 2, bias=False)\nmodel_ref.load_state_dict(model.state_dict())\nloss_fn = nn.MSELoss()\nopt = torch.optim.SGD(model.parameters(), lr=0.1)\nopt_ref = torch.optim.SGD(model_ref.parameters(), lr=0.1)\nx1, y1 = torch.randn(2, 4), torch.randn(2, 2)\nx2, y2 = torch.randn(2, 4), torch.randn(2, 2)\n{fn}(model, opt, loss_fn, [(x1, y1), (x2, y2)])\nopt_ref.zero_grad()\nloss_ref = loss_fn(model_ref(torch.cat([x1, x2])), torch.cat([y1, y2]))\nloss_ref.backward()\nopt_ref.step()\nassert torch.allclose(model.weight.data, model_ref.weight.data, atol=1e-5), 'Must match full batch'\n"
        },
        {
            "name": "Returns loss value",
            "code": "\nimport torch, torch.nn as nn\nmodel = nn.Linear(4, 2)\nopt = torch.optim.SGD(model.parameters(), lr=0.01)\nloss = {fn}(model, opt, nn.MSELoss(), [(torch.randn(2, 4), torch.randn(2, 2))])\nassert isinstance(loss, float), f'Should return float, got {type(loss)}'\nassert loss > 0, 'Loss should be positive'\n"
        },
        {
            "name": "Parameters actually update",
            "code": "\nimport torch, torch.nn as nn\nmodel = nn.Linear(4, 2)\nopt = torch.optim.SGD(model.parameters(), lr=0.1)\nw_before = model.weight.data.clone()\n{fn}(model, opt, nn.MSELoss(), [(torch.randn(2, 4), torch.randn(2, 2))])\nassert not torch.equal(model.weight.data, w_before), 'Should change'\n"
        },
        {
            "name": "Unequal micro-batches match a single full-batch update",
            "code": """
import torch
import torch.nn as nn
torch.manual_seed(12)
model = nn.Linear(4, 2, bias=False)
ref = nn.Linear(4, 2, bias=False)
ref.load_state_dict(model.state_dict())
x1, y1 = torch.randn(1, 4), torch.randn(1, 2)
x2, y2 = torch.randn(5, 4), torch.randn(5, 2)
loss_fn = nn.MSELoss()
opt = torch.optim.SGD(model.parameters(), lr=0.1)
opt_ref = torch.optim.SGD(ref.parameters(), lr=0.1)
loss = {fn}(model, opt, loss_fn, [(x1, y1), (x2, y2)])
opt_ref.zero_grad()
expected_loss = loss_fn(ref(torch.cat([x1, x2])), torch.cat([y1, y2]))
expected_loss.backward()
opt_ref.step()
assert torch.allclose(model.weight, ref.weight, atol=1e-6)
assert abs(loss - expected_loss.item()) < 1e-6
""",
        },
        {
            "name": "Empty micro-batch list is rejected",
            "code": """
import torch
import torch.nn as nn
model = nn.Linear(2, 1)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
try:
    {fn}(model, opt, nn.MSELoss(), [])
except ValueError:
    pass
else:
    raise AssertionError("Empty accumulation should raise ValueError")
""",
        },
    ],
    "solution": '''def accumulated_step(model, optimizer, loss_fn, micro_batches):
    if not micro_batches:
        raise ValueError("micro_batches must not be empty")
    total_samples = sum(x.shape[0] for x, _ in micro_batches)
    if total_samples <= 0:
        raise ValueError("micro_batches must contain samples")
    optimizer.zero_grad()
    total_loss = 0.0
    for x, y in micro_batches:
        if x.shape[0] == 0:
            continue
        # For mean-reduced losses, weight by the number of examples.
        weight = x.shape[0] / total_samples
        loss = loss_fn(model(x), y) * weight
        loss.backward()
        total_loss += loss.item()
    optimizer.step()
    return total_loss''',
    "demo": """model = nn.Linear(4, 2)
opt = torch.optim.SGD(model.parameters(), lr=0.01)
loss = accumulated_step(model, opt, nn.MSELoss(),
    [(torch.randn(2, 4), torch.randn(2, 2)) for _ in range(4)])
print('Accumulated loss:', loss)""",

}