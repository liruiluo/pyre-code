"""Linear Regression Three Ways task."""

TASK = {
    "title": "Linear Regression",
    "title_zh": "线性回归",
    "difficulty": "Medium",
    "description_en": "Implement linear regression three ways: closed-form, gradient descent, and nn.Linear.\n\nThis task covers the fundamental approaches to fitting a linear model, from the normal equation to autograd-based training.\n\n**Signature:** `LinearRegression()` (class)\n\n**Methods:**\n- `closed_form(X, y) -> (w, b)` — solve via normal equation\n- `gradient_descent(X, y, lr, steps) -> (w, b)` — manual gradient updates\n- `nn_linear(X, y, lr, steps) -> (w, b)` — PyTorch autograd training loop\n\n**Constraints:**\n- All three methods should converge to similar weights\n- Closed-form should not use autograd\n- X is (N, D), y is (N,), returns w (D,) and b (scalar)\n- Preserve `X.dtype` / `X.device` in all three methods: build the ones column and `w`/`b` with `dtype=X.dtype, device=X.device`, and construct `nn.Linear(D, 1, dtype=X.dtype, device=X.device)`. float64 input must stay float64",
    "description_zh": "用三种方式实现线性回归：闭式解、梯度下降和 nn.Linear。\n\n本题涵盖拟合线性模型的基本方法，从正规方程到基于 autograd 的训练。\n\n**签名:** `LinearRegression()`（类）\n\n**方法:**\n- `closed_form(X, y) -> (w, b)` — 通过正规方程求解\n- `gradient_descent(X, y, lr, steps) -> (w, b)` — 手动梯度更新\n- `nn_linear(X, y, lr, steps) -> (w, b)` — PyTorch autograd 训练循环\n\n**约束:**\n- 三种方法应收敛到相近的权重\n- 闭式解不应使用 autograd\n- X 为 (N, D)，y 为 (N,)，返回 w (D,) 和 b（标量）\n- 三个方法都要保持 `X.dtype` / `X.device`：ones 列与 `w`/`b` 用 `dtype=X.dtype, device=X.device` 构造，`nn.Linear(D, 1, dtype=X.dtype, device=X.device)`。float64 输入必须还是 float64",
    "function_name": "LinearRegression",
    "hint": "closed_form: augment X with ones col → `lstsq(X_aug, y)` → split `w, b`\ngradient_descent: `grad_w = (2/N) * X.T @ (X@w+b - y)`, `w -= lr*grad_w`\nnn_linear: `nn.Linear(D, 1, dtype=X.dtype, device=X.device)` + `MSELoss` + `optimizer.step()` loop\nAll three: keep X's dtype/device (ones column, w, b); float64 in, float64 out",
    "hint_zh": "closed_form：X 增加全 1 列 → `lstsq(X_aug, y)` → 拆分 `w, b`\ngradient_descent：`grad_w = (2/N) * X.T @ (X@w+b - y)`，`w -= lr*grad_w`\nnn_linear：`nn.Linear(D, 1, dtype=X.dtype, device=X.device)` + `MSELoss` + `optimizer.step()` 循环\n三个方法都要跟住 X 的 dtype/device（ones 列、w、b）；float64 进 float64 出",
    "tests": [
        {
            "name": "Closed-form returns correct shapes",
            "code": """
import torch
torch.manual_seed(42)
X = torch.randn(50, 3)
y = X @ torch.tensor([2.0, -1.0, 0.5]) + 3.0 + torch.randn(50) * 0.01
model = {fn}()
w, b = model.closed_form(X, y)
assert w.shape == (3,), f'w shape: {w.shape}, expected (3,)'
assert b.shape == (), f'b shape: {b.shape}, expected scalar'
""",
        },
        {
            "name": "Closed-form finds correct weights",
            "code": """
import torch
torch.manual_seed(42)
true_w = torch.tensor([2.0, -1.0, 0.5])
true_b = 3.0
X = torch.randn(100, 3)
y = X @ true_w + true_b
model = {fn}()
w, b = model.closed_form(X, y)
assert torch.allclose(w, true_w, atol=1e-4), f'w: {w} vs true: {true_w}'
assert torch.allclose(b, torch.tensor(true_b), atol=1e-4), f'b: {b.item():.4f} vs true: {true_b}'
""",
        },
        {
            "name": "Gradient descent converges",
            "code": """
import torch
torch.manual_seed(42)
true_w = torch.tensor([2.0, -1.0, 0.5])
true_b = 3.0
X = torch.randn(100, 3)
y = X @ true_w + true_b
model = {fn}()
w, b = model.gradient_descent(X, y, lr=0.05, steps=2000)
assert torch.allclose(w, true_w, atol=0.1), f'GD w: {w} vs true: {true_w}'
assert abs(b.item() - true_b) < 0.1, f'GD b: {b.item():.4f} vs true: {true_b}'
""",
        },
        {
            "name": "nn.Linear approach works",
            "code": """
import torch
torch.manual_seed(42)
true_w = torch.tensor([2.0, -1.0, 0.5])
true_b = 3.0
X = torch.randn(100, 3)
y = X @ true_w + true_b
model = {fn}()
w, b = model.nn_linear(X, y, lr=0.05, steps=2000)
assert torch.allclose(w, true_w, atol=0.1), f'nn w: {w} vs true: {true_w}'
assert abs(b.item() - true_b) < 0.1, f'nn b: {b.item():.4f} vs true: {true_b}'
""",
        },
        {
            "name": "All three methods agree",
            "code": """
import torch
torch.manual_seed(0)
X = torch.randn(200, 2)
true_w = torch.tensor([1.5, -2.0])
y = X @ true_w + 1.0 + torch.randn(200) * 0.1
model = {fn}()
w_cf, b_cf = model.closed_form(X, y)
w_gd, b_gd = model.gradient_descent(X, y, lr=0.05, steps=3000)
w_nn, b_nn = model.nn_linear(X, y, lr=0.05, steps=3000)
assert torch.allclose(w_cf, w_gd, atol=0.15), f'CF vs GD: max diff {(w_cf - w_gd).abs().max():.4f}'
assert torch.allclose(w_cf, w_nn, atol=0.15), f'CF vs NN: max diff {(w_cf - w_nn).abs().max():.4f}'
assert abs(b_cf.item() - b_gd.item()) < 0.15, f'Bias CF vs GD: {b_cf.item():.4f} vs {b_gd.item():.4f}'
assert abs(b_cf.item() - b_nn.item()) < 0.15, f'Bias CF vs NN: {b_cf.item():.4f} vs {b_nn.item():.4f}'
""",
        },
        {
            "name": "Closed-form uses no autograd",
            "code": """
import torch
X = torch.randn(30, 2)
y = X @ torch.tensor([1.0, 2.0]) + 0.5
model = {fn}()
w, b = model.closed_form(X, y)
assert not w.requires_grad, 'Closed-form w should not require grad'
""",
        },
        {
            "name": "float64 inputs work in all three methods",
            "code": """
import torch
torch.manual_seed(9)
X = torch.randn(60, 3, dtype=torch.float64)
w_true = torch.tensor([1.2, -0.4, 0.7], dtype=torch.float64)
y = X @ w_true + 0.8
m = {fn}()
for method in (m.closed_form, m.gradient_descent, m.nn_linear):
    kw = {} if method == m.closed_form else {"lr": 0.05, "steps": 600}
    w, b = method(X, y, **kw)
    assert w.dtype == X.dtype and b.dtype == X.dtype
    assert torch.allclose(w, w_true, atol=0.08)
    assert abs(b.item() - 0.8) < 0.08
""",
        },
    ],
    "solution": '''class LinearRegression:
    def closed_form(self, X: torch.Tensor, y: torch.Tensor):
        n, d = X.shape
        X_aug = torch.cat([X, torch.ones(n, 1, device=X.device, dtype=X.dtype)], dim=1)
        theta = torch.linalg.lstsq(X_aug, y).solution
        return theta[:d].detach(), theta[d].detach()

    def gradient_descent(self, X: torch.Tensor, y: torch.Tensor, lr=0.01, steps=1000):
        n, d = X.shape
        w = torch.zeros(d, device=X.device, dtype=X.dtype)
        b = torch.zeros((), device=X.device, dtype=X.dtype)
        for _ in range(steps):
            err = X @ w + b - y
            w = w - lr * (2.0 / n) * (X.T @ err)
            b = b - lr * (2.0 / n) * err.sum()
        return w, b

    def nn_linear(self, X: torch.Tensor, y: torch.Tensor, lr=0.01, steps=1000):
        layer = nn.Linear(X.shape[1], 1, device=X.device, dtype=X.dtype)
        optimizer = torch.optim.SGD(layer.parameters(), lr=lr)
        for _ in range(steps):
            optimizer.zero_grad()
            loss = F.mse_loss(layer(X).squeeze(-1), y)
            loss.backward()
            optimizer.step()
        return layer.weight.detach().squeeze(0).clone(), layer.bias.detach().squeeze(0).clone()''',
    "demo": """torch.manual_seed(42)
X = torch.randn(100, 3)
true_w = torch.tensor([2.0, -1.0, 0.5])
y = X @ true_w + 3.0

model = LinearRegression()
for name, method in [("Closed-form", model.closed_form),
                      ("Grad Descent", lambda X, y: model.gradient_descent(X, y, lr=0.05, steps=2000)),
                      ("nn.Linear", lambda X, y: model.nn_linear(X, y, lr=0.05, steps=2000))]:
    w, b = method(X, y)
    print(f"{name:13s}  w={w.tolist()}  b={b.item():.4f}")
print(f"{'True':13s}  w={true_w.tolist()}  b=3.0000")""",

}