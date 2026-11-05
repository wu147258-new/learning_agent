"""Calculus and automatic-differentiation foundations, pure Python.

Deterministic, zero network, no third-party packages. Run:
    python examples/foundation_calculus.py
"""
from __future__ import annotations

import math


def central_difference(func, x: float, h: float = 1e-5) -> float:
    if h <= 0 or not math.isfinite(h):
        raise ValueError("finite difference h must be positive and finite")
    return (func(x + h) - func(x - h)) / (2 * h)


def scalar_loss(w: float, b: float, x: float, y: float) -> float:
    error = w * x + b - y
    return 0.5 * error * error


def analytic_linear_grad(w: float, b: float, x: float, y: float):
    error = w * x + b - y
    return error * x, error


def numerical_linear_grad(w: float, b: float, x: float, y: float):
    return (
        central_difference(lambda z: scalar_loss(z, b, x, y), w),
        central_difference(lambda z: scalar_loss(w, z, x, y), b),
    )


def single_gradient_step(w, b, x, y, learning_rate):
    grad_w, grad_b = analytic_linear_grad(w, b, x, y)
    return w - learning_rate * grad_w, b - learning_rate * grad_b


def stable_logsumexp(values):
    if not values:
        raise ValueError("logsumexp requires nonempty sequence")
    if any(not math.isfinite(v) for v in values):
        raise ValueError("nonfinite input")
    pivot = max(values)
    return pivot + math.log(math.fsum(math.exp(v - pivot) for v in values))


def stable_softmax(values):
    log_normalizer = stable_logsumexp(values)
    probabilities = [math.exp(v - log_normalizer) for v in values]
    return probabilities


def cross_entropy_from_logits(logits, label):
    if type(label) is not int or not (0 <= label < len(logits)):
        raise ValueError("invalid class label")
    return stable_logsumexp(logits) - logits[label]


def analytic_cross_entropy_grad(logits, label):
    probs = stable_softmax(logits)
    return [p - (1 if index == label else 0)
            for index, p in enumerate(probs)]


def numerical_cross_entropy_grad(logits, label):
    results = []
    for index in range(len(logits)):
        def f(value, index=index):
            copied = list(logits)
            copied[index] = value
            return cross_entropy_from_logits(copied, label)
        results.append(central_difference(f, logits[index]))
    return results


def matrix_linear_grad(x, W, bias, target):
    """z = W x + bias, L=0.5 ||z-target||^2.

    x[d], W[m][d], bias[m], target[m]
    Return L, dL/dW[m][d], dL/dx[d].
    """
    d = len(x)
    m = len(W)
    if not d or not m or len(bias) != m or len(target) != m:
        raise ValueError("bad dimensions")
    if any(len(row) != d for row in W):
        raise ValueError("bad weight matrix shape")
    prediction = [
        sum(row[j] * x[j] for j in range(d)) + bias[i]
        for i, row in enumerate(W)
    ]
    error = [prediction[i] - target[i] for i in range(m)]
    loss = 0.5 * sum(e * e for e in error)
    grad_w = [[error[i] * x[j] for j in range(d)] for i in range(m)]
    grad_x = [sum(W[i][j] * error[i] for i in range(m))
              for j in range(d)]
    return loss, grad_w, grad_x


def check_derivatives():
    for x in (-3.5, -0.5, 0.0, 2.0, 10.0):
        estimate = central_difference(lambda t: t*t, x)
        assert math.isclose(estimate, 2*x, rel_tol=1e-8, abs_tol=1e-8)
        estimate_cubic = central_difference(lambda t: 3*t**3 - 2*t, x)
        assert math.isclose(estimate_cubic, 9*x*x - 2,
                            rel_tol=1e-7, abs_tol=1e-7)

    w, b, x, y = 1.0, 0.0, 2.0, 5.0
    grad = analytic_linear_grad(w, b, x, y)
    num = numerical_linear_grad(w, b, x, y)
    assert grad == (-6.0, -3.0)
    assert all(math.isclose(a, v, abs_tol=1e-7)
               for a, v in zip(grad, num))
    assert math.isclose(scalar_loss(w, b, x, y), 4.5)
    moderate = single_gradient_step(w, b, x, y, learning_rate=0.1)
    aggressive = single_gradient_step(w, b, x, y, learning_rate=1.0)
    assert all(math.isclose(a, b, abs_tol=1e-12)
               for a, b in zip(moderate, (1.6, 0.3)))
    assert all(math.isclose(a, b, abs_tol=1e-12)
               for a, b in zip(aggressive, (7.0, 3.0)))
    assert math.isclose(scalar_loss(*moderate, x, y), 1.125)
    assert math.isclose(scalar_loss(*aggressive, x, y), 72.0)


def check_matrix_gradients():
    x = [1.0, 2.0]
    W = [[1.0, -1.0], [0.5, 2.0]]
    bias = [0.2, -0.3]
    target = [2.0, 3.0]
    loss, grad_w, grad_x = matrix_linear_grad(x, W, bias, target)
    assert len(grad_w) == 2 and all(len(row) == 2 for row in grad_w)
    assert len(grad_x) == 2
    for i in range(len(W)):
        for j in range(len(x)):
            def perturbed(value, i=i, j=j):
                changed = [list(row) for row in W]
                changed[i][j] = value
                return matrix_linear_grad(x, changed, bias, target)[0]
            estimate = central_difference(perturbed, W[i][j])
            assert math.isclose(
                grad_w[i][j], estimate, abs_tol=1e-7, rel_tol=1e-7
            )
    for j in range(len(x)):
        def perturbed(value, j=j):
            changed = list(x)
            changed[j] = value
            return matrix_linear_grad(changed, W, bias, target)[0]
        estimate = central_difference(perturbed, x[j])
        assert math.isclose(grad_x[j], estimate, abs_tol=1e-7,
                            rel_tol=1e-7)
    print(f"PASS matrix calculus: loss={loss:.6f}, all gradients match")


def check_probability_gradients():
    logits = [2.0, 1.0, 0.0]
    label = 0
    analytic = analytic_cross_entropy_grad(logits, label)
    numeric = numerical_cross_entropy_grad(logits, label)
    assert math.isclose(sum(stable_softmax(logits)), 1, abs_tol=1e-12)
    assert math.isclose(sum(analytic), 0, abs_tol=1e-12)
    assert all(math.isclose(a, n, abs_tol=1e-7, rel_tol=1e-7)
               for a, n in zip(analytic, numeric))
    offset = 10000.0
    shifted = [x + offset for x in logits]
    assert all(math.isclose(a, b, abs_tol=1e-10)
               for a, b in zip(stable_softmax(logits),
                               stable_softmax(shifted)))
    assert math.isclose(cross_entropy_from_logits(logits, label),
                        cross_entropy_from_logits(shifted, label),
                        abs_tol=1e-10)
    assert math.isclose(
        central_difference(lambda z: math.log1p(math.exp(z)), 0),
        0.5, abs_tol=1e-8
    )


def check_nonconvex_example():
    def f(x):
        return (x*x - 1)**2
    for x in (-1.0, 1.0):
        assert f(x) == 0
        assert abs(central_difference(f, x)) < 1e-7
    assert math.isclose(central_difference(f, 0.0), 0, abs_tol=1e-10)
    assert f(0.0) > f(0.5)


if __name__ == "__main__":
    check_derivatives()
    check_matrix_gradients()
    check_probability_gradients()
    check_nonconvex_example()
    print("PASS scalar derivatives, linear gradients, learning-rate counterexample")
    print("PASS stable softmax/logsumexp, CE gradient, nonconvex stationary point")
