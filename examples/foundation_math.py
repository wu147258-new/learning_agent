"""Beginner math exercises: linear algebra, differentiation and probability.

Requires Python standard library only. Run:
  python examples/foundation_math.py

All numerical examples are synthetic, not observations about real Agent systems.
"""
from __future__ import annotations

import math
import random


def solve_two_by_two(a, b, c, d, e, f):
    """Solve ax+by=e, cx+dy=f with nonzero determinant."""
    det = a * d - b * c
    if det == 0:
        raise ValueError("singular coefficient matrix")
    x = (e * d - b * f) / det
    y = (a * f - e * c) / det
    return x, y


def dot(a, b):
    if len(a) != len(b):
        raise ValueError("dimension mismatch")
    return sum(x * y for x, y in zip(a, b))


def loss(w, b, x, y):
    error = w * x + b - y
    return 0.5 * error * error


def analytic_gradient(w, b, x, y):
    error = w * x + b - y
    return error * x, error


def numerical_gradient(w, b, x, y, epsilon=1e-5):
    dw = (loss(w+epsilon, b, x, y) -
          loss(w-epsilon, b, x, y)) / (2*epsilon)
    db = (loss(w, b+epsilon, x, y) -
          loss(w, b-epsilon, x, y)) / (2*epsilon)
    return dw, db


def softmax(logits):
    if not logits:
        raise ValueError("empty logit list")
    if not all(math.isfinite(x) for x in logits):
        raise ValueError("nonfinite logit")
    shift = max(logits)
    weights = [math.exp(x - shift) for x in logits]
    total = sum(weights)
    return [weight / total for weight in weights]


def posterior_positive(base_rate, detection, false_alarm):
    """P(true|+) from P(true), P(+|true), P(+|not true)."""
    if not all(0 <= v <= 1 for v in
               (base_rate, detection, false_alarm)):
        raise ValueError("probabilities must be in [0,1]")
    numerator = base_rate * detection
    denominator = numerator + (1-base_rate) * false_alarm
    if denominator == 0:
        raise ValueError("positive evidence has probability zero")
    return numerator / denominator


def row_matmul(row, matrix):
    n = len(row)
    if n == 0 or len(matrix) != n or any(len(r) != n for r in matrix):
        raise ValueError("matrix shape mismatch")
    for r in matrix:
        if any(p < 0 for p in r) or not math.isclose(
            sum(r), 1.0, abs_tol=1e-12
        ):
            raise ValueError("invalid transition probabilities")
    return [sum(row[i] * matrix[i][j] for i in range(n))
            for j in range(n)]


def simulate_positive_posterior(trials, seed=2026):
    rng = random.Random(seed)
    positives, true_positives = 0, 0
    for _ in range(trials):
        is_threat = rng.random() < 0.01
        alarm = rng.random() < (0.90 if is_threat else 0.05)
        if alarm:
            positives += 1
            if is_threat:
                true_positives += 1
    return true_positives / positives if positives else math.nan


def self_test():
    x, y = solve_two_by_two(2, 1, 1, 3, 5, 7)
    assert math.isclose(x, 1.6, abs_tol=1e-12)
    assert math.isclose(y, 1.8, abs_tol=1e-12)
    assert dot([1, 2, 3], [4, 5, 6]) == 32

    analytic = analytic_gradient(1, 0, 2, 5)
    numeric = numerical_gradient(1, 0, 2, 5)
    assert analytic == (-6, -3)
    assert all(math.isclose(a, b, abs_tol=1e-8)
               for a, b in zip(analytic, numeric))
    assert math.isclose(loss(1, 0, 2, 5), 4.5)
    assert math.isclose(loss(1.6, 0.3, 2, 5), 1.125)

    probabilities = softmax([1000, 1001, 1002])
    assert math.isclose(sum(probabilities), 1.0, abs_tol=1e-12)
    assert probabilities[0] < probabilities[1] < probabilities[2]
    assert all(math.isclose(a, b, abs_tol=1e-12)
               for a, b in zip(probabilities, softmax([0, 1, 2])))

    posterior = posterior_positive(0.01, 0.90, 0.05)
    assert math.isclose(posterior, 0.009/0.0585, abs_tol=1e-12)
    assert math.isclose(posterior, 0.153846153846, abs_tol=1e-10)
    expectation = 0.5*1 + 0.3*3 + 0.2*10
    variance = (0.5*1**2 + 0.3*3**2 + 0.2*10**2) - expectation**2
    assert math.isclose(expectation, 3.4)
    assert math.isclose(variance, 11.64, abs_tol=1e-12)

    P = [[0.8, 0.2], [0.3, 0.7]]
    step1 = row_matmul([1, 0], P)
    step2 = row_matmul(step1, P)
    assert all(math.isclose(a, b, abs_tol=1e-12)
               for a, b in zip(step2, [0.7, 0.3]))

    simulated = simulate_positive_posterior(50000)
    assert abs(simulated - posterior) < 0.03
    try:
        solve_two_by_two(1, 2, 2, 4, 3, 6)
        raise AssertionError("singular matrix was not rejected")
    except ValueError:
        pass
    print("PASS: systems, gradients, stable softmax, Bayes, moments, Markov")
    print(f"Bayes prediction: {posterior:.4f}; synthetic sample: {simulated:.4f}")


if __name__ == "__main__":
    self_test()
