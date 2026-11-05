"""Statistical optimization: logistic likelihood, unbiased minibatch gradients.

Pure Python, reproducible, synthetic binary classification only.
Run: python examples/foundation_optimization.py
"""
from __future__ import annotations

import math
import random


def sigmoid(z: float) -> float:
    # Avoid evaluating exp(1000) for a very negative logit.
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    scale = math.exp(z)
    return scale / (1.0 + scale)


def binary_ce_from_logit(z: float, y: int) -> float:
    if type(y) is not int or y not in (0, 1):
        raise ValueError("binary label required")
    return max(z, 0.0) - y * z + math.log1p(math.exp(-abs(z)))


def generate_dataset(seed: int, count: int):
    """Linearly separable latent rule with 5% symmetric label noise."""
    rng = random.Random(seed)
    records = []
    for _ in range(count):
        x1 = rng.uniform(-2.0, 2.0)
        x2 = rng.uniform(-2.0, 2.0)
        original_label = int(1.5 * x1 - 0.7 * x2 + 0.2 > 0)
        label = 1-original_label if rng.random() < 0.05 else original_label
        records.append(((x1, x2), label))
    return records


def logit(theta, x):
    # theta[0:2] weights, theta[2] intercept.
    return theta[0]*x[0] + theta[1]*x[1] + theta[2]


def single_loss_and_grad(theta, record):
    x, y = record
    z = logit(theta, x)
    loss = binary_ce_from_logit(z, y)
    error = sigmoid(z) - y
    return loss, (error*x[0], error*x[1], error)


def risk_and_gradient(theta, data):
    if not data:
        raise ValueError("Empty training data")
    totals = [0.0, 0.0, 0.0]
    losses = []
    for record in data:
        loss, gradient = single_loss_and_grad(theta, record)
        losses.append(loss)
        for j in range(3):
            totals[j] += gradient[j]
    n = len(data)
    return math.fsum(losses)/n, tuple(x/n for x in totals)


def numerical_gradient(theta, data, step=1e-5):
    output = []
    for j in range(3):
        plus = list(theta)
        minus = list(theta)
        plus[j] += step
        minus[j] -= step
        fp = risk_and_gradient(plus, data)[0]
        fm = risk_and_gradient(minus, data)[0]
        output.append((fp-fm)/(2*step))
    return output


def update(theta, gradient, learning_rate):
    return [theta[j]-learning_rate*gradient[j] for j in range(3)]


def train_full_batch(data, iterations=180, learning_rate=0.25):
    theta = [0.0, 0.0, 0.0]
    initial = risk_and_gradient(theta, data)[0]
    for _ in range(iterations):
        _, gradient = risk_and_gradient(theta, data)
        theta = update(theta, gradient, learning_rate)
    final = risk_and_gradient(theta, data)[0]
    return theta, initial, final


def train_minibatch(data, iterations=1600, batch_size=8, learning_rate=0.05,
                    seed=2026):
    theta = [0.0, 0.0, 0.0]
    rng = random.Random(seed)
    for _ in range(iterations):
        batch = [data[rng.randrange(len(data))] for _ in range(batch_size)]
        _, gradient = risk_and_gradient(theta, batch)
        theta = update(theta, gradient, learning_rate)
    return theta


def evaluate_accuracy(theta, data):
    return sum(
        (sigmoid(logit(theta, x)) >= 0.5) == bool(y)
        for x, y in data
    ) / len(data)


def check_gradient_and_stability():
    for label in (0, 1):
        for z in (-1000.0, -3.0, 0.0, 2.0, 1000.0):
            loss = binary_ce_from_logit(z, label)
            assert math.isfinite(loss) and loss >= 0
            if label == 0 and z == 1000:
                assert math.isclose(loss, 1000.0, abs_tol=1e-9)
            if label == 1 and z == 1000:
                assert abs(loss) < 1e-8

    samples = generate_dataset(48, 30)
    theta = [0.4, -0.2, 0.1]
    analytical = risk_and_gradient(theta, samples)[1]
    estimated = numerical_gradient(theta, samples)
    errors = [abs(a-b) for a, b in zip(analytical, estimated)]
    assert max(errors) < 1e-7, errors
    print(f"PASS analytic vs finite-difference gradient: max error={max(errors):.2e}")


def check_minibatch_unbiasedness():
    sample = generate_dataset(27, 12)
    theta = [0.2, 0.1, -0.3]
    full_grad = risk_and_gradient(theta, sample)[1]

    # The exact expectation for a uniformly sampled index is the arithmetic
    # mean of all per-sample gradients: no Monte Carlo assumption needed.
    per_example = [single_loss_and_grad(theta, record)[1]
                   for record in sample]
    exact = tuple(sum(g[j] for g in per_example)/len(per_example)
                  for j in range(3))
    assert all(math.isclose(a, b, abs_tol=1e-12)
               for a, b in zip(full_grad, exact))

    rng = random.Random(11)
    draws = 15000
    observed = [0.0, 0.0, 0.0]
    for _ in range(draws):
        g = per_example[rng.randrange(len(per_example))]
        for j in range(3):
            observed[j] += g[j]
    average = [v/draws for v in observed]
    assert max(abs(a-b) for a, b in zip(average, exact)) < 0.05
    print("PASS exact and Monte Carlo mean of stochastic gradient")


def check_train_and_test():
    train = generate_dataset(2026, 320)
    test = generate_dataset(2027, 120)
    full_theta, before, after = train_full_batch(train)
    sgd_theta = train_minibatch(train)

    test_accuracy_full = evaluate_accuracy(full_theta, test)
    test_accuracy_sgd = evaluate_accuracy(sgd_theta, test)
    assert before > after + 0.2, (before, after)
    assert test_accuracy_full > 0.82, test_accuracy_full
    assert test_accuracy_sgd > 0.78, test_accuracy_sgd

    # A different held-out synthetic test sample is used for evaluation.
    assert len(train) == 320 and len(test) == 120
    print(f"Full batch: training CE {before:.4f} -> {after:.4f}; "
          f"held-out synthetic accuracy {test_accuracy_full:.3f}")
    print(f"Mini batch: held-out synthetic accuracy {test_accuracy_sgd:.3f}")
    print("NOTE: These results are from synthetic data, not an Agent benchmark.")


if __name__ == "__main__":
    check_gradient_and_stability()
    check_minibatch_unbiasedness()
    check_train_and_test()
