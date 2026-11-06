"""Exact batch weighted gradient reduction without GPU dependencies.

All data are synthetic. Not a networking or distributed-runtime benchmark.
Run: python3 examples/distributed_gradient_lab.py
"""
from __future__ import annotations
from math import isclose

DATA = [([1.0, 0.0], 1.0),
        ([0.0, 1.0], 2.0),
        ([1.0, 1.0], 3.0),
        ([2.0, 1.0], 4.0),
        ([1.0, 2.0], 5.0),
        ([3.0, 1.0], 5.0)]


def gradient(samples, weights):
    if not samples:
        raise ValueError("empty microbatch")
    if not weights:
        raise ValueError("empty weights")
    out = [0.0] * len(weights)
    for x, y in samples:
        if len(x) != len(weights):
            raise ValueError("feature dimension mismatch")
        residual = sum(a*b for a, b in zip(x, weights)) - y
        for j, v in enumerate(x):
            out[j] += residual*v
    return [g / len(samples) for g in out]


def weighted_reduce(groups, weights):
    if not groups or any(not group for group in groups):
        raise ValueError("empty worker")
    n = sum(len(group) for group in groups)
    means = [gradient(group, weights) for group in groups]
    return [sum(len(g) * avg[j] for g, avg in zip(groups, means)) / n
            for j in range(len(weights))]


def loss(samples, weights):
    return sum((sum(a*b for a, b in zip(x, weights))-y)**2
               for x, y in samples) / (2 * len(samples))


def main():
    weights = [0.0, 0.0]
    groups = [DATA[:1], DATA[1:3], DATA[3:]]
    exact = gradient(DATA, weights)
    reduced = weighted_reduce(groups, weights)
    for x, y in zip(exact, reduced):
        assert isclose(x, y, abs_tol=1e-12)
    # This incorrect reduction assigns equal weight to each worker.
    naive_means = [gradient(g, weights) for g in groups]
    naive = [sum(g[j] for g in naive_means) / len(groups)
             for j in range(len(weights))]
    assert any(not isclose(a, b, abs_tol=1e-6)
               for a, b in zip(naive, exact))

    before = loss(DATA, weights)
    eta = 0.05
    after_weights = [w - eta*g for w, g in zip(weights, exact)]
    after = loss(DATA, after_weights)
    assert after < before
    try:
        gradient([], weights)
    except ValueError:
        pass
    else:
        raise AssertionError("empty batch accepted")
    try:
        gradient([([1.0], 1.0)], weights)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid feature width accepted")
    print("PASS weighted worker gradients equal global mean")
    print("PASS unequal worker means cannot be averaged uniformly")
    print(f"PASS loss decreases: {before:.6f} -> {after:.6f}")
    print("NOTE: exact synthetic CPU arithmetic; no GPU measurements.")


if __name__ == "__main__":
    main()
