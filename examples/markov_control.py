"""Finite-state probability, expected cost and discounted dynamic programming.

Only standard library. All probabilities use ROW-stochastic conventions.
"""
from __future__ import annotations

import math
import random


def matmul(a, b):
    assert a and b and len(a[0]) == len(b)
    return [[sum(a[i][k] * b[k][j] for k in range(len(b)))
             for j in range(len(b[0]))] for i in range(len(a))]


def row_times_matrix(row, matrix):
    assert len(row) == len(matrix)
    return [sum(row[i] * matrix[i][j] for i in range(len(row)))
            for j in range(len(matrix))]


def validate_transition(matrix, tol=1e-12):
    n = len(matrix)
    assert n > 0 and all(len(row) == n for row in matrix)
    for row in matrix:
        assert all(math.isfinite(x) and x >= 0 for x in row)
        assert math.isclose(sum(row), 1, rel_tol=0, abs_tol=tol)


def stationary_two_state(matrix):
    """For p01+p10>0, solve nu P=nu."""
    validate_transition(matrix)
    p01, p10 = matrix[0][1], matrix[1][0]
    if p01 + p10 <= 0:
        raise ValueError("Nonunique stationary distributions")
    return [p10/(p01+p10), p01/(p01+p10)]


def finite_horizon_values(states, actions, transition, cost, terminal, steps):
    """Backward recurrence. V[t][s] = minimum conditional expected remaining cost."""
    values = [dict(terminal)]
    for _ in range(steps):
        prev = values[-1]
        current = {}
        for state in states:
            candidates = [
                cost[(state, action)] +
                sum(prob * prev[next_state]
                    for next_state, prob in transition[(state, action)].items())
                for action in actions[state]
            ]
            current[state] = min(candidates)
        values.append(current)
    return values  # values[0] = horizon zero


def monte_carlo_expected_cost(trials, seed=2026):
    rng = random.Random(seed)
    results = []
    for _ in range(trials):
        recovered = rng.random() < 0.8
        results.append(3 + (0 if recovered else 10))
    mean = sum(results) / trials
    variance = sum((x - mean)**2 for x in results)/(trials - 1)
    return mean, math.sqrt(variance/trials)


def run_checks():
    p = [[0.8, 0.2], [0.3, 0.7]]
    validate_transition(p)
    squared = matmul(p, p)
    for got, expected in zip(squared[0], [0.70, 0.30]):
        assert math.isclose(got, expected, abs_tol=1e-12)
    for got, expected in zip(squared[1], [0.45, 0.55]):
        assert math.isclose(got, expected, abs_tol=1e-12)
    assert row_times_matrix([1, 0], squared) == squared[0]

    stationary = stationary_two_state(p)
    assert all(math.isclose(x, y, abs_tol=1e-12) for x, y in
               zip(stationary, [0.6, 0.4]))
    next_dist = row_times_matrix(stationary, p)
    assert all(math.isclose(x, y, abs_tol=1e-12)
               for x, y in zip(stationary, next_dist))

    # One decision; tomorrow's terminal outage loss is 10.
    states = ["F", "N"]
    actions = {"F": ["inspect", "wait"], "N": ["wait"]}
    trans = {
        ("F", "inspect"): {"N": 0.8, "F": 0.2},
        ("F", "wait"): {"N": 0.3, "F": 0.7},
        ("N", "wait"): {"N": 1.0},
    }
    cost = {("F", "inspect"): 3, ("F", "wait"): 0, ("N", "wait"): 0}
    vs = finite_horizon_values(states, actions, trans, cost,
                               terminal={"F": 10, "N": 0}, steps=1)
    assert math.isclose(vs[1]["F"], 5.0, abs_tol=1e-12)

    mean, se = monte_carlo_expected_cost(25000, seed=2026)
    assert abs(mean - 5.0) < 5 * se + 0.05

    flip = [[0, 1], [1, 0]]
    assert stationary_two_state(flip) == [0.5, 0.5]
    assert row_times_matrix([1, 0], flip) == [0, 1]

    print("PASS transition, distribution, stationary, periodic counterexample")
    print(f"PASS Bellman F: {vs[1]['F']:.1f}; Monte Carlo {mean:.3f} ± {1.96*se:.3f}")


if __name__ == "__main__":
    run_checks()
