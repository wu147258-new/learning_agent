"""Exact DPO-style preference objective and independent gradient checks.

Synthetic categorical responses only. No language-model training or online
reward modeling. Run: python3 examples/dpo_gradient_oracle.py
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, isclose
from hashlib import sha256
from random import Random


@dataclass(frozen=True)
class Preference:
    prompt_id: str
    group: str
    chosen: str
    rejected: str
    provenance: str


def stable_softplus(x: float) -> float:
    if x > 0:
        return x + log(1 + exp(-x))
    return log(1 + exp(x))


def logsigmoid(x: float) -> float:
    return -stable_softplus(-x)


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1 / (1 + exp(-x))
    z = exp(x)
    return z / (1 + z)


def log_softmax(logits: list[float]) -> list[float]:
    if not logits:
        raise ValueError("logits must be nonempty")
    m = max(logits)
    den = m + log(sum(exp(x - m) for x in logits))
    return [x-den for x in logits]


def dpo_loss(
    policy_logps: list[float],
    ref_logps: list[float],
    winner: int,
    loser: int,
    beta: float = 0.1,
) -> float:
    if (len(policy_logps) != len(ref_logps) or
            not 0 <= winner < len(policy_logps) or
            not 0 <= loser < len(policy_logps) or winner == loser):
        raise ValueError("invalid probabilities or chosen/rejected indices")
    if beta <= 0:
        raise ValueError("beta must be positive")
    difference = (
        (policy_logps[winner] - ref_logps[winner])
        - (policy_logps[loser] - ref_logps[loser])
    )
    return stable_softplus(-beta * difference)


def categorical_loss(
    logits: list[float], reference_logits: list[float],
    winner=0, loser=1, beta=0.1
) -> float:
    return dpo_loss(
        log_softmax(logits), log_softmax(reference_logits),
        winner, loser, beta
    )


def analytic_categorical_grad(
    logits, reference_logits, winner=0, loser=1, beta=0.1
) -> list[float]:
    logp = log_softmax(logits)
    logr = log_softmax(reference_logits)
    delta = ((logp[winner] - logr[winner])
             - (logp[loser] - logr[loser]))
    g = beta * (sigmoid(beta * delta) - 1)
    out = [0.0] * len(logits)
    out[winner] = g
    out[loser] = -g
    return out


def finite_diff(f, values: list[float], epsilon=1e-5):
    out = []
    for i in range(len(values)):
        left, right = values[:], values[:]
        left[i] -= epsilon
        right[i] += epsilon
        out.append((f(right) - f(left)) / (2*epsilon))
    return out


def conditional_pair_probability(utility_good, utility_bad, temp=1.0):
    if temp <= 0:
        raise ValueError("temperature must be positive")
    return sigmoid((utility_good - utility_bad) / temp)


def clean_pairs(pairs: list[Preference], allowed_provenance):
    seen = set()
    accepted, rejected = [], []
    for pair in pairs:
        if pair.provenance not in allowed_provenance:
            rejected.append((pair.prompt_id, "unapproved source"))
            continue
        if pair.chosen == pair.rejected or not pair.prompt_id:
            rejected.append((pair.prompt_id, "invalid contrast"))
            continue
        key = (pair.prompt_id, pair.chosen, pair.rejected)
        if key in seen:
            rejected.append((pair.prompt_id, "duplicate"))
            continue
        seen.add(key)
        accepted.append(pair)
    return accepted, rejected


def split_groups(pairs: list[Preference], train_fraction=0.7, seed=41):
    groups = sorted({pair.group for pair in pairs})
    if len(groups) < 2:
        raise ValueError("requires two groups")
    rng = Random(seed)
    rng.shuffle(groups)
    n = max(1, min(len(groups)-1, round(len(groups)*train_fraction)))
    train_groups = set(groups[:n])
    train = [p for p in pairs if p.group in train_groups]
    test = [p for p in pairs if p.group not in train_groups]
    assert {x.group for x in train}.isdisjoint(
        {x.group for x in test}
    )
    return train, test


def optimize_categorical(
    starting_logits: list[float], reference: list[float],
    chosen: int, rejected: int,
    steps=50, learning_rate=0.3, beta=0.2,
):
    theta = starting_logits[:]
    history = []
    for _ in range(steps):
        loss = categorical_loss(theta, reference, chosen, rejected, beta)
        history.append(loss)
        grad = analytic_categorical_grad(
            theta, reference, chosen, rejected, beta
        )
        theta = [v - learning_rate * g
                 for v, g in zip(theta, grad)]
    return theta, history


def score_response_tokens(
    token_logps: list[float], valid_mask: list[int],
    length_normalized=False,
):
    if len(token_logps) != len(valid_mask):
        raise ValueError("mask shape mismatch")
    count = sum(valid_mask)
    if not count or any(x not in (0, 1) for x in valid_mask):
        raise ValueError("invalid supervision mask")
    value = sum(v for v, flag in zip(token_logps, valid_mask) if flag)
    return value/count if length_normalized else value


def judge_checks():
    # The underlying examples are synthetic, not human preference data.
    pairs = [
        Preference("p1", "g1", "correct", "incorrect", "CC0"),
        Preference("p2", "g1", "safe", "unsafe", "CC0"),
        Preference("p3", "g2", "2", "3", "CC0"),
        Preference("p4", "g3", "proved", "guessed", "CC0"),
        Preference("p4", "g3", "proved", "guessed", "CC0"),
        Preference("p5", "g4", "same", "same", "CC0"),
        Preference("p6", "g5", "secret", "other", "restricted"),
    ]
    accepted, rejected = clean_pairs(pairs, {"CC0"})
    assert len(accepted) == 4
    assert len(rejected) == 3
    train, test = split_groups(accepted)
    assert train and test
    assert all(p.provenance == "CC0" for p in accepted)
    print("PASS pair validity, provenance and grouped split")


def gradient_checks():
    reference = [0.2, -0.4, 0.8, 0.0]
    policy = [1.2, -1.0, 0.5, -0.1]
    for beta in (0.01, 0.1, 1.0, 5.0):
        f = lambda z: categorical_loss(z, reference, 0, 1, beta)
        numeric = finite_diff(f, policy)
        analytic = analytic_categorical_grad(
            policy, reference, 0, 1, beta
        )
        for a, b in zip(analytic, numeric):
            assert isclose(a, b, abs_tol=1e-8, rel_tol=1e-8), (
                beta, a, b
            )
        assert analytic[2] == 0.0 and analytic[3] == 0.0
        assert isclose(sum(analytic), 0.0, abs_tol=1e-15)
    assert logsigmoid(0.0) == -log(2)
    assert conditional_pair_probability(10, -10) > 0.999
    print("PASS exact DPO categorical gradient vs finite differences")


def optimization_checks():
    ref = [0.0, 0.0, 0.0]
    optimized, history = optimize_categorical(
        [0., 0., 0.], ref, 0, 1, steps=80
    )
    assert history[-1] < history[0]
    assert optimized[0] > optimized[1]
    assert all(
        history[i+1] <= history[i] + 1e-12
        for i in range(len(history)-1)
    )
    assert isclose(
        score_response_tokens([-0.5, -1.0, -0.2], [0, 1, 1]),
        -1.2,
    )
    assert isclose(
        score_response_tokens([-0.5, -1.0, -0.2], [0, 1, 1], True),
        -0.6,
    )
    print("PASS toy optimization and sequence-length objective contrast")


def main():
    judge_checks()
    gradient_checks()
    optimization_checks()
    print("NOTE: synthetic scores; not a preference-aligned language model.")


if __name__ == "__main__":
    main()
