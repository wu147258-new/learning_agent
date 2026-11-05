"""Exact one-token speculative decoding via rejection sampling.

Uses synthetic categorical distributions, not a real draft or target LLM.
Run: python examples/speculative_decoding_lab.py

The rejection correction is essential: naive accept/reject of draft tokens
does not in general preserve the target distribution.
"""
from __future__ import annotations

import math
import random


def check_distribution(probabilities):
    if not probabilities or any(p < 0 or not math.isfinite(p)
                                for p in probabilities):
        raise ValueError("probabilities must be finite nonnegative")
    if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-12):
        raise ValueError("probabilities must sum to 1")


def acceptance_probability(target, draft):
    if len(target) != len(draft):
        raise ValueError("vocabulary sizes differ")
    check_distribution(target)
    check_distribution(draft)
    # Equivalent to 1 - total variation distance.
    common = sum(min(p, q) for p, q in zip(target, draft))
    tv = 0.5*sum(abs(p-q) for p, q in zip(target, draft))
    assert math.isclose(common, 1-tv, abs_tol=1e-12)
    return common


def one_token_speculative(target, draft, rng):
    """Return (target-distributed token, accepted-draft flag)."""
    acceptance_probability(target, draft)
    proposal = rng.choices(range(len(draft)), weights=draft)[0]
    q = draft[proposal]
    assert q > 0  # Cannot draw a zero-probability item.
    if rng.random() < min(1.0, target[proposal]/q):
        return proposal, True
    # Discard rejected proposal; sample from positive difference.
    residual = [max(p-q, 0.0) for p, q in zip(target, draft)]
    assert sum(residual) > 0
    token = rng.choices(range(len(residual)), weights=residual)[0]
    return token, False


def theoretical_output_distribution(target, draft):
    """P(final=x)=min(p_x,q_x)+(p_x-q_x)_+=p_x."""
    acceptance_probability(target, draft)
    return tuple(
        min(p, q)+max(p-q, 0.0)
        for p, q in zip(target, draft)
    )


def self_test():
    p = (0.6, 0.3, 0.1)
    q = (0.3, 0.2, 0.5)
    assert math.isclose(acceptance_probability(p, q), 0.6)
    assert all(math.isclose(x, y, abs_tol=1e-12)
               for x, y in zip(theoretical_output_distribution(p, q), p))
    assert math.isclose(acceptance_probability(p, p), 1.0)

    n = 60000
    rng = random.Random(2026)
    hist = [0]*len(p)
    accepted = 0
    for _ in range(n):
        token, yes = one_token_speculative(p, q, rng)
        hist[token] += 1
        accepted += yes
    estimate = tuple(c/n for c in hist)
    assert max(abs(a-b) for a, b in zip(estimate, p)) < 0.015
    assert abs(accepted/n-0.6) < 0.015

    # Even disjoint supports can be corrected by the residual distribution.
    opposite_p = (0.0, 1.0)
    opposite_q = (1.0, 0.0)
    assert acceptance_probability(opposite_p, opposite_q) == 0
    for _ in range(10):
        token, yes = one_token_speculative(opposite_p, opposite_q, rng)
        assert token == 1 and not yes

    try:
        acceptance_probability((0.7, 0.7), (0.5, 0.5))
    except ValueError:
        pass
    else:
        raise AssertionError("invalid target distribution was accepted")

    print("PASS speculative: exact target distribution and TV identity")
    print(f"Sampled target frequencies: {estimate}, draft acceptance: {accepted/n:.3f}")
    print("NOTE: Single-token categorical toy, not GPU speculative decoding.")


if __name__ == "__main__":
    self_test()
