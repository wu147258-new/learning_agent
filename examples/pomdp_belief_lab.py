"""Finite POMDP prediction, Bayesian correction, and value of information.

Only the standard library is used. All measurements below are synthetic.
Run: python3 examples/pomdp_belief_lab.py
"""
from __future__ import annotations

from math import isclose


def check_probabilities(v: list[float]) -> None:
    assert v and all(isinstance(x, (int, float)) for x in v)
    assert all(0 <= x <= 1 for x in v), v
    assert isclose(sum(v), 1.0, abs_tol=1e-10), v


def predict(b: list[float], transition: list[list[float]]) -> list[float]:
    """transition[s][sp] is P(next_state=sp | old_state=s, action)."""
    check_probabilities(b)
    n = len(b)
    assert len(transition) == n
    for row in transition:
        assert len(row) == n
        check_probabilities(row)
    return [sum(b[s] * transition[s][sp] for s in range(n))
            for sp in range(n)]


def correct(predicted: list[float], likelihood: list[float]) -> list[float]:
    """likelihood[sp] is P(observation | new_state=sp, action)."""
    check_probabilities(predicted)
    assert len(predicted) == len(likelihood)
    assert all(isinstance(x, (int, float)) and 0 <= x <= 1
               for x in likelihood)
    weights = [p * z for p, z in zip(predicted, likelihood)]
    evidence = sum(weights)
    if evidence == 0:
        raise ValueError("observed event has zero probability under model")
    ans = [x / evidence for x in weights]
    check_probabilities(ans)
    return ans


def expected_cost(b: list[float], costs: list[float]) -> float:
    check_probabilities(b)
    assert len(b) == len(costs)
    return sum(p * c for p, c in zip(b, costs))


def no_observation_cost(prior: list[float], mistake_cost: float) -> float:
    check_probabilities(prior)
    assert len(prior) == 2
    # Choose repair 0 or repair 1, whichever has lower expected loss.
    return min(prior[1] * mistake_cost, prior[0] * mistake_cost)


def observation_cost(prior: list[float], accuracy: float,
                     inspect_cost: float, mistake_cost: float) -> float:
    check_probabilities(prior)
    assert len(prior) == 2
    assert 0 <= accuracy <= 1
    # report is 0 or 1; observation likelihood is symmetric.
    expected_after = 0.0
    for report in (0, 1):
        likelihood = [
            accuracy if report == true_state else 1 - accuracy
            for true_state in (0, 1)
        ]
        p_report = sum(p * z for p, z in zip(prior, likelihood))
        if p_report == 0:
            continue
        posterior = correct(prior, likelihood)
        conditional_loss = min(posterior[0], posterior[1]) * mistake_cost
        expected_after += p_report * conditional_loss
    return inspect_cost + expected_after


def expect_raises(exc, action):
    try:
        action()
    except exc:
        return
    raise AssertionError(f"{exc.__name__} was not raised")


def main():
    # Asymmetric transition: old 0 always goes to new 1.
    transition = [[0.0, 1.0], [0.0, 1.0]]
    assert predict([1.0, 0.0], transition) == [0.0, 1.0]
    assert predict([0.3, 0.7], [[1.0, 0.0], [0.0, 1.0]]) == [0.3, 0.7]
    posterior = correct([0.6, 0.4], [0.1, 0.9])
    assert isclose(posterior[1], 6.0 / 7.0, abs_tol=1e-12)
    assert isclose(sum(posterior), 1.0, abs_tol=1e-12)

    prior = [0.5, 0.5]
    direct = no_observation_cost(prior, mistake_cost=10.0)
    cheap_inspection = observation_cost(prior, 0.9, 1.0, 10.0)
    expensive_inspection = observation_cost(prior, 0.9, 5.0, 10.0)
    assert isclose(direct, 5.0)
    assert isclose(cheap_inspection, 2.0)
    assert isclose(expensive_inspection, 6.0)
    assert cheap_inspection < direct < expensive_inspection

    # Negative tests are essential: a "normal-looking" vector is insufficient.
    expect_raises(ValueError, lambda: correct([1.0, 0.0], [0.0, 0.0]))
    expect_raises(AssertionError, lambda: predict(
        [1.0, 0.0], [[0.5, 0.2], [0.0, 1.0]]))
    expect_raises(AssertionError, lambda: predict(
        [1.0, 0.0], [[0.5, 0.5, 0.0], [0.0, 1.0]]))
    expect_raises(AssertionError, lambda: correct([0.5, 0.5], [0.1]))
    print("PASS POMDP: asymmetric transition, exact posterior=6/7")
    print("PASS value of information: inspect=2, direct=5, expensive=6")
    print("PASS invalid transition, dimensions and impossible observation")
    print("NOTE: synthetic numerical teaching model, not field telemetry.")


if __name__ == "__main__":
    main()
