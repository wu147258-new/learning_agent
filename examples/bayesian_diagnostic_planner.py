"""Finite-belief Bayesian diagnostic planner with exact value of information.

All fault distributions, diagnostic tests and losses are synthetic. This
module is a small POMDP-like problem, not a production incident agent.
Run: python3 examples/bayesian_diagnostic_planner.py
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import log2
from typing import Callable


FAULTS = ("cache", "configuration", "database")
SIGNALS = ("cache", "configuration", "database")


@dataclass(frozen=True)
class DiagnosticTest:
    name: str
    price: float
    likelihood: dict[str, dict[str, float]]
    # likelihood[hidden_fault][observed_signal]

    def validate(self):
        if self.price < 0:
            raise ValueError("negative test cost")
        if set(self.likelihood) != set(FAULTS):
            raise ValueError("missing hidden states")
        for hidden in FAULTS:
            dist = self.likelihood[hidden]
            if set(dist) != set(SIGNALS):
                raise ValueError("missing observation signals")
            if any(p < 0 or p > 1 for p in dist.values()):
                raise ValueError("invalid observation probability")
            if abs(sum(dist.values()) - 1) > 1e-12:
                raise ValueError("observation rows must normalize")


def make_test(name="diagnostic", accuracy=0.85, price=1.0):
    if not 0 < accuracy < 1:
        raise ValueError("accuracy must be between 0 and 1")
    other = (1-accuracy) / (len(FAULTS)-1)
    test = DiagnosticTest(
        name, price,
        {hidden: {signal: accuracy if hidden == signal else other
                  for signal in SIGNALS} for hidden in FAULTS}
    )
    test.validate()
    return test


def valid_belief(belief):
    if set(belief) != set(FAULTS):
        raise ValueError("belief missing possible states")
    if any(p < 0 or p > 1 for p in belief.values()):
        raise ValueError("invalid belief value")
    if abs(sum(belief.values()) - 1) > 1e-12:
        raise ValueError("belief not normalized")
    return belief


def signal_probability(belief, test: DiagnosticTest, signal: str):
    valid_belief(belief)
    test.validate()
    if signal not in SIGNALS:
        raise ValueError("unrecognized observation")
    return sum(
        belief[state] * test.likelihood[state][signal]
        for state in FAULTS
    )


def bayes_update(belief, test: DiagnosticTest, signal: str):
    probability = signal_probability(belief, test, signal)
    if probability == 0:
        raise ValueError("conditioning on zero-probability evidence")
    posterior = {
        state: belief[state] * test.likelihood[state][signal] / probability
        for state in FAULTS
    }
    valid_belief(posterior)
    return posterior


def entropy(belief):
    valid_belief(belief)
    return -sum(p * log2(p) for p in belief.values() if p > 0)


def risk_of_repair(belief, repair: str,
                   correct_cost=0.0, wrong_cost=20.0):
    valid_belief(belief)
    if repair not in FAULTS:
        raise ValueError("unrecognized repair")
    if min(correct_cost, wrong_cost) < 0:
        raise ValueError("invalid loss")
    return sum(
        belief[fault] * (correct_cost if fault == repair else wrong_cost)
        for fault in FAULTS
    )


def best_repair(belief):
    choices = sorted((risk_of_repair(belief, repair), repair)
                     for repair in FAULTS)
    return choices[0]  # risk, action; deterministic tie-break


def expected_posterior_risk(belief, test: DiagnosticTest):
    risk = 0.0
    for signal in SIGNALS:
        p = signal_probability(belief, test, signal)
        if p == 0:
            continue
        posterior = bayes_update(belief, test, signal)
        risk += p * best_repair(posterior)[0]
    return risk


def expected_information_gain(belief, test: DiagnosticTest):
    after = sum(
        signal_probability(belief, test, signal) *
        entropy(bayes_update(belief, test, signal))
        for signal in SIGNALS
        if signal_probability(belief, test, signal) > 0
    )
    gain = entropy(belief) - after
    if gain < -1e-12:
        raise AssertionError("discrete observation increased expected entropy")
    return gain


def value_of_information(belief, test: DiagnosticTest):
    immediate_risk, _ = best_repair(belief)
    risk_after = expected_posterior_risk(belief, test)
    return immediate_risk - test.price - risk_after


def choose_action(belief, tests, remaining_budget):
    valid_belief(belief)
    immediate_risk, direct = best_repair(belief)
    options = [(-immediate_risk, "repair:" + direct)]
    for test in tests:
        if test.price > remaining_budget:
            continue
        options.append((
            -test.price - expected_posterior_risk(belief, test),
            "test:" + test.name
        ))
    # Higher expected negative loss is better, tie-break by name.
    return sorted(options, key=lambda x: (-x[0], x[1]))[0]


def exact_two_step_policy(belief, tests, budget):
    """Return the complete conditional plan, not one guessed outcome."""
    score, choice = choose_action(belief, tests, budget)
    if choice.startswith("repair:"):
        return {
            "action": choice, "expected_cost": -score,
            "contingencies": {}
        }
    selected = {t.name: t for t in tests}[choice.split(":", 1)[1]]
    contingencies = {}
    for signal in SIGNALS:
        probability = signal_probability(belief, selected, signal)
        if probability == 0:
            continue
        posterior = bayes_update(belief, selected, signal)
        posterior_loss, repair = best_repair(posterior)
        contingencies[signal] = {
            "probability": probability,
            "posterior": posterior,
            "repair": repair,
            "conditional_expected_loss": posterior_loss,
        }
    return {
        "action": choice,
        "expected_cost": selected.price + sum(
            entry["probability"] * entry["conditional_expected_loss"]
            for entry in contingencies.values()
        ),
        "contingencies": contingencies,
    }


def enumerate_joint(belief, test: DiagnosticTest):
    return {
        (hidden, signal): belief[hidden] * test.likelihood[hidden][signal]
        for hidden, signal in product(FAULTS, SIGNALS)
    }


def check_bayes_identity(belief, test):
    joint = enumerate_joint(belief, test)
    for signal in SIGNALS:
        normalizer = sum(joint[hidden, signal] for hidden in FAULTS)
        if normalizer == 0:
            continue
        posterior = bayes_update(belief, test, signal)
        for state in FAULTS:
            assert abs(
                joint[state, signal] / normalizer - posterior[state]
            ) < 1e-12
    assert abs(sum(joint.values()) - 1) < 1e-12


def test_bayes():
    prior = {"cache": 0.5, "configuration": 0.3, "database": 0.2}
    test = make_test()
    check_bayes_identity(prior, test)
    post = bayes_update(prior, test, "database")
    assert post["database"] > prior["database"]
    assert post["cache"] < prior["cache"]
    assert expected_information_gain(prior, test) > 0
    assert abs(entropy(prior) - 1.4854752972273344) < 1e-9
    print("PASS conditional Bayes, joint mass, and expected information gain")


def test_action():
    prior = {"cache": 0.5, "configuration": 0.3, "database": 0.2}
    test = make_test()
    assert value_of_information(prior, test) > 0
    decision = exact_two_step_policy(prior, [test], budget=1.5)
    assert decision["action"] == "test:diagnostic"
    assert set(decision["contingencies"]) == set(SIGNALS)
    assert decision["expected_cost"] < best_repair(prior)[0]
    no_budget = exact_two_step_policy(prior, [test], budget=0)
    assert no_budget["action"].startswith("repair:")
    too_expensive = make_test("expensive", 0.85, price=100)
    assert value_of_information(prior, too_expensive) < 0
    assert exact_two_step_policy(
        prior, [too_expensive], budget=200
    )["action"].startswith("repair:")
    print("PASS exact 2-step information-versus-action planning")


def test_boundaries():
    prior = {"cache": 1., "configuration": 0., "database": 0.}
    test = make_test("perfect", accuracy=0.99)
    assert abs(value_of_information(prior, test) + test.price) < 1e-12
    try:
        valid_belief({"cache": 0.7, "configuration": 0.7, "database": 0.2})
    except ValueError:
        pass
    else:
        raise AssertionError("non-normalized belief accepted")
    print("PASS no-value-of-information and invalid-prior boundary cases")


if __name__ == "__main__":
    test_bayes()
    test_action()
    test_boundaries()
    print("NOTE: probabilities and losses are hypothetical, not field data.")
