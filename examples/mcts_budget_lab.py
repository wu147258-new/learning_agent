"""UCB selection and budget accounting for a synthetic two-armed search."""
from __future__ import annotations
from math import sqrt, log, isclose
import random


class Arm:
    def __init__(self):
        self.visits = 0
        self.total = 0.0

    @property
    def mean(self):
        return self.total / self.visits if self.visits else 0

    def observe(self, reward):
        if not 0 <= reward <= 1:
            raise ValueError("reward must be within [0,1]")
        self.visits += 1
        self.total += reward

    def ucb(self, total_visits, c=1.0):
        if self.visits == 0:
            return float("inf")
        return self.mean + c*sqrt(log(max(total_visits,1))/self.visits)


def simulate(seed, budget):
    if budget < 2:
        raise ValueError("need enough budget to visit both actions")
    rng = random.Random(seed)
    arms = [Arm(), Arm()]
    probabilities = (0.8, 0.5)
    for _ in range(budget):
        idx = max(range(2), key=lambda i: (arms[i].ucb(sum(a.visits for a in arms)), -i))
        reward = int(rng.random() < probabilities[idx])
        arms[idx].observe(reward)
    assert sum(a.visits for a in arms) == budget
    return [(a.visits,a.total) for a in arms]


def main():
    a=Arm()
    assert a.ucb(10)==float("inf")
    for reward in [0,1,1]:
        a.observe(reward)
    assert isclose(a.mean,2/3)
    a.observe(0)
    assert isclose(a.mean,0.5)
    for bad in [-0.1,1.1]:
        try:
            a.observe(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid reward accepted")
    result=simulate(2026,400)
    assert result == simulate(2026,400)
    assert all(n>0 for n, _ in result)
    print("PASS UCB unvisited priority, online mean and fixed seed")
    print("PASS budget invariant: 400 synthetic simulations")
    print("NOTE: no real tool calls or learned reward model.")


if __name__=="__main__":
    main()
