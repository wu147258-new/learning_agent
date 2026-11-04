"""Reproducible paired evaluation of task-completion Agents.

No network, ML packages, API keys or hidden tasks. Example data are SYNTHETIC.
Assumes a fixed task set, one binary completion outcome per Agent per task.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from statistics import mean


@dataclass(frozen=True)
class TaskResult:
    task_id: str
    baseline_success: bool
    candidate_success: bool
    baseline_cost: float
    candidate_cost: float


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054):
    """Approximate 95% CI for Bernoulli probability; not a causal comparison."""
    if trials <= 0 or not (0 <= successes <= trials):
        raise ValueError("Require 0 <= successes <= trials and trials > 0")
    if z < 0:
        raise ValueError("z must be nonnegative")
    p = successes / trials
    z2 = z*z
    denominator = 1 + z2/trials
    center = (p + z2/(2*trials)) / denominator
    half = (z / denominator) * math.sqrt(
        p*(1-p)/trials + z2/(4*trials*trials)
    )
    return max(0.0, center-half), min(1.0, center+half)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator from n samples, c verified successes."""
    if n < 1 or k < 1 or k > n or c < 0 or c > n:
        raise ValueError("Require 1 <= k <= n and 0 <= c <= n")
    if c == 0:
        return 0.0
    # 1 - binom(n-c,k)/binom(n,k); multiply ratios without factorials.
    if n - c < k:
        return 1.0
    miss_probability = 1.0
    for i in range(k):
        miss_probability *= (n-c-i) / (n-i)
    return 1.0 - miss_probability


def exact_mcnemar_pvalue(b: int, c: int) -> float:
    """Two-sided exact conditional binomial test, no continuity correction.

    b: baseline fails and candidate succeeds.
    c: baseline succeeds and candidate fails.
    """
    if b < 0 or c < 0:
        raise ValueError("Discordant counts must be nonnegative")
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c)+1))
    return min(1.0, 2.0 * tail / (2 ** n))


def percentile(sorted_values, quantile):
    """Linear interpolation, 0 <= quantile <= 1."""
    if not sorted_values or not 0 <= quantile <= 1:
        raise ValueError("Require nonempty sorted values and valid quantile")
    x = (len(sorted_values)-1)*quantile
    low = math.floor(x)
    high = math.ceil(x)
    return sorted_values[low]*(high-x) + sorted_values[high]*(x-low) if low != high else sorted_values[low]


def paired_bootstrap_delta_ci(
    records: list[TaskResult], trials: int = 4000,
    seed: int = 2026, alpha: float = 0.05
):
    """Percentile CI for mean paired success difference, resampling TASKS."""
    if not records or trials < 2 or not 0 < alpha < 1:
        raise ValueError("Require records, trials >=2, 0<alpha<1")
    diffs = [int(r.candidate_success)-int(r.baseline_success) for r in records]
    n = len(diffs)
    rng = random.Random(seed)
    estimates = sorted(
        sum(diffs[rng.randrange(n)] for _ in range(n))/n
        for _ in range(trials)
    )
    return percentile(estimates, alpha/2), percentile(estimates, 1-alpha/2)


def summarize(records: list[TaskResult]) -> dict:
    if not records:
        raise ValueError("The task set cannot be empty")
    ids = [r.task_id for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate task_id: evaluation unit must be explicit")
    if any(not r.task_id or r.baseline_cost < 0 or r.candidate_cost < 0 or
           not math.isfinite(r.baseline_cost) or not math.isfinite(r.candidate_cost)
           for r in records):
        raise ValueError("Invalid ID, cost or non-finite value")
    baseline = sum(r.baseline_success for r in records)
    candidate = sum(r.candidate_success for r in records)
    gained = sum(not r.baseline_success and r.candidate_success for r in records)
    lost = sum(r.baseline_success and not r.candidate_success for r in records)
    return {
        "n_tasks": len(records),
        "baseline_success_rate": baseline / len(records),
        "candidate_success_rate": candidate / len(records),
        "paired_improvement": (candidate - baseline) / len(records),
        "baseline_wilson95": wilson_interval(baseline, len(records)),
        "candidate_wilson95": wilson_interval(candidate, len(records)),
        "paired_bootstrap95": paired_bootstrap_delta_ci(records),
        "mcnemar_exact_p": exact_mcnemar_pvalue(gained, lost),
        "gained_tasks": gained,
        "lost_tasks": lost,
        "baseline_cost_per_task": mean(r.baseline_cost for r in records),
        "candidate_cost_per_task": mean(r.candidate_cost for r in records),
    }


def self_test():
    assert wilson_interval(0, 10)[0] == 0
    assert wilson_interval(10, 10)[1] == 1
    assert math.isclose(pass_at_k(10, 3, 2), 1 - (7*6)/(10*9))
    assert pass_at_k(10, 0, 5) == 0
    assert pass_at_k(10, 8, 5) == 1
    assert math.isclose(exact_mcnemar_pvalue(5, 0), 0.0625)
    assert exact_mcnemar_pvalue(0, 0) == 1

    # Intentionally small synthetic experiment: DO NOT report as model evidence.
    rows = [
        TaskResult(f"task-{i}", i % 3 != 0, i % 4 != 0,
                   0.02 + (i % 4)*0.003, 0.025 + (i % 5)*0.004)
        for i in range(24)
    ]
    result = summarize(rows)
    assert result["n_tasks"] == 24
    assert math.isclose(
        result["paired_improvement"],
        result["candidate_success_rate"]-result["baseline_success_rate"],
        abs_tol=1e-12
    )
    lo, hi = result["paired_bootstrap95"]
    assert -1 <= lo <= hi <= 1
    try:
        summarize(rows + [rows[0]])
        raise AssertionError("Duplicate task ID must be rejected")
    except ValueError:
        pass
    print("PASS Wilson, pass@k, McNemar, paired bootstrap, cost validation")
    for key, value in result.items():
        print(f"{key}: {value}")
    print("NOTE: Every task result above is SYNTHETIC, not Agent benchmark evidence.")


if __name__ == "__main__":
    self_test()
