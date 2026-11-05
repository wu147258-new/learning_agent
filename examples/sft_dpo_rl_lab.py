"""SFT, DPO, exact finite-action policy gradient, GAE and PPO clip.

Only synthetic two-action distributions; PyTorch CPU is sufficient.
Run: python examples/sft_dpo_rl_lab.py

No model downloads, no environment API, no claims about real LLM performance.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def sft_demo():
    """Fit the empirical 80%/20% action frequencies via maximum likelihood."""
    logits = torch.nn.Parameter(torch.zeros(2))
    target = torch.tensor([0]*8 + [1]*2)
    opt = torch.optim.Adam([logits], lr=0.12)
    initial = F.cross_entropy(logits[None, :].expand(len(target), -1),
                              target).item()
    for _ in range(140):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(
            logits[None, :].expand(len(target), -1), target
        )
        loss.backward()
        opt.step()
    with torch.no_grad():
        learned = F.softmax(logits, -1)
        final = F.cross_entropy(
            logits[None, :].expand(len(target), -1), target
        ).item()
    assert final < initial
    assert abs(learned[0].item() - 0.8) < 0.015
    return initial, final, learned.tolist()


def dpo_loss(train_logits, ref_logits, preferred=1, rejected=0,
             beta=1.0):
    """Finite-action DPO: -log sigmoid(beta * reference-relative log odds)."""
    if beta <= 0:
        raise ValueError("beta must be positive")
    pi = F.log_softmax(train_logits, dim=-1)
    pi_ref = F.log_softmax(ref_logits.detach(), dim=-1)
    difference = (
        (pi[preferred] - pi[rejected])
        - (pi_ref[preferred] - pi_ref[rejected])
    )
    return -F.logsigmoid(beta * difference)


def dpo_demo():
    logits = torch.nn.Parameter(torch.zeros(2))
    ref = torch.zeros(2)  # Immutable uniform reference.
    opt = torch.optim.SGD([logits], lr=0.2)
    initial = dpo_loss(logits, ref).item()
    for _ in range(60):
        opt.zero_grad(set_to_none=True)
        loss = dpo_loss(logits, ref)
        loss.backward()
        opt.step()
    final = dpo_loss(logits, ref).item()
    distribution = F.softmax(logits, -1)
    assert final < initial
    assert distribution[1].item() > 0.8
    # For pi=(0.6,0.4), ref=(0.2,0.8) and preferred=0:
    pi = torch.log(torch.tensor([0.6, 0.4]))
    pi_ref = torch.log(torch.tensor([0.2, 0.8]))
    d = (pi[0]-pi[1]) - (pi_ref[0]-pi_ref[1])
    assert math.isclose(d.item(), math.log(6), abs_tol=1e-6)
    analytic = -math.log(6/7)
    value = dpo_loss(pi, pi_ref, preferred=0, rejected=1).item()
    assert math.isclose(value, analytic, abs_tol=1e-6)
    return initial, final, distribution.tolist()


def exact_policy_gradient_demo():
    """Compare direct expected-reward gradient with the log-derivative form."""
    logits = torch.tensor([0.0, 0.0], dtype=torch.float64,
                          requires_grad=True)
    rewards = torch.tensor([1.0, 3.0], dtype=torch.float64)
    p = F.softmax(logits, -1)
    expected = torch.dot(p, rewards)
    analytic = torch.autograd.grad(expected, logits)[0]
    assert math.isclose(expected.item(), 2.0)
    assert torch.allclose(
        analytic, torch.tensor([-0.5, 0.5], dtype=torch.float64),
        atol=1e-12
    )
    # Score function gradient of a categorical distribution:
    # nabla log p(a) = one_hot(a) - p.
    scores = torch.eye(2, dtype=torch.float64) - p.detach()[None, :]
    direct = (p.detach()[:, None] * rewards[:, None] * scores).sum(0)
    baseline = 8.0  # Chosen to illustrate unbiasedness, not low variance.
    centered = (
        p.detach()[:, None] * (rewards[:, None]-baseline) * scores
    ).sum(0)
    assert torch.allclose(direct, analytic, atol=1e-12)
    assert torch.allclose(centered, analytic, atol=1e-12)
    return analytic.tolist()


def reinforce_expected_reward_demo():
    """Exact policy expectation optimization; no sampling claim."""
    rewards = torch.tensor([1.0, 3.0])
    logits = torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.Adam([logits], lr=0.12)
    initial = torch.dot(F.softmax(logits.detach(), -1), rewards).item()
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        objective = torch.dot(F.softmax(logits, -1), rewards)
        (-objective).backward()  # Adam minimizes, so negate reward.
        optimizer.step()
    final_p = F.softmax(logits.detach(), -1)
    final = torch.dot(final_p, rewards).item()
    assert final > initial
    assert final_p[1].item() > 0.95
    return initial, final


def gae(rewards, values, gamma=0.9, lam=0.8):
    """Finite terminated trajectory; values[-1] is terminal bootstrap=0."""
    if len(values) != len(rewards)+1:
        raise ValueError("values must include terminal state")
    if not 0 <= gamma <= 1 or not 0 <= lam <= 1:
        raise ValueError("invalid discount or lambda")
    output = [0.0] * len(rewards)
    later = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma*values[t+1] - values[t]
        later = delta + gamma*lam*later
        output[t] = later
    return output


def ppo_clipped_term(ratio, advantage, epsilon=0.2):
    """One PPO surrogate sample; PPO would MAXIMIZE this quantity."""
    if ratio <= 0 or epsilon <= 0:
        raise ValueError("invalid likelihood ratio or epsilon")
    clipped = max(1-epsilon, min(1+epsilon, ratio))
    return min(ratio*advantage, clipped*advantage)


def ppo_and_gae_demo():
    assert math.isclose(ppo_clipped_term(1.5, 2.0), 2.4)
    assert math.isclose(ppo_clipped_term(0.6, -2.0), -1.6)
    assert math.isclose(ppo_clipped_term(1.0, 2.0), 2.0)
    advantages = gae([1.0, 2.0], [0.5, 0.4, 0.0])
    assert math.isclose(advantages[1], 1.6, abs_tol=1e-12)
    assert math.isclose(advantages[0], 0.86 + 0.72*1.6, abs_tol=1e-12)
    return advantages


def main():
    torch.set_num_threads(1)
    torch.manual_seed(2026)
    sft = sft_demo()
    dpo = dpo_demo()
    gradient = exact_policy_gradient_demo()
    reinforcement = reinforce_expected_reward_demo()
    advantages = ppo_and_gae_demo()
    print("PASS SFT", sft)
    print("PASS DPO", dpo)
    print("PASS exact score-gradient and baseline", gradient)
    print("PASS expected-reward optimization", reinforcement)
    print("PASS PPO clipping and GAE", advantages)
    print("NOTE: Only synthetic two-action math, not actual model alignment.")


if __name__ == "__main__":
    main()
