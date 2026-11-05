"""Two-step RL rollout, exact policy gradient and clipped PPO toy experiment.

PyTorch CPU >=2.1. Run: python examples/multistep_rl_rollout_lab.py
The 3-action root decides inspect / blind-guess-0 / blind-guess-1.
Inspect reveals a hidden bit and permits a second, conditionally chosen action.
This is not language-model RLHF or a DeepSeek/OpenAI trainer.
"""
from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F


INSPECT_COST = 0.1


class ToyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.root_logits = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))
        self.observe_logits = torch.nn.Parameter(
            torch.zeros(2, 2, dtype=torch.float64)
        )

    def root_prob(self):
        return F.softmax(self.root_logits, dim=-1)

    def observed_prob(self, hidden: int):
        return F.softmax(self.observe_logits[hidden], dim=-1)


def trajectories(policy: ToyPolicy):
    """List all 8 joint outcomes (environment bit included).

    Each item contains (probability, reward, log POLICY probability, metadata).
    The hidden bit is sampled at probability 1/2 by the environment.
    """
    root = policy.root_prob()
    out = []
    for hidden in (0, 1):
        for action in (1, 2):
            reward = 1.0 if action-1 == hidden else 0.0
            p = 0.5 * root[action]
            out.append((p, reward, torch.log(root[action]),
                        (hidden, action, None)))
        finish = policy.observed_prob(hidden)
        for last in (0, 1):
            reward = (1.0 if last == hidden else 0.0)-INSPECT_COST
            p = 0.5*root[0]*finish[last]
            logp = torch.log(root[0])+torch.log(finish[last])
            out.append((p, reward, logp, (hidden, 0, last)))
    assert len(out) == 8
    return out


def expected_return(policy):
    terms = [(p*reward) for p, reward, _, _ in trajectories(policy)]
    return torch.stack(terms).sum()


def exact_score_surrogate(policy, baseline=0.0):
    """Score-function gradient: detached P times log(policy) times advantage."""
    terms = [
        p.detach()*(reward-baseline)*logp
        for p, reward, logp, _ in trajectories(policy)
    ]
    return torch.stack(terms).sum()


def clipped_ppo_surrogate(policy, old_probs, baseline, clip_epsilon=0.2):
    """PPO-style trajectory ratio in a toy finite-horizon setting.

    For a real rollout trainer: keep behavior logprobs and old policy version
    with every action. Per-step credit and GAE are not modeled here.
    """
    items = trajectories(policy)
    if len(items) != len(old_probs):
        raise ValueError("outcome support changed")
    terms = []
    for (new_prob, reward, _, _), old in zip(items, old_probs):
        if old <= 0:
            raise ValueError("old behavior probability must be positive")
        ratio = new_prob/old
        adv = reward-baseline
        clipped = ratio.clamp(1-clip_epsilon, 1+clip_epsilon)
        terms.append(old * torch.minimum(ratio*adv, clipped*adv))
    return torch.stack(terms).sum()


def sample_rollouts(policy, count: int, seed: int = 2026):
    """Generate real sampled two-step episodes from the toy policy."""
    rng = random.Random(seed)
    root = policy.root_prob().detach().tolist()
    episodes = []
    for _ in range(count):
        hidden = rng.randrange(2)
        selected = rng.choices(range(3), weights=root)[0]
        if selected == 0:
            second = policy.observed_prob(hidden).detach().tolist()
            guess = rng.choices((0, 1), weights=second)[0]
            reward = (1.0 if guess == hidden else 0.0)-INSPECT_COST
            episodes.append((hidden, 0, guess, reward))
        else:
            reward = float((selected-1) == hidden)
            episodes.append((hidden, selected, None, reward))
    return episodes


def score_gradient_from_rollouts(policy, episodes):
    """Manual categorical score gradient of a sampled trajectory batch."""
    root = policy.root_prob().detach()
    grad_root = torch.zeros_like(root)
    grad_observe = torch.zeros_like(policy.observe_logits)
    for hidden, action, final, reward in episodes:
        onehot_root = torch.zeros_like(root)
        onehot_root[action] = 1.0
        grad_root += reward*(onehot_root-root)
        if action == 0:
            onehot_obs = torch.zeros(2, dtype=torch.float64)
            onehot_obs[final] = 1.0
            grad_observe[hidden] += reward*(
                onehot_obs-policy.observed_prob(hidden).detach()
            )
    return grad_root/len(episodes), grad_observe/len(episodes)


def offpolicy_importance_identity(behavior, target):
    """Exact identity with full trajectory probabilities, not a practical
    promise of low variance for long off-policy rollouts."""
    b = trajectories(behavior)
    t = trajectories(target)
    estimate = sum(
        (pb.detach()*(pt.detach()/pb.detach())*reward)
        for ((pb, _, _, key_b), (pt, reward, _, key_t)) in zip(b, t)
        if key_b == key_t
    )
    assert len(b) == len(t)
    return float(estimate)


def self_test():
    torch.set_num_threads(1)
    torch.manual_seed(2026)
    policy = ToyPolicy()
    before = expected_return(policy)
    assert math.isclose(before.item(), (0.4+0.5+0.5)/3,
                        abs_tol=1e-12)
    terms = trajectories(policy)
    assert math.isclose(sum(float(p) for p, _, _, _ in terms), 1.0,
                        abs_tol=1e-12)

    grad_exact = torch.autograd.grad(
        before, (policy.root_logits, policy.observe_logits)
    )
    score = exact_score_surrogate(policy, baseline=0.45)
    grad_score = torch.autograd.grad(
        score, (policy.root_logits, policy.observe_logits)
    )
    for expected, actual in zip(grad_exact, grad_score):
        assert torch.allclose(expected, actual, atol=1e-12)

    sampled = sample_rollouts(policy, 6000)
    empirical = score_gradient_from_rollouts(policy, sampled)
    for expected, actual in zip(grad_exact, empirical):
        assert (expected-actual).abs().max().item() < 0.035, (
            expected, actual
        )
    assert any(final is not None for _, _, final, _ in sampled)
    assert any(final is None for _, _, final, _ in sampled)

    old_probs = tuple(p.detach().clone() for p, _, _, _ in terms)
    ppo_before = clipped_ppo_surrogate(policy, old_probs, baseline=0.45)
    assert math.isfinite(ppo_before.item())
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.4)
    optimizer.zero_grad(set_to_none=True)
    (-ppo_before).backward()
    optimizer.step()
    assert expected_return(policy).item() > before.item()

    target = ToyPolicy()
    with torch.no_grad():
        target.root_logits[:] = torch.tensor([2., -1., -1.])
        target.observe_logits[0, :] = torch.tensor([1.5, -1.])
        target.observe_logits[1, :] = torch.tensor([-1., 1.5])
    offpolicy = offpolicy_importance_identity(policy, target)
    assert math.isclose(offpolicy, expected_return(target).item(),
                        abs_tol=1e-12)

    optimizer = torch.optim.Adam(policy.parameters(), lr=0.2)
    for _ in range(85):
        optimizer.zero_grad(set_to_none=True)
        loss = -expected_return(policy)
        loss.backward()
        optimizer.step()
    after = expected_return(policy).item()
    assert after > 0.87, after
    assert policy.root_prob()[0].item() > 0.98
    for hidden in (0, 1):
        assert policy.observed_prob(hidden)[hidden].item() > 0.98
    print("PASS multi-step: 8 trajectory outcomes, score gradient = autograd")
    print("PASS multi-step: sampled rollouts, PPO clipping and off-policy identity")
    print(f"Exact expected return: {before.item():.6f} -> {after:.6f}")
    print("NOTE: Finite toy MDP, not an LLM RL trainer.")


if __name__ == "__main__":
    self_test()
