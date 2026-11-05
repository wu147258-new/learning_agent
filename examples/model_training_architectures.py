"""Offline GRPO-style group advantages and simplified clipped surrogate.

This script verifies only equations on synthetic rewards, not the complete
DeepSeek-R1 training procedure (token-wise loss and KL are not implemented).
No model downloads, API keys or files.
Run: python examples/model_training_architectures.py
"""
from __future__ import annotations

import math
import torch


def group_advantages(reward_groups: torch.Tensor,
                     eps: float = 1e-12) -> torch.Tensor:
    if reward_groups.ndim != 2:
        raise ValueError("each row must contain a group of responses")
    if reward_groups.shape[-1] < 2 or eps <= 0:
        raise ValueError("need at least 2 outputs and positive epsilon")
    if not torch.isfinite(reward_groups).all():
        raise ValueError("rewards must be finite")
    means = reward_groups.mean(dim=-1, keepdim=True)
    std = reward_groups.std(dim=-1, keepdim=True, unbiased=False)
    # A fully tied reward group carries no within-group ranking signal.
    # Set all advantages to zero and make the condition testable.
    return torch.where(
        std <= eps,
        torch.zeros_like(reward_groups),
        (reward_groups-means) / std.clamp_min(eps)
    )


def clipped_group_objective(ratios: torch.Tensor, advantages: torch.Tensor,
                            clip_eps: float = 0.2) -> torch.Tensor:
    if ratios.shape != advantages.shape or clip_eps <= 0:
        raise ValueError("shape mismatch or invalid clipping")
    if (ratios <= 0).any() or not torch.isfinite(ratios).all():
        raise ValueError("positive finite likelihood ratios required")
    capped = ratios.clamp(1-clip_eps, 1+clip_eps)
    # Maximize mean of clipped samples. No KL term in this toy.
    return torch.minimum(ratios * advantages, capped * advantages).mean()


def main():
    torch.set_num_threads(1)
    rewards = torch.tensor([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ], dtype=torch.float64)
    a = group_advantages(rewards)
    assert torch.allclose(
        a[0], torch.tensor([1., -1., 1., -1.], dtype=a.dtype)
    )
    assert torch.equal(a[1], torch.zeros(4, dtype=a.dtype))
    assert math.isclose(a[0].mean().item(), 0.0, abs_tol=1e-12)
    assert math.isclose(a[0].std(unbiased=False).item(), 1.0)
    ratios = torch.tensor([
        [1.5, 0.6, 1.0, 1.1],
        [1.0, 1.0, 1.0, 1.0],
    ], dtype=torch.float64)
    value = clipped_group_objective(ratios, a)
    # Sample-by-sample:
    # positive 1.5 clipped to 1.2
    # negative 0.6 clipped to 0.8 => minimum(-0.6,-0.8)=-0.8
    # positive 1.0 => 1.0
    # negative 1.1 => -1.1
    expected = (1.2 - 0.8 + 1.0 - 1.1) / 8
    assert math.isclose(value.item(), expected, abs_tol=1e-12)
    assert math.isfinite(value.item())
    # Distinct from an end-to-end language model's sequence/Tokens policy.
    print("PASS GRPO-style group normalization, ties and clipping")
    print(f"Toy clipped surrogate: {value.item():.6f}")
    print("NOTE: This is not an exact reproduction of DeepSeek-R1 GRPO.")


if __name__ == "__main__":
    main()
