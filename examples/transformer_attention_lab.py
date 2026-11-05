"""Verifiable attention mathematics, causal masking, RoPE and KV caching.

Requirements: PyTorch >= 2.1 (CPU sufficient).
Run: python examples/transformer_attention_lab.py

All data are synthetic. Passing tests verifies these implementations, not
generalization or production-scale language model performance.
"""
from __future__ import annotations

import math

import torch


def attention(q, k, v, causal=False):
    """(T,d_k) query/key and (T,d_v) value tensors; use float64 in tests."""
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise ValueError("expected matrices")
    if q.shape[1] != k.shape[1] or k.shape[0] != v.shape[0]:
        raise ValueError("incompatible Q/K/V dimensions")
    if causal and q.shape[0] != k.shape[0]:
        raise ValueError("causal example expects a full square sequence")
    scores = (q @ k.T) / math.sqrt(q.shape[-1])
    if causal:
        t = q.shape[0]
        allowed = torch.ones((t, t), dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~allowed, -torch.inf)
    weights = torch.softmax(scores, dim=-1)
    return weights @ v, weights


def decode_one(q, keys, values):
    """Single query attends to all keys seen so far, equivalent to a KV cache."""
    if q.shape[0] != 1:
        raise ValueError("exactly one new query expected")
    return torch.softmax(
        (q @ keys.T) / math.sqrt(q.shape[-1]), dim=-1
    ) @ values


def rotate_2d(v, theta):
    """RoPE's one-pair two-dimensional rotation."""
    c, s = math.cos(theta), math.sin(theta)
    rotation = torch.tensor(
        [[c, -s], [s, c]], dtype=v.dtype, device=v.device
    )
    return rotation @ v


def manual_two_key_check():
    q = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    k = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    v = torch.tensor([[2.0, 0.0], [0.0, 4.0]], dtype=torch.float64)
    result, weights = attention(q, k, v)
    w1 = math.exp(1 / math.sqrt(2))
    manual_weights = torch.tensor(
        [[w1 / (w1 + 1), 1 / (w1 + 1)]], dtype=torch.float64
    )
    manual_result = manual_weights @ v
    assert torch.allclose(weights, manual_weights, atol=1e-12)
    assert torch.allclose(result, manual_result, atol=1e-12)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1, dtype=q.dtype))


def causal_and_cache_check():
    torch.manual_seed(2026)
    x = torch.randn(7, 6, dtype=torch.float64)
    wq = torch.randn(6, 4, dtype=torch.float64)
    wk = torch.randn(6, 4, dtype=torch.float64)
    wv = torch.randn(6, 3, dtype=torch.float64)
    q, k, v = x @ wq, x @ wk, x @ wv
    full, weights = attention(q, k, v, causal=True)
    assert full.shape == (7, 3)
    assert weights.shape == (7, 7)
    assert torch.equal(
        weights.triu(diagonal=1),
        torch.zeros_like(weights)
    )

    # Every new query is evaluated using exactly its cached prefix of K/V.
    for t in range(len(x)):
        cached = decode_one(q[t:t+1], k[:t+1], v[:t+1])
        assert torch.allclose(cached[0], full[t], atol=1e-12)

    # A changed future must leave all earlier outputs invariant.
    x_modified = x.clone()
    x_modified[3:] += torch.randn_like(x_modified[3:])
    changed, _ = attention(
        x_modified @ wq, x_modified @ wk, x_modified @ wv, causal=True
    )
    assert torch.allclose(full[:3], changed[:3], atol=1e-12)


def finite_difference_gradient_check():
    # Use double precision; compare the analytic autograd derivative of
    # an attention output with a central finite difference.
    q = torch.tensor([[0.2, -0.3]], dtype=torch.float64, requires_grad=True)
    k = torch.tensor([[0.4, 0.1], [0.2, -0.5]], dtype=torch.float64)
    v = torch.tensor([[1.0, 2.0], [-2.0, 0.5]], dtype=torch.float64)
    output, _ = attention(q, k, v)
    analytic = torch.autograd.grad(output[0, 0], q)[0][0, 0].item()
    h = 1e-6
    with torch.no_grad():
        plus = q.detach().clone()
        minus = q.detach().clone()
        plus[0, 0] += h
        minus[0, 0] -= h
        fp = attention(plus, k, v)[0][0, 0].item()
        fm = attention(minus, k, v)[0][0, 0].item()
    numeric = (fp - fm) / (2*h)
    assert math.isclose(analytic, numeric, abs_tol=1e-8, rel_tol=1e-6)


def rope_relative_position_check():
    q = torch.tensor([0.8, -1.5], dtype=torch.float64)
    k = torch.tensor([1.2, 0.7], dtype=torch.float64)
    omega, m, n = 0.23, 4, 9
    absolute = torch.dot(rotate_2d(q, m*omega),
                         rotate_2d(k, n*omega)).item()
    relative = torch.dot(q, rotate_2d(k, (n-m)*omega)).item()
    assert math.isclose(absolute, relative, rel_tol=0, abs_tol=1e-12)


def stable_softmax_check():
    scores = torch.tensor([[1000.0, 999.0, 998.0]], dtype=torch.float64)
    probs = torch.softmax(scores, dim=-1)
    shifted = torch.softmax(scores - scores.max(dim=-1).values[:, None],
                            dim=-1)
    assert torch.isfinite(probs).all()
    assert torch.allclose(probs, shifted, atol=1e-12)


if __name__ == "__main__":
    torch.set_num_threads(1)
    manual_two_key_check()
    causal_and_cache_check()
    finite_difference_gradient_check()
    rope_relative_position_check()
    stable_softmax_check()
    print("PASS: hand attention, causal invariance, KV cache, gradients, RoPE")
    print("NOTE: These tests use synthetic matrices, not a trained LLM.")
