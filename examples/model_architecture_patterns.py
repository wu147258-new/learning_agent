"""Small, inspectable model architecture operators. PyTorch CPU is sufficient.

Demonstrates MHA/GQA equivalence under explicit KV sharing, top-k MoE routing,
generic latent KV compression and generic kernelized linear attention.
These are NOT exact implementations of Llama/Qwen/DeepSeek/gpt-oss weights.
Run: python examples/model_architecture_patterns.py
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def grouped_query_attention(q, k, v):
    """q (B,Hq,T,D), k/v (B,Hkv,T,D). Hq must be multiple of Hkv."""
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("expected 4-D tensors")
    b, hq, t, d = q.shape
    bk, hkv, tk, dk = k.shape
    if (b != bk or v.shape[:3] != (b, hkv, tk) or
            hq % hkv != 0 or dk != d):
        raise ValueError("incompatible head dimensions")
    if t != tk or v.shape[3] != d:
        raise ValueError("this example expects square causal attention")
    g = hq // hkv
    queries = q.reshape(b, hkv, g, t, d)
    keys = k[:, :, None, :, :]
    values = v[:, :, None, :, :]
    score = queries @ keys.transpose(-1, -2) / math.sqrt(d)
    mask = torch.ones((t, t), dtype=torch.bool, device=q.device).tril()
    score = score.masked_fill(~mask, float("-inf"))
    probabilities = torch.softmax(score, dim=-1)
    output = probabilities @ values
    return output.reshape(b, hq, t, d)


def gqa_vs_repeated_kv():
    torch.manual_seed(2026)
    q = torch.randn(2, 4, 5, 3, dtype=torch.float64)
    k = torch.randn(2, 2, 5, 3, dtype=torch.float64)
    v = torch.randn(2, 2, 5, 3, dtype=torch.float64)
    got = grouped_query_attention(q, k, v)
    # An intentionally memory-expensive reference: explicitly duplicate KV.
    k_full = k.repeat_interleave(2, dim=1)
    v_full = v.repeat_interleave(2, dim=1)
    s = q @ k_full.transpose(-1, -2) / math.sqrt(3)
    causal = torch.ones((5, 5), dtype=torch.bool).tril()
    s = s.masked_fill(~causal, float("-inf"))
    expected = torch.softmax(s, dim=-1) @ v_full
    assert torch.allclose(got, expected, atol=1e-12)
    assert k.shape == (2, 2, 5, 3)
    assert k_full.shape == (2, 4, 5, 3)
    assert k.numel() + v.numel() == (
        k_full.numel() + v_full.numel()
    ) // 2
    return float((got - expected).abs().max())


def topk_moe(x, router, experts, k):
    """Token-wise top-k MoE; all experts here are simple linear functions."""
    if x.ndim != 2 or router.ndim != 2:
        raise ValueError("token batch and router must be matrices")
    e = router.shape[1]
    if len(experts) != e or not 1 <= k <= e:
        raise ValueError("invalid experts or top-k")
    scores = x @ router
    chosen_scores, indices = torch.topk(scores, k=k, dim=-1)
    weights = torch.softmax(chosen_scores, dim=-1)
    output = torch.zeros_like(x)
    counts = [0] * e
    for token in range(x.shape[0]):
        for slot in range(k):
            expert = int(indices[token, slot])
            counts[expert] += 1
            output[token] += weights[token, slot] * (
                x[token] @ experts[expert]
            )
    return output, indices, weights, counts


def moe_check():
    x = torch.tensor([
        [3.0, 1.0], [1.0, 3.0], [-2.0, 0.5]
    ], dtype=torch.float64)
    router = torch.tensor([
        [3.0, 1.0, -1.0, 0.5],
        [0.1, 2.0, -2.0, -1.0],
    ], dtype=torch.float64)
    experts = [
        torch.tensor([[i+1.0, 0.0], [0.0, i+1.0]], dtype=x.dtype)
        for i in range(4)
    ]
    y, selected, weights, counts = topk_moe(
        x, router, experts, k=2
    )
    assert y.shape == x.shape
    assert selected.shape == (3, 2)
    assert torch.allclose(
        weights.sum(dim=-1), torch.ones(3, dtype=x.dtype)
    )
    assert sum(counts) == 6  # 3 tokens, 2 routed experts each.
    # Independent explicit construction for one token.
    s = x[0] @ router
    top = torch.topk(s, 2)
    w = torch.softmax(top.values, dim=-1)
    manual = sum(
        (w[j] * (x[0] @ experts[int(top.indices[j])])
         for j in range(2)),
        torch.zeros_like(x[0]),
    )
    assert torch.allclose(y[0], manual, atol=1e-12)
    return counts


def latent_kv_check():
    torch.manual_seed(7)
    t, d, r = 9, 32, 6
    x = torch.randn(t, d, dtype=torch.float64)
    down = torch.randn(d, r, dtype=torch.float64)
    up_key = torch.randn(r, d, dtype=torch.float64)
    up_value = torch.randn(r, d, dtype=torch.float64)
    latent = x @ down
    reconstructed_k = latent @ up_key
    reconstructed_v = latent @ up_value
    assert latent.shape == (t, r)
    assert reconstructed_k.shape == (t, d)
    assert reconstructed_v.shape == (t, d)
    naive_elements = 2*t*d
    latent_elements = t*r
    assert latent_elements < naive_elements
    assert torch.linalg.matrix_rank(down) <= r
    # This checks ONLY a toy low-rank memory representation; actual MLA
    # has special positional/content paths and kernel optimizations.
    return naive_elements, latent_elements


def kernel_linear_attention(q, k, v):
    """Nonnegative feature map and fixed-size recurrent state.

    This deliberately is NOT the Gated DeltaNet used by Qwen3.5.
    q,k: (T,D), v: (T,Dv).
    """
    if q.shape != k.shape or q.shape[0] != v.shape[0]:
        raise ValueError("sequence shapes mismatch")
    feature = lambda x: F.elu(x) + 1.0
    fq, fk = feature(q), feature(k)
    t, features = fq.shape
    dv = v.shape[-1]
    S = torch.zeros(features, dv, dtype=q.dtype)
    z = torch.zeros(features, dtype=q.dtype)
    outputs = []
    for i in range(t):
        S = S + torch.outer(fk[i], v[i])
        z = z + fk[i]
        denominator = fq[i] @ z
        if denominator <= 0:
            raise ValueError("normalization must be positive")
        outputs.append((fq[i] @ S) / denominator)
    return torch.stack(outputs)


def linear_attention_check():
    torch.manual_seed(42)
    q = torch.randn(5, 4, dtype=torch.float64)
    k = torch.randn(5, 4, dtype=torch.float64)
    v = torch.randn(5, 3, dtype=torch.float64)
    got = kernel_linear_attention(q, k, v)
    fq, fk = F.elu(q) + 1, F.elu(k) + 1
    ref = []
    for i in range(q.shape[0]):
        relevance = fk[:i+1] @ fq[i]
        ref.append((relevance @ v[:i+1]) / relevance.sum())
    expected = torch.stack(ref)
    assert torch.allclose(got, expected, atol=1e-12)
    assert got.shape == (5, 3)
    return float((got - expected).abs().max())


def main():
    torch.set_num_threads(1)
    error_gqa = gqa_vs_repeated_kv()
    experts = moe_check()
    full_elements, compact_elements = latent_kv_check()
    error_linear = linear_attention_check()
    print(f"PASS GQA vs explicit repeated-KV, max error {error_gqa:.2e}")
    print(f"PASS MoE top-2 routed experts, counts={experts}")
    print(f"PASS generic latent KV toy elements {full_elements}->{compact_elements}")
    print(f"PASS generic kernel attention recurrence error {error_linear:.2e}")
    print("NOTE: Components demonstrate mathematics, not vendor checkpoints.")


if __name__ == "__main__":
    main()
