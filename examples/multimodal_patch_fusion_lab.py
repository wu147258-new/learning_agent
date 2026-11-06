"""Image patch tokens, positional handling and toy early-fusion attention.

Uses PyTorch CPU. This is a transparent educational operator demonstration,
not a reproduction of Qwen3-VL, Gemma 3 or Llama 4 weights.
Run: python examples/multimodal_patch_fusion_lab.py
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def image_to_patches(images: torch.Tensor, patch: int) -> torch.Tensor:
    """Input (B,C,H,W), output (B,N,C*patch*patch), row-major."""
    if images.ndim != 4 or patch < 1:
        raise ValueError("images must be BCHW with a positive patch size")
    b, channels, height, width = images.shape
    if height % patch or width % patch:
        raise ValueError("height and width must be divisible by patch size")
    return F.unfold(images, kernel_size=patch, stride=patch).transpose(1, 2)


def patches_to_image(patches: torch.Tensor, channels: int, height: int,
                     width: int, patch: int) -> torch.Tensor:
    if patches.ndim != 3:
        raise ValueError("patches require BND")
    return F.fold(
        patches.transpose(1, 2), output_size=(height, width),
        kernel_size=patch, stride=patch
    )


def patch_coordinates(height: int, width: int, patch: int):
    if height % patch or width % patch:
        raise ValueError("invalid patch grid")
    return tuple(
        (row, col)
        for row in range(height // patch)
        for col in range(width // patch)
    )


def merge_visual_tokens(patch_vectors: torch.Tensor, hidden: int,
                        group_size: int):
    """Toy learned linear merger over consecutive patch feature groups.

    This is not a real model's dynamic tiling or 2D merge implementation.
    """
    b, n, dim = patch_vectors.shape
    if group_size <= 0 or n % group_size:
        raise ValueError("patch count must be divisible by merge grouping")
    merged = patch_vectors.reshape(b, n//group_size, group_size*dim)
    weight = torch.arange(
        group_size*dim*hidden,
        dtype=patch_vectors.dtype,
        device=patch_vectors.device
    ).reshape(group_size*dim, hidden)
    weight = weight / max(1, weight.numel())
    result = merged @ weight
    assert result.shape == (b, n//group_size, hidden)
    return result


def safe_fused_attention(text_tokens, image_tokens, causal_text=True):
    """Minimal fused attention with image-prefix, text causal visibility.

    Token order: all image tokens first, then text. Image prefix is fully
    visible to all tokens. Text queries cannot see future text tokens.
    This mask illustrates ONE choice, not every vendor's architecture.
    """
    if text_tokens.ndim != 3 or image_tokens.ndim != 3:
        raise ValueError("expected BTD and BND")
    if text_tokens.shape[0] != image_tokens.shape[0]:
        raise ValueError("batch mismatch")
    if text_tokens.shape[2] != image_tokens.shape[2]:
        raise ValueError("hidden dimensions differ")
    all_tokens = torch.cat([image_tokens, text_tokens], dim=1)
    b, n, d = all_tokens.shape
    visual_count = image_tokens.shape[1]
    mask = torch.zeros(n, n, dtype=torch.bool, device=all_tokens.device)
    mask[:, :visual_count] = True
    for row in range(visual_count, n):
        mask[row, visual_count:row+1] = True
    if not causal_text:
        mask[:, :] = True
    score = all_tokens @ all_tokens.transpose(-1, -2) / math.sqrt(d)
    score = score.masked_fill(~mask.unsqueeze(0), float("-inf"))
    weights = torch.softmax(score, dim=-1)
    output = weights @ all_tokens
    return output, weights, mask


def self_test():
    torch.manual_seed(2026)
    image = torch.arange(1*3*8*8, dtype=torch.float64).reshape(1, 3, 8, 8)
    patches = image_to_patches(image, patch=2)
    assert patches.shape == (1, 16, 12)
    restored = patches_to_image(patches, channels=3, height=8,
                                width=8, patch=2)
    assert torch.equal(restored, image)
    assert patch_coordinates(8, 8, 2) == tuple(
        (row, col) for row in range(4) for col in range(4)
    )
    merged = merge_visual_tokens(patches, hidden=6, group_size=4)
    assert merged.shape == (1, 4, 6)
    text = torch.randn(1, 3, 6, dtype=torch.float64)
    out, attention, mask = safe_fused_attention(text, merged)
    assert out.shape == (1, 7, 6)
    assert attention.shape == (1, 7, 7)
    assert mask.shape == (7, 7)
    assert bool(mask[:, :4].all())
    assert not mask[4, 6].item()  # first text token can't see later text.
    assert mask[6, 4].item()  # last text token can see preceding text.

    # Causal non-leakage: future text is changed, earlier output unchanged.
    changed = text.clone()
    changed[0, -1, :] += 100
    newer, _, _ = safe_fused_attention(changed, merged)
    assert torch.allclose(out[:, 4, :], newer[:, 4, :], atol=1e-12)
    assert torch.allclose(out[:, 5, :], newer[:, 5, :], atol=1e-12)

    # Visual patch count grows with resolution at fixed patch size.
    higher = torch.zeros(1, 3, 16, 16, dtype=torch.float64)
    assert image_to_patches(higher, 2).shape[1] == 4*patches.shape[1]
    try:
        image_to_patches(torch.zeros(1, 3, 7, 8), 2)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid input geometry was accepted")
    print("PASS multimodal: patchify/unpatchify and visual token projection")
    print("PASS multimodal: prefix early-fusion mask and causal non-leakage")
    print("NOTE: No real vision encoder, OCR model or multimodal checkpoint.")


if __name__ == "__main__":
    self_test()
