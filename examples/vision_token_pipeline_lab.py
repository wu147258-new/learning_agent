"""Vision-token pipeline and relative 2D rotary position example.

PyTorch CPU. No external images, pretrained checkpoints or model API.
Shows patchification, a 2x2 toy merger, dynamic token length, residual
feature injection and relative-position dot-product invariance.

This is a mathematical teaching analogue, NOT Qwen3-VL implementation.
Run: python examples/vision_token_pipeline_lab.py
"""
from __future__ import annotations

import math

import torch


def patchify(image: torch.Tensor, patch_size: int):
    """Return (rows, columns, [num_patches, flattened_patch_dim]).

    image shape: (C, H, W), without padding or interpolation.
    """
    if image.ndim != 3 or patch_size <= 0:
        raise ValueError("image must be CxHxW, patch size positive")
    c, height, width = image.shape
    if height % patch_size or width % patch_size:
        raise ValueError("height/width must be divisible by patch size")
    rows, columns = height // patch_size, width // patch_size
    patches = image.unfold(1, patch_size, patch_size).unfold(
        2, patch_size, patch_size
    )
    patches = patches.permute(1, 2, 0, 3, 4).contiguous()
    tokens = patches.reshape(rows*columns, c*patch_size*patch_size)
    return rows, columns, tokens


def reconstruct(rows, columns, patches, channels, patch_size):
    """Invert patchify exactly, using no learned weights."""
    if patches.shape != (rows*columns, channels*patch_size*patch_size):
        raise ValueError("wrong patch tensor dimensions")
    x = patches.reshape(
        rows, columns, channels, patch_size, patch_size
    )
    return x.permute(2, 0, 3, 1, 4).contiguous().reshape(
        channels, rows*patch_size, columns*patch_size
    )


def merge_neighboring_2x2(features: torch.Tensor, rows: int, columns: int,
                         merger: torch.nn.Linear):
    """Group spatially adjacent features, then apply learned projection.

    Input is already-encoded visual features, not raw image pixels.
    """
    if features.ndim != 2 or features.shape[0] != rows*columns:
        raise ValueError("feature grid shape mismatch")
    if rows % 2 or columns % 2:
        raise ValueError("teaching merger requires even patch grid")
    count, channels = features.shape
    if merger.in_features != 4*channels:
        raise ValueError("merger expects four concatenated visual features")
    grouped = features.reshape(rows, columns, channels)
    grouped = grouped.reshape(rows//2, 2, columns//2, 2, channels)
    grouped = grouped.permute(0, 2, 1, 3, 4).contiguous()
    fused = grouped.reshape(count//4, 4*channels)
    return merger(fused)


def rotate_2d(v: torch.Tensor, row: int, col: int):
    """Toy two-axis RoPE, 4-dimensional vector, one frequency per axis."""
    if v.shape[-1] != 4:
        raise ValueError("toy 2D RoPE needs 4 features")
    x0, x1, x2, x3 = v.unbind(-1)
    a, b = math.cos(row), math.sin(row)
    c, d = math.cos(col), math.sin(col)
    return torch.stack([
        a*x0-b*x1, b*x0+a*x1,
        c*x2-d*x3, d*x2+c*x3
    ], dim=-1)


def self_test():
    torch.set_num_threads(1)
    torch.manual_seed(2026)

    # 3 color channels, 8x12 pixels, 2x2 pixel patches.
    image = torch.arange(3*8*12, dtype=torch.float64).reshape(3, 8, 12)
    rows, cols, patches = patchify(image, 2)
    assert (rows, cols) == (4, 6)
    assert patches.shape == (24, 12)
    assert torch.equal(reconstruct(rows, cols, patches, 3, 2), image)

    # The synthetic vision encoder projects 12-dim patches to 8 features.
    encoder = torch.nn.Linear(12, 8, dtype=torch.float64)
    hidden = encoder(patches)
    assert hidden.shape == (24, 8)

    # Four adjacent tokens merged into one, with 4x8 -> 16 projection.
    merger = torch.nn.Linear(32, 16, dtype=torch.float64)
    merged = merge_neighboring_2x2(hidden, rows, cols, merger)
    assert merged.shape == (6, 16)
    assert merged.numel() == 6*16

    # The same encoder/merger can process a larger divisible resolution.
    larger = torch.zeros(3, 12, 16, dtype=torch.float64)
    rows2, cols2, patches2 = patchify(larger, 2)
    result2 = merge_neighboring_2x2(
        encoder(patches2), rows2, cols2, merger
    )
    assert (rows2, cols2) == (6, 8)
    assert result2.shape == (12, 16)

    # A single-pixel change affects exactly one raw patch.
    modified = image.clone()
    modified[0, 0, 0] += 11
    _, _, modified_patches = patchify(modified, 2)
    affected = (modified_patches != patches).any(dim=-1).nonzero().flatten()
    assert affected.tolist() == [0]

    # One 2x2 visual merger group changes; others stay unchanged.
    altered_hidden = encoder(modified_patches)
    altered_merged = merge_neighboring_2x2(
        altered_hidden, rows, cols, merger
    )
    merged_diff = (altered_merged-merged).abs().max(dim=-1).values
    assert (merged_diff > 1e-9).nonzero().flatten().tolist() == [0]

    # 3 frames each cost 6 merged visual tokens, plus 20 text tokens.
    assert 20+3*merged.shape[0] == 38
    # A larger frame costs 12 merged visual tokens instead of 6.
    assert 20+3*result2.shape[0] == 56

    # A toy DeepStack-like pathway: features can be injected at more
    # than one layer after projection, but this is NOT a full VLM.
    language_hidden = torch.randn(6, 16, dtype=torch.float64)
    first_visual = merger(hidden.reshape(4, 6, 8).permute(
        0, 1, 2
    ).contiguous().reshape(24, 8).reshape(
        4, 6, 8
    ).reshape(2, 2, 3, 2, 8).permute(0, 2, 1, 3, 4).reshape(6, 32))
    assert torch.allclose(first_visual, merged)
    second_projection = torch.nn.Linear(16, 16, dtype=torch.float64)
    injected_1 = language_hidden + merged
    injected_2 = injected_1 + second_projection(merged)
    assert injected_2.shape == language_hidden.shape
    assert not torch.allclose(injected_1, injected_2)

    # Relative two-axis rotary dot products depend on coordinate difference.
    q = torch.tensor([1.3, -0.5, 0.7, 2.0], dtype=torch.float64)
    k = torch.tensor([0.4, 1.1, -1.7, 0.2], dtype=torch.float64)
    dot1 = torch.dot(rotate_2d(q, 2, 5), rotate_2d(k, 7, 9))
    dot2 = torch.dot(rotate_2d(q, 102, 45), rotate_2d(k, 107, 49))
    assert torch.allclose(dot1, dot2, atol=1e-12)
    assert torch.allclose(
        rotate_2d(q, 0, 0), q, atol=1e-12
    )

    print("PASS vision: patchify/reconstruct 24 tokens and 2x2 merger 6")
    print("PASS vision: variable resolution 6 vs 12 visual tokens")
    print("PASS vision: 2D RoPE relative invariance, layered feature injection")
    print("NOTE: These are synthetic operators, not Qwen3-VL checkpoint ops.")


if __name__ == "__main__":
    self_test()
