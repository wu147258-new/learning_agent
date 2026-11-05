"""Quantization, KV-cache and paged allocation mathematics.

Standard library only. Run: python examples/inference_quantization_lab.py
The int4 values are logical integers, NOT packed GPU tensor kernels.
Predicted bytes are model formulas, NOT measured device allocations or speed.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isclose, sqrt


@dataclass(frozen=True)
class QuantizedGroup:
    codes: tuple[int, ...]
    scale: float

    def reconstruct(self) -> tuple[float, ...]:
        return tuple(x*self.scale for x in self.codes)


def symmetric_quantize_group(values, bits=4):
    if not values:
        raise ValueError("nonempty group required")
    if bits < 2 or bits > 8:
        raise ValueError("2 to 8 bits supported in this teaching example")
    q_max = 2**(bits-1)-1
    maximum = max(abs(v) for v in values)
    scale = maximum/q_max if maximum else 1.0
    codes = tuple(max(-q_max, min(q_max, round(v/scale))) for v in values)
    result = QuantizedGroup(codes, scale)
    # For this symmetric no-clipping setup, round-to-nearest error <=scale/2.
    assert all(
        abs(original-restored) <= scale/2+1e-10
        for original, restored in zip(values, result.reconstruct())
    )
    return result


def groupwise_quantize(values, group_size: int, bits=4):
    if group_size <= 0:
        raise ValueError("group size must be positive")
    chunks = [
        symmetric_quantize_group(values[i:i+group_size], bits=bits)
        for i in range(0, len(values), group_size)
    ]
    return chunks, tuple(v for group in chunks
                         for v in group.reconstruct())


def mean_squared_error(x, y):
    if len(x) != len(y) or not x:
        raise ValueError("nonempty vectors of same length required")
    return sum((a-b)**2 for a, b in zip(x, y))/len(x)


def logical_weight_bytes(count, bits, group_size, scale_bytes=2):
    if count <= 0 or group_size <= 0 or bits <= 0:
        raise ValueError("positive dimensions required")
    # This excludes kernel layout, alignment, zero points and tensor metadata.
    return ceil(count*bits/8)+ceil(count/group_size)*scale_bytes


def kv_cache_bytes(batch, layers, tokens, kv_heads, head_dim, dtype_bytes):
    if any(type(x) is not int or x < 0
           for x in (batch, layers, tokens, kv_heads, head_dim, dtype_bytes)):
        raise ValueError("nonnegative integer dimensions required")
    return 2*batch*layers*tokens*kv_heads*head_dim*dtype_bytes


def paged_cache_allocation(lengths, block_size):
    if block_size <= 0 or any(n < 0 for n in lengths):
        raise ValueError("invalid request length or page size")
    pages = sum(ceil(n/block_size) for n in lengths)
    capacity = pages*block_size
    return pages, capacity, capacity-sum(lengths)


def schedule_decode_arrivals(arrivals):
    """Token-level admission illustration, not vLLM scheduling code.

    Each request arrives at step t and needs its given number of next tokens.
    At each step we process at most one new token per admitted request.
    """
    if any(t < 0 or total < 1 for t, total in arrivals):
        raise ValueError("invalid arrival pattern")
    pending = [{"at": at, "remaining": total} for at, total in arrivals]
    timeline = []
    t = 0
    while any(item["remaining"] > 0 for item in pending):
        active = 0
        for item in pending:
            if item["at"] <= t and item["remaining"] > 0:
                item["remaining"] -= 1
                active += 1
        timeline.append(active)
        t += 1
        if t > 10000:
            raise RuntimeError("simulation exceeded step guard")
    return timeline


def self_test():
    values = [1.2, -0.9, 0.8, -1.1, -16., 13., -15., 9.]
    whole, restored_whole = groupwise_quantize(values, group_size=8)
    split, restored_split = groupwise_quantize(values, group_size=4)
    assert len(whole) == 1 and len(split) == 2
    mse_whole = mean_squared_error(values, restored_whole)
    mse_split = mean_squared_error(values, restored_split)
    assert mse_split < mse_whole, (mse_whole, mse_split)
    assert all(-7 <= x <= 7 for group in split for x in group.codes)

    # 8 logical 4-bit weights, scales in 2-byte format, 2 groups:
    # theoretical packed codes 4 bytes + two scales 4 bytes = 8 bytes.
    assert logical_weight_bytes(8, 4, 4) == 8
    assert 8*2 == 16  # theoretical FP16 weight-only storage.
    # For huge tensors group overhead remains explicit.
    assert logical_weight_bytes(4096, 4, 128) == 2048+32*2

    mha = kv_cache_bytes(1, 32, 8192, 32, 128, 2)
    gqa = kv_cache_bytes(1, 32, 8192, 8, 128, 2)
    assert mha == 4*1024**3  # 4 GiB, only KV cache.
    assert gqa == 1*1024**3
    assert mha/gqa == 4

    pages, capacity, wasted = paged_cache_allocation(
        lengths=[1, 17, 31], block_size=16
    )
    assert (pages, capacity, wasted) == (5, 80, 31)
    assert paged_cache_allocation([16, 32], 16) == (3, 48, 0)
    assert schedule_decode_arrivals([(0, 2), (1, 3), (2, 1)]) == [1, 2, 2, 1]
    assert isclose(sqrt(mean_squared_error(values, restored_split))**2,
                   mse_split, abs_tol=1e-12)

    print(f"PASS logical int4: whole-group MSE={mse_whole:.4f}, "
          f"two-group MSE={mse_split:.4f}")
    print("PASS KV: 4 GiB MHA vs 1 GiB GQA in specified toy shape")
    print("PASS paged allocation, token-level admission and storage formulas")
    print("NOTE: No actual int4 kernel, GPU benchmark or vLLM server used.")


if __name__ == "__main__":
    self_test()
