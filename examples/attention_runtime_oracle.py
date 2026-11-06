"""A pure-Python oracle for attention, rotary positions, and KV paging.

Not a GPU kernel or a substitute for a production serving engine.
Run: python3 examples/attention_runtime_oracle.py
"""
from __future__ import annotations

from dataclasses import dataclass
from math import cos, exp, isclose, sin, sqrt
from typing import Iterable


def dot(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("shape mismatch")
    return sum(x * y for x, y in zip(a, b))


def rotary(pair: tuple[float, float], pos: float, frequency: float):
    a, b = pair
    theta = pos * frequency
    return (a * cos(theta) - b * sin(theta),
            a * sin(theta) + b * cos(theta))


def rope(vec: list[float], position: float, base: float = 10000.0):
    if len(vec) % 2:
        raise ValueError("RoPE needs even head width")
    n = len(vec)
    out = []
    for j in range(0, n, 2):
        frequency = base ** (-j / n)
        out.extend(rotary((vec[j], vec[j + 1]), position, frequency))
    return out


def three_axis_rope(vec: list[float], t: float, row: float, col: float):
    if len(vec) != 12:
        raise ValueError("this lesson uses 12 dimensions")
    return (
        rope(vec[:4], t)
        + rope(vec[4:8], row)
        + rope(vec[8:], col)
    )


def normal_softmax(scores: list[float]) -> list[float]:
    if not scores:
        raise ValueError("empty scores")
    m = max(scores)
    weights = [exp(x - m) for x in scores]
    scale = sum(weights)
    return [w / scale for w in weights]


def dense_attention(query, keys, values, allowed=None):
    """One query and many keys; O(T) scalar weights."""
    if len(keys) != len(values) or not keys:
        raise ValueError("invalid KV sequence")
    if allowed is None:
        allowed = [True] * len(keys)
    if len(allowed) != len(keys):
        raise ValueError("invalid mask")
    chosen = [(k, v) for k, v, ok in zip(keys, values, allowed) if ok]
    if not chosen:
        raise ValueError("no permitted key")
    keys, values = zip(*chosen)
    dh = len(query)
    if not dh or any(len(k) != dh for k in keys):
        raise ValueError("Q/K dimension mismatch")
    dv = len(values[0])
    if any(len(v) != dv for v in values):
        raise ValueError("V dimension mismatch")
    probs = normal_softmax([dot(query, k) / sqrt(dh) for k in keys])
    return [sum(p * values[i][j] for i, p in enumerate(probs))
            for j in range(dv)]


@dataclass(frozen=True)
class OnlineAcc:
    maximum: float
    mass: float
    numerator: tuple[float, ...]

    @staticmethod
    def empty(width: int) -> "OnlineAcc":
        return OnlineAcc(float("-inf"), 0.0, (0.0,) * width)

    def push(self, score: float, value: list[float]) -> "OnlineAcc":
        if len(value) != len(self.numerator):
            raise ValueError("V width changed")
        m = max(self.maximum, score)
        old_scale = 0.0 if self.mass == 0 else exp(self.maximum - m)
        new_scale = exp(score - m)
        numerator = tuple(
            old_scale * self.numerator[j] + new_scale * value[j]
            for j in range(len(value))
        )
        return OnlineAcc(m, old_scale * self.mass + new_scale, numerator)

    def combine(self, other: "OnlineAcc") -> "OnlineAcc":
        """Stable combination of independent tile accumulators."""
        if len(self.numerator) != len(other.numerator):
            raise ValueError("accumulator width mismatch")
        if self.mass == 0:
            return other
        if other.mass == 0:
            return self
        m = max(self.maximum, other.maximum)
        aa, bb = exp(self.maximum - m), exp(other.maximum - m)
        numerator = tuple(
            aa * a + bb * b
            for a, b in zip(self.numerator, other.numerator)
        )
        return OnlineAcc(
            m, aa * self.mass + bb * other.mass, numerator
        )

    def result(self) -> list[float]:
        if self.mass <= 0:
            raise ValueError("empty accumulator")
        return [v / self.mass for v in self.numerator]


def tiled_attention(query, keys, values, tile=2, allowed=None):
    if tile <= 0:
        raise ValueError("tile must be positive")
    if len(keys) != len(values) or not keys:
        raise ValueError("invalid KV sequence")
    if allowed is None:
        allowed = [True] * len(keys)
    if len(allowed) != len(keys):
        raise ValueError("mask length mismatch")
    width = len(values[0])
    total = OnlineAcc.empty(width)
    for start in range(0, len(keys), tile):
        block = OnlineAcc.empty(width)
        for i in range(start, min(start + tile, len(keys))):
            if not allowed[i]:
                continue
            score = dot(query, keys[i]) / sqrt(len(query))
            block = block.push(score, values[i])
        total = total.combine(block)
    return total.result()


@dataclass(frozen=True)
class Block:
    block_id: int
    owner: str
    used: int
    capacity: int


class KVPagePool:
    """Toy physical block allocation. No GPU pointers or async kernels."""

    def __init__(self, block_size: int, physical_blocks: int):
        if block_size <= 0 or physical_blocks <= 0:
            raise ValueError("positive dimensions required")
        self.block_size = block_size
        self.free = set(range(physical_blocks))
        self.entries: dict[str, list[Block]] = {}

    def reserved(self, request: str) -> int:
        return sum(b.capacity for b in self.entries.get(request, []))

    def live(self, request: str) -> int:
        return sum(b.used for b in self.entries.get(request, []))

    def append(self, request: str, tokens: int):
        if tokens < 0 or not request:
            raise ValueError("invalid request")
        old = self.entries.setdefault(request, [])
        while tokens:
            if old and old[-1].used < old[-1].capacity:
                last = old.pop()
                delta = min(tokens, last.capacity - last.used)
                old.append(Block(last.block_id, request,
                                 last.used + delta, last.capacity))
                tokens -= delta
            else:
                if not self.free:
                    raise MemoryError("no spare KV blocks")
                block_id = min(self.free)
                self.free.remove(block_id)
                old.append(Block(block_id, request, 0, self.block_size))

    def release(self, request: str):
        for block in self.entries.pop(request, []):
            self.free.add(block.block_id)

    def assert_invariants(self):
        allocated = [b.block_id for blocks in self.entries.values()
                     for b in blocks]
        assert len(allocated) == len(set(allocated))
        assert not set(allocated).intersection(self.free)
        assert all(0 < b.used <= b.capacity
                   for blocks in self.entries.values() for b in blocks)


class TokenScheduler:
    """Deterministic token-step budget; no preemption side-effects."""

    def __init__(self, maximum: int, pool: KVPagePool):
        if maximum <= 0:
            raise ValueError("invalid token budget")
        self.maximum = maximum
        self.pool = pool
        self.remaining: dict[str, int] = {}
        self.completed: list[str] = []

    def admit(self, request: str, output_limit: int):
        if request in self.remaining or request in self.completed:
            raise ValueError("duplicate request")
        if output_limit <= 0:
            raise ValueError("output_limit must be positive")
        self.remaining[request] = output_limit

    def step(self) -> list[str]:
        scheduled = sorted(self.remaining)[:self.maximum]
        completed_now = []
        for req in scheduled:
            self.pool.append(req, 1)
            self.remaining[req] -= 1
            if self.remaining[req] == 0:
                completed_now.append(req)
        for req in completed_now:
            del self.remaining[req]
            self.completed.append(req)
            self.pool.release(req)
        return scheduled


def assert_close_vectors(a, b, tolerance=1e-10):
    assert len(a) == len(b)
    assert all(isclose(x, y, abs_tol=tolerance, rel_tol=tolerance)
               for x, y in zip(a, b)), (a, b)


def test_rotation():
    q, k = [1., 2., 3., 4.], [3., 1., -1., 2.]
    for i, j in [(0, 2), (6, 8), (100, 102), (12, 7)]:
        qi = rope(q, i)
        kj = rope(k, j)
        ref = dot(q, rope(k, j - i))
        assert abs(dot(qi, kj) - ref) < 1e-11
    x = list(range(1, 13))
    a = three_axis_rope(x, 5, 2, 3)
    b = three_axis_rope(x, 6, 2, 3)
    assert a != b
    assert a[4:8] == b[4:8]
    assert a[8:] == b[8:]
    print("PASS rotary relative offsets and axis separation")


def test_online_softmax():
    q = [1.0, 2.0, -0.5, 0.25]
    keys = [[float(j + i / 7) for j in range(4)] for i in range(13)]
    vals = [[float(i % 5), float(i*i % 11)] for i in range(13)]
    for mask in ([True] * 13, [i % 2 == 0 for i in range(13)]):
        dense = dense_attention(q, keys, vals, mask)
        for tile in (1, 2, 3, 5, 13, 20):
            tiled = tiled_attention(q, keys, vals, tile, mask)
            assert_close_vectors(dense, tiled)
    qlarge = [1000.0]
    keys_large = [[1.0], [0.999], [0.998]]
    vals_large = [[1.0], [2.0], [3.0]]
    assert_close_vectors(
        dense_attention(qlarge, keys_large, vals_large),
        tiled_attention(qlarge, keys_large, vals_large),
    )
    print("PASS online softmax vs independent dense oracle")


def test_pages_scheduler():
    pool = KVPagePool(4, 8)
    pool.append("a", 5)
    pool.append("b", 7)
    assert (pool.reserved("a"), pool.reserved("b")) == (8, 8)
    assert (pool.live("a"), pool.live("b")) == (5, 7)
    pool.assert_invariants()
    pool.release("a")
    pool.release("b")
    assert len(pool.free) == 8

    sched = TokenScheduler(2, pool)
    sched.admit("short", 2)
    sched.admit("long", 4)
    sched.admit("medium", 3)
    sequence = []
    while sched.remaining:
        sequence.append(sched.step())
        pool.assert_invariants()
    assert sorted(sched.completed) == ["long", "medium", "short"]
    assert len(pool.free) == 8
    assert sequence[0] == ["long", "medium"]
    assert "short" in sequence[-1]
    print("PASS KV allocation, release and deterministic scheduling")


def main():
    test_rotation()
    test_online_softmax()
    test_pages_scheduler()
    print("NOTE: reference mathematics only; no FlashAttention GPU benchmark.")


if __name__ == "__main__":
    main()
