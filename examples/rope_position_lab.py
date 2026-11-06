"""RoPE rotation identity and timestamp unit tests. Pure standard library."""
from __future__ import annotations
from math import cos, sin, hypot, isclose, pi
import random


def rotate(v: tuple[float, float], angle: float) -> tuple[float, float]:
    x, y = v
    return (cos(angle)*x - sin(angle)*y,
            sin(angle)*x + cos(angle)*y)


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def main():
    rng = random.Random(2026)
    for _ in range(250):
        q = (rng.uniform(-3, 3), rng.uniform(-3, 3))
        k = (rng.uniform(-3, 3), rng.uniform(-3, 3))
        m, n = rng.randrange(-100, 100), rng.randrange(-100, 100)
        frequency = rng.uniform(0.0001, 1.8)
        q_m = rotate(q, m*frequency)
        k_n = rotate(k, n*frequency)
        expected = dot(q, rotate(k, (n-m)*frequency))
        assert isclose(dot(q_m, k_n), expected,
                       rel_tol=1e-11, abs_tol=1e-11)
        assert isclose(hypot(*q), hypot(*q_m),
                       rel_tol=1e-11, abs_tol=1e-11)
    assert isclose(dot(rotate((1, 0), pi/2),
                       rotate((1, 0), pi)), 0.0, abs_tol=1e-11)
    # Unit conversion is a semantic obligation: milliseconds != seconds.
    seconds = (0.0, 0.04, 0.08)
    millis = tuple(x * 1000.0 for x in seconds)
    assert isclose(millis[2] / seconds[2], 1000)
    assert not isclose(rotate((1, 0), seconds[2]),
                       rotate((1, 0), millis[2]), abs_tol=1e-6)
    print("PASS RoPE identity: relative positions and preserved norms")
    print("PASS timestamp-unit mismatch causes changed phase")


if __name__ == "__main__":
    main()
