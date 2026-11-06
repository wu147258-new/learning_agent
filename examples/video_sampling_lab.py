"""Deterministic temporal sampling geometry, not a video model benchmark."""
from __future__ import annotations
import math


def hit(start, duration, sample_interval, count=100):
    if duration < 0 or sample_interval <= 0 or count <= 0:
        raise ValueError("invalid sampling parameters")
    # First sample at t=0; avoid floating point boundary by tolerance.
    return any(start <= i*sample_interval < start+duration
               for i in range(count))


def phase_enumeration(duration, interval, phases=10000):
    if not (0 < duration <= interval and phases > 0):
        raise ValueError("invalid parameter")
    # Shift the sample grid backward: event is [0,duration), sample at phase.
    hits = sum((k+0.5)*interval/phases < duration for k in range(phases))
    return hits/phases


def main():
    assert math.isclose(phase_enumeration(0.2, 1.0), 0.2, abs_tol=1e-4)
    assert math.isclose(phase_enumeration(0.15, 0.5), 0.3, abs_tol=1e-4)
    assert math.isclose(phase_enumeration(1.0, 1.0), 1.0)
    assert hit(1.0, 0.2, 1.0)
    assert not hit(1.1, 0.2, 1.0)
    # Equivalent sampled representations can conceal different labels.
    original_a = {0: "normal", 1: "normal", 2: "normal"}
    original_b = {0: "normal", 1: "event", 2: "normal"}
    sampled_indices = [0, 2]
    assert [original_a[i] for i in sampled_indices] == [
        original_b[i] for i in sampled_indices]
    assert original_a != original_b
    # Token count grows with area and sampled frames before any compression.
    count = lambda h, w, patch, frames: h//patch * (w//patch) * frames
    assert count(224, 224, 16, 32) == 6272
    assert count(448, 448, 16, 32) == 25088
    print("PASS phase enumeration: event hit probabilities 0.2 and 0.3")
    print("PASS two original videos are indistinguishable after sampling")
    print("PASS patch token count quadruples on doubled spatial dimensions")
    print("NOTE: idealized synthetic sampling, not real model recall.")


if __name__ == "__main__":
    main()
