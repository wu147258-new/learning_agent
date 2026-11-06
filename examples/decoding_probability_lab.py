"""Stable decoding distribution and local-vs-global path examples."""
from __future__ import annotations
import math


def softmax(xs, temperature=1.0):
    if not xs or temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("invalid logits/temperature")
    if not all(math.isfinite(x) for x in xs):
        raise ValueError("non-finite logit")
    m = max(xs)
    weights = [math.exp((x-m)/temperature) for x in xs]
    return [x/sum(weights) for x in weights]


def top_p(probs, threshold):
    if not 0 < threshold <= 1 or not probs:
        raise ValueError("invalid nucleus threshold")
    if any(x < 0 for x in probs) or not math.isclose(sum(probs), 1, abs_tol=1e-10):
        raise ValueError("invalid distribution")
    order = sorted(range(len(probs)), key=lambda i: (-probs[i], i))
    keep, total = [], 0.0
    for i in order:
        keep.append(i)
        total += probs[i]
        if total+1e-12 >= threshold:
            break
    return {i: probs[i]/total for i in keep}


def main():
    p = softmax([1000.0, 1001.0])
    assert math.isclose(sum(p), 1.0)
    assert p[1] > p[0]
    assert softmax([3.0, 1.0], 0.5)[0] > softmax([3.0, 1.0], 2.0)[0]
    nucl = top_p([0.4, 0.3, 0.2, 0.1], 0.65)
    assert set(nucl) == {0, 1}
    assert math.isclose(nucl[0], 4/7)
    assert math.isclose(0.6*0.5, 0.3)
    assert 0.4*0.95 > 0.6*0.5
    for op in (lambda: softmax([]), lambda: softmax([1], 0),
               lambda: top_p([0.5, 0.5], 0)):
        try:
            op()
        except ValueError:
            pass
        else:
            raise AssertionError("bad argument accepted")
    print("PASS stable softmax, temperature, top-p and two-step counterexample")


if __name__ == "__main__":
    main()
