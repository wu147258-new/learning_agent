"""Deterministic property tests for softmax with a deliberately bad mutation."""
from math import exp,isclose
import random


def stable_softmax(logits):
    if not logits:
        raise ValueError
    m=max(logits)
    z=[exp(x-m) for x in logits]
    return [v/sum(z) for v in z]


def bad_softmax(logits):
    # Intended mutant: valid-looking normalized distribution ignores logits.
    return [1/len(logits)]*len(logits)


def check(logits, fn):
    a=fn(logits)
    b=fn([x+100 for x in logits])
    assert all(isclose(x,y,abs_tol=1e-12) for x,y in zip(a,b))
    assert isclose(sum(a),1,abs_tol=1e-12)
    assert all(0 <= x <= 1 for x in a)


def main():
    rng=random.Random(2026)
    for length in [2,3,8,25]:
        for _ in range(25):
            x=[rng.uniform(-20,20) for i in range(length)]
            check(x,stable_softmax)
    assert bad_softmax([0,10])==[0.5,0.5]
    assert stable_softmax([0,10])[1]>0.999
    print("PASS 100 seeded stable-softmax shift invariants")
    print("PASS handcrafted counterexample catches normalized-but-wrong mutant")
    print("NOTE: test properties require independent examples and oracle checks.")


if __name__=="__main__":
    main()
