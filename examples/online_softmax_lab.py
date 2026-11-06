"""Streaming exact softmax (CPU math), not an accelerated GPU kernel."""
import math
import random


def reference(scores, values):
    if not scores or len(scores) != len(values):
        raise ValueError("empty or mismatched sequences")
    m = max(scores)
    weights = [math.exp(x-m) for x in scores]
    return sum(w*v for w,v in zip(weights,values))/sum(weights)


def online(scores, values):
    if not scores or len(scores) != len(values):
        raise ValueError("empty or mismatched sequences")
    m, denom, numerator = -math.inf, 0.0, 0.0
    for score, value in zip(scores, values):
        new_m = max(m, score)
        prev = math.exp(m-new_m) if m != -math.inf else 0
        cur = math.exp(score-new_m)
        denom = prev*denom + cur
        numerator = prev*numerator + cur*value
        m = new_m
    return numerator/denom


def main():
    rng=random.Random(28)
    for length in (1,2,3,7,31,100):
        for _ in range(30):
            scores=[rng.uniform(-1000,1000) for i in range(length)]
            values=[rng.uniform(-10,10) for i in range(length)]
            a,b=reference(scores,values),online(scores,values)
            assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10), (a,b)
    assert math.isclose(online([1000,1001],[2,4]),
                        (2+4*math.e)/(1+math.e),rel_tol=1e-12)
    for invalid in [([],[]),([1.0],[]),([], [1.0])]:
        try:
            online(*invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid input accepted")
    print("PASS online and stable reference attention agree over 180 cases")
    print("PASS extremes, singleton and invalid sequence handling")
    print("NOTE: no GPU compute or throughput benchmark.")


if __name__=="__main__":
    main()
