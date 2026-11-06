"""Exact single-state off-policy expectation and effective sample size."""
from math import isclose


def estimate_exact(behavior, target, rewards):
    assert len(behavior) == len(target) == len(rewards)
    assert all(x >= 0 for x in behavior+target)
    assert isclose(sum(behavior),1.0) and isclose(sum(target),1.0)
    if any(p == 0 and q > 0 for p, q in zip(behavior,target)):
        raise ValueError("target action not supported by behavior data")
    return sum(p*(q/p)*r for p,q,r in zip(behavior,target,rewards) if p)


def ess(weights):
    if not weights or any(w < 0 for w in weights) or sum(weights)==0:
        raise ValueError("invalid weights")
    return sum(weights)**2/sum(w*w for w in weights)


def main():
    mu=[0.9,0.1]
    pi=[0.5,0.5]
    reward=[1,0]
    assert isclose(sum(p*r for p,r in zip(mu,reward)),0.9)
    assert isclose(estimate_exact(mu,pi,reward),0.5)
    assert isclose(ess([1,1,1]),3)
    assert 1 < ess([100,1,1]) < 1.05
    try:
        estimate_exact([1,0],[0.9,0.1],reward)
    except ValueError:
        pass
    else:
        raise AssertionError("support mismatch accepted")
    print("PASS exact importance sampling changes old 0.9 to target 0.5")
    print("PASS effective sample size detects concentration")
    print("PASS zero support for required target action rejected")


if __name__=="__main__":
    main()
