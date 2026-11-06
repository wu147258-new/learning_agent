"""Generalized advantage estimation and time-limit bootstrap correctness."""
from __future__ import annotations
import math


def gae(rewards, values, terminals, gamma, lam):
    if not 0 <= gamma <= 1 or not 0 <= lam <= 1:
        raise ValueError("invalid discount/trace")
    if len(values) != len(rewards)+1 or len(terminals) != len(rewards):
        raise ValueError("length mismatch")
    result = [0.0]*len(rewards)
    next_gae = 0.0
    for t in range(len(rewards)-1,-1,-1):
        continuation = 0.0 if terminals[t] else 1.0
        delta = rewards[t] + gamma*continuation*values[t+1] - values[t]
        next_gae = delta + gamma*lam*continuation*next_gae
        result[t] = next_gae
    return result


def main():
    # When all V=0, GAE is a discounted sum of rewards with factor gamma*lambda.
    out = gae([1,2,3],[0,0,0,0],[False,False,True],1.0,0.5)
    assert all(math.isclose(a,b) for a,b in zip(out,[2.75,3.5,3.0]))
    # With true terminal, final estimated value must not leak into target.
    terminated = gae([0],[3,5],[True],0.9,0.95)[0]
    truncated = gae([0],[3,5],[False],0.9,0.95)[0]
    assert math.isclose(terminated,-3.0)
    assert math.isclose(truncated,1.5)
    # lambda=0 is one-step TD.
    assert gae([1],[2,3],[False],0.9,0.0) == [1+0.9*3-2]
    # Two episodes must not leak future rewards across first terminal.
    out = gae([5,8],[0,0,0],[True,True],1,1)
    assert out == [5,8],out
    for thunk in (lambda: gae([1],[0],[False],0.9,0.8),
                  lambda: gae([1],[0,0],[False],1.1,0.8)):
        try:
            thunk()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid input accepted")
    print("PASS GAE exact 3-step recurrence and lambda endpoint")
    print("PASS true terminal versus time truncation (-3 versus 1.5)")
    print("PASS no cross-episode reward leakage")
    print("NOTE: synthetic arithmetic, not learned policy improvement.")


if __name__ == "__main__":
    main()
