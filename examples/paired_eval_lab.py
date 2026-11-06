"""Paired binary score comparison with exact McNemar and task bootstrap."""
from __future__ import annotations
import math
import random


def paired_summary(rows):
    if not rows or any(a not in (0,1) or b not in (0,1) for a,b in rows):
        raise ValueError("invalid paired outcomes")
    gained=sum(a==0 and b==1 for a,b in rows)
    lost=sum(a==1 and b==0 for a,b in rows)
    n=len(rows)
    diff=(gained-lost)/n
    discordant=gained+lost
    if discordant == 0:
        exact_p=1.0
    else:
        lower=min(gained,lost)
        exact_p=min(1,2*sum(math.comb(discordant,i) for i in range(lower+1))
                         /2**discordant)
    return diff,exact_p,gained,lost


def bootstrap(rows,seed=2026,trials=1000):
    rng=random.Random(seed)
    out=[]
    for _ in range(trials):
        sample=[rows[rng.randrange(len(rows))] for _ in range(len(rows))]
        out.append(sum(b-a for a,b in sample)/len(rows))
    out.sort()
    return out[int(.025*trials)],out[int(.975*trials)]


def main():
    rows=[(0,1)]*20+[(1,0)]*10+[(1,1)]*40+[(0,0)]*30
    diff,p,b,c=paired_summary(rows)
    assert math.isclose(diff,.1)
    assert (b,c)==(20,10)
    assert 0 <= p <= 1
    low,high=bootstrap(rows)
    assert low < diff < high
    assert bootstrap(rows)==(low,high)
    assert paired_summary([(0,0),(1,1)])[1]==1
    print("PASS paired gain=20, regression=10, delta=0.10")
    print(f"PASS exact McNemar p={p:.6f}, fixed-seed paired bootstrap")
    print("NOTE: all results synthetic, not real Agent benchmarking.")


if __name__=="__main__":
    main()
