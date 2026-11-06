"""Exact 0-1 context packing and coverage counterexamples."""
from itertools import combinations


def exact(items,budget):
    if budget < 0 or any(i["cost"] <= 0 for i in items):
        raise ValueError("invalid budget/cost")
    best=(0,())
    for n in range(len(items)+1):
        for group in combinations(items,n):
            cost=sum(x["cost"] for x in group)
            utility=sum(x["utility"] for x in group)
            names=tuple(x["id"] for x in group)
            if cost <= budget and utility > best[0]:
                best=(utility,names)
    return best


def coverage(items):
    return len(set().union(*(i["facts"] for i in items))) if items else 0


def main():
    items=[
      {"id":"A","cost":6,"utility":12,"facts":{"x"}},
      {"id":"B","cost":5,"utility":11,"facts":{"x"}},
      {"id":"C","cost":5,"utility":11,"facts":{"y"}},
    ]
    score, selected=exact(items,10)
    assert score==22 and set(selected)=={"B","C"}
    assert coverage(items)==2
    assert coverage(items[:2])==1
    try:
        exact(items,-1)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid budget allowed")
    print("PASS exact packing B+C=22 beats naive A=12")
    print("PASS repeated evidence does not multiply fact coverage")


if __name__=="__main__":
    main()
