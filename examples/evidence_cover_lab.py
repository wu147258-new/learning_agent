"""Minimum cost verified evidence cover: tiny exact reference baseline."""
from itertools import combinations

def solve(claims,candidates,tenant):
    legal=[c for c in candidates if c["tenant"]==tenant and c["valid"]]
    for c in legal:
        if c["cost"]<=0:
            raise ValueError("cost must be positive")
    opts=[]
    for n in range(1,len(legal)+1):
        for subset in combinations(legal,n):
            covered=set().union(*(x["claims"] for x in subset))
            if set(claims) <= covered:
                opts.append((sum(x["cost"] for x in subset),
                             tuple(sorted(x["id"] for x in subset))))
    if not opts:
        raise ValueError("insufficient authorized evidence")
    return min(opts)


def main():
    candidates=[
      {"id":"E1","claims":{"A","B"},"cost":4,"tenant":"u","valid":True},
      {"id":"E2","claims":{"C"},"cost":2,"tenant":"u","valid":True},
      {"id":"E3","claims":{"A","B","C"},"cost":9,"tenant":"u","valid":True},
      {"id":"leak","claims":{"A","B","C"},"cost":1,"tenant":"other","valid":True},
      {"id":"stale","claims":{"A","B","C"},"cost":1,"tenant":"u","valid":False},
    ]
    assert solve({"A","B","C"},candidates,"u")== (6,("E1","E2"))
    try:
        solve({"A","B","C","D"},candidates,"u")
    except ValueError:
        pass
    else:
        raise AssertionError("unsupported claim was silently accepted")
    print("PASS exact minimum cost authorized evidence cover=6")
    print("PASS cross-tenant, stale and uncovered citations rejected")


if __name__=="__main__":
    main()
