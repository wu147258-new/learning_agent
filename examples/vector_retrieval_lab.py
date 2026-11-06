"""Exact 2D retrieval, two distinct recall metrics, and pre-filtered access."""
from __future__ import annotations
import math


def cosine(a,b):
    if len(a)!=len(b):
        raise ValueError("dimension mismatch")
    n1=math.sqrt(sum(x*x for x in a))
    n2=math.sqrt(sum(x*x for x in b))
    if n1==0 or n2==0:
        raise ValueError("zero embedding")
    return sum(x*y for x,y in zip(a,b))/(n1*n2)


def legal_topk(query,items,tenant,k):
    if k<1:
        raise ValueError("k must be positive")
    legal=[(name,vec) for name,owner,vec in items if owner==tenant]
    return [name for name,vec in sorted(legal,
        key=lambda x:(-cosine(query,x[1]),x[0]))[:k]]


def recall(retrieved,reference):
    if not reference:
        raise ValueError("empty relevant set")
    return len(set(retrieved)&set(reference))/len(reference)


def main():
    assert math.isclose(cosine([1,0],[0,1]),0)
    assert math.isclose(math.dist([1,0],[.8,.6])**2,
                        2-2*cosine([1,0],[.8,.6]))
    exact=["A","B","C","D","E"]
    ann=["A","B","C","F","G"]
    relevance=["A","F","H"]
    assert recall(ann,exact)==0.6
    assert math.isclose(recall(ann,relevance),2/3)
    corpus=[("leaked","other",[1,0]),
            ("allowed1","mine",[.8,.6]),
            ("allowed2","mine",[0,1])]
    top=legal_topk([1,0],corpus,"mine",2)
    assert top==["allowed1","allowed2"]
    assert "leaked" not in top
    for fn in (lambda: cosine([0,0],[1,0]),
               lambda: legal_topk([1,0],corpus,"mine",0)):
        try:
            fn()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid request accepted")
    print("PASS cosine and Euclidean unit-vector identity")
    print("PASS ANN recall@5=0.6 versus relevance recall@5=2/3")
    print("PASS tenant filter precedes candidate similarity search")
    print("NOTE: synthetic vectors, not an ANN engine benchmark.")


if __name__=="__main__":
    main()
