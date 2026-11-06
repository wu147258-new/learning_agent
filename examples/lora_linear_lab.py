"""Small exact LoRA low-rank branch and finite-difference gradient tests.

Standard-library exercise: no pretrained weights, GPU or model benchmarks.
"""
from __future__ import annotations
from math import isclose


def matvec(matrix, vector):
    if any(len(row)!=len(vector) for row in matrix):
        raise ValueError("shape mismatch")
    return [sum(x*y for x,y in zip(row,vector)) for row in matrix]


def lora_forward(w,a,b,x,alpha):
    if not a or not b or not x:
        raise ValueError("empty matrices")
    rank=len(a)
    if any(len(row)!=len(x) for row in a):
        raise ValueError("A shape mismatch")
    if any(len(row)!=rank for row in b):
        raise ValueError("B shape mismatch")
    base=matvec(w,x)
    low=matvec(b,matvec(a,x))
    if len(base)!=len(low):
        raise ValueError("output shape mismatch")
    return [v+(alpha/rank)*u for v,u in zip(base,low)]


def squared_loss(y,target):
    return sum((v-t)**2 for v,t in zip(y,target))/2


def main():
    w=[[1.,0.,0.],[0.,1.,0.]]
    a=[[.5,-.1,.2]]
    b=[[.3],[-.2]]
    x=[2.,3.,4.]
    alpha=2.0
    target=[1.,1.]
    actual=lora_forward(w,a,b,x,alpha)
    base=matvec(w,x)
    assert len(actual)==2 and any(v!=u for v,u in zip(actual,base))
    assert len(a)*len(x)+len(b)*len(a)==5
    step=1e-6
    derivs=[]
    for index in range(len(b)):
        bp=[row[:] for row in b]
        bm=[row[:] for row in b]
        bp[index][0]+=step
        bm[index][0]-=step
        lp=squared_loss(lora_forward(w,a,bp,x,alpha),target)
        lm=squared_loss(lora_forward(w,a,bm,x,alpha),target)
        derivs.append((lp-lm)/(2*step))
    ax=matvec(a,x)[0]
    expected=[(actual[i]-target[i])*(alpha/len(a))*ax
              for i in range(len(b))]
    assert all(isclose(v,e,rel_tol=1e-7,abs_tol=1e-7)
               for v,e in zip(derivs,expected)), (derivs,expected)
    zero_b=[[0.], [0.]]
    assert lora_forward(w,a,zero_b,x,alpha)==base
    print("PASS LoRA shapes, parameter count, zero-initialized adapter")
    print("PASS B gradient matches centered finite differences")
    print("NOTE: tiny synthetic matrix, not an LLM fine-tuning benchmark.")


if __name__=="__main__":
    main()
