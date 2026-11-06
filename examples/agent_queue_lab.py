"""Discrete bounded queue with explicit overload rejection, standard library."""
from collections import deque


def simulate(arrivals,service_per_tick,capacity):
    if capacity<=0 or service_per_tick<=0:
        raise ValueError
    queue=deque()
    accepted=rejected=completed=0
    for t,n in enumerate(arrivals):
        if n<0: raise ValueError
        for i in range(n):
            if len(queue)>=capacity:
                rejected+=1
            else:
                queue.append((t,i))
                accepted+=1
        for _ in range(min(len(queue),service_per_tick)):
            queue.popleft()
            completed+=1
        assert len(queue)<=capacity
        assert accepted==completed+len(queue)
    return accepted,rejected,completed,len(queue)


def main():
    a,r,c,q=simulate([12]*100,8,20)
    assert a+r==1200
    assert c+q==a
    assert 0<=q<=20
    assert r>0
    a2,r2,c2,q2=simulate([4]*100,8,20)
    assert r2==0 and q2==0 and c2==400
    print(f"PASS bounded overload: rejected={r}, accepted={a}, queued={q}")
    print("PASS conservation and non-overloaded control run")


if __name__=="__main__":
    main()
