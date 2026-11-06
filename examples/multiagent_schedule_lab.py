"""DAG critical path, topological sorting and optimistic version locks."""
from collections import deque


def topo_and_critical_path(nodes, dependencies, durations):
    indegree={n:0 for n in nodes}
    nexts={n:[] for n in nodes}
    for before,after in dependencies:
        if before not in indegree or after not in indegree:
            raise ValueError("unknown task")
        indegree[after]+=1
        nexts[before].append(after)
    queue=deque(n for n in nodes if not indegree[n])
    finish={n:durations[n] for n in nodes}
    order=[]
    while queue:
        n=queue.popleft()
        order.append(n)
        for v in nexts[n]:
            finish[v]=max(finish[v],finish[n]+durations[v])
            indegree[v]-=1
            if indegree[v]==0:
                queue.append(v)
    if len(order)!=len(nodes):
        raise ValueError("cycle")
    return order,max(finish.values())


class Blackboard:
    def __init__(self):
        self.version=0
        self.note=""

    def cas(self,expected,note):
        if expected!=self.version:
            return False
        self.version+=1
        self.note=note
        return True


def main():
    nodes=["A","B","C","D"]
    deps=[("A","B"),("A","C"),("B","D"),("C","D")]
    duration={"A":3,"B":5,"C":4,"D":2}
    order,t=topo_and_critical_path(nodes,deps,duration)
    assert order[0]=="A" and order[-1]=="D"
    assert t==10
    assert sum(duration.values())==14
    try:
        topo_and_critical_path(nodes,deps+[("D","A")],duration)
    except ValueError:
        pass
    else:
        raise AssertionError("cycle accepted")
    bb=Blackboard()
    assert bb.cas(0,"first")
    assert not bb.cas(0,"stale")
    assert bb.cas(1,"latest")
    assert bb.version==2 and bb.note=="latest"
    print("PASS critical path=10, total work=14")
    print("PASS dependency cycle rejection and version CAS")


if __name__=="__main__":
    main()
