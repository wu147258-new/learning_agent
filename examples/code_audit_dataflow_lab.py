"""Minimal source-to-sink flow graph as a deterministic static analysis toy."""
from collections import deque


def reachable(edges, start, stop, blocked=frozenset()):
    graph={}
    for a,b in edges:
        graph.setdefault(a,[]).append(b)
    todo=deque([(start,)])
    seen=set()
    while todo:
        path=todo.popleft()
        vertex=path[-1]
        if vertex==stop:
            return path
        if vertex in seen or vertex in blocked:
            continue
        seen.add(vertex)
        for nxt in graph.get(vertex,[]):
            todo.append(path+(nxt,))
    return None


def main():
    edges=[
      ("user_input","assign_y"),
      ("assign_y","execute_query"),
      ("constant","execute_fixed"),
      ("user_input","validated_parameter"),
      ("validated_parameter","parameterized_query"),
    ]
    path=reachable(edges,"user_input","execute_query")
    assert path == ("user_input","assign_y","execute_query")
    assert reachable(edges,"user_input","execute_query",{"assign_y"}) is None
    assert reachable(edges,"constant","execute_query") is None
    assert reachable(edges,"user_input","parameterized_query",
                     {"validated_parameter"}) is None
    print("PASS potential source-to-sink path and explicit barrier")
    print("PASS constant source cannot be confused with user-tainted source")
    print("NOTE: deliberately small graph; not a full language taint engine.")


if __name__=="__main__":
    main()
