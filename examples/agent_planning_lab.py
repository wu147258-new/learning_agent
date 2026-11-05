"""A* planning, permission checks, deterministic replanning and expected costs.

Pure standard library. No LLM, shell commands, filesystem writes or network.
Run: python examples/agent_planning_lab.py
The action failure injected here is explicitly KNOWN to have no side effect.
A real timed-out remote write must not be assumed to share this property.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
import math


@dataclass(frozen=True)
class Edge:
    action: str
    next_state: str
    cost: float
    permission: str


Graph = dict[str, list[Edge]]


def make_graph() -> Graph:
    return {
        "start": [
            Edge("quick_check", "risk", 1.0, "read"),
            Edge("direct_review", "audit", 3.0, "read"),
        ],
        "risk": [
            Edge("fast_deploy", "goal", 10.0, "deploy"),
            Edge("escalate_review", "audit", 1.0, "read"),
        ],
        "audit": [
            Edge("verified_deploy", "goal", 2.0, "deploy"),
        ],
        "goal": [],
    }


def validate_edges(graph: Graph) -> None:
    for source, edges in graph.items():
        for e in edges:
            if e.next_state not in graph:
                raise ValueError("edge points to missing state")
            if not math.isfinite(e.cost) or e.cost < 0:
                raise ValueError("A* requires finite nonnegative costs")
            if not e.action or not e.permission:
                raise ValueError("missing action or permission")


def bfs_edges(graph: Graph, start: str, goal: str) -> list[tuple[str, Edge]]:
    """Shortest number-of-edges path, ignoring heterogeneous costs."""
    queue = deque([start])
    parent: dict[str, tuple[str, Edge]] = {}
    seen = {start}
    while queue:
        state = queue.popleft()
        if state == goal:
            return reconstruct(parent, start, goal)
        for edge in graph[state]:
            if edge.next_state not in seen:
                seen.add(edge.next_state)
                parent[edge.next_state] = (state, edge)
                queue.append(edge.next_state)
    return []


def reconstruct(parent, start, goal):
    route = []
    state = goal
    while state != start:
        prior_state, action = parent[state]
        route.append((prior_state, action))
        state = prior_state
    return route[::-1]


def astar(graph: Graph, start: str, goal: str,
          allowed_permissions: frozenset[str],
          heuristic: dict[str, float] | None = None,
          blocked: frozenset[tuple[str, str]] = frozenset()
          ) -> list[tuple[str, Edge]]:
    """A* with reopening, versioned state assumptions and stable heap order.

    Optimal cost is guaranteed if edge weights are nonnegative and the
    supplied heuristic is admissible. Reopening supports inconsistent h.
    """
    validate_edges(graph)
    if start not in graph or goal not in graph:
        raise ValueError("unknown state")
    h = heuristic if heuristic is not None else {s: 0.0 for s in graph}
    if set(h) != set(graph) or any(not math.isfinite(z) or z < 0
                                  for z in h.values()):
        raise ValueError("invalid heuristic")
    if start == goal:
        return []

    serial = count()
    heap = [(h[start], 0.0, next(serial), start)]
    g_best = {start: 0.0}
    parent = {}
    while heap:
        _, g, _, state = heappop(heap)
        if g > g_best[state] + 1e-12:
            continue
        if state == goal:
            return reconstruct(parent, start, goal)
        for edge in graph[state]:
            if edge.permission not in allowed_permissions:
                continue
            if (state, edge.action) in blocked:
                continue
            tentative = g + edge.cost
            if tentative < g_best.get(edge.next_state, math.inf) - 1e-12:
                g_best[edge.next_state] = tentative
                parent[edge.next_state] = (state, edge)
                heappush(heap, (
                    tentative + h[edge.next_state], tentative,
                    next(serial), edge.next_state
                ))
    return []


def path_cost(route) -> float:
    return sum(edge.cost for _, edge in route)


def enforce_consistency(graph: Graph, h: dict[str, float]) -> None:
    for source, edges in graph.items():
        for edge in edges:
            if h[source] > edge.cost + h[edge.next_state] + 1e-12:
                raise ValueError("heuristic fails edge consistency")


def execute_with_replanning(
    graph: Graph, allowed: frozenset[str], budget: float,
    fail_once: frozenset[tuple[str, str]],
    heuristic: dict[str, float],
    max_steps: int = 12
):
    """Simulation; failed actions are known NOT to change external state."""
    state = "start"
    blocked: set[tuple[str, str]] = set()
    failures_seen = set()
    spent = 0.0
    audit_log = []
    for _ in range(max_steps):
        if state == "goal":
            return state, spent, audit_log
        route = astar(graph, state, "goal", allowed, heuristic,
                      frozenset(blocked))
        if not route:
            return "blocked_or_unreachable", spent, audit_log
        origin, edge = route[0]
        assert origin == state
        # Independent enforcement: do not trust the planner alone.
        if edge.permission not in allowed:
            raise PermissionError("unauthorized action")
        if spent + edge.cost > budget:
            return "budget_exhausted", spent, audit_log
        key = (state, edge.action)
        if key in fail_once and key not in failures_seen:
            failures_seen.add(key)
            blocked.add(key)
            audit_log.append((state, edge.action, "known_no_effect_failure"))
            continue
        # Only a simulated, independently known outcome can update state.
        spent += edge.cost
        audit_log.append((state, edge.action, "committed"))
        state = edge.next_state
    return "step_limit", spent, audit_log


def bayes_fault_given_alert(prior_fault, sensitivity, false_alarm):
    """Exact Bayes update for a binary observation."""
    if not all(0 <= z <= 1 for z in
               (prior_fault, sensitivity, false_alarm)):
        raise ValueError("all inputs must be probabilities")
    joint = prior_fault * sensitivity
    evidence = joint + (1 - prior_fault) * false_alarm
    if evidence == 0:
        raise ValueError("alert evidence has probability zero")
    return joint / evidence


def stochastic_choice_cost():
    """One step and one possible recovery: expectation, not worst case."""
    safe = 5.0
    risky = 1.0 + 0.8*0.0 + 0.2*10.0
    return safe, risky


def self_test():
    graph = make_graph()
    heuristic = {"start": 4, "risk": 3, "audit": 2, "goal": 0}
    validate_edges(graph)
    enforce_consistency(graph, heuristic)

    # BFS minimizes number of edges: start -> risk -> goal costs 11.
    hops = bfs_edges(graph, "start", "goal")
    assert [e.action for _, e in hops] == ["quick_check", "fast_deploy"]
    assert path_cost(hops) == 11.0

    # A* minimizes total cost, not edge count.
    allow = frozenset({"read", "deploy"})
    route = astar(graph, "start", "goal", allow, heuristic)
    assert [e.action for _, e in route] == [
        "quick_check", "escalate_review", "verified_deploy"
    ]
    assert path_cost(route) == 4.0
    assert path_cost(astar(graph, "start", "goal", allow)) == 4.0

    # Without deployment permission, no route to goal can be returned.
    assert astar(graph, "start", "goal", frozenset({"read"}),
                 heuristic) == []

    # When the first action is known to have failed without side effects,
    # the planner blocks that action and finds a verified alternative route.
    state, cost, log = execute_with_replanning(
        graph, allow, budget=7.0,
        fail_once=frozenset({("start", "quick_check")}),
        heuristic=heuristic
    )
    assert state == "goal"
    assert cost == 5.0
    assert [entry[2] for entry in log] == [
        "known_no_effect_failure", "committed", "committed"
    ]
    assert log[1][1] == "direct_review"

    assert execute_with_replanning(
        graph, allow, budget=2.0, fail_once=frozenset(),
        heuristic=heuristic
    )[0] == "budget_exhausted"

    safe, risky = stochastic_choice_cost()
    assert safe == 5 and math.isclose(risky, 3.0)
    assert math.isclose(
        bayes_fault_given_alert(0.1, 0.8, 0.05), 0.64
    )
    try:
        enforce_consistency(graph, {"start": 10, "risk": 3,
                                    "audit": 2, "goal": 0})
    except ValueError:
        pass
    else:
        raise AssertionError("inconsistent heuristic was accepted")

    print("PASS: BFS vs A*, optimal cost, permissions, replanning, budget")
    print("PASS: stochastic expected cost and Bayesian belief update")
    print("NOTE: Simulated tool failure has known no-side-effect semantics.")


if __name__ == "__main__":
    self_test()
