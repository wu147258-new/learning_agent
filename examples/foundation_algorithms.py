"""Offline examples for search, graphs, complexity and task dependencies.

Run: python examples/foundation_algorithms.py
"""
from __future__ import annotations

from collections import deque
from heapq import heappush, heappop
import math


def binary_search(items, target):
    """Return the first matching index in a NONDECREASING sequence."""
    left, right = 0, len(items)
    while left < right:
        middle = (left + right) // 2
        if items[middle] < target:
            left = middle + 1
        else:
            right = middle
    if left < len(items) and items[left] == target:
        return left
    return -1


def shortest_hops(graph, start, goal):
    """Shortest path by number of edges in an unweighted directed graph."""
    queue = deque([start])
    parent = {start: None}
    while queue:
        node = queue.popleft()
        if node == goal:
            result = []
            while node is not None:
                result.append(node)
                node = parent[node]
            return result[::-1]
        for nxt in graph.get(node, ()):
            if nxt not in parent:
                parent[nxt] = node
                queue.append(nxt)
    return None


def topological_sort(graph):
    """Topological ordering or ValueError when dependencies have a cycle."""
    nodes = set(graph)
    for children in graph.values():
        nodes.update(children)
    indegree = {node: 0 for node in nodes}
    for children in graph.values():
        for child in children:
            indegree[child] += 1
    queue = deque(sorted(n for n in nodes if indegree[n] == 0))
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for child in graph.get(node, ()):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if len(order) != len(nodes):
        raise ValueError("dependency graph contains a cycle")
    return order


def dijkstra(graph, source):
    """Shortest total cost; requires NONNEGATIVE finite edge weights."""
    distances = {source: 0.0}
    queue = [(0.0, source)]
    while queue:
        distance, node = heappop(queue)
        if distance > distances[node]:
            continue
        for neighbour, cost in graph.get(node, ()):
            if not math.isfinite(cost) or cost < 0:
                raise ValueError("Dijkstra requires nonnegative finite costs")
            candidate = distance + cost
            if candidate < distances.get(neighbour, math.inf):
                distances[neighbour] = candidate
                heappush(queue, (candidate, neighbour))
    return distances


def self_test():
    assert binary_search([], 5) == -1
    assert binary_search([1, 2, 2, 2, 4], 2) == 1
    assert binary_search([1, 3, 5], 4) == -1
    assert binary_search([1, 3, 5], 5) == 2

    graph = {"A": ["B", "C"], "B": ["D"], "C": ["D"], "D": []}
    assert shortest_hops(graph, "A", "D") == ["A", "B", "D"]
    assert shortest_hops(graph, "D", "A") is None
    order = topological_sort(graph)
    assert order.index("A") < order.index("B") < order.index("D")
    assert order.index("A") < order.index("C") < order.index("D")
    try:
        topological_sort({"A": ["B"], "B": ["A"]})
        raise AssertionError("Cycle must be detected")
    except ValueError:
        pass

    weighted = {
        "A": [("B", 4.0), ("C", 1.0)],
        "C": [("B", 1.0), ("D", 10.0)],
        "B": [("D", 2.0)],
    }
    assert dijkstra(weighted, "A")["D"] == 4.0
    try:
        dijkstra({"A": [("B", -1)]}, "A")
        raise AssertionError("Negative edge must be rejected")
    except ValueError:
        pass
    print("PASS: binary search, BFS, DAG ordering and Dijkstra")


if __name__ == "__main__":
    self_test()
