"""Offline multi-agent coordination: DAG scheduling, blackboard CAS, event dedup.

No LLMs, network, shell commands or external dependencies.
Run: python examples/multiagent_coordinator.py
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import sqlite3


@dataclass(frozen=True)
class Task:
    name: str
    duration: float
    predecessors: tuple[str, ...]


TASKS = (
    Task("A", 2, ()),
    Task("B", 4, ("A",)),
    Task("C", 3, ("A",)),
    Task("D", 5, ("B", "C")),
    Task("E", 2, ("D",)),
)


def topo_and_critical_path(tasks: tuple[Task, ...]):
    nodes = {t.name: t for t in tasks}
    if len(nodes) != len(tasks):
        raise ValueError("duplicate task ID")
    if any(t.duration < 0 for t in tasks):
        raise ValueError("negative duration")
    successors = {name: [] for name in nodes}
    indegree = {name: 0 for name in nodes}
    for task in tasks:
        for predecessor in task.predecessors:
            if predecessor not in nodes:
                raise ValueError("unknown predecessor")
            successors[predecessor].append(task.name)
            indegree[task.name] += 1
    queue = deque(sorted(n for n, degree in indegree.items() if degree == 0))
    order = []
    finish = {}
    while queue:
        node = queue.popleft()
        order.append(node)
        finish[node] = nodes[node].duration + max(
            (finish[p] for p in nodes[node].predecessors), default=0
        )
        for child in sorted(successors[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if len(order) != len(tasks):
        raise ValueError("dependency cycle")
    return order, max(finish.values(), default=0), finish


class Coordinator:
    def __init__(self):
        self.db = sqlite3.connect(":memory:")
        self.db.execute("""
            CREATE TABLE blackboard (
                id INTEGER PRIMARY KEY CHECK(id=1),
                version INTEGER NOT NULL,
                content TEXT NOT NULL
            )
        """)
        self.db.execute(
            "INSERT INTO blackboard VALUES(1,20,'initial')"
        )
        self.db.execute("""
            CREATE TABLE processed_events (
                event_id TEXT PRIMARY KEY,
                payload_hash TEXT NOT NULL,
                result TEXT NOT NULL
            )
        """)
        self.tasks = {task.name: "PENDING" for task in TASKS}
        self.db.commit()

    def close(self):
        self.db.close()

    def board(self):
        return self.db.execute(
            "SELECT version,content FROM blackboard WHERE id=1"
        ).fetchone()

    def compare_and_set(self, expected_version, content):
        with self.db:
            cursor = self.db.execute("""
                UPDATE blackboard
                SET version=version+1,content=?
                WHERE id=1 AND version=?
            """, (content, expected_version))
        return cursor.rowcount == 1

    def ready(self, task_name):
        task = next(t for t in TASKS if t.name == task_name)
        return (
            self.tasks[task_name] == "PENDING" and
            all(self.tasks[p] == "SUCCEEDED" for p in task.predecessors)
        )

    def accept_result(self, task_name, verified, artifact_version):
        if not self.ready(task_name):
            raise ValueError("dependencies incomplete or task already done")
        if not verified or not artifact_version:
            raise ValueError("independent validation/version required")
        self.tasks[task_name] = "SUCCEEDED"

    def process_event(self, event_id, payload):
        fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        entry = self.db.execute(
            "SELECT payload_hash,result FROM processed_events WHERE event_id=?",
            (event_id,)
        ).fetchone()
        if entry is not None:
            if entry[0] != fingerprint:
                raise ValueError("same event ID with different payload")
            return entry[1]
        response = "accepted:" + event_id
        with self.db:
            self.db.execute(
                "INSERT INTO processed_events(event_id,payload_hash,result) "
                "VALUES(?,?,?)", (event_id, fingerprint, response)
            )
        return response

    def event_count(self):
        return self.db.execute(
            "SELECT COUNT(*) FROM processed_events"
        ).fetchone()[0]


def self_test():
    order, critical, finish = topo_and_critical_path(TASKS)
    assert set(order) == {"A", "B", "C", "D", "E"}
    assert order.index("A") < order.index("B") < order.index("D")
    assert order.index("A") < order.index("C") < order.index("D")
    assert order.index("D") < order.index("E")
    assert sum(t.duration for t in TASKS) == 16
    assert critical == 13
    assert finish["E"] == 13
    assert max(critical, 16/2) == 13
    try:
        topo_and_critical_path((
            Task("A", 1, ("B",)), Task("B", 1, ("A",))
        ))
    except ValueError:
        pass
    else:
        raise AssertionError("cycle was not detected")

    system = Coordinator()
    try:
        assert system.ready("A")
        assert not system.ready("B")
        try:
            system.accept_result("B", True, "version-1")
        except ValueError:
            pass
        else:
            raise AssertionError("dependency barrier bypassed")

        version_a, _ = system.board()
        version_b, _ = system.board()
        assert version_a == version_b == 20
        assert system.compare_and_set(version_a, "new-test-plan")
        assert not system.compare_and_set(version_b, "stale-patch")
        assert system.board() == (21, "new-test-plan")

        for name in order:
            system.accept_result(name, True, f"commit-{name}")
        assert all(s == "SUCCEEDED" for s in system.tasks.values())
        assert not system.ready("E")

        first = system.process_event("event-7", "task completed")
        replay = system.process_event("event-7", "task completed")
        assert first == replay
        assert system.event_count() == 1
        try:
            system.process_event("event-7", "different payload")
        except ValueError:
            pass
        else:
            raise AssertionError("reused event ID accepted new payload")
    finally:
        system.close()
    print("PASS: DAG dependency, critical path 13, CAS version 20->21")
    print("PASS: validated artifacts and exactly one recorded event")


if __name__ == "__main__":
    self_test()
