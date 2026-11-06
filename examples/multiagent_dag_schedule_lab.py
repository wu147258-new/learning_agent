"""Deterministic multi-Agent DAG scheduler with versioned blackboard.

A teaching simulation, not a production concurrent runtime. Roles are
synthetic workers, not actual LLMs. Run:
    python3 examples/multiagent_dag_schedule_lab.py
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
import heapq


@dataclass(frozen=True)
class Job:
    name: str
    role: str
    duration: int
    dependencies: tuple[str, ...]
    artifact_version: str


@dataclass
class Worker:
    name: str
    roles: frozenset[str]
    occupied_until: int = 0
    current_job: str | None = None


@dataclass(frozen=True)
class Artifact:
    job: str
    version: int
    content: str
    source_version: str


class Conflict(Exception):
    pass


class DAG:
    def __init__(self, jobs: list[Job]):
        if len({j.name for j in jobs}) != len(jobs):
            raise ValueError("duplicate job names")
        self.jobs = {j.name: j for j in jobs}
        for job in jobs:
            if job.duration <= 0 or not job.name or not job.role:
                raise ValueError("invalid task duration, role or name")
            for dep in job.dependencies:
                if dep not in self.jobs:
                    raise ValueError(f"unknown dependency: {dep}")
        self.order = self.topological()

    def topological(self):
        incoming = {name: len(j.dependencies)
                    for name, j in self.jobs.items()}
        children = {name: [] for name in self.jobs}
        for job in self.jobs.values():
            for dependency in job.dependencies:
                children[dependency].append(job.name)
        queue = deque(sorted(name for name in self.jobs if not incoming[name]))
        result = []
        while queue:
            name = queue.popleft()
            result.append(name)
            for child in sorted(children[name]):
                incoming[child] -= 1
                if incoming[child] == 0:
                    queue.append(child)
        if len(result) != len(self.jobs):
            raise ValueError("dependency cycle")
        return result

    def critical_path(self):
        finish = {}
        for name in self.order:
            task = self.jobs[name]
            finish[name] = task.duration + max(
                (finish[dep] for dep in task.dependencies), default=0
            )
        return max(finish.values(), default=0), finish

    def required_predecessors(self, task_name):
        if task_name not in self.jobs:
            raise ValueError("unknown task")
        frontier = list(self.jobs[task_name].dependencies)
        result = set()
        while frontier:
            dep = frontier.pop()
            if dep not in result:
                result.add(dep)
                frontier.extend(self.jobs[dep].dependencies)
        return result


class Blackboard:
    """Optimistic append-only artifacts in a single process."""

    def __init__(self):
        self.entries: dict[str, Artifact] = {}
        self.events: set[str] = set()

    def publish(
        self, job: str, expected_version: int,
        content: str, source_version: str, event_id: str
    ):
        if not event_id or not content:
            raise ValueError("event and content required")
        current = self.entries.get(job)
        old_version = 0 if current is None else current.version
        if event_id in self.events:
            if (current and current.content == content
                    and current.source_version == source_version):
                return current
            raise Conflict("duplicate event with changed payload")
        if expected_version != old_version:
            raise Conflict("compare-and-swap failed")
        artifact = Artifact(job, old_version + 1, content, source_version)
        self.entries[job] = artifact
        self.events.add(event_id)
        return artifact

    def require(self, job: str, source_version: str):
        a = self.entries.get(job)
        if a is None or a.source_version != source_version:
            raise Conflict("missing artifact or incompatible source")
        return a


@dataclass(frozen=True)
class Event:
    time: int
    worker: str
    job: str
    action: str
    detail: str


class Scheduler:
    def __init__(self, dag: DAG, workers: list[Worker], deadline: int):
        if deadline <= 0:
            raise ValueError("deadline must be positive")
        if len({w.name for w in workers}) != len(workers):
            raise ValueError("duplicate worker names")
        self.dag = dag
        self.workers = {w.name: w for w in workers}
        self.deadline = deadline
        self.blackboard = Blackboard()
        self.time = 0
        self.running: dict[str, tuple[str, int, int]] = {}
        self.completed: set[str] = set()
        self.events: list[Event] = []
        self._fence = 0

    def _ready(self, job: Job):
        return (
            job.name not in self.running
            and job.name not in self.completed
            and all(dep in self.completed for dep in job.dependencies)
        )

    def _pick(self, job: Job):
        candidates = sorted(
            (w for w in self.workers.values()
             if job.role in w.roles and w.current_job is None),
            key=lambda w: w.name,
        )
        return candidates[0] if candidates else None

    def dispatch(self):
        dispatched = 0
        for job_name in self.dag.order:
            job = self.dag.jobs[job_name]
            if not self._ready(job):
                continue
            worker = self._pick(job)
            if worker is None:
                continue
            self._fence += 1
            token = self._fence
            end_at = self.time + job.duration
            self.running[job_name] = (worker.name, end_at, token)
            worker.current_job = job_name
            worker.occupied_until = end_at
            self.events.append(Event(
                self.time, worker.name, job_name, "start",
                f"fence={token} end={end_at}"
            ))
            dispatched += 1
        return dispatched

    def finish(self, job_name: str, worker_name: str,
               token: int, content: str):
        execution = self.running.get(job_name)
        if execution is None:
            raise Conflict("task not running")
        worker, end_at, fence = execution
        if worker != worker_name or token != fence:
            raise Conflict("stale worker or stale fencing token")
        if self.time != end_at:
            raise Conflict("completion at wrong virtual time")
        job = self.dag.jobs[job_name]
        for dep in job.dependencies:
            self.blackboard.require(dep, self.dag.jobs[dep].artifact_version)
        self.blackboard.publish(
            job_name, 0, content, job.artifact_version,
            f"{job_name}:commit:{token}"
        )
        self.completed.add(job_name)
        del self.running[job_name]
        self.workers[worker].current_job = None
        self.events.append(Event(
            self.time, worker, job_name, "finish",
            f"fence={token}"
        ))

    def run(self):
        expected, _ = self.dag.critical_path()
        if expected > self.deadline:
            raise Conflict("critical path alone exceeds deadline")
        while len(self.completed) < len(self.dag.jobs):
            self.dispatch()
            if not self.running:
                raise Conflict("no compatible workers or unsatisfied task")
            next_time = min(end for _, end, _ in self.running.values())
            self.time = next_time
            ending = sorted(
                (name, worker, token)
                for name, (worker, end, token) in self.running.items()
                if end == next_time
            )
            for name, worker, token in ending:
                self.finish(
                    name, worker, token,
                    f"synthetic output for {name}"
                )
            if self.time > self.deadline:
                raise Conflict("deadline exceeded")
        return self.time

    def event_summary(self):
        return [f"{ev.time}:{ev.worker}:{ev.job}:{ev.action}"
                for ev in self.events]


def expect(error, fn):
    try:
        fn()
    except error:
        return
    raise AssertionError(f"expected {error.__name__}")


def demo_jobs():
    return [
        Job("read", "research", 3, (), "r1"),
        Job("design", "architect", 4, ("read",), "r1"),
        Job("code", "developer", 5, ("design",), "r1"),
        Job("unit_test", "tester", 2, ("code",), "r1"),
        Job("security", "tester", 3, ("design",), "r1"),
        Job("release", "lead", 1, ("unit_test", "security"), "r1"),
    ]


def test_dag():
    dag = DAG(demo_jobs())
    length, finishes = dag.critical_path()
    assert length == 3 + 4 + 5 + 2 + 1 == 15
    assert finishes["security"] == 10
    assert "read" in dag.required_predecessors("release")
    expect(ValueError, lambda: DAG([
        Job("a", "x", 1, ("b",), "v1"),
        Job("b", "x", 1, ("a",), "v1"),
    ]))
    expect(ValueError, lambda: DAG([
        Job("a", "x", 1, ("missing",), "v1")
    ]))
    print("PASS DAG, predecessors, cycle and critical path")


def test_blackboard():
    board = Blackboard()
    first = board.publish("a", 0, "result", "v1", "event-1")
    assert first.version == 1
    assert board.publish("a", 0, "result", "v1", "event-1") == first
    expect(Conflict, lambda: board.publish(
        "a", 0, "changed", "v1", "event-1"
    ))
    expect(Conflict, lambda: board.publish(
        "a", 0, "new", "v1", "event-2"
    ))
    second = board.publish("a", 1, "new", "v1", "event-2")
    assert second.version == 2
    expect(Conflict, lambda: board.require("a", "v2"))
    print("PASS blackboard version CAS and event idempotency")


def test_schedule():
    workers = [
        Worker("researcher", frozenset({"research"})),
        Worker("architect", frozenset({"architect"})),
        Worker("coder", frozenset({"developer"})),
        Worker("tester-1", frozenset({"tester"})),
        Worker("tester-2", frozenset({"tester"})),
        Worker("lead", frozenset({"lead"})),
    ]
    dag = DAG(demo_jobs())
    sim = Scheduler(dag, workers, deadline=18)
    elapsed = sim.run()
    assert elapsed == 15
    assert len(sim.completed) == len(dag.jobs)
    assert len(sim.blackboard.entries) == len(dag.jobs)
    assert len(sim.events) == 2 * len(dag.jobs)
    assert sim.blackboard.require("release", "r1").version == 1
    expect(Conflict, lambda: sim.finish(
        "release", "lead", 0, "stale"
    ))
    expect(Conflict, lambda: Scheduler(
        dag, workers, deadline=10
    ).run())
    print("PASS parallel DAG schedule, dependency gates and deadline")


if __name__ == "__main__":
    test_dag()
    test_blackboard()
    test_schedule()
    print("NOTE: simulated roles and time, no LLMs or real concurrency.")
