"""Agent evaluation as a versioned, trace-backed paired experiment.

All episodes and tasks are synthetic. This is an evaluator for records,
not a real LLM benchmark or a substitute for external judge reliability.
Run: python3 examples/agent_trace_evaluation_lab.py
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from math import comb, sqrt
from random import Random
import json


@dataclass(frozen=True)
class Task:
    task_id: str
    category: str
    tenant: str
    input_hash: str
    expected_sha: str
    max_cost: float


@dataclass(frozen=True)
class Span:
    task_id: str
    step: int
    kind: str
    start_ms: int
    end_ms: int
    amount: float
    source_sha: str
    approved: bool
    payload_hash: str


@dataclass(frozen=True)
class Result:
    task_id: str
    candidate: str
    passed_tests: bool
    accepted: bool
    produced_sha: str
    spans: tuple[Span, ...]


@dataclass(frozen=True)
class Score:
    valid: bool
    successful: bool
    cost: float
    duration_ms: int
    reason: str


class TraceChecker:
    """Fail closed: invalid traces cannot count as successful episodes."""

    ALLOWED = {"read", "reason", "patch", "test", "review", "commit"}

    def __init__(self, tasks: list[Task]):
        if len({t.task_id for t in tasks}) != len(tasks):
            raise ValueError("duplicate task")
        self.tasks = {t.task_id: t for t in tasks}

    def check(self, result: Result) -> Score:
        task = self.tasks.get(result.task_id)
        if task is None:
            return Score(False, False, 0, 0, "unknown task")
        spans = result.spans
        if not spans:
            return Score(False, False, 0, 0, "empty trace")
        ordered = sorted(spans, key=lambda s: s.step)
        if [s.step for s in ordered] != list(range(len(spans))):
            return Score(False, False, 0, 0, "nonconsecutive steps")
        if tuple(ordered) != spans:
            return Score(False, False, 0, 0, "trace order changed")
        for s in spans:
            if (s.task_id != task.task_id or s.kind not in self.ALLOWED
                    or s.start_ms < 0 or s.end_ms < s.start_ms
                    or s.amount < 0 or not s.payload_hash
                    or s.source_sha != task.input_hash):
                return Score(False, False, 0, 0, "invalid span")
            if s.kind in {"patch", "commit"} and not s.approved:
                return Score(False, False, 0, 0, "unapproved side effect")
        if any(spans[i].start_ms < spans[i-1].start_ms
               for i in range(1, len(spans))):
            return Score(False, False, 0, 0, "nonmonotone start time")
        if "commit" in [s.kind for s in spans]:
            if "test" not in [s.kind for s in spans]:
                return Score(False, False, 0, 0, "no test before commit")
            first_commit = next(
                i for i, s in enumerate(spans) if s.kind == "commit"
            )
            if not any(s.kind == "test" for s in spans[:first_commit]):
                return Score(False, False, 0, 0, "test follows commit")
        cost = sum(s.amount for s in spans)
        duration = max(s.end_ms for s in spans) - min(
            s.start_ms for s in spans
        )
        if cost > task.max_cost:
            return Score(True, False, cost, duration, "budget exceeded")
        if result.produced_sha != task.expected_sha:
            return Score(True, False, cost, duration, "wrong source version")
        if not result.passed_tests:
            return Score(True, False, cost, duration, "regression tests failed")
        if not result.accepted:
            return Score(True, False, cost, duration, "not approved")
        return Score(True, True, cost, duration, "passed all independent gates")


def sha(value: str):
    return sha256(value.encode("utf-8")).hexdigest()


def spans_for(task: Task, passed: bool, candidate: str):
    kinds = ["read", "reason", "patch", "test", "review", "commit"]
    spans = []
    for i, kind in enumerate(kinds):
        spans.append(Span(
            task.task_id, i, kind,
            1000 + i*40, 1025 + i*40,
            0.002 if candidate == "base" else 0.003,
            task.input_hash, True, sha(kind + task.task_id),
        ))
    return tuple(spans)


def paired_results(n=40, seed=17):
    rng = Random(seed)
    categories = ("syntax", "logic", "dependency", "security")
    tasks = []
    baseline, candidate = [], []
    for i in range(n):
        task_id = f"task-{i:03}"
        category = categories[i % len(categories)]
        content = sha("synthetic-source-" + task_id)
        task = Task(task_id, category, "tenant-demo", content,
                    sha("target-" + task_id), 1.0)
        tasks.append(task)
        # Related outcomes are deliberate: some candidate regressions exist.
        draw = rng.random()
        base_ok = draw < (0.65 if category != "security" else 0.45)
        can_ok = (
            draw < (0.72 if category != "security" else 0.52)
            and i % 13 != 0
        )
        for name, ok, storage in [
            ("base", base_ok, baseline),
            ("candidate", can_ok, candidate),
        ]:
            storage.append(Result(
                task_id, name, ok, True, task.expected_sha,
                spans_for(task, ok, name)
            ))
    return tasks, baseline, candidate


def wilson_interval(successes, n, z=1.96):
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError("invalid sample")
    p = successes/n
    den = 1 + z*z/n
    center = (p + z*z/(2*n))/den
    margin = z*sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return max(0, center-margin), min(1, center+margin)


def mcnemar_exact(discordant_base_only, discordant_candidate_only):
    b, c = discordant_base_only, discordant_candidate_only
    if b < 0 or c < 0:
        raise ValueError("counts must be nonnegative")
    n = b + c
    if n == 0:
        return 1.0
    lower = min(b, c)
    # Under H0 each discordant task has probability 1/2 in either direction.
    tail = sum(comb(n, i) for i in range(lower+1)) / 2**n
    return min(1.0, 2*tail)


def stratified_delta_ci(records, iterations=1000, seed=13):
    if not records or iterations <= 0:
        raise ValueError("invalid bootstrap input")
    by_group: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for category, base_ok, new_ok in records:
        by_group[category].append((base_ok, new_ok))
    rng = Random(seed)
    deltas = []
    for _ in range(iterations):
        sampled = []
        for category, values in sorted(by_group.items()):
            sampled += [values[rng.randrange(len(values))]
                        for _ in range(len(values))]
        deltas.append(sum(int(c)-int(b) for b, c in sampled)
                      /len(sampled))
    deltas.sort()
    return (deltas[int(0.025*iterations)],
            deltas[min(iterations-1, int(0.975*iterations))])


def summarise(tasks, baseline, candidate):
    checker = TraceChecker(tasks)
    if not (len(tasks) == len(baseline) == len(candidate)):
        raise ValueError("task length mismatch")
    task_by_id = {t.task_id: t for t in tasks}
    base_map = {r.task_id: checker.check(r) for r in baseline}
    cand_map = {r.task_id: checker.check(r) for r in candidate}
    if set(base_map) != set(task_by_id) or set(cand_map) != set(task_by_id):
        raise ValueError("task coverage differs")
    records = []
    gained = lost = 0
    costs = {}
    for task_id, task in task_by_id.items():
        base_score, cand_score = base_map[task_id], cand_map[task_id]
        assert base_score.valid and cand_score.valid
        b, c = base_score.successful, cand_score.successful
        records.append((task.category, b, c))
        gained += int(not b and c)
        lost += int(b and not c)
        costs[task_id] = (base_score.cost, cand_score.cost)
    n = len(tasks)
    base_pass = sum(b for _, b, _ in records)
    cand_pass = sum(c for _, _, c in records)
    return {
        "tasks": n,
        "baseline_successes": base_pass,
        "candidate_successes": cand_pass,
        "baseline_wilson95": wilson_interval(base_pass, n),
        "candidate_wilson95": wilson_interval(cand_pass, n),
        "paired_delta": (cand_pass - base_pass)/n,
        "stratified_bootstrap95": stratified_delta_ci(records),
        "gained": gained,
        "lost": lost,
        "mcnemar_exact_p": mcnemar_exact(lost, gained),
        "baseline_cost": sum(x for x, _ in costs.values()),
        "candidate_cost": sum(x for _, x in costs.values()),
    }


def test_gate_failures():
    tasks, base, new = paired_results(20)
    checker = TraceChecker(tasks)
    good = base[0]
    assert checker.check(good).valid
    bad_span = Span(
        good.task_id, 2, "patch", 1080, 1105,
        0.002, tasks[0].input_hash, False, "hash"
    )
    broken = Result(
        good.task_id, good.candidate, good.passed_tests, good.accepted,
        good.produced_sha,
        good.spans[:2] + (bad_span,) + good.spans[3:],
    )
    assert checker.check(broken).reason == "unapproved side effect"
    wrong_version = Result(
        good.task_id, good.candidate, True, True, "wrong", good.spans
    )
    assert checker.check(wrong_version).reason == "wrong source version"
    no_approval = Result(
        good.task_id, good.candidate, True, False,
        good.produced_sha, good.spans
    )
    assert checker.check(no_approval).reason == "not approved"
    print("PASS independent safety gates and versioned trace checks")


def test_stats():
    tasks, base, new = paired_results(40)
    report = summarise(tasks, base, new)
    assert report["tasks"] == 40
    assert report["gained"] + report["lost"] <= 40
    assert -1 <= report["paired_delta"] <= 1
    assert 0 <= report["mcnemar_exact_p"] <= 1
    assert 0 <= report["baseline_wilson95"][0] <= 1
    assert 0 <= report["candidate_wilson95"][1] <= 1
    assert report["candidate_cost"] > report["baseline_cost"]
    again = summarise(tasks, base, new)
    assert report == again
    print("PASS paired bootstrap, Wilson, exact McNemar and cost accounting")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    test_gate_failures()
    test_stats()
    print("NOTE: 40 synthetic cases; NOT evidence of real Agent performance.")
