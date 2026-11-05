"""Offline Agent orchestration patterns: ReAct, Plan-and-Execute, approvals,
graph pause/resume, manager-as-tools and handoff.

No LLM, remote API, filesystem write or authorization escalation.
Run: python examples/agent_architecture_patterns.py
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Environment:
    phase: str = "new"
    version: int = 0
    patch_effects: int = 0
    events: list[str] = field(default_factory=list)
    applied_keys: set[str] = field(default_factory=set)
    allowed_tools: frozenset[str] = frozenset(
        {"inspect", "propose", "apply", "verify"}
    )

    def execute(self, action: str, expected_version: int,
                operation_key: str = "op-1") -> str:
        if action not in self.allowed_tools:
            raise PermissionError("tool not authorized")
        # Idempotent replay is permitted only for the same named operation.
        if action == "apply" and operation_key in self.applied_keys:
            return "already_committed"
        if expected_version != self.version:
            raise RuntimeError("stale state version")
        transitions = {
            ("new", "inspect"): ("inspected", "file_checked"),
            ("inspected", "propose"): ("proposed", "patch_prepared"),
            ("approved", "apply"): ("patched", "patch_committed"),
            ("patched", "verify"): ("verified", "tests_passed"),
        }
        key = (self.phase, action)
        if key not in transitions:
            raise RuntimeError("invalid phase/action combination")
        next_phase, outcome = transitions[key]
        self.phase = next_phase
        self.version += 1
        self.events.append(action + ":" + outcome)
        if action == "apply":
            self.patch_effects += 1
            self.applied_keys.add(operation_key)
        return outcome

    def human_approve(self, approved: bool) -> str:
        if self.phase != "proposed":
            raise RuntimeError("no pending proposal")
        if not approved:
            self.phase = "rejected"
            self.version += 1
            return "rejected"
        self.phase = "approved"
        self.version += 1
        self.events.append("human_approval")
        return "approved"


def react_step(env: Environment) -> str:
    """Pick ONE action using an observed phase. The model is replaced
    by a deterministic planner to test the tool protocol."""
    next_action = {
        "new": "inspect",
        "inspected": "propose",
        "approved": "apply",
        "patched": "verify",
    }.get(env.phase)
    if next_action is None:
        return "pause_for_human" if env.phase == "proposed" else env.phase
    return env.execute(next_action, env.version)


def react_until_pause_or_done(env: Environment):
    for _ in range(8):
        if env.phase in ("proposed", "verified", "rejected"):
            return env.phase
        react_step(env)
    raise RuntimeError("agent loop exceeded step limit")


def plan_and_execute(env: Environment, include_human: bool):
    """A fixed DAG-style plan. External approval is an explicit barrier."""
    plan = ("inspect", "propose", "apply", "verify")
    for action in plan:
        if action == "apply" and env.phase == "proposed":
            if not include_human:
                return "paused_for_human", env.version
            env.human_approve(True)
        env.execute(action, env.version)
    return env.phase, env.version


@dataclass
class GraphRun:
    """A simple simulation of checkpointed state, NOT a LangGraph API."""
    env: Environment = field(default_factory=Environment)
    checkpoint_count: int = 0

    def run_until_interrupt(self):
        if self.env.phase == "new":
            self.env.execute("inspect", self.env.version)
        if self.env.phase == "inspected":
            self.env.execute("propose", self.env.version)
        if self.env.phase == "proposed":
            self.checkpoint_count += 1
            return {"interrupted": True, "phase": self.env.phase,
                    "version": self.env.version}
        return {"interrupted": False, "phase": self.env.phase,
                "version": self.env.version}

    def resume_after_human_review(self, approval: bool,
                                  expected_version: int):
        if self.env.version != expected_version:
            raise RuntimeError("checkpoint version changed")
        self.env.human_approve(approval)
        if not approval:
            return "rejected"
        self.env.execute("apply", self.env.version, operation_key="graph-1")
        self.env.execute("verify", self.env.version)
        return self.env.phase


def simulate_handoff(owner: str, specialist: str):
    """Control ownership changes to the specialist."""
    if owner != "triage" or specialist != "billing":
        raise ValueError("unknown handoff")
    return {"owner": specialist, "handoff_count": 1}


def simulate_agents_as_tools(owner: str, specialist: str):
    """Manager retains final ownership while consuming specialist output."""
    if owner != "manager" or specialist != "analysis":
        raise ValueError("unknown specialist")
    return {"owner": owner, "specialist_result": "candidate_evidence",
            "delegation_count": 1}


def self_test():
    # ReAct-style observation-action loop pauses before any write.
    r = Environment()
    assert react_until_pause_or_done(r) == "proposed"
    assert r.patch_effects == 0
    r.human_approve(True)
    assert react_until_pause_or_done(r) == "verified"
    assert r.patch_effects == 1
    assert r.events == [
        "inspect:file_checked", "propose:patch_prepared",
        "human_approval", "apply:patch_committed", "verify:tests_passed"
    ]

    # Deterministic Plan-and-Execute has an explicit approval barrier.
    p = Environment()
    state, _ = plan_and_execute(p, include_human=False)
    assert state == "paused_for_human"
    assert p.patch_effects == 0
    try:
        p.execute("apply", p.version)
    except RuntimeError:
        pass
    else:
        raise AssertionError("approval bypassed")
    p.human_approve(True)
    assert p.execute("apply", p.version) == "patch_committed"
    assert p.execute("verify", p.version) == "tests_passed"
    assert p.patch_effects == 1

    # Simulate a checkpointed graph pause and an authorized resume.
    g = GraphRun()
    pause = g.run_until_interrupt()
    assert pause["interrupted"]
    assert g.env.patch_effects == 0
    assert g.run_until_interrupt()["version"] == pause["version"]
    assert g.resume_after_human_review(True, pause["version"]) == "verified"
    assert g.env.patch_effects == 1
    # Replaying an operation with the exact same idempotency key is harmless.
    assert g.env.execute("apply", 0, operation_key="graph-1") == "already_committed"
    assert g.env.patch_effects == 1

    # An operation without permission must be blocked, even if model proposes it.
    try:
        Environment().execute("delete_private_files", 0)
    except PermissionError:
        pass
    else:
        raise AssertionError("unauthorized tool was executed")

    assert simulate_handoff("triage", "billing")["owner"] == "billing"
    manager = simulate_agents_as_tools("manager", "analysis")
    assert manager["owner"] == "manager"
    assert manager["specialist_result"] == "candidate_evidence"
    print("PASS ReAct loop, planned DAG, pause/resume, approval and idempotency")
    print("PASS Handoff changes owner; agents-as-tools keeps manager ownership")
    print("NOTE: Protocol simulation, not a tested framework SDK integration.")


if __name__ == "__main__":
    self_test()
