"""MCP-style tool contract and A2A-style task lifecycle teaching simulator.

Python standard library only. No network access, SDK dependencies, actual
OAuth or real MCP/A2A server. The mock deliberately tests authorization and
idempotency independently of protocol syntax.

Run: python examples/agent_protocol_contracts.py
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import re


@dataclass(frozen=True)
class Principal:
    identity: str
    tenant: str
    permissions: frozenset[str]


def digest(payload: str) -> str:
    return sha256(payload.encode("utf-8")).hexdigest()


class ToolGateway:
    """MCP-like JSON-RPC 2.0 validation, not a full MCP implementation."""

    ALERT_PATTERN = re.compile(r"A-[0-9]{1,8}\Z")

    def __init__(self):
        # Identical resource names in two different tenants are deliberate.
        self.alerts = {
            ("tenant-a", "A-102"): "synthetic alert for tenant A",
            ("tenant-b", "A-102"): "synthetic alert for tenant B",
        }
        self.report_results: dict[tuple[str, str], tuple[str, str]] = {}
        self.write_effects = 0

    def call(self, request: dict, user: Principal) -> dict:
        if type(request) is not dict or request.get("jsonrpc") != "2.0":
            raise ValueError("invalid JSON-RPC version")
        request_id = request.get("id")
        if type(request_id) not in (str, int) or request_id == "":
            raise ValueError("invalid request identifier")
        if request.get("method") != "tools/call":
            raise ValueError("unsupported method in toy gateway")
        params = request.get("params")
        if type(params) is not dict or set(params) != {"name", "arguments"}:
            raise ValueError("invalid params")
        name, args = params["name"], params["arguments"]
        if type(name) is not str or type(args) is not dict:
            raise ValueError("invalid tool name or arguments")

        if name == "read_alert":
            if set(args) != {"alert_id"}:
                raise ValueError("unknown or missing read argument")
            alert_id = args["alert_id"]
            if type(alert_id) is not str or not self.ALERT_PATTERN.fullmatch(
                alert_id
            ):
                raise ValueError("invalid alert identifier")
            if "alert:read" not in user.permissions:
                raise PermissionError("tool scope denied")
            # Tenant is derived from authenticated principal, not arguments.
            value = self.alerts.get((user.tenant, alert_id))
            if value is None:
                raise PermissionError("resource unavailable")
            return {
                "jsonrpc": "2.0", "id": request_id,
                "result": {"content": value}
            }

        if name == "submit_report":
            if set(args) != {"operation_key", "report"}:
                raise ValueError("unknown or missing write argument")
            key, report = args["operation_key"], args["report"]
            if (type(key) is not str or not 1 <= len(key) <= 64 or
                    type(report) is not str or not 1 <= len(report) <= 300):
                raise ValueError("invalid report request")
            if "report:write" not in user.permissions:
                raise PermissionError("write scope denied")
            compound = (user.tenant, key)
            payload_digest = digest(report)
            if compound in self.report_results:
                old_digest, result_id = self.report_results[compound]
                if old_digest != payload_digest:
                    raise ValueError("operation key payload conflict")
            else:
                self.write_effects += 1
                result_id = f"report-{self.write_effects}"
                self.report_results[compound] = (payload_digest, result_id)
            return {
                "jsonrpc": "2.0", "id": request_id,
                "result": {"report_id": result_id}
            }

        # Tool metadata from external sources cannot register new operations.
        raise PermissionError("tool is not in the trusted registry")


def tool_request(request_id, name, arguments):
    return {
        "jsonrpc": "2.0", "id": request_id, "method": "tools/call",
        "params": {"name": name, "arguments": arguments}
    }


@dataclass(frozen=True)
class AgentCard:
    agent_id: str
    capabilities: frozenset[str]
    trusted_tenants: frozenset[str]


@dataclass
class Task:
    task_id: str
    owner_tenant: str
    capability: str
    request_hash: str
    status: str = "SUBMITTED"
    artifact: str | None = None


class TaskCoordinator:
    """Illustrative A2A-like task model, not protocol schema or transport."""

    TRANSITIONS = {
        "SUBMITTED": frozenset({"WORKING", "REJECTED", "CANCELED"}),
        "WORKING": frozenset({
            "INPUT_REQUIRED", "COMPLETED", "FAILED", "CANCELED"
        }),
        "INPUT_REQUIRED": frozenset({"WORKING", "CANCELED"}),
        "COMPLETED": frozenset(),
        "FAILED": frozenset(),
        "CANCELED": frozenset(),
        "REJECTED": frozenset(),
    }

    def __init__(self):
        self.cards = {
            "audit-specialist": AgentCard(
                "audit-specialist",
                frozenset({"review_code"}),
                frozenset({"tenant-a"})
            )
        }
        self.tasks: dict[tuple[str, str], Task] = {}

    def submit(self, tenant: str, agent_id: str, task_id: str,
               capability: str, request_text: str) -> Task:
        card = self.cards.get(agent_id)
        if card is None or capability not in card.capabilities:
            raise PermissionError("capability unavailable")
        if tenant not in card.trusted_tenants:
            raise PermissionError("delegation not authorized")
        if not task_id or not request_text:
            raise ValueError("task ID and request required")
        key = (tenant, task_id)
        current_hash = digest(agent_id + "\0" + capability + "\0" + request_text)
        if key in self.tasks:
            existing = self.tasks[key]
            if existing.request_hash != current_hash:
                raise ValueError("duplicate task ID with changed request")
            return existing
        task = Task(task_id, tenant, capability, current_hash)
        self.tasks[key] = task
        return task

    def update(self, tenant: str, task_id: str, next_status: str,
               artifact: str | None = None):
        task = self.tasks.get((tenant, task_id))
        if task is None:
            raise PermissionError("task unavailable")
        if next_status not in self.TRANSITIONS[task.status]:
            raise ValueError("illegal lifecycle transition")
        if next_status == "COMPLETED":
            if type(artifact) is not str or not artifact:
                raise ValueError("completed task requires a versioned artifact")
            task.artifact = artifact
        elif artifact is not None:
            raise ValueError("artifact only accepted on completion")
        task.status = next_status
        return task

    def get(self, tenant: str, task_id: str) -> Task:
        task = self.tasks.get((tenant, task_id))
        if task is None:
            raise PermissionError("task unavailable")
        return task


def expect_error(fn, exception_type):
    try:
        fn()
    except exception_type:
        return
    raise AssertionError(f"expected {exception_type.__name__}")


def test_mcp():
    gateway = ToolGateway()
    reader = Principal("alice", "tenant-a", frozenset({"alert:read"}))
    other = Principal("bob", "tenant-b", frozenset({"alert:read"}))
    writer = Principal("alice", "tenant-a", frozenset({"report:write"}))
    req = tool_request(7, "read_alert", {"alert_id": "A-102"})
    assert gateway.call(req, reader)["result"]["content"].endswith("tenant A")
    assert gateway.call(req, other)["result"]["content"].endswith("tenant B")
    assert gateway.call(req, reader)["id"] == 7
    expect_error(lambda: gateway.call(
        tool_request(8, "read_alert", {"alert_id": "A-102", "tenant": "tenant-b"}),
        reader), ValueError)
    expect_error(lambda: gateway.call(
        tool_request(9, "read_alert", {"alert_id": "A-102"}), writer),
        PermissionError)
    expect_error(lambda: gateway.call(
        tool_request(9, "read_alert", {"alert_id": 102}), reader), ValueError)
    expect_error(lambda: gateway.call(
        tool_request(10, "unregistered_tool", {}), reader), PermissionError)
    request = tool_request(15, "submit_report", {
        "operation_key": "op-42", "report": "synthetic report"
    })
    first = gateway.call(request, writer)["result"]["report_id"]
    second = gateway.call({**request, "id": 16}, writer)["result"]["report_id"]
    assert first == second == "report-1"
    assert gateway.write_effects == 1
    expect_error(lambda: gateway.call(tool_request(
        17, "submit_report",
        {"operation_key": "op-42", "report": "different report"}
    ), writer), ValueError)
    expect_error(lambda: gateway.call(request, reader), PermissionError)
    print("PASS MCP-like: JSON-RPC, strict args, trusted scopes, dedup")


def test_a2a():
    coord = TaskCoordinator()
    task = coord.submit("tenant-a", "audit-specialist", "task-41",
                        "review_code", "check synthetic source")
    assert task.status == "SUBMITTED"
    again = coord.submit("tenant-a", "audit-specialist", "task-41",
                         "review_code", "check synthetic source")
    assert task is again
    expect_error(lambda: coord.submit(
        "tenant-a", "audit-specialist", "task-41",
        "review_code", "changed task"), ValueError)
    expect_error(lambda: coord.submit(
        "tenant-b", "audit-specialist", "task-41",
        "review_code", "check synthetic source"), PermissionError)
    coord.update("tenant-a", "task-41", "WORKING")
    coord.update("tenant-a", "task-41", "INPUT_REQUIRED")
    assert coord.get("tenant-a", "task-41").status == "INPUT_REQUIRED"
    coord.update("tenant-a", "task-41", "WORKING")
    expect_error(lambda: coord.update(
        "tenant-a", "task-41", "COMPLETED"), ValueError)
    coord.update("tenant-a", "task-41", "COMPLETED",
                 artifact="report@commit-abc")
    assert coord.get("tenant-a", "task-41").artifact == "report@commit-abc"
    expect_error(lambda: coord.update(
        "tenant-a", "task-41", "WORKING"), ValueError)
    expect_error(lambda: coord.get("tenant-b", "task-41"), PermissionError)
    print("PASS A2A-like: discovery, lifecycle, duplicate task, tenant isolation")


if __name__ == "__main__":
    test_mcp()
    test_a2a()
    print("NOTE: Protocol boundary simulations; no live MCP or A2A integration.")
