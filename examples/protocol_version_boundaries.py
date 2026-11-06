"""Teaching-only MCP 2026-07-28 and A2A 1.0 trust-boundary simulator.

NOT a complete implementation of either protocol. Network transport,
OAuth flows, SDK compatibility and conformance are deliberately absent.
Run: python3 examples/protocol_version_boundaries.py
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from typing import Any


class ProtocolFault(Exception):
    pass


class AuthFault(Exception):
    pass


@dataclass(frozen=True)
class Identity:
    subject: str
    tenant: str
    issuer: str
    audience: str
    scopes: frozenset[str]
    expires_at: int


@dataclass(frozen=True)
class Envelope:
    version: str
    method: str
    name: str | None
    payload: dict[str, Any]
    headers: dict[str, str]


@dataclass
class AuditEvent:
    operation: str
    tenant: str
    approved: bool
    reason: str


def check_token(
    token: Identity, now: int, expected_issuer: str,
    expected_audience: str, scope: str,
):
    if token.expires_at <= now:
        raise AuthFault("expired token")
    if token.issuer != expected_issuer:
        raise AuthFault("wrong authorization issuer")
    if token.audience != expected_audience:
        raise AuthFault("wrong token audience")
    if scope not in token.scopes:
        raise AuthFault("insufficient scope")
    if not token.subject or not token.tenant:
        raise AuthFault("missing authenticated identity")
    return token.tenant


def cache_key(
    catalog_name: str,
    version: str,
    tenant: str,
    scope: str,
    cache_scope: str,
):
    if cache_scope not in ("private", "public"):
        raise ValueError("unrecognized toy cache scope")
    if cache_scope == "private":
        return (version, catalog_name, tenant, scope)
    return (version, catalog_name, "*", scope)


@dataclass
class ToolServer:
    """A small stateless request validator; not an MCP HTTP server."""

    trust_issuer: str = "https://identity.example.test"
    audience: str = "mcp-service"
    data: dict[tuple[str, str], str] = field(default_factory=lambda: {
        ("tenant-A", "A-1"): "synthetic A alert",
        ("tenant-B", "A-1"): "synthetic B alert",
    })
    audit: list[AuditEvent] = field(default_factory=list)

    def call(self, envelope: Envelope, token: Identity, now=100) -> dict:
        if envelope.version != "2026-07-28":
            raise ProtocolFault("unsupported version for this teaching server")
        if envelope.headers.get("MCP-Protocol-Version") != envelope.version:
            raise ProtocolFault("mismatched protocol version header")
        if envelope.headers.get("Mcp-Method") != envelope.method:
            raise ProtocolFault("mismatched routing method header")
        if envelope.method == "tools/call":
            if envelope.headers.get("Mcp-Name") != envelope.name:
                raise ProtocolFault("mismatched tool name header")
            if envelope.name != "read_alert":
                raise ProtocolFault("unknown trusted tool")
            body = envelope.payload
            if (type(body) is not dict or
                    set(body) != {"alert_id"} or
                    type(body["alert_id"]) is not str):
                raise ProtocolFault("invalid input schema")
            tenant = check_token(
                token, now, self.trust_issuer, self.audience, "alert:read"
            )
            result = self.data.get((tenant, body["alert_id"]))
            if result is None:
                self.audit.append(AuditEvent(
                    "read_alert", tenant, False, "resource unavailable"
                ))
                raise AuthFault("resource unavailable")
            self.audit.append(AuditEvent(
                "read_alert", tenant, True, "authorized read"
            ))
            return {"result": result, "tenant": tenant}
        if envelope.method == "server/discover":
            check_token(
                token, now, self.trust_issuer, self.audience, "catalog:read"
            )
            return {"tools": ["read_alert"], "cacheScope": "private",
                    "ttlMs": 1000}
        raise ProtocolFault("unsupported method")


def make_call(alert_id: str, name="read_alert", method="tools/call"):
    return Envelope(
        "2026-07-28", method, name,
        {"alert_id": alert_id},
        {
            "MCP-Protocol-Version": "2026-07-28",
            "Mcp-Method": method,
            "Mcp-Name": name,
        },
    )


@dataclass(frozen=True)
class AgentCard:
    agent_id: str
    capabilities: frozenset[str]
    tenant_allowlist: frozenset[str]


@dataclass
class AgentTask:
    task_id: str
    tenant: str
    requester: str
    capability: str
    fingerprint: str
    state: str = "SUBMITTED"
    artifact: str | None = None


class A2ACoordinator:
    """Simplified 1.0-style lifecycle, not the official wire schema."""

    EDGES = {
        "SUBMITTED": frozenset({"WORKING", "REJECTED", "CANCELED"}),
        "WORKING": frozenset({
            "INPUT_REQUIRED", "AUTH_REQUIRED", "COMPLETED",
            "FAILED", "CANCELED"
        }),
        "INPUT_REQUIRED": frozenset({"WORKING", "CANCELED"}),
        "AUTH_REQUIRED": frozenset({"WORKING", "CANCELED"}),
        "COMPLETED": frozenset(),
        "FAILED": frozenset(),
        "CANCELED": frozenset(),
        "REJECTED": frozenset(),
    }

    def __init__(self):
        self.cards = {
            "reviewer": AgentCard(
                "reviewer", frozenset({"review_code"}),
                frozenset({"tenant-A"})
            )
        }
        self.tasks: dict[tuple[str, str], AgentTask] = {}

    def submit(
        self, user: Identity, now: int, task_id: str,
        agent_id: str, capability: str, payload: str
    ) -> AgentTask:
        check_token(
            user, now, "https://identity.example.test",
            "agent-coordinator", "task:delegate"
        )
        card = self.cards.get(agent_id)
        if card is None or capability not in card.capabilities:
            raise ProtocolFault("unknown target agent or capability")
        if user.tenant not in card.tenant_allowlist:
            raise AuthFault("cross-tenant delegation")
        if not task_id or not payload:
            raise ValueError("task id and payload required")
        key = (user.tenant, task_id)
        request_hash = sha256(
            json.dumps([agent_id, capability, payload],
                       ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        if key in self.tasks:
            task = self.tasks[key]
            if task.fingerprint != request_hash:
                raise ProtocolFault("task idempotency conflict")
            return task
        task = AgentTask(task_id, user.tenant, user.subject,
                         capability, request_hash)
        self.tasks[key] = task
        return task

    def transition(
        self, user: Identity, now: int, task_id: str,
        new_state: str, artifact: str | None = None
    ) -> AgentTask:
        check_token(
            user, now, "https://identity.example.test",
            "agent-coordinator", "task:manage"
        )
        task = self.tasks.get((user.tenant, task_id))
        if task is None:
            raise AuthFault("task not available")
        if new_state not in self.EDGES[task.state]:
            raise ProtocolFault("illegal lifecycle transition")
        if new_state == "COMPLETED":
            if not artifact or not isinstance(artifact, str):
                raise ProtocolFault("completed task needs artifact")
            task.artifact = artifact
        elif artifact is not None:
            raise ProtocolFault("artifact allowed only at completion")
        task.state = new_state
        return task

    def get(self, user: Identity, now: int, task_id: str) -> AgentTask:
        check_token(
            user, now, "https://identity.example.test",
            "agent-coordinator", "task:read"
        )
        task = self.tasks.get((user.tenant, task_id))
        if task is None:
            raise AuthFault("task not available")
        return task


def identity(tenant="tenant-A", issuer="https://identity.example.test",
             audience="mcp-service", scopes=("alert:read",), expiry=200):
    return Identity("user-1", tenant, issuer, audience,
                    frozenset(scopes), expiry)


def expect(exc, thunk):
    try:
        thunk()
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__}")


def test_mcp_boundary():
    server = ToolServer()
    good = identity()
    request = make_call("A-1")
    assert server.call(request, good)["result"] == "synthetic A alert"
    other = identity(tenant="tenant-B")
    assert server.call(request, other)["result"] == "synthetic B alert"
    expect(AuthFault, lambda: server.call(request, identity(
        tenant="tenant-C"
    )))
    expect(AuthFault, lambda: server.call(request, identity(
        issuer="https://untrusted.example.test"
    )))
    expect(AuthFault, lambda: server.call(request, identity(
        audience="other-service"
    )))
    expect(AuthFault, lambda: server.call(request, identity(
        scopes=("catalog:read",)
    )))
    expect(AuthFault, lambda: server.call(request, identity(
        expiry=100
    )))
    broken = Envelope(
        request.version, request.method, "read_alert",
        request.payload,
        {**request.headers, "Mcp-Name": "write_alert"}
    )
    expect(ProtocolFault, lambda: server.call(broken, good))
    old = Envelope("2025-11-25", request.method, request.name,
                   request.payload, request.headers)
    expect(ProtocolFault, lambda: server.call(old, good))
    assert cache_key("tools", "2026", "A", "read", "private") != (
        cache_key("tools", "2026", "B", "read", "private")
    )
    assert cache_key("tools", "2026", "A", "read", "public") == (
        cache_key("tools", "2026", "B", "read", "public")
    )
    print("PASS modern MCP-style header, identity and catalog boundaries")


def test_a2a_boundary():
    co = A2ACoordinator()
    u = identity(
        audience="agent-coordinator",
        scopes=("task:delegate", "task:manage", "task:read")
    )
    task = co.submit(u, 100, "task-1", "reviewer",
                     "review_code", "synthetic source")
    assert task.state == "SUBMITTED"
    assert co.submit(u, 100, "task-1", "reviewer",
                     "review_code", "synthetic source") is task
    expect(ProtocolFault, lambda: co.submit(
        u, 100, "task-1", "reviewer", "review_code", "different source"
    ))
    intruder = identity(
        tenant="tenant-B", audience="agent-coordinator",
        scopes=("task:delegate", "task:manage", "task:read")
    )
    expect(AuthFault, lambda: co.submit(
        intruder, 100, "task-1", "reviewer",
        "review_code", "synthetic source"
    ))
    co.transition(u, 100, "task-1", "WORKING")
    co.transition(u, 100, "task-1", "AUTH_REQUIRED")
    assert co.get(u, 100, "task-1").state == "AUTH_REQUIRED"
    co.transition(u, 100, "task-1", "WORKING")
    co.transition(u, 100, "task-1", "COMPLETED",
                  "artifact@verified-sha")
    expect(ProtocolFault, lambda: co.transition(
        u, 100, "task-1", "WORKING"
    ))
    expect(AuthFault, lambda: co.get(intruder, 100, "task-1"))
    print("PASS A2A-style lifecycle, idempotency and tenant delegation")


if __name__ == "__main__":
    test_mcp_boundary()
    test_a2a_boundary()
    print("NOTE: no real SDK, transport, OAuth, or protocol conformance test.")
