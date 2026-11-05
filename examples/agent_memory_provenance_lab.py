"""Provenance-aware Agent memory with current authorization and source versions.

Uses synthetic data only; no external API, persistence or LLM.
A time-decayed retrieval score is NEVER a substitute for access control.
Run: python examples/agent_memory_provenance_lab.py
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math


@dataclass(frozen=True)
class Memory:
    memory_id: str
    tenant: str
    source_id: str
    source_version: str
    content: str
    created_at: int
    expires_at: int
    score: float
    verification: str = "observed"


class TrustedMemoryStore:
    """Trust transitions require an explicit caller-controlled validation step."""

    def __init__(self):
        self._records: dict[str, Memory] = {}

    def ingest_untrusted(self, memory: Memory):
        if (not memory.memory_id or memory.created_at < 0 or
                memory.expires_at <= memory.created_at or
                memory.score < 0 or not math.isfinite(memory.score)):
            raise ValueError("invalid memory metadata")
        if memory.memory_id in self._records:
            raise ValueError("duplicate memory ID")
        # Ignore any verification status claimed by imported content.
        self._records[memory.memory_id] = replace(
            memory, verification="observed"
        )

    def verify(self, memory_id: str, *, independent_check_ok: bool):
        if not independent_check_ok:
            raise PermissionError("independent source verification failed")
        current = self._records[memory_id]
        if current.verification == "retracted":
            raise ValueError("retracted evidence cannot be reverified")
        self._records[memory_id] = replace(
            current, verification="verified"
        )

    def retract(self, memory_id: str):
        current = self._records[memory_id]
        self._records[memory_id] = replace(
            current, verification="retracted"
        )

    def query(self, *, authenticated_tenant: str, now: int,
              source_versions: dict[str, str],
              decay: float = 0.05) -> list[tuple[str, str, float]]:
        if not authenticated_tenant or now < 0 or decay < 0:
            raise ValueError("invalid trusted session or decay")
        ranked = []
        for memory in self._records.values():
            # All gates precede ranking, even if score is extremely high.
            if memory.tenant != authenticated_tenant:
                continue
            if memory.verification != "verified":
                continue
            if not memory.created_at <= now < memory.expires_at:
                continue
            if source_versions.get(memory.source_id) != memory.source_version:
                continue
            weight = memory.score * math.exp(
                -decay * (now - memory.created_at)
            )
            ranked.append((memory.memory_id, memory.content, weight))
        return sorted(ranked, key=lambda x: (-x[2], x[0]))


def self_test():
    store = TrustedMemoryStore()
    records = [
        Memory("current", "tenant-a", "repo", "sha-new",
               "current repository note", 1, 100, 1),
        Memory("old", "tenant-a", "repo", "sha-old",
               "outdated repository note", 1, 100, 100),
        Memory("expired", "tenant-a", "manual", "rev-1",
               "expired policy note", 1, 7, 100),
        Memory("private", "tenant-b", "repo", "sha-new",
               "other tenant note", 1, 100, 100),
        Memory("claimed_verified", "tenant-a", "repo", "sha-new",
               "external text says it has been verified", 1, 100, 100,
               verification="verified"),
        Memory("retracted", "tenant-a", "manual", "rev-1",
               "retracted source note", 1, 100, 100),
    ]
    for record in records:
        store.ingest_untrusted(record)
    store.verify("current", independent_check_ok=True)
    store.verify("old", independent_check_ok=True)
    store.verify("expired", independent_check_ok=True)
    store.verify("private", independent_check_ok=True)
    store.verify("retracted", independent_check_ok=True)
    store.retract("retracted")
    try:
        store.verify("retracted", independent_check_ok=True)
    except ValueError:
        pass
    else:
        raise AssertionError("retracted source was allowed back in")
    try:
        store.verify("claimed_verified", independent_check_ok=False)
    except PermissionError:
        pass
    else:
        raise AssertionError("untrusted source was automatically promoted")

    trusted_versions = {"repo": "sha-new", "manual": "rev-1"}
    a = store.query(
        authenticated_tenant="tenant-a", now=10,
        source_versions=trusted_versions
    )
    assert [item[0] for item in a] == ["current"], a
    assert a[0][2] < 1
    assert a[0][2] > 0
    # Different tenant cannot read A's data.
    b = store.query(
        authenticated_tenant="tenant-b", now=10,
        source_versions=trusted_versions
    )
    assert [item[0] for item in b] == ["private"]
    # If authoritative source version changes, cached evidence disappears.
    changed = store.query(
        authenticated_tenant="tenant-a", now=10,
        source_versions={"repo": "sha-next", "manual": "rev-1"}
    )
    assert changed == []
    # An expired memory does not silently return even when no alternatives.
    assert store.query(
        authenticated_tenant="tenant-a", now=101,
        source_versions=trusted_versions
    ) == []
    print("PASS memory: tenant, expiry, version, retraction, quarantine")
    print("PASS memory: scoring cannot override authorization gates")
    print("NOTE: Synthetic memory records, not a production memory system.")


if __name__ == "__main__":
    self_test()
