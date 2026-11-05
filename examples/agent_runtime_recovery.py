"""Durable Agent step protocol using an in-memory SQLite DB and fake remote.

No actual network requests or side effects. This is a protocol simulation.
Run: python examples/agent_runtime_recovery.py
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import sqlite3


class ResponseLost(TimeoutError):
    """The server executed the request but its response was not delivered."""


@dataclass(frozen=True)
class Response:
    operation_key: str
    result_id: str


class FakeIdempotentRemote:
    def __init__(self):
        self.results: dict[str, tuple[str, Response]] = {}
        self.side_effects = 0
        self.requests = 0

    def execute(self, operation_key: str, payload: str,
                drop_response=False) -> Response:
        self.requests += 1
        fingerprint = hashlib.sha256(payload.encode()).hexdigest()
        if operation_key in self.results:
            prior_hash, prior_response = self.results[operation_key]
            if prior_hash != fingerprint:
                raise ValueError("idempotency key reused for different payload")
            response = prior_response
        else:
            self.side_effects += 1
            response = Response(operation_key, f"remote-{self.side_effects}")
            self.results[operation_key] = (fingerprint, response)
        if drop_response:
            raise ResponseLost("response was lost after the remote commit")
        return response

    def check(self, operation_key: str) -> Response | None:
        item = self.results.get(operation_key)
        return item[1] if item else None


def make_db():
    db = sqlite3.connect(":memory:")
    db.execute("""
        CREATE TABLE steps(
            task_id TEXT NOT NULL,
            step_id TEXT NOT NULL,
            status TEXT NOT NULL,
            version INTEGER NOT NULL,
            operation_key TEXT NOT NULL UNIQUE,
            result_id TEXT,
            PRIMARY KEY(task_id,step_id)
        )
    """)
    db.execute("""
        INSERT INTO steps(task_id,step_id,status,version,operation_key)
        VALUES('task-1','send','PENDING',0,'op-123')
    """)
    db.commit()
    return db


def transition(db, expected: str, new: str, version: int,
               result_id: str | None = None) -> bool:
    allowed = {
        "PENDING": {"PREPARED"},
        "PREPARED": {"RUNNING"},
        "RUNNING": {"UNKNOWN", "FAILED", "SUCCEEDED"},
        "UNKNOWN": {"SUCCEEDED", "FAILED"},
        "FAILED": set(), "SUCCEEDED": set(),
    }
    if new not in allowed.get(expected, set()):
        raise ValueError(f"invalid transition {expected}->{new}")
    with db:
        cursor = db.execute("""
            UPDATE steps SET status=?, version=version+1, result_id=?
            WHERE task_id='task-1' AND step_id='send'
            AND status=? AND version=?
        """, (new, result_id, expected, version))
    return cursor.rowcount == 1


def read_step(db):
    return db.execute("""
        SELECT status,version,operation_key,result_id
        FROM steps WHERE task_id='task-1' AND step_id='send'
    """).fetchone()


def authorized_dispatch(db, remote, payload, *, allowed, budget, cost,
                        drop_response=False):
    if not allowed:
        raise PermissionError("tool authorization denied")
    if cost > budget or cost < 0:
        raise ValueError("budget limit exceeded or invalid cost")
    status, version, key, _ = read_step(db)
    if status != "PREPARED":
        raise ValueError("step not prepared")
    assert transition(db, "PREPARED", "RUNNING", version)
    try:
        response = remote.execute(key, payload, drop_response=drop_response)
    except ResponseLost:
        current = read_step(db)
        assert transition(db, "RUNNING", "UNKNOWN", current[1])
        return None
    current = read_step(db)
    assert transition(db, "RUNNING", "SUCCEEDED", current[1], response.result_id)
    return response


def recover_unknown(db, remote, payload: str) -> Response:
    status, version, key, _ = read_step(db)
    if status != "UNKNOWN":
        raise ValueError("step is not in unknown state")
    # Production systems need an authoritative remote lookup or a documented
    # server-side idempotency contract. This fake implements both.
    response = remote.check(key)
    if response is None:
        response = remote.execute(key, payload)
    assert transition(db, "UNKNOWN", "SUCCEEDED", version, response.result_id)
    return response


def self_test():
    db = make_db()
    remote = FakeIdempotentRemote()
    try:
        # Unauthorized and over-budget operations must not advance the state.
        assert transition(db, "PENDING", "PREPARED", 0)
        for options, expected_error in [
            ({"allowed": False, "budget": 3, "cost": 1}, PermissionError),
            ({"allowed": True, "budget": 0, "cost": 1}, ValueError),
        ]:
            try:
                authorized_dispatch(db, remote, "send-safe-message",
                                    **options)
            except expected_error:
                pass
            else:
                raise AssertionError("unsafe dispatch was accepted")
        assert remote.side_effects == 0
        assert read_step(db)[:2] == ("PREPARED", 1)

        # Remote succeeds; response is lost, so local state is UNKNOWN.
        result = authorized_dispatch(
            db, remote, "send-safe-message",
            allowed=True, budget=3, cost=1, drop_response=True
        )
        assert result is None
        assert read_step(db)[0] == "UNKNOWN"
        assert remote.side_effects == 1

        # Recover using the same operation ID; no second side effect occurs.
        recovered = recover_unknown(db, remote, "send-safe-message")
        assert recovered.result_id == "remote-1"
        assert read_step(db) == ("SUCCEEDED", 4, "op-123", "remote-1")
        assert remote.side_effects == 1
        assert not transition(db, "PENDING", "PREPARED", 0)

        # Repeated remote requests with same key and payload yield same result.
        duplicate = remote.execute("op-123", "send-safe-message")
        assert duplicate == recovered and remote.side_effects == 1
        try:
            remote.execute("op-123", "a-different-message")
        except ValueError:
            pass
        else:
            raise AssertionError("idempotency key collision was ignored")
        try:
            transition(db, "SUCCEEDED", "PENDING", 4)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal terminal transition accepted")
    finally:
        db.close()
    print("PASS runtime: unknown result, idempotent recovery, CAS, permissions")
    print("Remote effect count: 1 (despite retries and lost response)")


if __name__ == "__main__":
    self_test()
