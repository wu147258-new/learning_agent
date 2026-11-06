"""A durable, deterministic teaching runtime for Agent tasks.

This lab uses SQLite transactions, leases, monotonically increasing fencing
tokens, a transactional outbox, tenant isolation, and replay-safe effects.
No LLM API or network access. Run: python examples/durable_agent_queue_lab.py
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json
import sqlite3
import tempfile
import time
from typing import Callable


class Conflict(Exception):
    """A precondition, version, lease or permission was not satisfied."""


def digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TaskView:
    task_id: str
    tenant: str
    state: str
    version: int
    fence: int
    worker: str | None
    lease_until: float | None
    result: str | None


class DurableQueue:
    """One SQLite file is authoritative; wall clock is an injected dependency."""

    def __init__(self, path: Path, clock: Callable[[], float] = time.time):
        self.path = str(path)
        self.clock = clock
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA busy_timeout=5000")
        return db

    def initialize(self) -> None:
        with self.connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY, tenant TEXT NOT NULL,
                    request_hash TEXT NOT NULL, state TEXT NOT NULL,
                    version INTEGER NOT NULL DEFAULT 0,
                    worker TEXT, fence INTEGER NOT NULL DEFAULT 0,
                    lease_until REAL, result TEXT
                );
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL REFERENCES tasks(id),
                    kind TEXT NOT NULL, fence INTEGER NOT NULL,
                    description TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS effects (
                    tenant TEXT NOT NULL, operation_key TEXT NOT NULL,
                    payload_hash TEXT NOT NULL, result TEXT NOT NULL,
                    PRIMARY KEY(tenant, operation_key)
                );
                CREATE TABLE IF NOT EXISTS outbox (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL REFERENCES tasks(id),
                    kind TEXT NOT NULL, payload TEXT NOT NULL,
                    sent INTEGER NOT NULL DEFAULT 0
                );
                """
            )

    def transaction(self, work):
        db = self.connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            answer = work(db)
            db.execute("COMMIT")
            return answer
        except BaseException:
            if db.in_transaction:
                db.execute("ROLLBACK")
            raise
        finally:
            db.close()

    def add(self, tenant: str, task_id: str, request: str) -> TaskView:
        if not tenant or not task_id or not request:
            raise ValueError("tenant, ID and request are required")
        request_hash = digest(request)

        def run(db):
            row = db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
            if row:
                if row["tenant"] != tenant or row["request_hash"] != request_hash:
                    raise Conflict("task identity reused with another request")
            else:
                db.execute(
                    "INSERT INTO tasks(id,tenant,request_hash,state) "
                    "VALUES(?,?,?,'PENDING')",
                    (task_id, tenant, request_hash),
                )
                db.execute(
                    "INSERT INTO events(task_id,kind,fence,description) "
                    "VALUES(?,'create',0,'request accepted')", (task_id,)
                )
            return self._view(db, tenant, task_id)

        return self.transaction(run)

    @staticmethod
    def _view(db, tenant: str, task_id: str) -> TaskView:
        row = db.execute(
            "SELECT * FROM tasks WHERE tenant=? AND id=?", (tenant, task_id)
        ).fetchone()
        if row is None:
            raise Conflict("task unavailable")
        return TaskView(
            row["id"], row["tenant"], row["state"], row["version"],
            row["fence"], row["worker"], row["lease_until"], row["result"],
        )

    def get(self, tenant: str, task_id: str) -> TaskView:
        with self.connect() as db:
            return self._view(db, tenant, task_id)

    def claim(self, tenant: str, task_id: str, worker: str,
              duration: float = 20) -> TaskView:
        if duration <= 0 or not worker:
            raise ValueError("invalid worker or lease duration")

        def run(db):
            current = self._view(db, tenant, task_id)
            now = self.clock()
            if current.state in ("DONE", "CANCELED", "FAILED"):
                raise Conflict("task terminal")
            if current.state == "RUNNING" and current.lease_until > now:
                raise Conflict("another worker holds an active lease")
            new_fence = current.fence + 1
            count = db.execute(
                "UPDATE tasks SET state='RUNNING',version=version+1,"
                "worker=?,fence=?,lease_until=? "
                "WHERE tenant=? AND id=? AND version=?",
                (worker, new_fence, now + duration, tenant, task_id,
                 current.version),
            ).rowcount
            if count != 1:
                raise Conflict("stale task version")
            db.execute(
                "INSERT INTO events(task_id,kind,fence,description) "
                "VALUES(?,'claim',?,?)", (task_id, new_fence, worker)
            )
            return self._view(db, tenant, task_id)

        return self.transaction(run)

    def heartbeat(self, tenant: str, task_id: str, worker: str,
                  fence: int, duration: float = 20) -> TaskView:
        if duration <= 0:
            raise ValueError("invalid duration")

        def run(db):
            task = self._view(db, tenant, task_id)
            now = self.clock()
            if (task.state != "RUNNING" or task.worker != worker
                    or task.fence != fence or task.lease_until <= now):
                raise Conflict("lease no longer valid")
            db.execute(
                "UPDATE tasks SET lease_until=?,version=version+1 "
                "WHERE id=?", (now + duration, task_id)
            )
            return self._view(db, tenant, task_id)

        return self.transaction(run)

    def complete(self, tenant: str, task_id: str, worker: str,
                 fence: int, result: str) -> TaskView:
        if not isinstance(result, str) or not result:
            raise ValueError("nonempty result required")

        def run(db):
            task = self._view(db, tenant, task_id)
            now = self.clock()
            if (task.state != "RUNNING" or task.worker != worker
                    or task.fence != fence or task.lease_until <= now):
                raise Conflict("stale worker cannot complete task")
            db.execute(
                "UPDATE tasks SET state='DONE',version=version+1,result=?,"
                "worker=NULL,lease_until=NULL WHERE id=?", (result, task_id)
            )
            db.execute(
                "INSERT INTO events(task_id,kind,fence,description) "
                "VALUES(?,'complete',?,?)", (task_id, fence, result)
            )
            db.execute(
                "INSERT INTO outbox(task_id,kind,payload) VALUES(?,?,?)",
                (task_id, "completed", json.dumps({"result": result}))
            )
            return self._view(db, tenant, task_id)

        return self.transaction(run)

    def effect_once(self, tenant: str, key: str, payload: str) -> str:
        """Simulate a remote idempotent write using the local atomic store.

        This guarantees dedup only for this database, not a real remote API.
        """
        if not tenant or not key or not payload:
            raise ValueError("invalid effect arguments")
        h = digest(payload)

        def run(db):
            prior = db.execute(
                "SELECT * FROM effects WHERE tenant=? AND operation_key=?",
                (tenant, key)
            ).fetchone()
            if prior:
                if prior["payload_hash"] != h:
                    raise Conflict("idempotency key reused with new content")
                return prior["result"]
            result = "effect-" + digest(tenant + ":" + key)[:12]
            db.execute(
                "INSERT INTO effects VALUES(?,?,?,?)", (tenant, key, h, result)
            )
            return result

        return self.transaction(run)

    def deliver_outbox(self, sender) -> int:
        """At-least-once delivery: sender must implement its own dedup."""
        delivered = 0
        with self.connect() as db:
            rows = db.execute(
                "SELECT event_id,kind,payload FROM outbox "
                "WHERE sent=0 ORDER BY event_id"
            ).fetchall()
        for row in rows:
            sender(row["event_id"], row["kind"], row["payload"])
            with self.connect() as db:
                db.execute(
                    "UPDATE outbox SET sent=1 WHERE event_id=?",
                    (row["event_id"],)
                )
            delivered += 1
        return delivered

    def events(self, task_id: str) -> list[tuple[str, int]]:
        with self.connect() as db:
            return [(r["kind"], r["fence"]) for r in db.execute(
                "SELECT kind,fence FROM events WHERE task_id=? ORDER BY seq",
                (task_id,)
            )]


def expect(exc, fn):
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__}")


def self_test() -> None:
    now = [1000.0]
    clock = lambda: now[0]
    with tempfile.TemporaryDirectory(prefix="agent-runtime-") as tmp:
        path = Path(tmp) / "runtime.sqlite"
        q1 = DurableQueue(path, clock)
        q2 = DurableQueue(path, clock)
        t = q1.add("A", "task-1", "fix synthetic failing test")
        assert t.state == "PENDING" and t.version == 0
        assert q2.add("A", "task-1", "fix synthetic failing test") == t
        expect(Conflict, lambda: q1.add("B", "task-1", "another tenant"))
        expect(Conflict, lambda: q2.get("B", "task-1"))

        first = q1.claim("A", "task-1", "worker-old", duration=10)
        assert first.fence == 1 and first.version == 1
        expect(Conflict, lambda: q2.claim("A", "task-1", "worker-new"))
        now[0] += 11
        second = q2.claim("A", "task-1", "worker-new", duration=20)
        assert second.fence == 2
        expect(Conflict, lambda: q1.heartbeat(
            "A", "task-1", "worker-old", first.fence
        ))
        expect(Conflict, lambda: q1.complete(
            "A", "task-1", "worker-old", first.fence, "stale result"
        ))
        after_heartbeat = q2.heartbeat(
            "A", "task-1", "worker-new", second.fence
        )
        assert after_heartbeat.lease_until == now[0] + 20
        finished = q2.complete(
            "A", "task-1", "worker-new", second.fence, "tests passed"
        )
        assert finished.state == "DONE"
        assert q1.get("A", "task-1").result == "tests passed"
        expect(Conflict, lambda: q1.claim("A", "task-1", "worker-old"))
        assert q1.events("task-1") == [
            ("create", 0), ("claim", 1), ("claim", 2), ("complete", 2)
        ]

        a = q1.effect_once("A", "op-1", "write report")
        b = q2.effect_once("A", "op-1", "write report")
        assert a == b
        expect(Conflict, lambda: q2.effect_once(
            "A", "op-1", "overwrite something else"
        ))
        assert q2.effect_once("B", "op-1", "different tenant") != a
        with q1.connect() as db:
            assert db.execute("SELECT count(*) FROM effects").fetchone()[0] == 2

        seen: list[int] = []

        def send(event_id, kind, payload):
            assert kind == "completed"
            assert json.loads(payload)["result"] == "tests passed"
            seen.append(event_id)

        assert q1.deliver_outbox(send) == 1
        assert q2.deliver_outbox(send) == 0
        assert len(seen) == 1

        q3 = DurableQueue(path, clock)
        assert q3.get("A", "task-1").state == "DONE"
        print("PASS durable queue: tenant, CAS, lease, fencing, outbox, replay")
        print("NOTE: external API exactly-once is NOT established here.")


if __name__ == "__main__":
    self_test()
