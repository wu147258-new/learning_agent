"""Offline HTTP URL and SQLite persistence exercise; no network required.

Run: python examples/foundation_http_sql.py
"""
from __future__ import annotations

from urllib.parse import parse_qs, urlsplit
import sqlite3


def check_url():
    url = "https://example.org/api/tasks?status=running&limit=10"
    parts = urlsplit(url)
    assert parts.scheme == "https"
    assert parts.hostname == "example.org"
    assert parts.path == "/api/tasks"
    query = parse_qs(parts.query)
    assert query["status"] == ["running"]
    assert query["limit"] == ["10"]


def check_sqlite():
    db = sqlite3.connect(":memory:")
    try:
        db.execute(
            "CREATE TABLE tasks ("
            "task_id TEXT PRIMARY KEY,"
            "status TEXT NOT NULL,"
            "version INTEGER NOT NULL,"
            "attempts INTEGER NOT NULL DEFAULT 0)"
        )
        db.execute(
            "INSERT INTO tasks(task_id,status,version) VALUES (?,?,?)",
            ("t-001", "pending", 0),
        )
        db.commit()

        row = db.execute(
            "SELECT status,version FROM tasks WHERE task_id=?",
            ("t-001",),
        ).fetchone()
        assert row == ("pending", 0)

        # Executor A has read version 0, and performs a successful update.
        with db:
            first = db.execute(
                "UPDATE tasks SET status=?, version=version+1 "
                "WHERE task_id=? AND version=?",
                ("running", "t-001", 0),
            )
            assert first.rowcount == 1

        # Executor B's old version 0 is rejected by the database.
        with db:
            stale = db.execute(
                "UPDATE tasks SET status=?, version=version+1 "
                "WHERE task_id=? AND version=?",
                ("done", "t-001", 0),
            )
            assert stale.rowcount == 0
        assert db.execute(
            "SELECT status,version FROM tasks WHERE task_id=?",
            ("t-001",),
        ).fetchone() == ("running", 1)

        # Primary-key collision must be rejected, not silently overwritten.
        try:
            with db:
                db.execute(
                    "INSERT INTO tasks(task_id,status,version) VALUES (?,?,?)",
                    ("t-001", "pending", 0),
                )
        except sqlite3.IntegrityError:
            pass
        else:
            raise AssertionError("duplicate primary key accepted")

        # An exception in a transactional block rolls back the change.
        try:
            with db:
                db.execute(
                    "UPDATE tasks SET attempts=attempts+1 WHERE task_id=?",
                    ("t-001",),
                )
                raise RuntimeError("synthetic downstream failure")
        except RuntimeError:
            pass
        attempts = db.execute(
            "SELECT attempts FROM tasks WHERE task_id=?", ("t-001",)
        ).fetchone()[0]
        assert attempts == 0
        print("PASS: URL parsing, SQL parameters, optimistic versioning, rollback")
    finally:
        db.close()


if __name__ == "__main__":
    check_url()
    check_sqlite()
