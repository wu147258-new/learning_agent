"""Reopen a real SQLite file and test transactional task claims.

This exercise proves application-level restart/reopen behavior, not power-loss
durability or remote exactly-once semantics.
"""
from __future__ import annotations
import sqlite3
from tempfile import TemporaryDirectory
from pathlib import Path


SCHEMA = """CREATE TABLE task(
 id TEXT PRIMARY KEY, status TEXT NOT NULL, version INTEGER NOT NULL
);
CREATE TABLE event(
 seq INTEGER PRIMARY KEY AUTOINCREMENT,
 task_id TEXT NOT NULL,
 note TEXT NOT NULL
);"""


def claim(conn, task_id, expected_version):
    with conn:
        cur=conn.execute(
            "UPDATE task SET status='RUNNING', version=version+1 "
            "WHERE id=? AND version=? AND status='PENDING'",
            (task_id,expected_version))
        if cur.rowcount == 1:
            conn.execute("INSERT INTO event(task_id,note) VALUES (?,?)",
                         (task_id,"CLAIMED"))
        return cur.rowcount


def main():
    with TemporaryDirectory() as temp:
        path=Path(temp)/"workflow.sqlite"
        con=sqlite3.connect(path)
        con.executescript(SCHEMA)
        con.execute("INSERT INTO task VALUES('t1','PENDING',0)")
        con.commit()
        con.close()

        first=sqlite3.connect(path,timeout=2)
        second=sqlite3.connect(path,timeout=2)
        assert claim(first,"t1",0)==1
        assert claim(second,"t1",0)==0
        assert first.execute("SELECT count(*) FROM event").fetchone()[0]==1
        assert first.execute("SELECT status,version FROM task").fetchone()==(
            "RUNNING",1)
        first.close()
        second.close()

        recovered=sqlite3.connect(path)
        assert recovered.execute("SELECT status,version FROM task").fetchone()==(
            "RUNNING",1)
        try:
            with recovered:
                recovered.execute("INSERT INTO event(task_id,note) VALUES('t1','ROLLBACK')")
                raise RuntimeError("simulate failed business operation")
        except RuntimeError:
            pass
        assert recovered.execute("SELECT count(*) FROM event").fetchone()[0]==1
        recovered.close()
    print("PASS real SQLite file re-open and CAS task claim")
    print("PASS second connection loses race; event update is atomic")
    print("PASS failed transaction rolls back its inserted event")


if __name__=="__main__":
    main()
