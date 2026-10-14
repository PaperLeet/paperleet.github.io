"""Single-instance, durable FIFO jobs; HTTP polling never executes a job twice."""
from contextlib import contextmanager
import json
import logging
from pathlib import Path
import re
import sqlite3
import threading
import time


class QueueFull(ValueError):
    pass


class JobConflict(ValueError):
    pass


class WorkerBusy(RuntimeError):
    """The worker explicitly refused the job before execution; safe to wait."""


class JobFailure(RuntimeError):
    """A safe, user-facing execution/service failure."""


class JudgeQueue:
    def __init__(self, path, execute, scope="local", capacity=8, max_wait=1200, start=True):
        self.path, self.execute, self.scope = Path(path), execute, scope
        self.capacity, self.max_wait = capacity, max_wait
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stopping, self.wake = threading.Event(), threading.Event()
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS judge_jobs (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE,
                    scope TEXT NOT NULL, owner TEXT NOT NULL, problem TEXT NOT NULL,
                    payload TEXT NOT NULL, state TEXT NOT NULL,
                    created_at REAL NOT NULL, updated_at REAL NOT NULL,
                    report TEXT, error TEXT
                );
                CREATE UNIQUE INDEX IF NOT EXISTS judge_one_active_owner
                    ON judge_jobs(scope,owner) WHERE state IN ('queued','running');
                CREATE INDEX IF NOT EXISTS judge_pending ON judge_jobs(scope,state,seq);
            """)
            # A running task has an unknown outcome after process loss. Never
            # re-execute it automatically; queued tasks can safely resume.
            db.execute("UPDATE judge_jobs SET state='failed',error=?,updated_at=? WHERE scope=? AND state='running'",
                       ("The service restarted during this run. Your draft is preserved; submit again.", time.time(), scope))
        self.thread = threading.Thread(target=self.work, name="paperleet-judge-queue", daemon=True)
        if start:
            self.thread.start()

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def snapshot(self, db, row):
        ahead = 0
        if row["state"] == "queued":
            ahead = db.execute("SELECT COUNT(*) FROM judge_jobs WHERE scope=? AND (state='running' OR (state='queued' AND seq<?))",
                               (self.scope, row["seq"])).fetchone()[0]
        result = {"id": row["id"], "problem": row["problem"], "status": row["state"],
                  "ahead": ahead, "created_at": row["created_at"]}
        if row["report"]:
            result["report"] = json.loads(row["report"])
        if row["error"]:
            result["error"] = row["error"]
        return result

    def get(self, identifier, owner):
        with self.connect() as db:
            row = db.execute("SELECT * FROM judge_jobs WHERE scope=? AND id=? AND owner=?", (self.scope, identifier, owner)).fetchone()
            if row is None:
                raise KeyError("Unknown or expired judge job")
            return self.snapshot(db, row)

    def submit(self, identifier, owner, payload):
        if not isinstance(identifier, str) or not re.fullmatch(r"[0-9a-f]{32}", identifier):
            raise ValueError("A 32-character request ID is required")
        encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM judge_jobs WHERE scope=? AND state NOT IN ('queued','running') AND updated_at<?", (self.scope, time.time() - 86400))
            existing = db.execute("SELECT * FROM judge_jobs WHERE id=?", (identifier,)).fetchone()
            if existing:
                if existing["scope"] != self.scope or existing["owner"] != owner or existing["payload"] != encoded:
                    raise JobConflict("This request ID has already been used")
                return self.snapshot(db, existing)
            if db.execute("SELECT 1 FROM judge_jobs WHERE scope=? AND owner=? AND state IN ('queued','running')", (self.scope, owner)).fetchone():
                raise JobConflict("You already have a queued or running submission. Wait for it to finish.")
            if db.execute("SELECT COUNT(*) FROM judge_jobs WHERE scope=? AND state IN ('queued','running')", (self.scope,)).fetchone()[0] >= self.capacity:
                raise QueueFull("The judge queue is full. Your draft is preserved; try again shortly.")
            now = time.time()
            db.execute("INSERT INTO judge_jobs(id,scope,owner,problem,payload,state,created_at,updated_at) VALUES (?,?,?,?,?,'queued',?,?)",
                       (identifier, self.scope, owner, payload["problem"], encoded, now, now))
            row = db.execute("SELECT * FROM judge_jobs WHERE id=?", (identifier,)).fetchone()
            result = self.snapshot(db, row)
        self.wake.set()
        return result

    def cancel(self, identifier, owner):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM judge_jobs WHERE scope=? AND id=? AND owner=?", (self.scope, identifier, owner)).fetchone()
            if row is None:
                raise KeyError("Unknown or expired judge job")
            if row["state"] == "running":
                raise JobConflict("This job has already started and cannot be removed from the queue.")
            if row["state"] == "queued":
                db.execute("UPDATE judge_jobs SET state='cancelled',updated_at=? WHERE id=?", (time.time(), identifier))
            return self.snapshot(db, db.execute("SELECT * FROM judge_jobs WHERE id=?", (identifier,)).fetchone())

    def finish(self, identifier, state, report=None, error=None):
        with self.connect() as db:
            db.execute("UPDATE judge_jobs SET state=?,report=?,error=?,updated_at=? WHERE id=?",
                       (state, json.dumps(report, allow_nan=False) if report is not None else None, error, time.time(), identifier))

    def work(self):
        while not self.stopping.is_set():
            try:
                with self.connect() as db:
                    db.execute("BEGIN IMMEDIATE")
                    row = db.execute("SELECT * FROM judge_jobs WHERE scope=? AND state='queued' ORDER BY seq LIMIT 1", (self.scope,)).fetchone()
                    if row:
                        if time.time() - row["created_at"] > self.max_wait:
                            db.execute("UPDATE judge_jobs SET state='failed',error=?,updated_at=? WHERE id=?",
                                       ("The queue wait limit was reached. Your draft is preserved; submit again.", time.time(), row["id"]))
                            continue
                        db.execute("UPDATE judge_jobs SET state='running',updated_at=? WHERE id=?", (time.time(), row["id"]))
                if row is None:
                    self.wake.wait(.5)
                    self.wake.clear()
                    continue
                try:
                    report = self.execute(row["owner"], json.loads(row["payload"]))
                    self.finish(row["id"], "succeeded", report=report)
                except WorkerBusy:
                    self.finish(row["id"], "queued")
                    self.stopping.wait(1)
                except JobFailure as exc:
                    self.finish(row["id"], "failed", error=str(exc))
                except Exception:
                    logging.exception("Judge job %s failed", row["id"])
                    self.finish(row["id"], "failed", error="The judge could not complete this job. Your draft is preserved.")
            except Exception:
                logging.exception("Judge queue storage unavailable")
                self.stopping.wait(1)

    def close(self):
        self.stopping.set()
        self.wake.set()
        if self.thread.is_alive():
            self.thread.join(timeout=.5)
