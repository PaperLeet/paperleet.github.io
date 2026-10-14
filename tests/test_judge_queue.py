"""FIFO, durable recovery, deduplication, ownership and bounded resource use."""
import threading
import time

import pytest

from paperleet.judge.queue import JudgeQueue, QueueFull, JobConflict, WorkerBusy


def job(number=1):
    return {"problem": "rl/gae", "code": f"source {number}", "layers": ["T0"]}


def identifier(number):
    return f"{number:032x}"


def terminal(queue, number, owner, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = queue.get(identifier(number), owner)
        if result["status"] not in ("queued", "running"):
            return result
        time.sleep(.01)
    pytest.fail("Queue job did not finish")


def test_fifo_positions_cancel_and_private_snapshots(tmp_path):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def execute(owner, payload):
        calls.append((owner, payload["code"]))
        if owner == "alice":
            entered.set()
            assert release.wait(5)
        return {"problem": payload["problem"], "passed": False}

    queue = JudgeQueue(tmp_path / "db.sqlite3", execute)
    try:
        queue.submit(identifier(1), "alice", job(1))
        assert entered.wait(2)
        assert queue.get(identifier(1), "alice")["status"] == "running"
        second = queue.submit(identifier(2), "bob", job(2))
        third = queue.submit(identifier(3), "carol", job(3))
        assert second["ahead"] == 1 and third["ahead"] == 2
        assert not {"code", "payload", "owner"} & second.keys()
        with pytest.raises(KeyError):
            queue.get(identifier(1), "bob")
        with pytest.raises(KeyError):
            queue.cancel(identifier(1), "bob")
        with pytest.raises(JobConflict):
            queue.cancel(identifier(1), "alice")
        assert queue.cancel(identifier(2), "bob")["status"] == "cancelled"
        assert queue.get(identifier(3), "carol")["ahead"] == 1
        release.set()
        assert terminal(queue, 1, "alice")["status"] == "succeeded"
        assert terminal(queue, 3, "carol")["status"] == "succeeded"
        assert calls == [("alice", "source 1"), ("carol", "source 3")]
    finally:
        release.set()
        queue.close()


def test_idempotent_submission_and_capacity_are_atomic(tmp_path):
    calls = []
    queue = JudgeQueue(tmp_path / "db.sqlite3", lambda owner, payload: calls.append(owner) or {}, capacity=2, start=False)
    try:
        original = queue.submit(identifier(1), "alice", job())
        assert queue.submit(identifier(1), "alice", job()) == original
        with pytest.raises(JobConflict):
            queue.submit(identifier(1), "alice", job(2))
        with pytest.raises(JobConflict):
            queue.submit(identifier(1), "bob", job())
        with pytest.raises(JobConflict):
            queue.submit(identifier(2), "alice", job())
        queue.submit(identifier(2), "bob", job())
        with pytest.raises(QueueFull):
            queue.submit(identifier(3), "carol", job())
        queue.thread.start()
        assert terminal(queue, 1, "alice")["status"] == "succeeded"
        assert terminal(queue, 2, "bob")["status"] == "succeeded"
        assert queue.submit(identifier(1), "alice", job())["status"] == "succeeded"
        assert calls == ["alice", "bob"]
    finally:
        queue.close()


def test_concurrent_submissions_cannot_bypass_one_job_per_owner(tmp_path):
    queue = JudgeQueue(tmp_path / "db.sqlite3", lambda *_: {}, start=False)
    outcomes = []
    barrier = threading.Barrier(3)

    def submit(number):
        barrier.wait()
        try:
            queue.submit(identifier(number), "alice", job(number))
            outcomes.append("accepted")
        except JobConflict:
            outcomes.append("conflict")

    threads = [threading.Thread(target=submit, args=(n,)) for n in (1, 2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(2)
    assert sorted(outcomes) == ["accepted", "conflict"]
    queue.close()


def test_restart_resumes_only_unstarted_jobs_and_preserves_reports(tmp_path):
    path = tmp_path / "db.sqlite3"
    queue = JudgeQueue(path, lambda *_: {}, start=False)
    queue.submit(identifier(1), "alice", job(1))
    queue.submit(identifier(2), "bob", job(2))
    with queue.connect() as db:
        db.execute("UPDATE judge_jobs SET state='running' WHERE id=?", (identifier(1),))
    queue.close()
    calls = []
    restarted = JudgeQueue(path, lambda owner, payload: calls.append(payload["code"]) or {"saved": True})
    try:
        assert restarted.get(identifier(1), "alice")["status"] == "failed"
        result = terminal(restarted, 2, "bob")
        assert result["report"] == {"saved": True}
        assert calls == ["source 2"]
    finally:
        restarted.close()


def test_queue_wait_is_bounded_and_does_not_run_expired_source(tmp_path):
    calls = []
    queue = JudgeQueue(tmp_path / "db.sqlite3", lambda *_: calls.append(True), max_wait=1, start=False)
    try:
        queue.submit(identifier(1), "alice", job())
        with queue.connect() as db:
            db.execute("UPDATE judge_jobs SET created_at=?", (time.time() - 5,))
        queue.thread.start()
        assert terminal(queue, 1, "alice")["status"] == "failed"
        assert not calls
    finally:
        queue.close()


def test_busy_worker_waits_without_creating_another_job(tmp_path):
    calls = []

    def execute(owner, payload):
        calls.append(payload)
        if len(calls) == 1:
            raise WorkerBusy()
        return {"done": True}

    queue = JudgeQueue(tmp_path / "db.sqlite3", execute)
    try:
        queue.submit(identifier(1), "alice", job())
        assert terminal(queue, 1, "alice")["report"] == {"done": True}
        assert len(calls) == 2
        with queue.connect() as db:
            assert db.execute("SELECT COUNT(*) FROM judge_jobs").fetchone()[0] == 1
    finally:
        queue.close()


def test_execution_failure_is_terminal_and_does_not_block_next_owner(tmp_path):
    def execute(owner, payload):
        if owner == "alice":
            raise RuntimeError("internal detail must not be returned")
        return {"done": True}
    queue = JudgeQueue(tmp_path / "db.sqlite3", execute)
    try:
        queue.submit(identifier(1), "alice", job())
        queue.submit(identifier(2), "bob", job())
        result = terminal(queue, 1, "alice")
        assert result["status"] == "failed" and "internal detail" not in result["error"]
        assert terminal(queue, 2, "bob")["status"] == "succeeded"
    finally:
        queue.close()
