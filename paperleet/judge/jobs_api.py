"""Queued local judging with browser client isolation and bounded requests."""
import re
import logging
import threading

from paperleet.catalog import problem, root, workspace_path
from paperleet.config import data_directory
from paperleet.judge.runner import normalize_layers
from .queue import JudgeQueue, JobConflict, JobFailure, QueueFull, WorkerBusy

_creation_lock = threading.Lock()
_route = re.compile(r"/api/judge/jobs(?:/([0-9a-f]{32})(/cancel)?)?")


def is_job_path(path):
    return bool(_route.fullmatch(path))


def queue_for(handler):
    return start_queue(handler.server)


def start_queue(server):
    with _creation_lock:
        if not hasattr(server, "judge_queue"):
            server.judge_queue = JudgeQueue(data_directory() / "community.sqlite3", execute,
                                            scope="local")
        return server.judge_queue


def execute(owner, job):
    entry = problem(job["problem"])
    # Share the execution lock with the legacy synchronous local endpoint.
    from paperleet import server
    if not server._judge_lock.acquire(blocking=False):
        raise WorkerBusy()
    try:
        path = workspace_path(entry)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(job["code"], encoding="utf-8")
        destination = root() / "reports" / (entry["id"].replace("/", "-") + ".json")
        return server.run_problem(entry["id"], path, layers=job["layers"], report_path=destination)
    finally:
        server._judge_lock.release()


def dispatch(handler, method, payload=None):
    payload = payload or {}
    match = _route.fullmatch(handler.path)
    if not match:
        return handler.respond(404, {"error": "Unknown job endpoint"})
    identifier, cancellation = match.groups()
    if (method == "GET" and (not identifier or cancellation)) or (method == "POST" and identifier and not cancellation):
        return handler.respond(404, {"error": "Unknown job endpoint"})
    try:
        client = handler.headers.get("X-PaperLeet-Client-ID", "")
        if not re.fullmatch(r"[0-9a-f]{32}", client):
            raise ValueError("A local browser client ID is required")
        owner = "local:" + client
        if method == "GET":
            return handler.respond(200, queue_for(handler).get(identifier, owner))
        if cancellation:
            return handler.respond(200, queue_for(handler).cancel(identifier, owner))
        entry = problem(payload.get("problem"))
        code = payload.get("code")
        if not isinstance(code, str) or not code.strip() or len(code.encode("utf-8")) > 128 * 1024:
            raise ValueError("Python code must be nonempty and at most 128 KB")
        job = {"problem": entry["id"], "code": code, "layers": normalize_layers(payload.get("layers"))}
        return handler.respond(202, queue_for(handler).submit(payload.get("id"), owner, job))
    except QueueFull as exc:
        return handler.respond(429, {"error": str(exc), "code": "queue_full"}, {"Retry-After": "5"})
    except JobConflict as exc:
        return handler.respond(409, {"error": str(exc)})
    except KeyError:
        return handler.respond(404, {"error": "This job does not exist or has expired. Your draft is preserved."})
    except (TypeError, ValueError) as exc:
        return handler.respond(400, {"error": str(exc)})
    except Exception:
        logging.exception("Judge jobs API unavailable")
        return handler.respond(503, {"error": "The judge job service is temporarily unavailable. Your draft is preserved."})
