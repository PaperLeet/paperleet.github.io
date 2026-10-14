"""Short-lived HTTP requests for private queued jobs in local and hosted modes."""
import re
import logging
import threading

from paperleet.catalog import problem, root, workspace_path
from paperleet.config import data_directory, hosted
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
                                            scope="hosted" if hosted() else "local")
        return server.judge_queue


def execute(owner, job):
    entry = problem(job["problem"])
    if hosted():
        from paperleet import social_api
        from paperleet.config import auth_mode
        from paperleet.hosting.judge_client import JudgeBusy, JudgeUnavailable, run_hosted_problem
        from paperleet.practice import PracticeService
        accounts, _ = social_api.services()
        user_id = owner.removeprefix("user:")
        practice = PracticeService(accounts.db_path)
        try:
            practice.workspace(user_id)  # Recheck that the account is still active.
            user = accounts.get_user(user_id)
            if auth_mode() == "institutional":
                accounts._check_invitation(user["email"])
                if not user["institution_verified"] or user["is_test_user"]:
                    raise PermissionError("Your account is no longer eligible for hosted judging.")
            elif not user["is_test_user"]:
                raise PermissionError("Sign in with a test account to run this job.")
            report = run_hosted_problem(entry, job["code"], job["layers"])
            run = practice.save_run(user_id, entry, job["code"], report)
            return {**report, "run_id": run["id"]}
        except JudgeBusy as exc:
            raise WorkerBusy() from exc
        except (JudgeUnavailable, PermissionError, KeyError, ValueError) as exc:
            raise JobFailure(str(exc)) from exc
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
    from paperleet import social_api
    from paperleet.accounts import AccountError, RateLimited
    payload = payload or {}
    match = _route.fullmatch(handler.path)
    if not match:
        return handler.respond(404, {"error": "Unknown job endpoint"})
    identifier, cancellation = match.groups()
    if (method == "GET" and (not identifier or cancellation)) or (method == "POST" and identifier and not cancellation):
        return handler.respond(404, {"error": "Unknown job endpoint"})
    try:
        if hosted():
            accounts, _ = social_api.services()
            token = social_api.session_token(handler)
            user = accounts.get_session(token) if token else None
            if not user:
                return handler.respond(401, {"error": "Sign in to view or submit judge jobs."})
            if method == "POST" and payload.get("expected_user_id") != user["id"]:
                return handler.respond(409, {"error": "Your account changed. Reload before submitting."})
            owner = "user:" + user["id"]
        else:
            client = handler.headers.get("X-PaperLeet-Client-ID", "")
            if not re.fullmatch(r"[0-9a-f]{32}", client):
                raise ValueError("A local browser client ID is required")
            owner = "local:" + client
        if method == "GET":
            return handler.respond(200, queue_for(handler).get(identifier, owner))
        if not social_api._throttle(handler, handler.path):
            return handler.respond(429, {"error": "Too many requests. Please wait before submitting again."})
        if cancellation:
            return handler.respond(200, queue_for(handler).cancel(identifier, owner))
        entry = problem(payload.get("problem"))
        code = payload.get("code")
        if not isinstance(code, str) or not code.strip() or len(code.encode("utf-8")) > 128 * 1024:
            raise ValueError("Python code must be nonempty and at most 128 KB")
        job = {"problem": entry["id"], "code": code, "layers": normalize_layers(payload.get("layers"))}
        if hosted():
            from paperleet.hosting.judge_client import validate_job
            validate_job(job)
            accounts._limit("judge", user["id"], maximum=10, seconds=60, global_maximum=100)
        return handler.respond(202, queue_for(handler).submit(payload.get("id"), owner, job))
    except QueueFull as exc:
        return handler.respond(429, {"error": str(exc), "code": "queue_full"}, {"Retry-After": "5"})
    except JobConflict as exc:
        return handler.respond(409, {"error": str(exc)})
    except RateLimited as exc:
        return handler.respond(429, {"error": str(exc)})
    except KeyError:
        return handler.respond(404, {"error": "This job does not exist or has expired. Your draft is preserved."})
    except (AccountError, TypeError, ValueError) as exc:
        return handler.respond(400, {"error": str(exc)})
    except Exception:
        logging.exception("Judge jobs API unavailable")
        return handler.respond(503, {"error": "The judge job service is temporarily unavailable. Your draft is preserved."})
