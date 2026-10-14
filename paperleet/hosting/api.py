"""Authenticated hosted routes; never dispatch learner code to the local runner."""
import threading
from urllib.parse import parse_qs, urlsplit

from paperleet.catalog import problem
from paperleet.config import auth_mode
from paperleet.judge.runner import normalize_layers
from paperleet import social_api
from paperleet.submissions import store


_active_users = set()
_active_lock = threading.Lock()
PAPER_WORKFLOW_REASON = "During the beta, paper proposals are saved for maintainers. Generation, validation, and publication are managed offline."


def current_user(handler):
    accounts, _ = social_api.services()
    token = social_api.session_token(handler)
    return accounts.get_session(token) if token else None


def dispatch(handler, method, payload=None):
    from paperleet.accounts import AccountError, RateLimited
    from paperleet.practice import PracticeService, PracticeConflict
    from paperleet.hosting.judge_client import JudgeBusy, JudgeUnavailable, run_hosted_problem
    payload = payload or {}
    parsed = urlsplit(handler.path)
    path = parsed.path
    try:
        if path == "/api/practice/stats" and method == "GET":
            accounts, _ = social_api.services()
            return handler.respond(200, {"available": True, "stats": PracticeService(accounts.db_path).problem_stats()})
        user = current_user(handler)
        if path == "/api/submissions/capabilities" and method == "GET":
            return handler.respond(200, {"mode": "hosted", "can_submit": bool(user),
                "can_generate": False, "can_review": False, "human_review_required": True,
                "reason": PAPER_WORKFLOW_REASON,
                "generation": {"ready": False, "reason": PAPER_WORKFLOW_REASON},
                "sandbox": {"ready": False, "reason": PAPER_WORKFLOW_REASON}})
        if not user:
            return handler.respond(401, {"error": "Sign in with your test account to continue." if auth_mode() == "test" else "Sign in with your verified school or company email to continue."})
        accounts, _ = social_api.services()
        practice = PracticeService(accounts.db_path)
        if path == "/api/practice/workspace" and method == "GET":
            return handler.respond(200, {"user_id": user["id"], "workspace": practice.workspace(user["id"])})
        if path == "/api/practice/runs" and method == "GET":
            query = parse_qs(parsed.query)
            return handler.respond(200, {"runs": practice.list_runs(user["id"],
                limit=int(query.get("limit", ["30"])[0]), offset=int(query.get("offset", ["0"])[0]))})
        if method == "POST" and not social_api._throttle(handler, path):
            return handler.respond(429, {"error": "Too many requests. Please try again shortly."})
        if method == "POST" and path in ("/api/practice/draft", "/api/test") and payload.get("expected_user_id") != user["id"]:
            return handler.respond(409, {"error": "Your account changed. Reload before saving or running code."})
        if path == "/api/practice/draft" and method == "POST":
            draft = practice.save_draft(user["id"], payload.get("problem"), payload.get("code"),
                                        expected_updated=payload.get("expected_updated"))
            return handler.respond(200, {"draft": draft})
        if path == "/api/test" and method == "POST":
            entry = problem(payload.get("problem"))
            code = payload.get("code")
            if not isinstance(code, str) or not code.strip() or len(code.encode("utf-8")) > 128 * 1024:
                raise ValueError("Python code must be nonempty and at most 128 KB")
            layers = normalize_layers(payload.get("layers"))
            accounts._limit("judge", user["id"], maximum=10, seconds=60, global_maximum=100)
            with _active_lock:
                if user["id"] in _active_users:
                    return handler.respond(429, {"error": "Your previous judge run is still in progress. Please wait."})
                _active_users.add(user["id"])
            try:
                report = run_hosted_problem(entry, code, layers)
                run = practice.save_run(user["id"], entry, code, report)
                return handler.respond(200, {**report, "run_id": run["id"]})
            finally:
                with _active_lock:
                    _active_users.discard(user["id"])
        if path == "/api/submissions":
            if method == "GET":
                rows = store().list()
                if user["role"] != "admin":
                    rows = [row for row in rows if row.get("owner_id") == user["id"]]
                return handler.respond(200, {"submissions": rows})
            if method == "POST":
                accounts._limit("paper", user["id"], maximum=5, seconds=3600, global_maximum=50)
                return handler.respond(201, store().create(payload, owner_id=user["id"]))
        parts = path.strip("/").split("/")
        if parts[:2] == ["api", "submissions"] and len(parts) in (3, 4):
            record = store().get(parts[2])
            if user["role"] != "admin" and record.get("owner_id") != user["id"]:
                raise KeyError("Unknown submission")
            if len(parts) == 3 and method == "GET":
                return handler.respond(200, record)
            if len(parts) == 4 and method == "POST" and parts[3] in ("generate", "validate", "review"):
                if user["role"] != "admin":
                    raise PermissionError("Only administrators can manage paper review and publication.")
                return handler.respond(503, {"error": PAPER_WORKFLOW_REASON})
        return handler.respond(404, {"error": "Unknown endpoint"})
    except PracticeConflict as exc:
        return handler.respond(409, {"error": str(exc)})
    except (RateLimited, JudgeBusy) as exc:
        return handler.respond(429, {"error": str(exc)})
    except JudgeUnavailable as exc:
        return handler.respond(503, {"error": str(exc)})
    except PermissionError as exc:
        return handler.respond(403, {"error": str(exc)})
    except KeyError:
        return handler.respond(404, {"error": "The requested content does not exist."})
    except (AccountError, ValueError, TypeError) as exc:
        return handler.respond(400, {"error": str(exc)})
    except Exception:
        return handler.respond(503, {"error": "The service is temporarily unavailable. Please retry shortly."})
