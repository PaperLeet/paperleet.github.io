"""HTTP adapter for institutional accounts and the persisted community."""
from collections import defaultdict, deque
from http.cookies import SimpleCookie
import os
import sqlite3
import threading
import time
from urllib.parse import parse_qs, unquote, urlsplit

from paperleet.config import auth_mode, data_directory, hosted


COOKIE = "paperleet_session"
_services = {}
_service_lock = threading.Lock()
_rate_lock = threading.Lock()
_requests = defaultdict(deque)


def services():
    from paperleet.accounts import AccountService
    from paperleet.community import CommunityService
    database = data_directory() / "community.sqlite3"
    with _service_lock:
        if database not in _services:
            accounts = AccountService(database)
            _services[database] = (accounts, CommunityService(database, accounts.institutions))
        return _services[database]


def is_social_path(path):
    path = urlsplit(path).path
    return path.startswith("/api/account/") or path.startswith("/api/community/")


def session_token(handler):
    try:
        cookie = SimpleCookie()
        cookie.load(handler.headers.get("Cookie", ""))
        return cookie[COOKIE].value if COOKIE in cookie else ""
    except Exception:
        return ""


def session_cookie(token=""):
    cookie = SimpleCookie()
    cookie[COOKIE] = token
    cookie[COOKIE]["path"] = "/api"
    cookie[COOKIE]["httponly"] = True
    cookie[COOKIE]["samesite"] = "Lax"
    cookie[COOKIE]["max-age"] = 7 * 86400 if token else 0
    if hosted() or os.environ.get("PAPERLEET_SECURE_COOKIES") == "1":
        cookie[COOKIE]["secure"] = True
    return cookie.output(header="").strip()


def _throttle(handler, path):
    # Counts only writes, not profile/feed reads. Per-email limits live in auth.
    now = time.monotonic()
    group = "auth" if path.startswith("/api/account/") else "community"
    client = handler.client_address[0]
    if hosted() and os.environ.get("PAPERLEET_TRUST_PROXY") == "1":
        # The deployment proxy overwrites this header; the API has no public port.
        import ipaddress
        try:
            client = str(ipaddress.ip_address(handler.headers.get("X-PaperLeet-Client-IP", client)))
        except ValueError:
            pass
    key = (client, group)
    window, limit = (600, 40) if group == "auth" else (60, 60)
    with _rate_lock:
        # Evict inactive clients rather than retaining an IP history.
        for stale in [k for k, values in _requests.items() if not values or now - values[-1] > 600]:
            del _requests[stale]
        queue = _requests[key]
        while queue and now - queue[0] > window:
            queue.popleft()
        if len(queue) >= limit:
            return False
        queue.append(now)
    return True


def dispatch(handler, method, data=None):
    """Only the server calls this, after configured-origin and JSON checks."""
    from paperleet.accounts import AccountError, AuthenticationError, AuthUnavailable, RateLimited
    parsed = urlsplit(handler.path)
    path = parsed.path.rstrip("/")
    query = parse_qs(parsed.query)
    parts = [unquote(part) for part in path.split("/") if part]
    payload = data or {}
    try:
        accounts, community = services()
        token = session_token(handler)
        user = accounts.get_session(token) if token else None
        viewer = user["id"] if user else None
        if method == "POST" and not _throttle(handler, path):
            return handler.respond(429, {"error": "Too many requests. Please try again shortly."})
        if method == "GET" and path == "/api/account/me":
            return handler.respond(200, {"user": user, "capabilities": accounts.capabilities()})
        if method == "POST" and path == "/api/account/test-login":
            new_token, account = accounts.test_login(payload.get("username", ""), payload.get("access_code", ""))
            if token:
                accounts.logout(token)
            return handler.respond(200, {"user": account}, {"Set-Cookie": session_cookie(new_token)})
        if method == "POST" and path == "/api/account/send-code":
            return handler.respond(200, accounts.request_code(payload.get("email", "")))
        if method == "POST" and path == "/api/account/verify":
            new_token, account = accounts.verify_code(payload.get("email", ""), payload.get("code", ""))
            if token:
                accounts.logout(token)
            return handler.respond(200, {"user": account}, {"Set-Cookie": session_cookie(new_token)})
        if method == "POST" and path == "/api/account/logout":
            if token:
                accounts.logout(token)
            return handler.respond(200, {"ok": True}, {"Set-Cookie": session_cookie()})
        if method == "POST" and path == "/api/account/institution-requests":
            return handler.respond(201, accounts.request_institution(payload))
        if method == "POST" and not user:
            return handler.respond(401, {"error": "Sign in with your test account to continue." if auth_mode() == "test" else "Sign in with a verified institutional email to continue."})
        if method == "POST" and path == "/api/account/profile":
            return handler.respond(200, {"user": accounts.update_profile(viewer, payload)})

        def option(name, default=""):
            return query.get(name, [default])[0]

        def active():
            value = payload.get("active")
            if type(value) is not bool:
                raise ValueError("active must be a boolean")
            return value

        if parts[:3] == ["api", "community", "posts"]:
            if len(parts) == 3:
                if method == "GET":
                    following = option("following") in {"1", "true"}
                    if following and not viewer:
                        return handler.respond(401, {"error": "Sign in to see your following feed."})
                    result = community.list_posts(viewer_id=viewer, category=option("category") or None,
                        query=option("q"), following=following, problem_id=option("problem_id") or None,
                        limit=int(option("limit", "30")), offset=int(option("offset", "0")))
                    return handler.respond(200, result)
                if method == "POST":
                    return handler.respond(201, community.create_post(viewer, payload))
            if len(parts) == 4 and method == "GET":
                return handler.respond(200, community.get_post(parts[3], viewer_id=viewer))
            if len(parts) == 5 and method == "POST":
                identifier, action = parts[3:]
                if action == "replies":
                    return handler.respond(201, community.reply(viewer, identifier, payload))
                if action in {"like", "repost"}:
                    return handler.respond(200, community.set_reaction(viewer, identifier, action, active()))
                if action == "delete":
                    return handler.respond(200, community.delete_post(viewer, identifier))
        if len(parts) == 5 and parts[:3] == ["api", "community", "replies"] and parts[4] == "delete" and method == "POST":
            return handler.respond(200, community.delete_reply(viewer, parts[3]))
        if len(parts) == 4 and parts[:3] == ["api", "community", "follow"] and method == "POST":
            return handler.respond(200, community.follow(viewer, parts[3], active()))
        if parts[:3] == ["api", "community", "profiles"] and method == "GET":
            if len(parts) == 4:
                return handler.respond(200, community.profile(parts[3], viewer_id=viewer))
            if len(parts) == 5 and parts[4] == "practice":
                return handler.respond(200, community.practice_progress(parts[3], viewer_id=viewer,
                    year=option("year") or None, timezone_name=option("timezone", "UTC")))
            if len(parts) == 5 and parts[4] == "activity":
                return handler.respond(200, community.activity(parts[3], option("tab", "posts"), viewer_id=viewer,
                    limit=int(option("limit", "30")), offset=int(option("offset", "0"))))
            if len(parts) == 5 and parts[4] == "connections":
                return handler.respond(200, community.connections(parts[3], option("kind", "followers"), viewer_id=viewer,
                    limit=int(option("limit", "30")), offset=int(option("offset", "0"))))
        return handler.respond(404, {"error": "Unknown endpoint"})
    except AuthUnavailable as exc:
        return handler.respond(503, {"error": str(exc)})
    except RateLimited as exc:
        return handler.respond(429, {"error": str(exc)})
    except AuthenticationError as exc:
        return handler.respond(401, {"error": str(exc)})
    except PermissionError as exc:
        return handler.respond(403, {"error": str(exc)})
    except KeyError:
        return handler.respond(404, {"error": "The requested content or user does not exist."})
    except (AccountError, ValueError, TypeError) as exc:
        return handler.respond(400, {"error": str(exc)})
    except (sqlite3.Error, OSError):
        return handler.respond(503, {"error": "The account and community service is temporarily unavailable."})
