"""Standalone loopback-only developer bridge; no accounts or online services."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import re
from urllib.parse import urlparse, urlsplit

from paperleet.catalog import problem, root, workspace_path
from paperleet.judge.runner import normalize_layers, run_problem
from paperleet.submissions import Conflict, store
from paperleet.submissions.model import ModelUnavailable
from paperleet.config import validate_local_settings
from paperleet.judge.jobs_api import is_job_path, dispatch as dispatch_jobs


_judge_lock = threading.Lock()


class JudgeHandler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(15)

    def log_message(self, format, *args):
        pass

    def respond(self, status, payload, headers=None):
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(body)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def valid_origin(self):
        host = urlparse("http://" + self.headers.get("Host", "")).hostname
        supplied_origin = self.headers.get("Origin")
        return host in ("127.0.0.1", "localhost", "::1") and (not supplied_origin or urlparse(supplied_origin).hostname in ("127.0.0.1", "localhost", "::1"))

    def do_GET(self):
        if not self.valid_origin():
            return self.respond(403, {"error": "This request origin is not allowed"})
        if urlsplit(self.path).path == "/api/practice/stats":
            return self.respond(200, {"available": False, "stats": {}})
        if self.path == "/api/health":
            result = {"ok": True, "mode": "local", "version": "0.1.0",
                      "auth_required": False, "judge": {"ready": True}}
            return self.respond(200, result)
        if is_job_path(self.path):
            return dispatch_jobs(self, "GET")
        try:
            if self.path == "/api/submissions/capabilities":
                return self.respond(200, store().capabilities())
            if self.path == "/api/submissions":
                return self.respond(200, {"submissions": store().list()})
            match = re.fullmatch(r"/api/submissions/([0-9a-f]{32})", self.path)
            if match:
                return self.respond(200, store().get(match[1]))
        except KeyError:
            return self.respond(404, {"error": "Unknown submission"})
        except ValueError as exc:
            return self.respond(400, {"error": str(exc)})
        return self.respond(404, {"error": "Unknown endpoint"})

    def do_POST(self):
        if not self.valid_origin():
            return self.respond(403, {"error": "This request origin is not allowed"})
        submission_action = re.fullmatch(r"/api/submissions/([0-9a-f]{32})/(generate|validate|review)", self.path)
        if self.path not in ("/api/test", "/api/submissions", "/api/practice/draft") and not submission_action and not is_job_path(self.path):
            return self.respond(404, {"error": "Unknown endpoint"})
        if self.headers.get_content_type() != "application/json":
            return self.respond(415, {"error": "Expected application/json"})
        if self.headers.get("Transfer-Encoding") or len(self.headers.get_all("Content-Length", [])) != 1:
            return self.respond(400, {"error": "Expected one Content-Length header"})
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 256 * 1024:
                return self.respond(413, {"error": "Request must be between 1 byte and 256 KB"})
            try:
                data = json.loads(self.rfile.read(size))
            except RecursionError:
                raise ValueError("Request JSON is too deeply nested") from None
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object")
            if is_job_path(self.path):
                return dispatch_jobs(self, "POST", data)
            if self.path == "/api/practice/draft":
                return self.respond(404, {"error": "Cloud practice belongs to the separate online application"})
            if self.path == "/api/submissions":
                return self.respond(201, store().create(data))
            if submission_action:
                identifier, action = submission_action.groups()
                if action == "review":
                    return self.respond(200, store().review(identifier, data))
                return self.respond(202, store().start(identifier, action))
            entry = problem(data["problem"])
            code = data["code"]
            if not isinstance(code, str) or not code.strip():
                raise ValueError("Python code cannot be empty")
            layers = normalize_layers(data.get("layers"))
        except ModelUnavailable as exc:
            return self.respond(503, {"error": str(exc)})
        except Conflict as exc:
            return self.respond(409, {"error": str(exc)})
        except (ValueError, KeyError, TypeError) as exc:
            return self.respond(400, {"error": str(exc)})
        if not _judge_lock.acquire(blocking=False):
            return self.respond(409, {"error": "A judge run is already in progress. Please wait."})
        try:
            path = workspace_path(entry)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(code, encoding="utf-8")
            destination = root() / "reports" / (entry["id"].replace("/", "-") + ".json")
            report = run_problem(entry["id"], path, layers=layers, report_path=destination)
            self.respond(200, report)
        except Exception as exc:
            self.respond(500, {"error": str(exc)})
        finally:
            _judge_lock.release()


class BoundedHTTPServer(ThreadingHTTPServer):
    """Limit concurrent connections; the bridge only binds loopback."""
    daemon_threads = True

    def __init__(self, *args, **kwargs):
        self.slots = threading.BoundedSemaphore(32)
        super().__init__(*args, **kwargs)

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            request.close()
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()

    def server_close(self):
        if hasattr(self, "judge_queue"):
            self.judge_queue.close()
        super().server_close()


def serve(port=8765):
    validate_local_settings()
    address = "127.0.0.1"
    server = BoundedHTTPServer((address, port), JudgeHandler)
    from paperleet.judge.jobs_api import start_queue
    start_queue(server)
    print(f"PaperLeet local judge: http://{address}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
