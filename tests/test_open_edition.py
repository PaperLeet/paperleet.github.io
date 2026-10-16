"""Standalone local execution and the public/private source boundary."""
from http.client import HTTPConnection
import importlib.util
import json
from pathlib import Path
import threading
import time
import uuid

import pytest

from paperleet import server
from paperleet.catalog import problem
from paperleet.config import validate_local_settings


def test_open_tree_has_no_online_implementation():
    repository = Path(__file__).resolve().parents[1]
    for path in ["paperleet/accounts.py", "paperleet/community.py", "paperleet/social_api.py",
                 "paperleet/hosting", "paperleet/test_accounts.py", "paperleet/backup.py",
                 "paperleet/practice.py", "paperleet/practice_progress.py", "institutions.yaml", "deploy"]:
        assert not (repository / path).exists(), path
    assert importlib.util.find_spec("paperleet_cloud") is None
    for path in (repository / "paperleet").rglob("*.py"):
        assert "paperleet_cloud" not in path.read_text(), path


def test_public_service_cannot_be_promoted_to_hosted(monkeypatch):
    monkeypatch.setenv("PAPERLEET_MODE", "hosted")
    with pytest.raises(ValueError, match="only runs locally"):
        validate_local_settings()


@pytest.fixture
def local_api(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPERLEET_MODE", "local")
    monkeypatch.setenv("PAPERLEET_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(server, "workspace_path", lambda entry: tmp_path / "submission.py")
    monkeypatch.setattr(server, "root", lambda: tmp_path)
    httpd = server.BoundedHTTPServer(("127.0.0.1", 0), server.JudgeHandler)
    worker = threading.Thread(target=httpd.serve_forever, daemon=True)
    worker.start()

    def request(path, body=None, client="a" * 32, origin="http://127.0.0.1:5173", *, raw=None):
        connection = HTTPConnection("127.0.0.1", httpd.server_port, timeout=20)
        encoded = raw if raw is not None else json.dumps(body) if body is not None else None
        connection.request("POST" if encoded is not None else "GET", "/api/" + path, encoded,
                           {"Content-Type": "application/json", "Origin": origin,
                            "X-PaperLeet-Client-ID": client})
        response = connection.getresponse()
        result = response.status, json.loads(response.read())
        connection.close()
        return result
    yield request
    httpd.shutdown()
    httpd.server_close()
    worker.join(timeout=2)


def test_deep_json_is_rejected_before_local_execution(local_api, monkeypatch):
    def forbidden_execution(*args, **kwargs):
        raise AssertionError("Malformed JSON must not execute learner code")

    monkeypatch.setattr(server, "run_problem", forbidden_execution)
    nested = '{"layers":' + '[' * 10000 + '0' + ']' * 10000 + '}'
    status, payload = local_api("test", raw=nested)
    assert status == 400
    assert "error" in payload
    assert local_api("health")[0] == 200


def test_local_judge_runs_without_accounts_and_rejects_online_routes(local_api):
    health = local_api("health")[1]
    assert health["mode"] == "local" and health["auth_required"] is False
    for path in ["account/me", "community/posts", "practice/workspace"]:
        assert local_api(path)[0] == 404
    assert local_api("account/send-code", {"email": "person@school.edu"})[0] == 404
    assert local_api("health", origin="https://attacker.example")[0] == 403
    entry = problem("rl/gae")
    status, report = local_api("test", {"problem": entry["id"], "code": (entry["directory"] / "solution.py").read_text()})
    assert status == 200 and report["passed"] is True, report


def test_local_queue_keeps_browser_jobs_private_without_login(local_api, monkeypatch):
    monkeypatch.setattr(server, "run_problem", lambda *args, **kwargs: {"passed": True})
    identifier = uuid.uuid4().hex
    assert local_api("judge/jobs", {"id": identifier, "problem": "rl/gae", "code": "trusted fixture"})[0] == 202
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        status, job = local_api("judge/jobs/" + identifier)
        if job["status"] in ("succeeded", "failed"):
            break
        time.sleep(.01)
    assert status == 200 and job["status"] == "succeeded", job
    assert local_api("judge/jobs/" + identifier, client="b" * 32)[0] == 404
    assert local_api("judge/jobs", {"id": uuid.uuid4().hex, "problem": "rl/gae", "code": "x"}, client="invalid")[0] == 400
