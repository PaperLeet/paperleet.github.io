"""Hosted HTTP boundaries with real sessions/storage and mocked mail/judge I/O."""
import hashlib
from http.client import HTTPConnection
from http.cookies import SimpleCookie
import json
import sqlite3
import threading
import time
import uuid

import pytest

from paperleet import server as api_server, social_api, submissions
from paperleet.accounts import AuthenticationError, SupabaseEmailProvider
from paperleet.catalog import problem
from paperleet.config import validate_hosted_settings
from paperleet.hosting import api as hosted_api, judge_client


PUBLIC_ORIGIN = "https://paperleet.test"


@pytest.fixture
def hosted_api_server(tmp_path, monkeypatch):
    settings = {
        "PAPERLEET_REGISTRATION_POLICY": "invite_only", "PAPERLEET_BLOCKED_EMAIL_DOMAINS": "",
        "PAPERLEET_MODE": "hosted", "PAPERLEET_AUTH_MODE": "institutional", "PAPERLEET_PUBLIC_URL": PUBLIC_ORIGIN,
        "PAPERLEET_DATA_DIR": str(tmp_path / "persistent-data"),
        "PAPERLEET_INVITE_EMAILS": "alice@mit.edu,bob@mit.edu,admin@mit.edu,invited@gmail.com",
        "PAPERLEET_ADMIN_EMAILS": "admin@mit.edu",
        "SUPABASE_URL": "https://project.supabase.co", "SUPABASE_PUBLISHABLE_KEY": "test-publishable-key",
        "PAPERLEET_JUDGE_URL": "http://isolated-judge:8766", "PAPERLEET_JUDGE_TOKEN": "test-shared-token-" * 3,
    }
    for key, value in settings.items():
        monkeypatch.setenv(key, value)
    for key in ("SUPABASE_ANON_KEY", "PAPERLEET_SECURE_COOKIES", "OPENAI_API_KEY", "PAPERLEET_MODEL", "PAPERLEET_TRUST_PROXY"):
        monkeypatch.delenv(key, raising=False)
    validate_hosted_settings()
    pending_codes = set()

    def send_code(provider, email):
        pending_codes.add(email)

    def verify_code(provider, email, code):
        if email not in pending_codes or code != "123456":
            raise AuthenticationError("Invalid or consumed verification code")
        pending_codes.remove(email)
        return {"id": "remote-" + email, "email": email, "email_confirmed_at": "2026-10-13T00:00:00Z"}

    monkeypatch.setattr(SupabaseEmailProvider, "request_code", send_code)
    monkeypatch.setattr(SupabaseEmailProvider, "verify_code", verify_code)
    remote_calls, hosted_calls = [], []
    remote_state = {"ready": True, "unavailable": False}

    def remote_judge(endpoint, payload=None, timeout=5):
        judge_client._configuration()
        remote_calls.append((endpoint, payload))
        if remote_state["unavailable"]:
            raise judge_client.JudgeUnavailable("The isolated judge is unavailable")
        if endpoint == "/health":
            return {"ready": remote_state["ready"]}
        assert endpoint == "/run"
        if payload["code"] == "runtime_error":
            return {"problem": payload["problem"], "layers": payload["layers"], "tests": [],
                    "source_sha256": hashlib.sha256(payload["code"].encode()).hexdigest(),
                    "created_at": "2026-10-13T00:00:00Z", "passed": False,
                    "error": "The implementation raised a runtime error before completing any tests"}
        entry = problem(payload["problem"])
        tests = []
        for (name, layer), count in judge_client.expected_tests(entry, payload["layers"]).items():
            for case in range(count):
                passing = "broken" not in payload["code"]
                tests.append({"id": name, "nodeid": f"tests/test_module.py::{name}[{case}]", "layer": layer,
                              "passed": passing, "outcome": "passed" if passing else "failed", "duration_s": 0.001,
                              "artifacts": {}, "message": "", "why": "A trusted test checks this property."})
        return {"problem": payload["problem"], "layers": payload["layers"], "tests": tests,
                "source_sha256": hashlib.sha256(payload["code"].encode()).hexdigest(),
                "created_at": "2026-10-13T00:00:00Z", "passed": all(row["passed"] for row in tests)}

    real_hosted_run = judge_client.run_hosted_problem

    def hosted_run(entry, code, layers=None):
        hosted_calls.append(entry["id"])
        return real_hosted_run(entry, code, layers)

    def forbidden_host_run(*args, **kwargs):
        raise AssertionError("Hosted requests must never execute Python through the local runner")

    monkeypatch.setattr(judge_client, "_request", remote_judge)
    monkeypatch.setattr(judge_client, "run_hosted_problem", hosted_run)
    monkeypatch.setattr(api_server, "run_problem", forbidden_host_run)
    forbidden_workspace = tmp_path / "forbidden-host-workspace.py"
    monkeypatch.setattr(api_server, "workspace_path", lambda entry: forbidden_workspace)
    social_api._requests.clear()
    hosted_api._active_users.clear()
    server = api_server.BoundedHTTPServer(("127.0.0.1", 0), api_server.JudgeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def request(path, body=None, cookie="", origin=PUBLIC_ORIGIN, host="paperleet.test"):
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        headers = {"Content-Type": "application/json", "Host": host}
        if origin is not None:
            headers["Origin"] = origin
        if cookie:
            headers["Cookie"] = cookie
        connection.request("GET" if body is None else "POST", "/api/" + path,
                           json.dumps(body) if body is not None else None, headers)
        response = connection.getresponse()
        result = response.status, json.loads(response.read()), dict(response.getheaders())
        connection.close()
        return result

    request.remote_calls = remote_calls
    request.hosted_calls = hosted_calls
    request.remote_state = remote_state
    request.settings = settings
    request.forbidden_workspace = forbidden_workspace
    yield request
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)
    social_api._services.pop(tmp_path / "persistent-data" / "community.sqlite3", None)
    for key in list(submissions._stores):
        if key[1] == tmp_path / "persistent-data":
            submissions._stores.pop(key)
    assert not forbidden_workspace.exists(), "Hosted request wrote to the local execution workspace"


def test_open_company_registration_real_http_private_workspace_and_queued_judging(hosted_api_server, monkeypatch):
    api = hosted_api_server
    monkeypatch.setenv("PAPERLEET_REGISTRATION_POLICY", "work_email")
    monkeypatch.delenv("PAPERLEET_INVITE_EMAILS")
    validate_hosted_settings()
    caps = api("account/me")[1]["capabilities"]
    assert caps["work_email_registration"] and not caps["invite_only"]
    assert api("account/send-code", {"email": "person@gmail.com"})[0] == 400
    email = "developer@company.example"
    assert api("account/send-code", {"email": email})[0] == 200
    assert api("account/verify", {"email": email, "code": "000000"})[0] == 401
    status, payload, headers = api("account/verify", {"email": email, "code": "123456"})
    assert status == 200, payload
    user = payload["user"]
    cookie = headers["Set-Cookie"].split(";", 1)[0]
    assert "Secure" in headers["Set-Cookie"] and "HttpOnly" in headers["Set-Cookie"]
    assert user["email_verified"] and not user["institution_verified"] and user["institution"] is None
    assert api("account/verify", {"email": email, "code": "123456"})[0] == 401
    assert api("practice/draft", {"problem": "rl/gae", "code": "private draft", "expected_user_id": user["id"]}, cookie)[0] == 200
    assert api("practice/workspace", cookie=cookie)[1]["workspace"]["rl/gae"]["code"] == "private draft"
    assert api("practice/workspace")[0] == 401
    identifier = uuid.uuid4().hex
    assert api("judge/jobs", {"id": identifier, "problem": "rl/gae", "code": "correct", "expected_user_id": user["id"]}, cookie)[0] == 202
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline:
        result = api(f"judge/jobs/{identifier}", cookie=cookie)[1]
        if result["status"] in ("succeeded", "failed"):
            break
        time.sleep(.01)
    assert result["status"] == "succeeded", result
    assert result["report"]["passed"]
    monkeypatch.setenv("PAPERLEET_BLOCKED_EMAIL_DOMAINS", "company.example")
    assert api("account/me", cookie=cookie)[1]["user"] is None
    assert api("practice/workspace", cookie=cookie)[0] == 401


def test_open_registration_still_requires_email_provider_settings(hosted_api_server, monkeypatch):
    monkeypatch.setenv("PAPERLEET_REGISTRATION_POLICY", "work_email")
    monkeypatch.delenv("PAPERLEET_INVITE_EMAILS")
    validate_hosted_settings()
    monkeypatch.delenv("SUPABASE_PUBLISHABLE_KEY")
    with pytest.raises(ValueError, match="SUPABASE_PUBLISHABLE_KEY"):
        validate_hosted_settings()


def sign_in(api, name):
    email = name + "@mit.edu"
    assert api("account/send-code", {"email": email})[0] == 200
    status, payload, headers = api("account/verify", {"email": email, "code": "123456"})
    assert status == 200, payload
    cookie = SimpleCookie(headers["Set-Cookie"])[social_api.COOKIE]
    assert cookie["secure"] and cookie["httponly"] and cookie["samesite"] == "Lax"
    assert cookie["path"] == "/api"
    return payload["user"], social_api.COOKIE + "=" + cookie.value


def test_queued_judging_is_private_cancellable_and_saves_once(hosted_api_server, monkeypatch):
    api = hosted_api_server
    alice, alice_cookie = sign_in(api, "alice")
    bob, bob_cookie = sign_in(api, "bob")
    entered, release = threading.Event(), threading.Event()
    original = judge_client.run_hosted_problem

    def waiting_judge(entry, code, layers=None):
        entered.set()
        assert release.wait(5)
        return original(entry, code, layers)

    monkeypatch.setattr(judge_client, "run_hosted_problem", waiting_judge)
    alice_id, bob_id = uuid.uuid4().hex, uuid.uuid4().hex
    alice_body = {"id": alice_id, "problem": "rl/gae", "code": "correct", "expected_user_id": alice["id"]}
    try:
        assert api("judge/jobs", alice_body)[0] == 401
        assert api("judge/jobs", alice_body, alice_cookie, origin=None)[0] == 403
        assert api("judge/jobs", {**alice_body, "expected_user_id": bob["id"]}, alice_cookie)[0] == 409
        assert api("judge/jobs", alice_body, alice_cookie)[0] == 202
        assert entered.wait(2)
        status, first, _ = api(f"judge/jobs/{alice_id}", cookie=alice_cookie)
        assert status == 200 and first["status"] == "running"
        assert not {"code", "owner", "payload"} & first.keys()
        assert api(f"judge/jobs/{alice_id}", cookie=bob_cookie)[0] == 404
        assert api(f"judge/jobs/{alice_id}/cancel", {"expected_user_id": bob["id"]}, bob_cookie)[0] == 404
        status, second, _ = api("judge/jobs", {"id": bob_id, "problem": "rl/dpo", "code": "broken", "expected_user_id": bob["id"]}, bob_cookie)
        assert status == 202 and second["status"] == "queued" and second["ahead"] == 1
        assert api(f"judge/jobs/{bob_id}/cancel", {"expected_user_id": bob["id"]}, bob_cookie)[1]["status"] == "cancelled"
        release.set()
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            result = api(f"judge/jobs/{alice_id}", cookie=alice_cookie)[1]
            if result["status"] == "succeeded":
                break
            time.sleep(.01)
        assert result["status"] == "succeeded" and result["report"]["run_id"]
        assert api("judge/jobs", alice_body, alice_cookie)[1]["id"] == alice_id
        for _ in range(2):
            assert api(f"judge/jobs/{alice_id}", cookie=alice_cookie)[1]["report"]["passed"]
        assert len(api("practice/runs", cookie=alice_cookie)[1]["runs"]) == 1
        assert api("practice/runs", cookie=bob_cookie)[1]["runs"] == []
        assert api.hosted_calls == ["rl/gae"]
    finally:
        release.set()


def test_queued_job_rechecks_revoked_invitation_before_execution(hosted_api_server, monkeypatch):
    api = hosted_api_server
    alice, alice_cookie = sign_in(api, "alice")
    bob, bob_cookie = sign_in(api, "bob")
    entered, release = threading.Event(), threading.Event()
    original = judge_client.run_hosted_problem

    def waiting_judge(entry, code, layers=None):
        entered.set()
        assert release.wait(5)
        return original(entry, code, layers)

    monkeypatch.setattr(judge_client, "run_hosted_problem", waiting_judge)
    alice_id, bob_id = uuid.uuid4().hex, uuid.uuid4().hex
    try:
        assert api("judge/jobs", {"id": alice_id, "problem": "rl/gae", "code": "correct", "expected_user_id": alice["id"]}, alice_cookie)[0] == 202
        assert entered.wait(2)
        assert api("judge/jobs", {"id": bob_id, "problem": "rl/dpo", "code": "correct", "expected_user_id": bob["id"]}, bob_cookie)[0] == 202
        monkeypatch.setenv("PAPERLEET_INVITE_EMAILS", "alice@mit.edu")
        release.set()
        deadline = time.monotonic() + 4
        status = None
        while time.monotonic() < deadline:
            if api(f"judge/jobs/{alice_id}", cookie=alice_cookie)[1]["status"] == "succeeded":
                # Polling Bob is forbidden after revocation, so inspect storage
                # through the authenticated API only after restoring the invite.
                with sqlite3.connect(api.settings["PAPERLEET_DATA_DIR"] + "/community.sqlite3") as db:
                    status = db.execute("SELECT state FROM judge_jobs WHERE id=?", (bob_id,)).fetchone()[0]
                if status == "failed":
                    break
            time.sleep(.01)
        assert status == "failed"
        assert api(f"judge/jobs/{bob_id}", cookie=bob_cookie)[0] == 401
        monkeypatch.setenv("PAPERLEET_INVITE_EMAILS", "alice@mit.edu,bob@mit.edu")
        assert api(f"judge/jobs/{bob_id}", cookie=bob_cookie)[1]["status"] == "failed"
        assert api("practice/runs", cookie=bob_cookie)[1]["runs"] == []
        assert api.hosted_calls == ["rl/gae"]
    finally:
        release.set()


@pytest.mark.parametrize("missing", ["PAPERLEET_PUBLIC_URL", "PAPERLEET_DATA_DIR", "PAPERLEET_INVITE_EMAILS", "SUPABASE_URL", "SUPABASE_PUBLISHABLE_KEY", "PAPERLEET_JUDGE_URL", "PAPERLEET_JUDGE_TOKEN"])
def test_hosted_startup_requires_explicit_configuration(hosted_api_server, monkeypatch, missing):
    monkeypatch.delenv(missing)
    with pytest.raises(ValueError, match="Hosted mode requires"):
        validate_hosted_settings()


@pytest.mark.parametrize("key,value", [("PAPERLEET_MODE", "public"), ("PAPERLEET_PUBLIC_URL", "http://paperleet.test"),
                                      ("PAPERLEET_DATA_DIR", "relative-path"), ("PAPERLEET_JUDGE_TOKEN", "short")])
def test_hosted_startup_rejects_invalid_settings(hosted_api_server, monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError):
        validate_hosted_settings()


def test_host_and_origin_are_enforced_before_any_hosted_action(hosted_api_server):
    api = hosted_api_server
    assert api("health", origin=None)[0] == 200
    assert api("health", host="localhost")[0] == 403
    assert api("health", host="paperleet.test.attacker.example")[0] == 403
    assert api("health", origin="https://attacker.example")[0] == 403
    for origin in (None, "https://attacker.example", "http://paperleet.test", "null"):
        assert api("account/send-code", {"email": "alice@mit.edu"}, origin=origin)[0] == 403
    assert api("account/send-code", {"email": "alice@mit.edu"}, host="localhost")[0] == 403


def test_invitation_and_institution_gates_secure_cookie_and_revocation(hosted_api_server, monkeypatch):
    api = hosted_api_server
    assert api("account/me")[1]["capabilities"]["invite_only"] is True
    assert api("account/send-code", {"email": "uninvited@mit.edu"})[0] == 401
    assert api("account/verify", {"email": "uninvited@mit.edu", "code": "123456"})[0] == 401
    assert api("account/send-code", {"email": "invited@gmail.com"})[0] == 400
    user, cookie = sign_in(api, "alice")
    assert api("account/me", cookie=cookie)[1]["user"]["id"] == user["id"]
    assert api("account/verify", {"email": "alice@mit.edu", "code": "123456"})[0] == 401
    monkeypatch.setenv("PAPERLEET_INVITE_EMAILS", "bob@mit.edu")
    assert api("account/me", cookie=cookie)[1]["user"] is None
    assert api("practice/workspace", cookie=cookie)[0] == 401


def test_hosted_judging_uses_only_isolated_runner_and_saves_private_history(hosted_api_server):
    api = hosted_api_server
    assert api("test", {"problem": "1", "code": "correct"})[0] == 401
    assert api("practice/workspace")[0] == 401
    assert api("practice/runs")[0] == 401
    assert api.hosted_calls == [] and api.remote_calls == []
    alice, alice_cookie = sign_in(api, "alice")
    bob, bob_cookie = sign_in(api, "bob")
    status, partial, _ = api("test", {"problem": "1", "code": "partial", "layers": ["T0"], "user_id": bob["id"], "expected_user_id": alice["id"]}, alice_cookie)
    assert status == 200 and partial["passed"] is True and partial["run_id"]
    workspace = api("practice/workspace", cookie=alice_cookie)[1]["workspace"]
    assert api("practice/workspace", cookie=alice_cookie)[1]["user_id"] == alice["id"]
    assert workspace["rl/gae"]["code"] == "partial"
    assert "acceptedReport" not in workspace["rl/gae"]
    assert api("practice/workspace", cookie=bob_cookie)[1]["workspace"] == {}
    assert api("practice/runs", cookie=bob_cookie)[1]["runs"] == []
    assert api("test", {"problem": "rl/gae", "code": "correct", "expected_user_id": alice["id"]}, alice_cookie)[0] == 200
    assert api("test", {"problem": "rl/gae", "code": "broken", "expected_user_id": alice["id"]}, alice_cookie)[0] == 200
    workspace = api("practice/workspace", cookie=alice_cookie)[1]["workspace"]["rl/gae"]
    assert workspace["acceptedReport"]["passed"] is True
    assert workspace["report"]["passed"] is False
    history = api("practice/runs?limit=2&offset=0", cookie=alice_cookie)[1]["runs"]
    assert len(history) == 2 and [item["code"] for item in history] == ["broken", "correct"]
    assert api.hosted_calls == ["rl/gae"] * 3
    assert [endpoint for endpoint, _ in api.remote_calls] == ["/run"] * 3


def test_unavailable_isolated_judge_never_falls_back_or_creates_a_pass(hosted_api_server):
    api = hosted_api_server
    alice, cookie = sign_in(api, "alice")
    api.remote_state["unavailable"] = True
    status, result, _ = api("test", {"problem": "rl/gae", "code": "correct", "expected_user_id": alice["id"]}, cookie)
    assert status == 503 and "isolated judge" in result["error"]
    assert api("practice/runs", cookie=cookie)[1]["runs"] == []
    assert api("practice/workspace", cookie=cookie)[1]["workspace"] == {}


def test_draft_conflicts_and_actor_fields_cannot_write_another_account(hosted_api_server):
    api = hosted_api_server
    alice, alice_cookie = sign_in(api, "alice")
    bob, bob_cookie = sign_in(api, "bob")
    body = {"problem": "1", "code": "alice draft", "expected_updated": "", "user_id": bob["id"], "owner_id": bob["id"], "expected_user_id": alice["id"]}
    status, saved, _ = api("practice/draft", body, alice_cookie)
    assert status == 200
    assert api("practice/workspace", cookie=bob_cookie)[1]["workspace"] == {}
    assert api("practice/draft", {**body, "code": "stale write"}, alice_cookie)[0] == 409
    current = saved["draft"]["updated"]
    assert api("practice/draft", {**body, "code": "newer draft", "expected_updated": current}, alice_cookie)[0] == 200
    assert api("test", {"problem": "1", "code": "old running source", "expected_user_id": alice["id"]}, alice_cookie)[0] == 200
    workspace = api("practice/workspace", cookie=alice_cookie)[1]["workspace"]["rl/gae"]
    assert workspace["code"] == "newer draft"
    assert api("practice/runs?user_id=" + bob["id"], cookie=alice_cookie)[1]["runs"][0]["code"] == "old running source"


def test_stale_account_identity_cannot_save_or_run_after_account_switch(hosted_api_server):
    api = hosted_api_server
    alice, alice_cookie = sign_in(api, "alice")
    bob, _ = sign_in(api, "bob")
    for endpoint in ("practice/draft", "test"):
        payload = {"problem": "1", "code": "from another account"}
        assert api(endpoint, payload, alice_cookie)[0] == 409
        assert api(endpoint, {**payload, "expected_user_id": bob["id"]}, alice_cookie)[0] == 409
    assert api("practice/workspace", cookie=alice_cookie)[1] == {"user_id": alice["id"], "workspace": {}}
    assert api.hosted_calls == [] and api.remote_calls == []


def test_paper_proposal_ownership_and_explicit_offline_review_policy(hosted_api_server):
    api = hosted_api_server
    alice, alice_cookie = sign_in(api, "alice")
    bob, bob_cookie = sign_in(api, "bob")
    admin, admin_cookie = sign_in(api, "admin")
    assert admin["role"] == "admin"
    proposal = {"title": "GAE", "paper_url": "https://arxiv.org/abs/1506.02438", "focus": "Core recurrence", "owner_id": bob["id"]}
    assert api("submissions", proposal)[0] == 401
    status, alice_paper, _ = api("submissions", proposal, alice_cookie)
    assert status == 201 and alice_paper["owner_id"] == alice["id"]
    status, bob_paper, _ = api("submissions", {**proposal, "title": "Bob's proposal"}, bob_cookie)
    assert status == 201
    assert [item["id"] for item in api("submissions", cookie=alice_cookie)[1]["submissions"]] == [alice_paper["id"]]
    assert [item["id"] for item in api("submissions", cookie=bob_cookie)[1]["submissions"]] == [bob_paper["id"]]
    assert {item["id"] for item in api("submissions", cookie=admin_cookie)[1]["submissions"]} == {alice_paper["id"], bob_paper["id"]}
    assert api("submissions/" + bob_paper["id"], cookie=alice_cookie)[0] == 404
    for action in ("generate", "validate", "review"):
        path = "submissions/" + alice_paper["id"] + "/" + action
        assert api(path, {}, alice_cookie)[0] == 403
        status, response, _ = api(path, {}, admin_cookie)
        assert status == 503 and "managed offline" in response["error"]
    assert api("submissions/" + alice_paper["id"], cookie=alice_cookie)[1]["status"] == "submitted"
    capabilities = api("submissions/capabilities")[1]
    assert capabilities["can_submit"] is False and capabilities["can_review"] is False
    assert capabilities["generation"]["ready"] is False
    assert api("submissions/capabilities", cookie=admin_cookie)[1]["can_submit"] is True


def test_public_health_reports_hosted_sandbox_readiness_without_exposing_credentials(hosted_api_server, monkeypatch):
    api = hosted_api_server
    status, health, _ = api("health", origin=None)
    assert status == 200 and health["mode"] == "hosted" and health["auth_required"] is True
    assert health["judge"]["ready"] is True and health["judge"]["mode"] == "isolated"
    public = json.dumps(health)
    assert api.settings["PAPERLEET_JUDGE_TOKEN"] not in public
    assert api.settings["SUPABASE_PUBLISHABLE_KEY"] not in public
    assert api.settings["PAPERLEET_JUDGE_URL"] not in public
    api.remote_state["ready"] = False
    assert api("health")[1]["judge"]["ready"] is False
    monkeypatch.delenv("PAPERLEET_JUDGE_TOKEN")
    assert api("health")[1]["judge"]["ready"] is False


def test_public_problem_stats_return_real_zeroes_and_no_user_data(hosted_api_server):
    api = hosted_api_server
    status, data, _ = api("practice/stats", cookie="paperleet_session=invalid", origin=None)
    assert status == 200 and data["available"] is True
    assert "rl/gae" in data["stats"]
    expected_keys = {"attempted_users", "solved_users", "total_submissions",
                     "accepted_submissions", "failed_submissions", "acceptance_rate"}
    for stats in data["stats"].values():
        assert set(stats) == expected_keys
        assert stats["acceptance_rate"] is None
        assert all(value == 0 for key, value in stats.items() if key != "acceptance_rate")
    assert api.remote_calls == [] and api.hosted_calls == []
    assert api("practice/stats", origin="https://attacker.example")[0] == 403
    assert api("practice/stats", {"accepted_submissions": 100})[0] == 404


def test_public_problem_stats_refresh_after_full_runs_and_exclude_infrastructure_errors(hosted_api_server):
    api = hosted_api_server
    alice, alice_cookie = sign_in(api, "alice")
    bob, bob_cookie = sign_in(api, "bob")

    def run(user, cookie, code, layers=None):
        body = {"problem": "rl/gae", "code": code, "expected_user_id": user["id"]}
        if layers is not None:
            body["layers"] = layers
        return api("test", body, cookie)

    assert run(alice, alice_cookie, "private partial source", ["T0"])[0] == 200
    assert api("practice/stats")[1]["stats"]["rl/gae"]["total_submissions"] == 0
    assert run(alice, alice_cookie, "private correct source")[0] == 200
    assert run(alice, alice_cookie, "private repeated correct source")[0] == 200
    assert run(bob, bob_cookie, "private broken source")[0] == 200
    assert run(bob, bob_cookie, "runtime_error")[0] == 200
    public = api("practice/stats")[1]
    assert public["stats"]["rl/gae"] == {
        "attempted_users": 2, "solved_users": 1, "total_submissions": 4,
        "accepted_submissions": 2, "failed_submissions": 2, "acceptance_rate": 50.0,
    }
    encoded = json.dumps(public)
    for private_value in (alice["id"], bob["id"], alice["email"], bob["email"], "private correct source", "runtime_error"):
        assert private_value not in encoded
    api.remote_state["unavailable"] = True
    assert run(bob, bob_cookie, "no judge available")[0] == 503
    assert api("practice/stats")[1] == public
