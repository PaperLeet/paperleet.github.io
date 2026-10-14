"""Real HTTP/session/SQLite integration; only the remote mail provider is stubbed."""
import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from http.cookies import SimpleCookie

import pytest

from paperleet.accounts import AccountService, AuthenticationError, SupabaseEmailProvider
from paperleet.community import CommunityService
from paperleet.server import JudgeHandler
from paperleet import social_api


class MailProvider:
    available = True

    def __init__(self):
        self.pending = set()

    def request_code(self, email):
        self.pending.add(email)

    def verify_code(self, email, code):
        if email not in self.pending or code != "123456":
            raise AuthenticationError("Invalid or expired code")
        self.pending.remove(email)
        return {"id": "test-" + email, "email": email, "email_confirmed_at": "2026-10-13T00:00:00Z"}


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.delenv("PAPERLEET_ADMIN_EMAILS", raising=False)
    monkeypatch.delenv("PAPERLEET_SECURE_COOKIES", raising=False)
    institutions = tmp_path / "institutions.yaml"
    institutions.write_text("institutions:\n  - id: test\n    name: Test Institute\n    domains: [research.edu]\n")
    accounts = AccountService(tmp_path / "social.sqlite3", institutions, MailProvider())
    community = CommunityService(accounts.db_path, accounts.institutions)
    monkeypatch.setattr(social_api, "services", lambda: (accounts, community))
    social_api._requests.clear()
    server = ThreadingHTTPServer(("127.0.0.1", 0), JudgeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def request(path, body=None, cookie="", origin=None, host=None):
        conn = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        headers = {"Content-Type": "application/json"}
        if cookie:
            headers["Cookie"] = cookie
        if origin:
            headers["Origin"] = origin
        if host:
            headers["Host"] = host
        conn.request("GET" if body is None else "POST", "/api/" + path,
                     json.dumps(body) if body is not None else None, headers)
        response = conn.getresponse()
        result = response.status, json.loads(response.read()), dict(response.getheaders())
        conn.close()
        return result

    request.accounts = accounts
    request.community = community
    yield request
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def login(api, name):
    email = name + "@research.edu"
    assert api("account/send-code", {"email": email})[0] == 200
    status, data, headers = api("account/verify", {"email": email, "code": "123456"})
    assert status == 200
    cookie = SimpleCookie(headers["Set-Cookie"])
    value = cookie[social_api.COOKIE]
    assert value["httponly"] and value["samesite"] == "Lax" and value["path"] == "/api"
    assert int(value["max-age"]) > 0
    return data["user"], social_api.COOKIE + "=" + value.value


def test_community_mutations_reject_a_changed_account_before_writing(api):
    alice, alice_cookie = login(api, "original-author")
    bob, bob_cookie = login(api, "other-author")
    payload = {"title": "Original draft", "body": "Written under Alice's identity", "expected_user_id": alice["id"]}
    status, error, _ = api("community/posts", payload, bob_cookie)
    assert status == 409 and error["code"] == "account_changed"
    assert api.community.list_posts()["total"] == 0
    status, post, _ = api("community/posts", payload, alice_cookie)
    assert status == 201 and post["author"]["id"] == alice["id"]
    for action, body in (("replies", {"body": "Private draft reply"}), ("like", {"active": True}),
                         ("repost", {"active": True}), ("delete", {})):
        status, error, _ = api(f"community/posts/{post['id']}/{action}", {**body, "expected_user_id": alice["id"]}, bob_cookie)
        assert status == 409 and error["code"] == "account_changed"
    detail = api.community.get_post(post["id"])
    assert detail["counts"] == {"likes": 0, "reposts": 0, "replies": 0}
    # Existing clients which omit the optional concurrency field stay compatible.
    assert api("community/posts", {"title": "Legacy post", "body": "No owner assertion"}, bob_cookie)[0] == 201
    assert api("community/posts", {**payload, "expected_user_id": bob["id"]}, bob_cookie)[0] == 201


def test_module_threads_http_session_filtering_and_persistence(api):
    _, cookie = login(api, "module-author")
    payload = {"title": "Attention mask question", "body": "How should fully masked rows behave?",
               "category": "questions", "problem_id": "architecture/attention"}
    assert api("community/posts", payload)[0] == 401
    status, post, _ = api("community/posts", payload, cookie)
    assert status == 201 and post["problem"]["number"] == 22
    assert api("community/posts", {**payload, "problem_id": "22"}, cookie)[0] == 400
    assert api("community/posts", {**payload, "problem_id": "finetuning/lora"}, cookie)[0] == 201
    status, page, _ = api("community/posts?problem_id=architecture%2Fattention&category=questions&q=mask&limit=1")
    assert status == 200 and page["total"] == 1
    assert page["items"][0]["id"] == post["id"]
    assert page["items"][0]["viewer"]["can_delete"] is False
    assert api("community/posts?problem_id=architecture%2Fattention&category=papers")[1]["total"] == 0
    assert api("community/posts?problem_id=..%2Fattention")[0] == 400
    assert api("community/posts?problem_id=architecture%2Fattention&following=true")[0] == 401
    assert api("community/posts/" + post["id"])[1]["problem"] == post["problem"]
    restarted = CommunityService(api.community.db_path, api.accounts.institutions)
    assert restarted.list_posts(problem_id="architecture/attention")["items"][0]["id"] == post["id"]


def test_anonymous_reads_and_verified_mailbox_write_gate(api):
    status, data, _ = api("account/me")
    assert status == 200 and data["user"] is None
    assert api("community/posts")[1]["items"] == []
    assert api("community/posts", {"title": "Test", "body": "Body", "category": "papers"})[0] == 401
    assert api("community/posts?following=true")[0] == 401
    assert api("account/send-code", {"email": "test@gmail.com"})[0] == 400
    assert api("account/verify", {"email": "test@research.edu", "code": "123456"})[0] == 401
    user, cookie = login(api, "alice")
    assert api("account/me", cookie=cookie)[1]["user"]["id"] == user["id"]
    assert api("account/verify", {"email": "alice@research.edu", "code": "123456"})[0] == 401


def test_public_identity_activities_and_follows_with_two_real_sessions(api):
    alice, alice_cookie = login(api, "alice")
    bob, bob_cookie = login(api, "bob")
    status, profile, _ = api("account/profile", {"display_name": "Alice", "username": "alice", "country": "US",
                                                "show_institution": False, "show_country": False}, alice_cookie)
    assert status == 200 and profile["user"]["email"] == "alice@research.edu"
    status, post, _ = api("community/posts", {"title": "DDPM implementation", "body": "Which variance?", "category": "papers",
                                              "user_id": bob["id"], "author": bob}, alice_cookie)
    assert status == 201 and post["author"]["id"] == alice["id"]
    path = "community/posts/" + post["id"]
    for kind in ("like", "repost"):
        assert api(path + "/" + kind, {"active": True}, bob_cookie)[1]["counts"][kind + "s"] == 1
        assert api(path + "/" + kind, {"active": True}, bob_cookie)[1]["counts"][kind + "s"] == 1
    reply = api(path + "/replies", {"body": "Use the posterior variance."}, bob_cookie)
    assert reply[0] == 201
    assert api(path)[1]["counts"]["replies"] == 1
    follow = api("community/follow/alice", {"active": True}, bob_cookie)
    assert follow[0] == 200 and follow[1]["counts"]["followers"] == 1
    assert api("community/posts?following=true", cookie=bob_cookie)[1]["items"][0]["id"] == post["id"]
    assert api("community/profiles/alice/connections?kind=followers")[1]["items"][0]["id"] == bob["id"]
    page = api("community/profiles/alice/connections?kind=followers&limit=1&offset=1")[1]
    assert page["total"] == 1 and page["items"] == []
    for tab in ("likes", "reposts", "replies"):
        rows = api(f"community/profiles/{bob['username']}/activity?tab={tab}")[1]
        assert rows["total"] == 1 and rows["items"][0]["post"]["id"] == post["id"]
    for url in ("community/posts", path, "community/profiles/alice", "community/profiles/alice/activity"):
        payload = json.dumps(api(url)[1])
        assert "alice@research.edu" not in payload and "provider_id" not in payload
        assert '"institution"' not in payload and '"country"' not in payload
        assert "Alice" in payload and "avatar_color" in payload
    api("account/profile", {"show_institution": True, "show_country": True}, alice_cookie)
    public = api("community/profiles/alice")[1]
    assert public["institution"]["id"] == "test" and public["country"] == "US"
    api("account/profile", {"show_institution": False, "show_country": False}, alice_cookie)
    assert "institution" not in api(path)[1]["author"]
    assert "country" not in api("community/profiles/alice", cookie=alice_cookie)[1]
    assert api("community/follow/alice", {"active": True}, alice_cookie)[0] == 400
    assert api("community/follow/alice", {"active": False}, bob_cookie)[1]["counts"]["followers"] == 0


def test_ownership_role_spoof_and_removal(api):
    alice, cookie_a = login(api, "alice")
    _, cookie_b = login(api, "bob")
    assert api("account/profile", {"role": "admin"}, cookie_a)[0] == 400
    assert api("account/profile", {"show_avatar": False}, cookie_a)[0] == 400
    post = api("community/posts", {"title": "Title", "body": "Body", "category": "questions"}, cookie_a)[1]
    path = "community/posts/" + post["id"]
    assert api(path + "/delete", {"user_id": alice["id"]}, cookie_b)[0] == 403
    assert api(path + "/like", {"active": "true"}, cookie_b)[0] == 400
    assert api(path + "/delete", {}, cookie_a)[0] == 200
    assert api(path)[0] == 404


def test_cross_site_origin_rejected_and_logout_revokes_session(api):
    _, cookie = login(api, "alice")
    assert api("account/profile", {"display_name": "Bad"}, cookie, origin="https://evil.example")[0] == 403
    assert api("account/me", cookie=cookie, host="evil.example")[0] == 403
    status, _, headers = api("account/logout", {}, cookie)
    assert status == 200 and "Max-Age=0" in headers["Set-Cookie"]
    assert api("account/me", cookie=cookie)[1]["user"] is None
    assert api("community/posts", {"title": "Title", "body": "Body", "category": "papers"}, cookie)[0] == 401


def test_setup_required_errors_and_validation_use_json(api):
    api.accounts.provider = SupabaseEmailProvider(url="", key="")
    assert api("account/me")[1]["capabilities"]["auth_available"] is False
    assert api("account/send-code", {"email": "student@research.edu"})[0] == 503
    assert api("community/posts?limit=bad")[0] == 400
    assert api("community/posts/missing")[0] == 404


def test_secure_cookie_option(monkeypatch):
    monkeypatch.setenv("PAPERLEET_SECURE_COOKIES", "1")
    assert "Secure" in social_api.session_cookie("test")
    assert "Secure" in social_api.session_cookie()


def test_local_problem_statistics_are_explicitly_unavailable(api, monkeypatch):
    monkeypatch.setenv("PAPERLEET_MODE", "local")
    assert api("practice/stats")[0:2] == (200, {"available": False, "stats": {}})
    assert api("practice/stats?problem=rl%2Fgae")[1] == {"available": False, "stats": {}}


def test_profile_practice_route_privacy_timezone_and_validation(api):
    alice, owner = login(api, "alice")
    _, other = login(api, "bob")
    path = f"community/profiles/{alice['username']}/practice"
    status, data, _ = api(path + "?year=2024&timezone=America%2FNew_York")
    assert status == 200 and data["visible"] is True
    assert data["year"] == 2024 and data["timezone"] == "America/New_York"
    assert len(data["calendar"]["days"]) == 366
    assert data["total_submissions"] == 0
    assert api(path + "?year=bad")[0] == 400
    assert api(path + "?timezone=not-a-timezone")[0] == 400
    assert api("community/profiles/missing/practice")[0] == 404
    assert api("account/profile", {"show_progress": False}, owner)[0] == 200
    assert api(path)[0:2] == (200, {"visible": False})
    assert api(path, cookie=other)[1] == {"visible": False}
    assert api(path, cookie=owner)[1]["visible"] is True
