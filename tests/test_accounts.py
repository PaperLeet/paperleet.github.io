import hashlib
import json
import sqlite3
import time
from urllib.error import HTTPError

import pytest

from paperleet.accounts import (
    AccountError, AccountService, AuthenticationError, AuthUnavailable,
    RateLimited, SupabaseEmailProvider, public_user,
)


class EmailProvider:
    available = True

    def __init__(self):
        self.sent = []
        self.response = None

    def request_code(self, email):
        self.sent.append(email)

    def verify_code(self, email, code):
        if self.response is not None:
            return self.response
        if code != "123456":
            raise AuthenticationError("Invalid code")
        return {"id": "provider-" + email, "email": email, "email_confirmed_at": "2026-01-01T00:00:00Z"}


@pytest.fixture
def accounts(tmp_path, monkeypatch):
    monkeypatch.delenv("PAPERLEET_REGISTRATION_POLICY", raising=False)
    monkeypatch.delenv("PAPERLEET_BLOCKED_EMAIL_DOMAINS", raising=False)
    monkeypatch.delenv("PAPERLEET_ADMIN_EMAILS", raising=False)
    monkeypatch.setenv("PAPERLEET_MODE", "local")
    monkeypatch.delenv("PAPERLEET_INVITE_EMAILS", raising=False)
    config = tmp_path / "institutions.yaml"
    config.write_text("""institutions:
  - id: institute
    name: 研究大学
    name_en: Research University
    domains: [research.edu]
""", encoding="utf-8")
    return AccountService(tmp_path / "community.sqlite3", config, provider=EmailProvider())


def sign_in(accounts, email="student@research.edu"):
    return accounts.verify_code(email, "123456")


@pytest.mark.parametrize("email", ["student@another.edu", "student@school.edu.cn", "student@dept.university.ac.uk", "engineer@company.example", "engineer@mail.company.example"])
def test_open_registration_verifies_unlisted_school_and_company_mailboxes(accounts, monkeypatch, email):
    monkeypatch.setenv("PAPERLEET_MODE", "hosted")
    monkeypatch.setenv("PAPERLEET_REGISTRATION_POLICY", "work_email")
    capabilities = accounts.capabilities()
    assert capabilities["auth_available"] and capabilities["work_email_registration"]
    assert not capabilities["invite_only"] and not capabilities["institution_verification_required"]
    accounts.request_code(email)
    with sqlite3.connect(accounts.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0
    token, user = sign_in(accounts, email)
    assert user["email_verified"] is True and user["institution_verified"] is False
    assert user["institution"] is None and user["role"] == "member"
    assert accounts.get_session(token)["id"] == user["id"]
    public = accounts.update_profile(user["id"], {"show_institution": True})
    assert public["institution"] is None
    assert not {"email", "email_verified", "provider_id", "institution"}.intersection(accounts.public_user(user["id"]))


@pytest.mark.parametrize("domain", ["GMAIL.COM", "qq.com", "foxmail.com", "outlook.com", "icloud.com", "proton.me", "163.com", "yahoo.co.jp", "mail.gmail.com", "mailinator.com", "duck.com"])
def test_open_registration_blocks_personal_email_before_any_provider_call(accounts, monkeypatch, domain):
    monkeypatch.setenv("PAPERLEET_REGISTRATION_POLICY", "work_email")
    def forbidden(*args):
        pytest.fail("Ineligible mailbox contacted the email provider")
    accounts.provider.request_code = accounts.provider.verify_code = forbidden
    for action in (accounts.request_code, lambda email: sign_in(accounts, email)):
        with pytest.raises(AccountError, match="Personal email"):
            action("person@" + domain)
    with sqlite3.connect(accounts.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0


def test_open_registration_keeps_reviewed_badges_and_blocks_policy_changes(accounts, monkeypatch):
    monkeypatch.setenv("PAPERLEET_REGISTRATION_POLICY", "work_email")
    _, school = sign_in(accounts)
    assert school["institution_verified"] and school["institution"]["id"] == "institute"
    token, company = sign_in(accounts, "engineer@company.example")
    monkeypatch.setenv("PAPERLEET_BLOCKED_EMAIL_DOMAINS", "company.example")
    assert accounts.get_session(token) is None
    with pytest.raises(AccountError, match="Personal email"):
        accounts.check_email_access(company)
    monkeypatch.setenv("PAPERLEET_BLOCKED_EMAIL_DOMAINS", "")
    monkeypatch.setenv("PAPERLEET_REGISTRATION_POLICY", "invite_only")
    assert accounts.get_session(token) is None


@pytest.mark.parametrize("setting,value", [("PAPERLEET_REGISTRATION_POLICY", "anything"), ("PAPERLEET_BLOCKED_EMAIL_DOMAINS", "*@example.com")])
def test_open_registration_invalid_policy_fails_closed(accounts, monkeypatch, setting, value):
    monkeypatch.setenv("PAPERLEET_REGISTRATION_POLICY", "work_email")
    monkeypatch.setenv(setting, value)
    assert accounts.capabilities()["auth_available"] is False
    with pytest.raises(AuthUnavailable):
        sign_in(accounts, "engineer@company.example")


def test_open_registration_wrong_or_mismatched_verification_creates_no_account(accounts, monkeypatch):
    monkeypatch.setenv("PAPERLEET_REGISTRATION_POLICY", "work_email")
    with pytest.raises(AuthenticationError):
        accounts.verify_code("engineer@company.example", "000000")
    accounts.provider.response = {"id": "provider", "email": "other@company.example", "email_confirmed_at": "now"}
    with pytest.raises(AuthenticationError, match="does not match"):
        sign_in(accounts, "engineer@company.example")
    with sqlite3.connect(accounts.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM account_sessions").fetchone()[0] == 0


def test_no_credentials_never_creates_a_verified_account(accounts):
    accounts.provider = SupabaseEmailProvider(url="", key="")
    assert accounts.capabilities()["auth_available"] is False
    for action in (lambda: accounts.request_code("student@research.edu"),
                   lambda: accounts.verify_code("student@research.edu", "123456")):
        with pytest.raises(AuthUnavailable):
            action()
    with sqlite3.connect(accounts.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0


@pytest.mark.parametrize("email", ["student@gmail.com", "student@other.edu", "student@sub.research.edu", "student@research.edu.evil.com"])
def test_exact_institution_domain_required_before_provider_contact(accounts, email):
    with pytest.raises(AccountError, match="not approved"):
        accounts.request_code(email)
    with pytest.raises(AccountError, match="not approved"):
        accounts.verify_code(email, "123456")
    assert accounts.provider.sent == []


def test_mailbox_request_is_normalized_and_rate_limited(accounts):
    assert accounts.request_code(" Student@RESEARCH.EDU ")["sent"] is True
    assert accounts.provider.sent == ["student@research.edu"]
    with pytest.raises(RateLimited):
        accounts.request_code("student@research.edu")
    assert len(accounts.provider.sent) == 1


def test_rate_limit_samples_time_under_lock_and_zero_cooldown_ignores_clock_order(accounts, monkeypatch):
    from contextlib import contextmanager
    original_connect = accounts._connect
    state = {"locked": False}
    clock_values = iter([200.0, 199.0])

    class LockedConnection:
        def __init__(self, conn):
            self.conn = conn

        def execute(self, sql, *args):
            result = self.conn.execute(sql, *args)
            if sql == "BEGIN IMMEDIATE":
                state["locked"] = True
            return result

        def executemany(self, *args):
            return self.conn.executemany(*args)

    @contextmanager
    def instrumented_connection():
        state["locked"] = False
        with original_connect() as conn:
            yield LockedConnection(conn)

    def locked_clock():
        assert state["locked"], "Request timestamps must be sampled after acquiring the write lock"
        return next(clock_values)

    monkeypatch.setattr(accounts, "_connect", instrumented_connection)
    monkeypatch.setattr("paperleet.accounts.time.time", locked_clock)
    # A backwards wall-clock adjustment must not introduce a cooldown when none
    # was configured. Sampling before the lock also allowed this order under load.
    accounts._limit("test", "request", maximum=3, seconds=3600, cooldown=0)
    accounts._limit("test", "request", maximum=3, seconds=3600, cooldown=0)


@pytest.mark.parametrize("response", [
    {"id": "123", "email": "student@research.edu"},
    {"id": "123", "email": "other@research.edu", "email_confirmed_at": "now"},
    {"id": "123", "email": "bad email", "email_confirmed_at": "now"},
    {"email": "student@research.edu", "email_confirmed_at": "now"},
    {"id": "123", "email": "student@research.edu", "email_confirmed_at": "now", "is_anonymous": True},
])
def test_provider_must_attest_to_the_requested_verified_identity(accounts, response):
    accounts.provider.response = response
    with pytest.raises(AuthenticationError):
        sign_in(accounts)
    with sqlite3.connect(accounts.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0


def test_failed_verification_attempts_are_limited(accounts):
    for _ in range(10):
        with pytest.raises(AuthenticationError):
            accounts.verify_code("student@research.edu", "000000")
    with pytest.raises(RateLimited):
        sign_in(accounts)


def test_session_is_opaque_hashed_expiring_and_revocable(accounts):
    token, user = sign_in(accounts)
    assert user["email"] == "student@research.edu"
    assert user["institution_verified"] is True
    assert accounts.get_session(token)["id"] == user["id"]
    with sqlite3.connect(accounts.db_path) as conn:
        row = conn.execute("SELECT token_hash,expires_at FROM account_sessions").fetchone()
        assert row[0] == hashlib.sha256(token.encode()).hexdigest()
        assert token != row[0]
        assert 6 * 86400 < row[1] - time.time() <= 7 * 86400
        conn.execute("UPDATE account_sessions SET expires_at=?", (time.time() - 1,))
    assert accounts.get_session(token) is None
    new_token, second = sign_in(accounts)
    assert second["id"] == user["id"]
    assert new_token != token
    accounts.logout(new_token)
    assert accounts.get_session(new_token) is None


def test_suspended_or_no_longer_approved_users_cannot_use_sessions(accounts):
    token, user = sign_in(accounts)
    accounts.institutions["institute"]["domains"] = ["new.research.edu"]
    assert accounts.get_session(token) is None
    accounts.institutions["institute"]["domains"] = ["research.edu"]
    with sqlite3.connect(accounts.db_path) as conn:
        conn.execute("UPDATE users SET status='suspended' WHERE id=?", (user["id"],))
    assert accounts.get_session(token) is None
    with pytest.raises(AuthenticationError, match="not active"):
        sign_in(accounts)


def test_provider_identity_conflict_never_links_accounts_by_email_alone(accounts):
    sign_in(accounts)
    accounts.provider.response = {"id": "new-provider-id", "email": "student@research.edu", "email_confirmed_at": "now"}
    with pytest.raises(AuthenticationError, match="identity changed"):
        sign_in(accounts)


def test_public_identity_is_mandatory_and_sensitive_fields_default_hidden(accounts):
    _, private = sign_in(accounts)
    public = accounts.public_user(private["id"])
    assert public["display_name"]
    assert public["avatar_color"].startswith("#")
    assert public["username"]
    assert public["show_progress"] is True
    assert not ({"email", "institution", "country", "role", "status", "provider_id"} & public.keys())
    assert private["show_institution"] is False
    assert private["show_country"] is False
    assert private["institution"]["id"] == "institute"
    with pytest.raises(AccountError):
        accounts.update_profile(private["id"], {"display_name": " "})
    with pytest.raises(AccountError):
        accounts.update_profile(private["id"], {"avatar_color": ""})
    with pytest.raises(AccountError):
        accounts.update_profile(private["id"], {"show_avatar": False})


def test_privacy_switches_apply_to_shared_author_serializer(accounts):
    _, user = sign_in(accounts)
    accounts.update_profile(user["id"], {"display_name": "Ada", "country": "gb", "show_country": True, "show_institution": True, "show_progress": False})
    public = accounts.public_user(user["id"])
    assert public["display_name"] == "Ada"
    assert public["country"] == "GB"
    assert public["institution"]["name_en"] == "Research University"
    assert public["show_progress"] is False
    with accounts._connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE id=?", (user["id"],)).fetchone()
        assert public_user(conn, row, accounts.institutions) == public
    accounts.update_profile(user["id"], {"show_country": False, "show_institution": False})
    assert "country" not in accounts.public_user(user["id"])
    assert "institution" not in accounts.public_user(user["id"])
    assert accounts.get_user(user["id"])["country"] == "GB"


@pytest.mark.parametrize("payload", [
    {"role": "admin"}, {"institution_id": "mit"}, {"email": "another@research.edu"},
    {"show_country": "false"}, {"country": "ZZ"}, {"avatar_color": "url(javascript:x)"},
    {"username": "bad-user"}, {"bio": "a" * 501}, {"display_name": "a" * 41},
])
def test_profile_rejects_privilege_changes_and_invalid_fields(accounts, payload):
    _, user = sign_in(accounts)
    with pytest.raises(AccountError):
        accounts.update_profile(user["id"], payload)


def test_usernames_unique_without_case_variants(accounts):
    _, first = sign_in(accounts)
    _, second = sign_in(accounts, "other@research.edu")
    accounts.update_profile(first["id"], {"username": "Ada_Lovelace"})
    with pytest.raises(AccountError, match="already taken"):
        accounts.update_profile(second["id"], {"username": "ada_lovelace"})


def test_admin_bootstrap_requires_verified_allowlisted_mailbox(accounts, monkeypatch):
    monkeypatch.setenv("PAPERLEET_ADMIN_EMAILS", " Student@RESEARCH.EDU ,person@gmail.com")
    _, user = sign_in(accounts)
    assert user["role"] == "admin"
    assert "role" not in accounts.public_user(user["id"])
    with pytest.raises(AccountError):
        sign_in(accounts, "person@gmail.com")


def test_hosted_requires_nonempty_exact_invites_and_still_requires_institution(accounts, monkeypatch):
    monkeypatch.setenv("PAPERLEET_MODE", "hosted")
    assert accounts.capabilities()["invite_only"] is True
    assert accounts.capabilities()["auth_available"] is False
    with pytest.raises(AuthUnavailable, match="nonempty"):
        accounts.request_code("student@research.edu")
    with pytest.raises(AuthUnavailable, match="nonempty"):
        sign_in(accounts)
    monkeypatch.setenv("PAPERLEET_INVITE_EMAILS", " Student@RESEARCH.EDU , invited@gmail.com")
    assert accounts.capabilities()["auth_available"] is True
    with pytest.raises(AuthenticationError, match="invite-only"):
        accounts.request_code("other@research.edu")
    with pytest.raises(AuthenticationError, match="invite-only"):
        sign_in(accounts, "other@research.edu")
    with pytest.raises(AccountError, match="not approved"):
        accounts.request_code("invited@gmail.com")
    with pytest.raises(AccountError, match="not approved"):
        sign_in(accounts, "invited@gmail.com")
    assert accounts.provider.sent == []
    accounts.request_code("student@research.edu")
    token, user = sign_in(accounts)
    assert accounts.get_session(token)["id"] == user["id"]


def test_invite_removal_revokes_existing_sessions_and_admin_cannot_bypass(accounts, monkeypatch):
    monkeypatch.setenv("PAPERLEET_ADMIN_EMAILS", "student@research.edu")
    token, user = sign_in(accounts)
    assert user["role"] == "admin"
    monkeypatch.setenv("PAPERLEET_MODE", "hosted")
    monkeypatch.setenv("PAPERLEET_INVITE_EMAILS", "student@research.edu")
    assert accounts.get_session(token)
    monkeypatch.setenv("PAPERLEET_INVITE_EMAILS", "other@research.edu")
    assert accounts.get_session(token) is None
    with pytest.raises(AuthenticationError, match="invite-only"):
        sign_in(accounts)
    monkeypatch.setenv("PAPERLEET_INVITE_EMAILS", "")
    assert accounts.get_session(token) is None


def test_invalid_invitation_config_fails_closed(accounts, monkeypatch):
    monkeypatch.setenv("PAPERLEET_MODE", "hosted")
    monkeypatch.setenv("PAPERLEET_INVITE_EMAILS", "*@research.edu")
    # Wildcards cannot grant eligibility to all institutional users.
    assert accounts.capabilities()["auth_available"] is False
    with pytest.raises(AuthUnavailable, match="wildcards"):
        sign_in(accounts)
    monkeypatch.setenv("PAPERLEET_INVITE_EMAILS", "research.edu")
    assert accounts.capabilities()["auth_available"] is False
    with pytest.raises(AuthUnavailable):
        sign_in(accounts)


def test_pending_institution_request_never_grants_access(accounts):
    payload = {"name": "New Research Institute", "website": "https://new.edu", "email_domain": "new.edu"}
    request = accounts.request_institution(payload)
    assert request["status"] == "pending"
    assert accounts.request_institution(payload)["id"] == request["id"]
    with pytest.raises(AccountError, match="not approved"):
        accounts.request_code("student@new.edu")
    with sqlite3.connect(accounts.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0


def test_concurrent_institution_requests_return_one_intake_record(accounts, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    barrier = Barrier(2)
    original_limit = accounts._limit

    def limited(*args, **kwargs):
        original_limit(*args, **kwargs)
        barrier.wait(timeout=5)

    monkeypatch.setattr(accounts, "_limit", limited)
    payload = {"name": "New Institute", "website": "https://new.edu", "email_domain": "new.edu"}
    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(accounts.request_institution, [payload, payload]))
    assert first["id"] == second["id"]
    with sqlite3.connect(accounts.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM institution_requests").fetchone()[0] == 1


def test_request_maintenance_command_is_read_only(accounts, capsys):
    from paperleet.accounts import main
    accounts.request_institution({"name": "New Institute", "website": "https://new.edu", "email_domain": "new.edu"})
    before = accounts.db_path.read_bytes()
    main(["institution-requests", "--database", str(accounts.db_path)])
    output = json.loads(capsys.readouterr().out)
    assert output[0]["email_domain"] == "new.edu"
    assert output[0]["status"] == "pending"
    assert output[0]["domain_approved"] is False
    assert accounts.db_path.read_bytes() == before


def test_supabase_requests_use_otp_and_authoritative_verify_response(monkeypatch):
    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, limit):
            return json.dumps({"user": {"id": "provider", "email": "a@research.edu", "email_confirmed_at": "now"}}).encode()

    def open_request(request, timeout):
        calls.append((request, timeout))
        return Response()

    monkeypatch.setattr("paperleet.accounts.urlopen", open_request)
    provider = SupabaseEmailProvider("https://project.supabase.co", "public-key")
    provider.request_code("a@research.edu")
    assert calls[0][0].full_url == "https://project.supabase.co/auth/v1/otp"
    assert json.loads(calls[0][0].data) == {"email": "a@research.edu", "create_user": True}
    assert calls[0][0].headers["Apikey"] == "public-key"
    assert calls[0][1] == 15
    assert provider.verify_code("a@research.edu", "123456")["id"] == "provider"
    assert json.loads(calls[1][0].data)["type"] == "email"
    assert calls[1][0].full_url.endswith("/verify")


def test_provider_errors_do_not_disclose_secrets_or_raw_response(monkeypatch):
    def failed(*args, **kwargs):
        raise HTTPError("https://project.supabase.co", 401, "secret-email-token", {}, None)

    monkeypatch.setattr("paperleet.accounts.urlopen", failed)
    with pytest.raises(AuthenticationError) as error:
        SupabaseEmailProvider("https://project.supabase.co", "public-key").request_code("a@research.edu")
    assert "secret-email-token" not in str(error.value)
