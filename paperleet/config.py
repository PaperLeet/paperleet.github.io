"""Explicit local/hosted boundaries and durable application data paths."""
import os
from pathlib import Path
from urllib.parse import urlsplit

from paperleet.catalog import root


def hosted():
    return os.environ.get("PAPERLEET_MODE", "local") == "hosted"


def auth_mode():
    mode = os.environ.get("PAPERLEET_AUTH_MODE", "institutional")
    if mode not in ("institutional", "test"):
        raise ValueError("PAPERLEET_AUTH_MODE must be institutional or test")
    return mode


def registration_policy():
    policy = os.environ.get("PAPERLEET_REGISTRATION_POLICY", "invite_only")
    if policy not in ("invite_only", "work_email"):
        raise ValueError("PAPERLEET_REGISTRATION_POLICY must be invite_only or work_email")
    return policy


def data_directory():
    return Path(os.environ.get("PAPERLEET_DATA_DIR", root() / ".paperleet")).expanduser().resolve()


def origin(value):
    """Normalize only an HTTP origin, rejecting URLs with credentials or paths."""
    try:
        parsed = urlsplit(value)
        if (parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username
                or parsed.password or parsed.path not in ("", "/") or parsed.query or parsed.fragment):
            raise ValueError("Expected an HTTP origin")
        port = parsed.port
        host = parsed.hostname.lower()
        if ":" in host:
            host = "[" + host + "]"
        suffix = ":" + str(port) if port and port != (443 if parsed.scheme == "https" else 80) else ""
        return parsed.scheme + "://" + host + suffix
    except (ValueError, AttributeError):
        raise ValueError("Expected an HTTP origin") from None


def public_origin():
    return origin(os.environ.get("PAPERLEET_PUBLIC_URL", ""))


def validate_hosted_settings():
    authentication = auth_mode()
    registration = registration_policy()
    mode = os.environ.get("PAPERLEET_MODE", "local")
    if mode not in ("local", "hosted"):
        raise ValueError("PAPERLEET_MODE must be local or hosted")
    if not hosted():
        return
    required = ["PAPERLEET_PUBLIC_URL", "PAPERLEET_DATA_DIR", "PAPERLEET_JUDGE_URL", "PAPERLEET_JUDGE_TOKEN"]
    if authentication == "institutional":
        required += ["SUPABASE_URL"]
        if registration == "invite_only":
            required += ["PAPERLEET_INVITE_EMAILS"]
    missing = [key for key in required if not os.environ.get(key, "").strip()]
    if authentication == "institutional" and not (os.environ.get("SUPABASE_PUBLISHABLE_KEY") or os.environ.get("SUPABASE_ANON_KEY")):
        missing.append("SUPABASE_PUBLISHABLE_KEY")
    if missing:
        raise ValueError("Hosted mode requires: " + ", ".join(missing))
    if not public_origin().startswith("https://"):
        raise ValueError("PAPERLEET_PUBLIC_URL must use HTTPS in hosted mode")
    if not Path(os.environ["PAPERLEET_DATA_DIR"]).is_absolute():
        raise ValueError("PAPERLEET_DATA_DIR must be an absolute persistent directory")
    token = os.environ["PAPERLEET_JUDGE_TOKEN"]
    if len(token) < 32 or not token.isascii() or any(c.isspace() or ord(c) < 33 or ord(c) > 126 for c in token):
        raise ValueError("PAPERLEET_JUDGE_TOKEN must contain at least 32 printable ASCII characters without whitespace")
    origin(os.environ["PAPERLEET_JUDGE_URL"])
    from paperleet.hosting.judge_client import job_timeout
    job_timeout()
    if authentication == "institutional":
        from paperleet.accounts import SupabaseEmailProvider
        if not SupabaseEmailProvider().available:
            raise ValueError("Configure a valid Supabase URL and publishable key")
        from paperleet.accounts import blocked_email_domains
        blocked_email_domains()
