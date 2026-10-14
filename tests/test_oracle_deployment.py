"""Oracle profile must coexist with the host's Nginx and keep private services private."""
import json
import os
import shutil
import subprocess
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
def test_oracle_profile_only_publishes_loopback_proxy(tmp_path):
    if not shutil.which("docker"):
        pytest.skip("Docker CLI is unavailable")
    settings = tmp_path / "oracle.env"
    settings.write_text("PAPERLEET_DOMAIN=app.paperleet.example.test\nPAPERLEET_AUTH_MODE=test\nPAPERLEET_JUDGE_TOKEN=" + "x" * 64 + "\n")
    env = {key: value for key, value in os.environ.items() if not key.startswith(("PAPERLEET_", "SUPABASE_", "COMPOSE_"))}
    if subprocess.run(["docker", "compose", "version"], capture_output=True, env=env).returncode:
        pytest.skip("Docker Compose is unavailable")
    result = subprocess.run(["docker", "compose", "--env-file", str(settings), "-f", str(ROOT / "deploy/compose.yaml"), "-f", str(ROOT / "deploy/compose.oracle.yaml"), "config", "--format", "json"], capture_output=True, text=True, env=env, timeout=15)
    assert result.returncode == 0, result.stderr
    services = json.loads(result.stdout)["services"]
    ports = services["web"]["ports"]
    assert len(ports) == 1
    assert ports[0]["host_ip"] == "127.0.0.1" and str(ports[0]["published"]) == "18880"
    assert not services["api"].get("ports") and not services["judge"].get("ports")
    assert services["judge"]["networks"] == {"judge_internal": None}
    assert services["api"]["environment"]["PAPERLEET_PUBLIC_URL"] == "https://app.paperleet.example.test"
    assert any(volume["target"] == "/etc/caddy/Caddyfile" and volume["source"].endswith("Caddyfile.oracle") for volume in services["web"]["volumes"])
