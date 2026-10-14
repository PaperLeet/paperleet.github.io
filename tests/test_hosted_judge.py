"""Isolation/protocol regressions; live Docker smoke is explicitly opt-in."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from paperleet.catalog import problem, root
from paperleet.hosting.judge_client import (
    CURATED_PROBLEMS, InvalidReport, JudgeBusy, JudgeUnavailable, expected_tests,
    job_timeout, judge_capabilities, run_hosted_problem, validate_job, validate_report,
)
from paperleet.hosting.judge_service import handler_for
from paperleet.hosting.sandbox import JudgeSandbox, OutputLimitError, _run_bounded


@pytest.fixture
def job():
    return validate_job({"problem": "rl/gae", "code": "def gae(*args):\n    raise NotImplementedError\n", "layers": ["T0"]})


def report_for(job, passed=True):
    rows = []
    for (name, layer), count in expected_tests(problem(job["problem"]), job["layers"]).items():
        for i in range(count):
            rows.append({"id": name, "nodeid": f"trusted_test.py::{name}[{i}]", "layer": layer,
                "passed": passed, "outcome": "passed" if passed else "failed", "duration_s": .01,
                "why": "Trusted test", "message": "", "artifacts": {}})
    return {"problem": job["problem"], "layers": job["layers"], "passed": passed,
        "source_sha256": hashlib.sha256(job["code"].encode()).hexdigest(), "tests": rows,
        "created_at": "2026-01-01T00:00:00Z", "summary": {"T0": [999, 999]}}


@pytest.mark.parametrize("payload", [None, {}, {"problem": "community/new", "code": "x"},
    {"problem": "../accounts", "code": "x"}, {"problem": "rl/gae", "code": " "},
    {"problem": "rl/gae", "code": "x" * (128 * 1024 + 1)},
    {"problem": "rl/gae", "code": "x", "image": "evil"},
    {"problem": "rl/gae", "code": "x", "layers": {"T0": True}}])
def test_job_rejects_unmaintained_packages_paths_and_resource_controls(payload):
    with pytest.raises(ValueError):
        validate_job(payload)


def test_manifest_covers_curated_catalog_and_parametrized_tests():
    from paperleet.paper_catalog import paper_catalog
    from paperleet.catalog import problems
    cards, _ = paper_catalog(problems())
    assert CURATED_PROBLEMS == {module for card in cards for module in card.get("curated_module_ids", [])}
    assert not any(module.startswith("community/") for module in CURATED_PROBLEMS)
    for identifier in CURATED_PROBLEMS:
        assert expected_tests(problem(identifier), ["T0", "T1", "T2", "T3"])
    assert sum(expected_tests(problem("diffusion/ddpm_sampling"), ["T3"]).values()) == 2


def test_report_acceptance_and_summary_are_derived_not_trusted(job):
    raw = report_for(job)
    report = validate_report(raw, job)
    assert report["passed"] is True
    assert report["summary"]["T0"] == [len(raw["tests"])] * 2
    assert report["impl"] == "submission.py"
    assert not validate_report(report_for(job, False), job)["passed"]


@pytest.mark.parametrize("mutation", [
    lambda report: report.update(problem="rl/ppo_clip"),
    lambda report: report.update(source_sha256="0" * 64),
    lambda report: report.update(layers=["T1"]),
    lambda report: report.update(tests=[]),
    lambda report: report.update(tests=report["tests"] * 2),
    lambda report: report["tests"][0].update(id="invented_test"),
    lambda report: report["tests"][0].update(outcome="failed"),
    lambda report: report["tests"][0].update(passed="true"),
    lambda report: report["tests"][0].update(duration_s=float("nan")),
    lambda report: report.update(error="Timed out but maliciously claims acceptance"),
])
def test_malformed_mismatched_incomplete_and_forged_acceptance_is_rejected(job, mutation):
    report = report_for(job)
    mutation(report)
    with pytest.raises(InvalidReport):
        validate_report(report, job)


def test_incomplete_error_report_cannot_be_accepted(job):
    report = report_for(job)
    report.update(tests=[], passed=False, error="Import failed")
    result = validate_report(report, job)
    assert result["passed"] is False and result["error"] == "Import failed"


def test_optional_comparisons_survive_hosted_validation_without_affecting_outcomes(job):
    raw=report_for(job,False)
    comparison={'kind':'value','expected_shape':[],'actual_shape':[],'index':[], 'expected':1.,'actual':'NaN','absolute_error':'Infinity','rtol':1e-5,'atol':1e-8}
    raw['tests'][0]['comparison']=comparison
    result=validate_report(raw,job)
    assert not result['passed'] and result['tests'][0]['comparison']==comparison
    raw['tests'][0]['comparison']={'kind':'invented','actual':'x'*10000}
    assert 'comparison' not in validate_report(raw,job)['tests'][0]
    accepted=report_for(job)
    accepted['tests'][0]['comparison']=comparison
    assert validate_report(accepted,job)['passed']
    assert 'comparison' not in validate_report(accepted,job)['tests'][0]


@pytest.mark.parametrize("configured,expected", [(None, 150), ("1", 1), ("90", 90), ("150", 150)])
def test_judge_timeout_retains_default_and_allows_shorter_deadlines(monkeypatch, configured, expected):
    monkeypatch.delenv("PAPERLEET_JUDGE_TIMEOUT_SECONDS", raising=False)
    if configured is not None:
        monkeypatch.setenv("PAPERLEET_JUDGE_TIMEOUT_SECONDS", configured)
    assert job_timeout() == expected
    assert JudgeSandbox().timeout == expected


@pytest.mark.parametrize("configured", ["", "0", "-1", "151", "90.5", "inf", "not-a-number"])
def test_judge_timeout_rejects_invalid_config_at_manager_startup(monkeypatch, configured):
    monkeypatch.setenv("PAPERLEET_JUDGE_TIMEOUT_SECONDS", configured)
    with pytest.raises(ValueError, match="PAPERLEET_JUDGE_TIMEOUT_SECONDS"):
        job_timeout()
    with pytest.raises(ValueError, match="PAPERLEET_JUDGE_TIMEOUT_SECONDS"):
        JudgeSandbox()


@pytest.mark.parametrize("configured,expected", [(None, 170), ("90", 110)])
def test_client_timeout_includes_bounded_manager_cleanup_margin(job, monkeypatch, configured, expected):
    import paperleet.hosting.judge_client as module
    monkeypatch.delenv("PAPERLEET_JUDGE_TIMEOUT_SECONDS", raising=False)
    if configured is not None:
        monkeypatch.setenv("PAPERLEET_JUDGE_TIMEOUT_SECONDS", configured)
    calls = []
    def request(endpoint, payload, timeout):
        calls.append((endpoint, payload, timeout))
        return report_for(job)
    monkeypatch.setattr(module, "_request", request)
    assert run_hosted_problem(problem(job["problem"]), job["code"], job["layers"])["passed"]
    assert calls == [("/run", job, expected)]


def test_sandbox_uses_configured_deadline_and_reports_it_after_cleanup(job, monkeypatch):
    import paperleet.hosting.sandbox as module
    monkeypatch.setenv("PAPERLEET_JUDGE_TIMEOUT_SECONDS", "90")
    sandbox = JudgeSandbox()
    monkeypatch.setattr(sandbox, "capability", lambda: {"ready": True, "image_id": "sha256:" + "b" * 64})
    calls, cleanup = [], []
    def execute(args, payload, timeout):
        calls.append((args, json.loads(payload), timeout))
        raise subprocess.TimeoutExpired(args, timeout)
    monkeypatch.setattr(module, "_run_bounded", execute)
    monkeypatch.setattr(module.subprocess, "run", lambda args, **kwargs: cleanup.append(args))
    with pytest.raises(JudgeUnavailable, match="90-second limit"):
        sandbox.run(job)
    args, payload, timeout = calls[0]
    assert payload == job and timeout == 90
    assert cleanup == [["docker", "rm", "-f", args[args.index("--name") + 1]]]


def test_sandbox_pins_image_uses_stdin_and_never_mounts_host_or_forwards_env(job, monkeypatch):
    import paperleet.hosting.sandbox as module
    image_id = "sha256:" + "a" * 64
    sandbox = JudgeSandbox()
    monkeypatch.setattr(sandbox, "capability", lambda: {"ready": True, "image_id": image_id})
    calls, cleanup = [], []
    def execute(args, payload, timeout):
        assert timeout == sandbox.timeout
        calls.append((args, json.loads(payload)))
        return 0, json.dumps(report_for(job)).encode(), b""
    monkeypatch.setattr(module, "_run_bounded", execute)
    monkeypatch.setattr(module.subprocess, "run", lambda args, **kwargs: cleanup.append((args, kwargs)))
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-forward")
    monkeypatch.setenv("PAPERLEET_JUDGE_TOKEN", "also-not-forward")
    assert sandbox.run(job)["passed"]
    args, payload = calls[0]
    assert payload == job and args[-1] == image_id
    for flag in ("--network=none", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges", "--memory=2g", "--memory-swap=2g", "--cpus=2", "--pids-limit=128", "--log-driver=none"):
        assert flag in args
    assert args[args.index("--user") + 1] == "65534:65534"
    assert "--mount" not in args and "-v" not in args and "--env" not in args and "-e" not in args
    assert cleanup[0][0][:3] == ["docker", "rm", "-f"]
    assert cleanup[0][0][-1] == args[args.index("--name") + 1]
    assert "OPENAI_API_KEY" not in cleanup[0][1]["env"]
    assert "PAPERLEET_JUDGE_TOKEN" not in cleanup[0][1]["env"]


@pytest.mark.parametrize("failure", [subprocess.TimeoutExpired("docker", 150), OutputLimitError("flood"), OSError("offline")])
def test_sandbox_cleans_up_container_even_when_client_fails(job, monkeypatch, failure):
    import paperleet.hosting.sandbox as module
    sandbox = JudgeSandbox()
    monkeypatch.setattr(sandbox, "capability", lambda: {"ready": True, "image_id": "sha256:" + "b" * 64})
    def fail(*args, **kwargs):
        raise failure
    monkeypatch.setattr(module, "_run_bounded", fail)
    calls = []
    monkeypatch.setattr(module.subprocess, "run", lambda args, **kwargs: calls.append(args))
    with pytest.raises(JudgeUnavailable):
        sandbox.run(job)
    assert calls[0][:3] == ["docker", "rm", "-f"]


def test_unavailable_docker_never_executes_python_fallback(job, monkeypatch):
    sandbox = JudgeSandbox()
    monkeypatch.setattr(sandbox, "capability", lambda: {"ready": False, "reason": "Docker unavailable"})
    monkeypatch.setattr("paperleet.hosting.sandbox._run_bounded", lambda *_: pytest.fail("No process may execute"))
    with pytest.raises(JudgeUnavailable, match="Docker unavailable"):
        sandbox.run(job)


def test_bounded_process_handles_large_stdin_and_separate_stderr():
    payload = b"x" * 180000
    code, stdout, stderr = _run_bounded([sys.executable, "-c", "import sys; data=sys.stdin.buffer.read(); sys.stdout.write(str(len(data))); sys.stderr.write('diagnostic')"], payload, timeout=5)
    assert code == 0 and stdout == b"180000" and stderr == b"diagnostic"


def test_bounded_process_stops_output_flood_and_walltime():
    with pytest.raises(OutputLimitError):
        _run_bounded([sys.executable, "-c", "import sys; sys.stdout.write('x'*100000)"], b"", timeout=5, max_bytes=1000)
    with pytest.raises(subprocess.TimeoutExpired):
        _run_bounded([sys.executable, "-c", "import time; time.sleep(3)"], b"", timeout=.1)


@pytest.fixture
def internal_service(monkeypatch):
    class FakeSandbox:
        entered = threading.Event()
        release = threading.Event()
        block = False
        calls = []
        def capability(self):
            return {"ready": True}
        def run(self, job):
            self.calls.append(job)
            self.entered.set()
            if self.block:
                assert self.release.wait(5)
            return report_for(job)
    sandbox = FakeSandbox()
    token = "private-manager-token-" + "x" * 32
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(sandbox, token))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setenv("PAPERLEET_JUDGE_URL", url)
    monkeypatch.setenv("PAPERLEET_JUDGE_TOKEN", token)
    try:
        yield sandbox, url, token
    finally:
        sandbox.release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_client_uses_authenticated_internal_http_and_validates_report(job, internal_service):
    sandbox, url, _ = internal_service
    with pytest.raises(HTTPError) as error:
        urlopen(url + "/health")
    assert error.value.code == 401
    assert judge_capabilities()["ready"] is True
    assert run_hosted_problem(problem(job["problem"]), job["code"], job["layers"])["passed"]
    assert sandbox.calls == [job]


def test_internal_service_rejects_unsupported_jobs_before_execution(internal_service):
    sandbox, url, token = internal_service
    request = Request(url + "/run", data=json.dumps({"problem": "community/unreviewed", "code": "x"}).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + token})
    with pytest.raises(HTTPError) as error:
        urlopen(request)
    assert error.value.code == 400 and sandbox.calls == []


def test_manager_concurrency_is_bounded_and_busy_is_actionable(job, internal_service):
    sandbox, _, _ = internal_service
    sandbox.block = True
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(run_hosted_problem, problem(job["problem"]), job["code"], job["layers"])
        assert sandbox.entered.wait(3)
        with pytest.raises(JudgeBusy):
            run_hosted_problem(problem(job["problem"]), job["code"], job["layers"])
        sandbox.release.set()
        assert future.result()["passed"] is True
    assert len(sandbox.calls) == 1


def test_missing_client_configuration_does_not_use_local_runner(job, monkeypatch):
    monkeypatch.delenv("PAPERLEET_JUDGE_URL", raising=False)
    monkeypatch.delenv("PAPERLEET_JUDGE_TOKEN", raising=False)
    assert judge_capabilities()["ready"] is False
    with pytest.raises(JudgeUnavailable, match="not configured"):
        run_hosted_problem(problem(job["problem"]), job["code"], job["layers"])


@pytest.mark.parametrize("identifier", ["rl/gae", "diffusion/ddpm_sampling"])
def test_worker_protocol_with_trusted_reference_only(identifier):
    """Checks trusted numerical fixtures locally; this is not Docker validation."""
    entry = problem(identifier)
    job = validate_job({"problem": identifier, "code": (entry["directory"] / "solution.py").read_text(), "layers": ["T0", "T1", "T2", "T3"]})
    process = subprocess.run([sys.executable, "-m", "paperleet.hosting.worker"], input=json.dumps(job).encode(), capture_output=True, cwd=root(), timeout=145)
    assert process.returncode == 0, process.stderr.decode()
    assert validate_report(json.loads(process.stdout), job)["passed"] is True


@pytest.mark.skipif(os.environ.get("PAPERLEET_DOCKER_SMOKE") != "1", reason="Requires running Docker and built paperleet-worker:local image")
@pytest.mark.parametrize("identifier", sorted(CURATED_PROBLEMS))
def test_live_docker_references(identifier):
    sandbox = JudgeSandbox()
    assert sandbox.capability()["ready"]
    entry = problem(identifier)
    report = sandbox.run({"problem": entry["id"], "code": (entry["directory"] / "solution.py").read_text(), "layers": ["T0", "T1", "T2", "T3"]})
    assert report["passed"] is True, report.get("error") or [(row["id"], row["message"]) for row in report["tests"] if not row["passed"]]


@pytest.mark.skipif(os.environ.get("PAPERLEET_DOCKER_SMOKE") != "1", reason="Requires running Docker and built worker image")
def test_live_docker_starter_fails():
    entry = problem("rl/gae")
    report = JudgeSandbox().run({"problem": entry["id"], "code": (entry["directory"] / "starter.py").read_text(), "layers": ["T0", "T1", "T2", "T3"]})
    assert report["passed"] is False
    assert any(not row["passed"] for row in report["tests"])


@pytest.mark.skipif(os.environ.get("PAPERLEET_DOCKER_SMOKE") != "1", reason="Requires running Docker and built worker image")
def test_live_docker_blocks_network_private_files_secrets_and_privileges(monkeypatch):
    monkeypatch.setenv("PAPERLEET_ISOLATION_SENTINEL", "host-only-test-value")
    entry = problem("rl/gae")
    probes = '''import os, socket
from pathlib import Path
assert os.getuid() == 65534
assert "PAPERLEET_ISOLATION_SENTINEL" not in os.environ
assert "PAPERLEET_JUDGE_TOKEN" not in os.environ
assert "OPENAI_API_KEY" not in os.environ
for private in ("/opt/paperleet/.env", "/opt/paperleet/.paperleet", "/opt/paperleet/paperleet/accounts.py", "/var/run/docker.sock"):
    assert not Path(private).exists(), private
try:
    Path("/opt/paperleet/unauthorized-write").write_text("x")
except OSError:
    pass
else:
    raise AssertionError("Root filesystem was writable")
try:
    connection = socket.create_connection(("1.1.1.1", 53), timeout=.25)
except OSError:
    pass
else:
    connection.close()
    raise AssertionError("Outbound network was available")
capabilities = next(line for line in Path("/proc/self/status").read_text().splitlines() if line.startswith("CapEff:"))
assert int(capabilities.split()[1], 16) == 0
'''
    report = JudgeSandbox().run({"problem": entry["id"], "code": probes + (entry["directory"] / "solution.py").read_text(), "layers": ["T0"]})
    assert report["passed"] is True, [(row["id"], row["message"]) for row in report["tests"] if not row["passed"]]


@pytest.mark.skipif(os.environ.get("PAPERLEET_DOCKER_SMOKE") != "1", reason="Requires running Docker and built worker image")
def test_live_docker_timeout_removes_the_exact_container(monkeypatch):
    import paperleet.hosting.sandbox as module
    original, names = module._run_bounded, []
    def short_deadline(args, payload, timeout):
        names.append(args[args.index("--name") + 1])
        return original(args, payload, timeout=4)
    monkeypatch.setattr(module, "_run_bounded", short_deadline)
    with pytest.raises(JudgeUnavailable, match="limit"):
        JudgeSandbox().run({"problem": "rl/gae", "code": "while True:\n    pass\n", "layers": ["T0"]})
    assert names
    status = subprocess.run(["docker", "container", "inspect", names[0]], capture_output=True, timeout=5)
    assert status.returncode != 0, "Timed-out job container was not removed"


def test_worker_build_excludes_private_application_code_and_data():
    dockerfile = (root() / "deploy" / "Dockerfile.judge").read_text()
    copies = [line for line in dockerfile.splitlines() if line.startswith("COPY ")]
    assert all(not line.startswith("COPY . ") and "accounts.py" not in line and ".paperleet" not in line and ".env" not in line for line in copies)
    assert "USER 65534:65534" in dockerfile
