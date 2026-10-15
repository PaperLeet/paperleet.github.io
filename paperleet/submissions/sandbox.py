"""All generated test execution happens inside a bounded, offline container."""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import selectors
import time
import uuid

from .checks import package_hash, static_checks
from .artifacts import validate_report_artifacts


class OutputLimitError(RuntimeError):
    pass


def _run_bounded(args, timeout=90, max_bytes=1024 * 1024):
    """Bound both memory and host disk; a container's stdout is not subject to its fsize ulimit."""
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    deadline, chunks, length = time.monotonic() + timeout, [], 0
    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(args, timeout)
            for key, _ in selector.select(min(remaining, .25)):
                chunk = os.read(key.fileobj.fileno(), 16384)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                length += len(chunk)
                if length > max_bytes:
                    raise OutputLimitError("Sandbox output exceeded 1 MB")
                chunks.append(chunk)
        code = process.wait(timeout=max(.01, deadline - time.monotonic()))
        return code, b"".join(chunks).decode("utf-8", errors="replace")
    finally:
        selector.close()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        process.stdout.close()


class Sandbox:
    def __init__(self):
        self.image = os.environ.get("PAPERLEET_SANDBOX_IMAGE", "paperleet-harness:local")

    def capability(self):
        if not shutil.which("docker"):
            return {"ready": False, "image": self.image, "reason": "Install Docker and build the PaperLeet harness image to validate generated code."}
        try:
            result = subprocess.run(["docker", "image", "inspect", self.image, "--format", "{{.Id}}"], capture_output=True, text=True, timeout=4)
            if result.returncode:
                return {"ready": False, "image": self.image, "reason": "Start Docker and build the image: docker build -t paperleet-harness:local paperleet/submissions"}
            return {"ready": True, "image": self.image, "image_id": result.stdout.strip()}
        except (OSError, subprocess.TimeoutExpired):
            return {"ready": False, "image": self.image, "reason": "Docker is unavailable. Generated code has not been executed."}

    def run(self, candidate, implementation, layers=None):
        capability = self.capability()
        if not capability["ready"]:
            return {"exit_code": 125, "tests": [], "error": capability["reason"], "output": ""}
        # A worker container talking to the host Docker daemon needs an
        # identically mounted host/container directory for candidate bind mounts.
        # This administrator setting is never taken from a request or model.
        work_directory = os.environ.get("PAPERLEET_SANDBOX_WORK_DIR", "").strip()
        if work_directory and (not Path(work_directory).is_absolute() or not Path(work_directory).is_dir()):
            return {"exit_code": 125, "tests": [], "error": "The configured sandbox working directory is unavailable.", "output": ""}
        with tempfile.TemporaryDirectory(prefix="paperleet-sandbox-", dir=work_directory or None) as temp:
            directory = Path(temp)
            directory.chmod(0o755)
            (directory / "tests").mkdir(mode=0o755)
            (directory / "implementation.py").write_text(implementation, encoding="utf-8")
            if candidate.get("toy"):
                (directory / "artifact_context.json").write_text(json.dumps(candidate["toy"], allow_nan=False), encoding="utf-8")
            selected = set(layers or ["T0", "T1", "T2", "T3"])
            for test in candidate["tests"]:
                if test["layer"] in selected:
                    (directory / "tests" / f"test_{test['layer']}_{test['name']}.py").write_text(test["code"], encoding="utf-8")
            name = "paperleet-" + uuid.uuid4().hex
            args = ["docker", "run", "--rm", "--pull=never", "--name", name,
                    "--network=none", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
                    "--pids-limit=128", "--memory=1g", "--memory-swap=1g", "--cpus=2",
                    "--ulimit", "nofile=128:128", "--ulimit", "fsize=16777216:16777216",
                    "--tmpfs", "/tmp:rw,noexec,nosuid,size=128m", "--user", "65534:65534",
                    "--mount", f"type=bind,source={directory},target=/candidate,readonly", capability["image_id"]]
            try:
                returncode, text = _run_bounded(args)
                marker = "PAPERLEET_RESULT="
                rows = [line[len(marker):] for line in text.splitlines() if line.startswith(marker)]
                result = json.loads(rows[-1]) if rows else {"exit_code": returncode or 125, "tests": []}
                validate_report_artifacts(result.get("tests", []))
                result["output"] = text[-20000:]
                if returncode not in (0, 1) or not rows:
                    result["error"] = "Sandbox did not complete all tests."
                return result
            except subprocess.TimeoutExpired:
                return {"exit_code": 124, "tests": [], "error": "Sandbox exceeded the 90-second wall-clock limit.", "output": ""}
            except OutputLimitError:
                return {"exit_code": 125, "tests": [], "error": "Sandbox exceeded the 1 MB output limit.", "output": ""}
            except (OSError, ValueError, TypeError, AttributeError):
                return {"exit_code": 125, "tests": [], "error": "Sandbox returned no valid test report.", "output": ""}
            finally:
                # Killing the Docker client alone does not terminate a timed-out container.
                try:
                    subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    pass

    def validate(self, candidate):
        digest = package_hash(candidate)
        checks = static_checks(candidate)
        report = {"package_hash": digest, "passed": False, "status": "failed", "checks": checks,
                  "created_at": datetime.now(timezone.utc).isoformat()}
        if not all(row["passed"] for row in checks):
            return report
        capability = self.capability()
        report["sandbox"] = capability
        if not capability["ready"]:
            report.update(status="unavailable", output=capability["reason"])
            return report
        expected = len(candidate["tests"])
        expected_ids = {row["name"] for row in candidate["tests"]}
        def complete(result):
            rows = result.get("tests", [])
            return not result.get("error") and len(rows) == expected and {r.get("id") for r in rows} == expected_ids and all(r.get("outcome") in ("passed", "failed") for r in rows)
        results = {}
        for label, source in [("reference", candidate["solution"]), ("reference_repeat", candidate["solution"]), ("starter", candidate["starter"])]:
            result = results[label] = self.run(candidate, source)
            detail = f"{sum(bool(t.get('passed')) for t in result.get('tests', []))}/{expected} tests passed"
            if label == "starter":
                passed = complete(result) and result["exit_code"] == 1 and any(not t["passed"] for t in result["tests"])
            else:
                passed = complete(result) and result["exit_code"] == 0 and all(t["passed"] for t in result["tests"])
                if candidate.get("toy"):
                    toy = candidate["toy"]
                    provenance = {"kind": toy["kind"], "source": toy["source_url"],
                                  "locator": toy["locator"], "adaptation": toy["adaptation"]}
                    has_toy = any(
                        isinstance(value, dict) and value.get("kind") == "toy_experiment"
                        and value.get("provenance") == provenance
                        and any(value.get(key) for key in ("metrics", "loss_curve", "samples"))
                        for row in result.get("tests", []) if row.get("layer") == "T3"
                        for value in row.get("artifacts", {}).values())
                    passed = passed and has_toy
                    if not has_toy:
                        detail += "; missing a measured T3 toy artifact with the declared provenance"
            checks.append({"name": label, "passed": passed, "detail": detail})
        for mutant in candidate["mutants"]:
            result = results[mutant["name"]] = self.run(candidate, mutant["code"])
            killed = complete(result) and result["exit_code"] == 1 and any(t.get("outcome") == "failed" and t.get("error_type") == "AssertionError" for t in result["tests"])
            checks.append({"name": "mutant:" + mutant["name"], "passed": killed, "detail": mutant["bug"]})
        report.update(passed=all(c["passed"] for c in checks), runs=results)
        report["status"] = "passed" if report["passed"] else "failed"
        report["output"] = "\n".join(f"{name}: {result.get('error', str(result['exit_code']))}" for name, result in results.items())
        return report

    def try_candidate(self, candidate, code):
        """Run a pool attempt in the same offline container as published work.

        The caller loads a private, hash-bound validated package. This method
        never imports a generated module on the host. Raw pytest stdout and
        tracebacks may contain private test sources, so they never appear in
        the learner report returned here.
        """
        if not isinstance(code, str) or not code.strip() or len(code.encode("utf-8")) > 64000:
            raise ValueError("Code must be nonempty Python source under 64 KB")
        checks = static_checks(candidate)
        if not all(row["passed"] for row in checks):
            raise ValueError("Candidate failed structural validation")
        digest = package_hash(candidate)
        source_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
        result = self.run(candidate, code)
        tests = result.get("tests", [])
        expected = {row["name"] for row in candidate["tests"]}
        valid = (isinstance(tests, list) and all(isinstance(row, dict) for row in tests)
                 and len(tests) == len(expected) and {row.get("id") for row in tests} == expected
                 and all(row.get("outcome") in {"passed", "failed", "error"} for row in tests))
        try:
            validate_report_artifacts(tests if valid else [])
        except (ValueError, TypeError, AttributeError):
            valid = False
        if result.get("error") or result.get("exit_code") not in (0, 1) or not valid:
            return {"package_hash": digest, "source_sha256": source_hash, "passed": False, "tests": [],
                    "execution_completed": False,
                    "error": "The isolated runner could not complete this attempt. Please retry later.",
                    "status": "unavailable" if result.get("exit_code") == 125 else "error"}
        sanitized = []
        layers = {row["name"]: row["layer"] for row in candidate["tests"]}
        for row in tests:
            outcome = row["outcome"]
            passed = row.get("passed") is True and outcome == "passed"
            duration = row.get("duration_s", 0)
            duration = duration if isinstance(duration, (int, float)) and math.isfinite(duration) and 0 <= duration <= 90 else 0
            sanitized.append({"id": row["id"], "layer": layers[row["id"]], "passed": passed,
                              "outcome": outcome, "duration_s": duration,
                              "message": "" if passed else ("This case raised an error." if outcome == "error" else "This case did not match the required result."),
                              "artifacts": row.get("artifacts", {})})
        passed = result["exit_code"] == 0 and all(row["passed"] for row in sanitized)
        return {"package_hash": digest, "source_sha256": source_hash, "passed": passed,
                "tests": sanitized, "status": "accepted" if passed else "failed", "error": None,
                "execution_completed": True}


def run_published(entry, implementation, layers=None):
    """Judge a reviewed community package without running its tests on the host."""
    from paperleet.judge.runner import make_report, normalize_layers
    layers = normalize_layers(layers)
    manifest = json.loads((entry["directory"] / "candidate.json").read_text())
    if package_hash(manifest) != entry.get("package_hash"):
        raise ValueError("Published candidate has changed; human review and validation must be repeated")
    source = Path(implementation).read_text(encoding="utf-8")
    result = Sandbox().run(manifest, source, layers)
    records = result.get("tests", [])
    expected = {row["name"] for row in manifest["tests"] if row["layer"] in layers}
    error = result.get("error")
    if result["exit_code"] not in (0, 1) or len(records) != len(expected) or {r.get("id") for r in records} != expected:
        error = error or "Sandbox did not complete the selected tests."
    report = make_report(entry["id"], implementation, records, layers, error)
    report["source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
    return report
