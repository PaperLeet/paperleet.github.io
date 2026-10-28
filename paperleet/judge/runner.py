"""Run the very same pytest suite for learners, references, and mutants."""
from datetime import datetime, timezone
import json
import base64
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from paperleet.catalog import problem, problems, root


LAYERS = ("T0", "T1", "T2", "T3")


def normalize_layers(layers=None):
    if layers is None:
        return list(LAYERS)
    if isinstance(layers, str):
        layers = layers.split(",")
    normalized = list(dict.fromkeys(str(layer).strip().upper() for layer in layers))
    if not normalized or any(layer not in LAYERS for layer in normalized):
        raise ValueError("Layers must be a nonempty selection of T0,T1,T2,T3")
    return normalized


def run_batch(plan, layers=None, timeout=600):
    layers = normalize_layers(layers)
    # Resolve against one batch-local snapshot: problem() reloads every YAML.
    # Keep its first-match alias order and refresh the snapshot on the next run.
    entries = problems() if plan else []
    batch_entries = {}
    for identifier in plan:
        entry = next((entry for entry in entries if identifier in
                      (entry["id"], entry.get("short_id"), str(entry["number"]))), None)
        if entry is None:
            raise ValueError(f"Unknown problem {identifier!r}. Run paperleet list.")
        if entry["id"].startswith("community/"):
            raise ValueError("Generated community packages must run in the submission sandbox. Use the web editor or paperleet test for a single implementation; batch CI/mutate currently supports maintained packages only.")
        batch_entries[identifier] = entry
    # Freeze every source before pytest begins. Saving a file during a run must
    # not make different tests judge different versions of the implementation.
    sources = {}
    frozen_plan = {}
    for identifier, variants in plan.items():
        frozen_plan[identifier] = []
        for variant in variants:
            path = str(Path(variant["path"]).resolve())
            if path not in sources:
                content = Path(path).read_bytes()
                sources[path] = {"source_b64": base64.b64encode(content).decode("ascii"),
                                 "source_sha256": hashlib.sha256(content).hexdigest()}
            frozen_plan[identifier].append({**variant, "path": path, **sources[path]})
    with tempfile.TemporaryDirectory(prefix="paperleet-") as directory:
        plan_path, result_path = Path(directory) / "plan.json", Path(directory) / "result.json"
        plan_path.write_text(json.dumps(frozen_plan))
        paths = [str(batch_entries[identifier]["directory"] / "tests") for identifier in plan]
        args = [sys.executable, "-m", "pytest", *paths, "-q", "--tb=short", "-p", "pytest_timeout",
                "--paperleet-plan", str(plan_path), "--paperleet-result", str(result_path),
                "-m", " or ".join(layer.lower() for layer in layers)]
        env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTHONHASHSEED="0", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        env["PYTHONPATH"] = str(root()) + os.pathsep + env.get("PYTHONPATH", "")
        def recover():
            if result_path.exists():
                return json.loads(result_path.read_text())
            journal = result_path.with_suffix(".jsonl")
            records = []
            if journal.exists():
                for line in journal.read_text().splitlines():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        break  # A crash can interrupt the very last append.
            return {"tests": records, "incomplete": True}
        try:
            process = subprocess.run(args, cwd=root(), env=env, text=True, capture_output=True, timeout=timeout)
            result = recover()
            output = (process.stdout + process.stderr)[-100000:]
            if result.get("incomplete") or process.returncode not in (0, 1):
                output = f"Judge process exited with code {process.returncode}; completed test results are preserved.\n" + output
            result.update(exit_code=process.returncode, output=output)
        except subprocess.TimeoutExpired as exc:
            result = recover()
            result.update(exit_code=124, incomplete=True, output=f"Judge exceeded {timeout}s wall-clock limit; completed test results are preserved.")
        return result


def make_report(identifier, implementation, records, layers=None, error=None):
    layers = normalize_layers(layers)
    summary = {layer: [sum(t["passed"] for t in records if t["layer"] == layer), sum(t["layer"] == layer for t in records)] for layer in LAYERS}
    tests = [{key: value for key, value in row.items() if key not in ("problem", "impl", "variant", "repeat")} for row in records]
    report = {"problem": identifier, "impl": str(implementation), "created_at": datetime.now(timezone.utc).isoformat(),
              "summary": summary, "tests": tests, "mutation": None, "layers": layers,
              "passed": bool(tests) and all(t["passed"] for t in tests) and not error}
    if records and records[0].get("source_sha256"):
        report["source_sha256"] = records[0]["source_sha256"]
    if error:
        report["error"] = error
    return report


def write_report(report, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(destination)


def run_problem(identifier, implementation, layers=None, report_path=None):
    entry = problem(identifier)
    implementation = Path(implementation).resolve()
    if not implementation.is_file():
        raise ValueError(f"Missing implementation {implementation}. Run paperleet init {entry['id']} first.")
    if entry["id"].startswith("community/"):
        from paperleet.submissions.sandbox import run_published
        report = run_published(entry, implementation, layers)
        if report_path:
            write_report(report, report_path)
        return report
    result = run_batch({entry["id"]: [{"path": str(implementation), "variant": "implementation"}]}, layers, timeout=120)
    error = result["output"] if result.get("incomplete") or result["exit_code"] not in (0, 1) or not result["tests"] else None
    report = make_report(entry["id"], implementation, result["tests"], layers, error)
    if report_path:
        write_report(report, report_path)
    return report
