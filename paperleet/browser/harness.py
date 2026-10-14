"""Run the unchanged pilot pytest suites in a browser Python runtime."""
import contextlib
from datetime import datetime, timezone
import hashlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import sys
import types
import warnings

import numpy as np
import pytest


def _jsonable(value):
    if isinstance(value, getattr(sys.modules.get("torch"), "Tensor", ())):
        value = value.detach().tolist()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


class Recorder:
    def __init__(self, source, reference):
        self.source, self.reference, self.rows = source, reference, {}

    @pytest.fixture(autouse=True)
    def deterministic(self):
        np.random.seed(42)

    @pytest.fixture
    def impl(self):
        module = types.ModuleType("browser_learner")
        exec(compile(self.source, "submission.py", "exec"), module.__dict__)
        return module

    @pytest.fixture
    def ref(self):
        module = types.ModuleType("browser_reference")
        exec(compile(self.reference, "reference.py", "exec"), module.__dict__)
        return module

    @pytest.fixture
    def artifact(self, request):
        request.node.browser_artifacts = {}
        def save(name, data):
            request.node.browser_artifacts[name] = _jsonable(data)
        return save

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item, call):
        result = yield
        report = result.get_result()
        row = self.rows.setdefault(item.nodeid, {
            "id": item.originalname or item.name, "nodeid": item.nodeid,
            "layer": next((f"T{i}" for i in range(4) if item.get_closest_marker(f"t{i}")), "T0"),
            "passed": True, "outcome": "passed", "duration_s": 0,
            "why": (inspect.getdoc(item.obj) or "Implementation contract").splitlines()[0], "message": "", "artifacts": {},
        })
        row["duration_s"] += report.duration
        if report.failed or report.skipped:
            row["passed"] = False
            row["outcome"] = "failed" if report.when == "call" and report.failed else "error"
            row["message"] += str(report.longrepr)[-4000:]
            # Imported after prepare has written the bundle's package files.
            from paperleet.common.comparisons import comparison_from_exception
            comparison = comparison_from_exception(call.excinfo.value) if call.excinfo else None
            if comparison is not None: row["comparison"] = comparison
        row["artifacts"] = getattr(item, "browser_artifacts", {})


def prepare(bundle, adapter_source, directory):
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    for name, source in bundle["files"].items():
        target = folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8")
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    namespace = {"__name__": "browser_numpy"}
    exec(compile(adapter_source, "browser_numpy.py", "exec"), namespace)
    sys.modules["torch"] = namespace["install"]()
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    return folder


def run_suite(bundle, source, directory):
    recorder = Recorder(source, bundle["reference"])
    # A bounded sink avoids unbounded pytest/learner output in the worker.
    class Sink(io.TextIOBase):
        def __init__(self): self.text = ""
        def write(self, text):
            self.text = (self.text + text)[-5000:]
            return len(text)
        def flush(self): pass
    sink = Sink()
    args = [str(Path(directory) / bundle["test_file"]), "-q", "-s", "--tb=short", "--noconftest", "-p", "no:cacheprovider"]
    args.extend(["-o", "markers=" + "\n".join(f"t{i}: layer" for i in range(4))])
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        code = pytest.main(args, plugins=[recorder])
    rows = list(recorder.rows.values())
    expected = bundle["tests"]
    actual = {layer: sum(row["layer"] == layer for row in rows) for layer in expected}
    complete = actual == expected and int(code) in (0, 1)
    summary = {f"T{i}": [sum(row["passed"] for row in rows if row["layer"] == f"T{i}"), sum(row["layer"] == f"T{i}" for row in rows)] for i in range(4)}
    result = {"problem": bundle["problem"], "impl": "browser/submission.py", "summary": summary, "tests": rows,
              "runtime": "browser-numpy-wasm", "passed": complete and bool(rows) and all(row["passed"] for row in rows),
              "source_sha256": hashlib.sha256(source.encode()).hexdigest(), "mutation": None,
              "created_at": datetime.now(timezone.utc).isoformat()}
    if not complete:
        result["error"] = "Browser test collection did not match the maintained suite: " + sink.text
    return result


def calibrate(bundle, directory):
    reference = run_suite(bundle, bundle["reference"], directory)
    if not reference["passed"]:
        return {"compatible": False, "reason": "reference_failed", "reference": reference}
    caught = {}
    for name, source in bundle["mutants"].items():
        report = run_suite(bundle, source, directory)
        if any("BrowserUnsupported" in row["message"] for row in report["tests"]):
            return {"compatible": False, "reason": "mutant_api_unsupported", "mutant": name}
        failures = [row["id"] for row in report["tests"] if row["outcome"] == "failed"]
        if not failures:
            return {"compatible": False, "reason": "mutant_not_caught", "mutant": name, "report": report}
        caught[name] = failures
    return {"compatible": True, "reference_passed": len(reference["tests"]), "mutants_caught": caught}
