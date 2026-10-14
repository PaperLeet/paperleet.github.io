import inspect
import base64
import json
import math
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from paperleet.catalog import root
from paperleet.judge.loader import load_implementation
from paperleet.common.comparisons import comparison_from_exception


def pytest_addoption(parser):
    group = parser.getgroup("paperleet")
    group.addoption("--impl", default=None, help="Path to the learner implementation")
    group.addoption("--paperleet-plan", default=None, help="Internal batch implementation plan")
    group.addoption("--paperleet-result", default=None, help="Internal JSON result destination")


def _problem_dir(path):
    for parent in Path(path).resolve().parents:
        if (parent / "problem.yaml").is_file():
            return parent
    return None


def pytest_configure(config):
    config._paperleet_records = {}
    config._paperleet_errors = []
    path = config.getoption("--paperleet-plan")
    config._paperleet_plan = json.loads(Path(path).read_text()) if path else {}


def pytest_generate_tests(metafunc):
    directory = _problem_dir(metafunc.definition.path)
    if directory is None or "impl" not in metafunc.fixturenames:
        return
    identifier = directory.relative_to(root() / "problems").as_posix()
    plan = metafunc.config._paperleet_plan.get(identifier)
    if plan:
        metafunc.parametrize("impl", plan, indirect=True, ids=[f"{p['variant']}@{p.get('repeat', 0)}" for p in plan])


def pytest_collection_modifyitems(config, items):
    for item in items:
        layer = _layer(item)
        if _problem_dir(item.path) is not None:
            item.add_marker(pytest.mark.timeout(15 if layer == "T3" else 5))


def _layer(item):
    return next((f"T{i}" for i in range(4) if item.get_closest_marker(f"t{i}")), "T0")


@pytest.fixture(autouse=True)
def deterministic():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)


@pytest.fixture
def impl(request):
    directory = _problem_dir(request.node.path)
    if directory is None:
        raise RuntimeError("impl fixture must be used inside a problem package")
    param = getattr(request, "param", None)
    path = param["path"] if param else request.config.getoption("--impl")
    if not path:
        identifier = directory.relative_to(root() / "problems")
        path = root() / "workspace" / identifier.with_suffix(".py")
    source = base64.b64decode(param["source_b64"]) if param and "source_b64" in param else None
    return load_implementation(path, source=source)


@pytest.fixture
def ref(request):
    directory = _problem_dir(request.node.path)
    return load_implementation(directory / "solution.py")


def jsonable(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.ndim >= 2 and len(value) > 2000:
            value = value[np.linspace(0, len(value) - 1, 2000, dtype=int)]
        value = value.tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    return value


@pytest.fixture
def artifact(request):
    request.node._paperleet_artifacts = {}

    def save(name, data):
        request.node._paperleet_artifacts[name] = jsonable(data)
    return save


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    directory = _problem_dir(item.path)
    if directory is None:
        return
    params = getattr(getattr(item, "callspec", None), "params", {}).get("impl", {})
    identifier = directory.relative_to(root() / "problems").as_posix()
    records = item.config._paperleet_records
    row = records.setdefault(item.nodeid, {
        "id": item.originalname or item.name, "nodeid": item.nodeid,
        "problem": identifier, "layer": _layer(item), "passed": True,
        "why": (inspect.getdoc(item.obj) or "Test the declared implementation contract.").splitlines()[0],
        "message": "", "duration_s": 0., "artifacts": {},
        "impl": params.get("path", item.config.getoption("--impl") or str(root() / "workspace" / (identifier + ".py"))),
        "variant": params.get("variant", "implementation"), "repeat": params.get("repeat", 0),
        "source_sha256": params.get("source_sha256"),
        "outcome": "passed",
    })
    row["duration_s"] += report.duration
    if report.failed or report.skipped:
        row["passed"] = False
        row["outcome"] = "failed" if report.failed and report.when == "call" else ("skipped" if report.skipped else "error")
        row["message"] += str(report.longrepr)[-16000:]
        comparison = comparison_from_exception(call.excinfo.value) if call.excinfo else None
        if comparison is not None: row["comparison"] = comparison
    row["artifacts"] = getattr(item, "_paperleet_artifacts", {})
    destination = item.config.getoption("--paperleet-result")
    if destination and report.when == "teardown":
        with Path(destination).with_suffix(".jsonl").open("a", encoding="utf-8") as journal:
            journal.write(json.dumps(row, ensure_ascii=False) + "\n")


def pytest_collectreport(report):
    # Collection failures are also represented by pytest's nonzero process exit.
    pass


def pytest_sessionfinish(session, exitstatus):
    destination = session.config.getoption("--paperleet-result")
    if destination:
        Path(destination).write_text(json.dumps({"exit_code": int(exitstatus), "tests": list(session.config._paperleet_records.values())}, ensure_ascii=False), encoding="utf-8")
