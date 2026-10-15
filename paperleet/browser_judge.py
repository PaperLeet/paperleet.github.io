"""Static, public compatibility bundles for validated CPU/WASM browser modules."""
import hashlib
import json
from pathlib import Path

from paperleet.catalog import root

from paperleet.browser_config import BACKENDS, browser_config
from paperleet._vendor.torch.runtime import source_files

SHARED_FILES = (
    "paperleet/__init__.py", "paperleet/common/__init__.py",
    "paperleet/common/asserts.py", "paperleet/common/comparisons.py",
)


def require_browser_catalog(entries):
    """Publication requires browser support for every discoverable module."""
    missing = sorted(entry["id"] for entry in entries if browser_config(entry) is None)
    if missing:
        raise ValueError("Browser support is required before publishing: " + ", ".join(missing)
                         + ". Adapt the CPU/WASM runtime, export a browser bundle, and add full-suite/reference/mutant validation; see docs/browser-judge.md.")


def _write_bundle(output, prefix, bundle):
    raw = json.dumps(bundle, ensure_ascii=False, separators=(",", ":"))
    name = f"browser-judge/{prefix}-{hashlib.sha256(raw.encode()).hexdigest()[:20]}.json"
    path = output / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(raw)
    return name


def export_browser_bundle(entry, output):
    config = browser_config(entry)
    if config is None:
        return None
    repository, directory = root(), entry["directory"]
    relative = directory.relative_to(repository).as_posix()
    tests = sorted((directory / "tests").glob("test_*.py"))
    files = {f"{relative}/tests/{test.name}": test.read_text() for test in tests}
    experiment = directory / "experiment.py"
    if experiment.exists():
        files[f"{relative}/experiment.py"] = experiment.read_text()
    toy = directory / 'toy.py'
    if toy.exists():
        files[f"{relative}/toy.py"] = toy.read_text()
    package = Path(__file__).resolve().parent
    files["paperleet/common/asserts.py"] = (package / "common/asserts.py").read_text()
    files["paperleet/common/comparisons.py"] = (package / "common/comparisons.py").read_text()
    for name in config['resources']:
        # Common helpers belong to the installed runtime package; problem data
        # belongs to PAPERLEET_ROOT, which may deliberately link a catalog tree.
        boundary = package / 'common' if name.startswith('paperleet/common/') else repository / 'problems'
        path = boundary / name.removeprefix('paperleet/common/' if name.startswith('paperleet/common/') else 'problems/')
        if not path.resolve().is_relative_to(boundary.resolve()):
            raise ValueError(f"{entry['id']}: browser resource escapes the repository: {name}")
        if name in files: raise ValueError(f"{entry['id']}: duplicate browser resource: {name}")
        files[name] = path.read_text(encoding='utf-8')
    for name in list(files):
        parts = name.split("/")[:-1]
        for length in range(1, len(parts) + 1):
            files.setdefault("/".join(parts[:length]) + "/__init__.py", "")
    browser = package / "browser"
    backend = config["backend"]
    runtime = {
        "version": 1, "backend": backend,
        "files": {**{name: files.pop(name) for name in SHARED_FILES},
                  **source_files('schedules' if backend == 'numpy-wasm' else 'autograd')},
        "adapter": (browser / BACKENDS[backend]).read_text(),
        "harness": (browser / "harness.py").read_text(),
    }
    runtime_url = _write_bundle(output, "runtime", runtime)
    if not tests: raise ValueError(f"{entry['id']}: browser suites require at least one test file")
    bundle = {"version": 3, "problem": entry["id"], "tests": entry["tests"],
              "test_file": f"{relative}/tests/{tests[0].name}",
              "test_files": [f"{relative}/tests/{test.name}" for test in tests],
              "packages": config["packages"], "budgets": config["budgets"], "files": files,
              "reference": (directory / "solution.py").read_text(),
              "mutants": {p.stem: p.read_text() for p in sorted((directory / "mutants").glob("*.py"))},
              "runtime_bundle_url": runtime_url}
    return {"bundle_url": _write_bundle(output, str(entry['number']), bundle), "backend": backend, "budgets": config["budgets"]}
