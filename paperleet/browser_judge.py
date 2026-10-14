"""Static, public compatibility bundles for the three browser-judge pilots."""
import hashlib
import json
from pathlib import Path

from paperleet.catalog import root

PILOTS = {"diffusion/schedules", "architecture/attention", "finetuning/lora"}


def export_browser_bundle(entry, output):
    if entry["id"] not in PILOTS:
        return None
    repository, directory = root(), entry["directory"]
    relative = directory.relative_to(repository).as_posix()
    tests = sorted((directory / "tests").glob("test_*.py"))
    files = {f"{relative}/tests/{test.name}": test.read_text() for test in tests}
    experiment = directory / "experiment.py"
    if experiment.exists():
        files[f"{relative}/experiment.py"] = experiment.read_text()
    package = Path(__file__).resolve().parent
    files["paperleet/common/asserts.py"] = (package / "common/asserts.py").read_text()
    for name in list(files):
        parts = name.split("/")[:-1]
        for length in range(1, len(parts) + 1):
            files.setdefault("/".join(parts[:length]) + "/__init__.py", "")
    browser = package / "browser"
    bundle = {"version": 1, "problem": entry["id"], "tests": entry["tests"],
              "test_file": f"{relative}/tests/{tests[0].name}", "files": files,
              "reference": (directory / "solution.py").read_text(),
              "mutants": {p.stem: p.read_text() for p in sorted((directory / "mutants").glob("*.py"))},
              "adapter": (browser / ("schedules_numpy.py" if entry["id"] == "diffusion/schedules" else "numpy_autograd.py")).read_text(),
              "harness": (browser / "harness.py").read_text()}
    raw = json.dumps(bundle, ensure_ascii=False, separators=(",", ":"))
    name = f"browser-judge/{entry['number']}-{hashlib.sha256(raw.encode()).hexdigest()[:20]}.json"
    path = output / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(raw)
    return {"bundle_url": name, "backend": "numpy-wasm" if entry["id"] == "diffusion/schedules" else "numpy-autograd-wasm"}
