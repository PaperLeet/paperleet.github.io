"""Static, public compatibility bundles for validated CPU/WASM browser modules."""
import hashlib
import json
from pathlib import Path

from paperleet.catalog import root

PILOTS = {
    "diffusion/schedules", "architecture/attention", "finetuning/lora",
    "diffusion/param_conversion", "architecture/layer_norm",
    "architecture/positional_encoding", "rl/dpo",
    "architecture/multihead_attention", "architecture/positionwise_ffn", "optimization/adamw",
    "rl/gae", "rl/ppo_clip", "rl/grpo", "rl/dr_grpo",
    "rl/dapo", "rl/gspo", "representation/clip", "architecture/situ_glu",
    "finetuning/qlora", "architecture/block_attention_residual",
    "architecture/differential_attention", "representation/vicreg",
}

SHARED_FILES = (
    "paperleet/__init__.py", "paperleet/common/__init__.py",
    "paperleet/common/asserts.py", "paperleet/common/comparisons.py",
)


def _write_bundle(output, prefix, bundle):
    raw = json.dumps(bundle, ensure_ascii=False, separators=(",", ":"))
    name = f"browser-judge/{prefix}-{hashlib.sha256(raw.encode()).hexdigest()[:20]}.json"
    path = output / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(raw)
    return name


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
    files["paperleet/common/comparisons.py"] = (package / "common/comparisons.py").read_text()
    if entry["id"] in {"rl/gae", "rl/ppo_clip", "rl/grpo", "rl/dr_grpo"}:
        files["paperleet/common/rl_envs.py"] = (package / "common/rl_envs.py").read_text()
    if entry["id"] == "rl/dr_grpo":
        for name in ("solution.py", "experiment.py"):
            relative_name = f"problems/rl/grpo/{name}"
            files[relative_name] = (repository / relative_name).read_text()
    for name in list(files):
        parts = name.split("/")[:-1]
        for length in range(1, len(parts) + 1):
            files.setdefault("/".join(parts[:length]) + "/__init__.py", "")
    browser = package / "browser"
    schedules = entry["id"] == "diffusion/schedules"
    backend = "numpy-wasm" if schedules else "numpy-autograd-wasm"
    runtime = {
        "version": 1, "backend": backend,
        "files": {name: files.pop(name) for name in SHARED_FILES},
        "adapter": (browser / ("schedules_numpy.py" if schedules else "numpy_autograd.py")).read_text(),
        "harness": (browser / "harness.py").read_text(),
    }
    runtime_url = _write_bundle(output, "runtime", runtime)
    bundle = {"version": 2, "problem": entry["id"], "tests": entry["tests"],
              "test_file": f"{relative}/tests/{tests[0].name}", "files": files,
              "reference": (directory / "solution.py").read_text(),
              "mutants": {p.stem: p.read_text() for p in sorted((directory / "mutants").glob("*.py"))},
              "runtime_bundle_url": runtime_url}
    return {"bundle_url": _write_bundle(output, str(entry['number']), bundle), "backend": backend}
