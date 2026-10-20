"""Reviewable runtime adoption must preserve product source and unfinished work."""
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sync_stridetorch", ROOT / "scripts/sync_stridetorch.py")
sync = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync)


def commit(root):
    subprocess.run(["git", "-C", str(root), "add", "."], check=True, capture_output=True)
    subprocess.run(["git", "-c", "user.name=Runtime test", "-c", "user.email=runtime@example.test",
                    "-C", str(root), "commit", "-qm", "Runtime update"], check=True, capture_output=True)
    return sync.git(root, "rev-parse", "HEAD").decode().strip()


@pytest.fixture
def repositories(tmp_path):
    source = tmp_path / "runtime"
    source.mkdir()
    subprocess.run(["git", "-C", str(source), "init", "-q"], check=True)
    for name in ["LICENSE", "NOTICE", "pyproject.toml", "src/stridetorch/__init__.py", "tests/test_sample.py"]:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("initial runtime content\n")
    revision = commit(source)
    root = tmp_path / "product"
    root.mkdir()
    files = sync.snapshot(source, revision)
    for name, data in files.items():
        path = root / sync.PACKAGE / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (root / sync.LOCK).write_text(json.dumps({"version": 1, "repository": sync.REPOSITORY,
                                            "revision": revision,
                                            "files": {name: sync.digest(data) for name, data in files.items()}}))
    return source, root, revision


def test_standalone_change_updates_runtime_not_product(repositories):
    source, root, _ = repositories
    owned = root / "web/App.tsx"
    owned.parent.mkdir()
    owned.write_text("local product UI")
    (source / "src/stridetorch/__init__.py").write_text("reviewed runtime fix\n")
    unrelated = source / ".github/workflows/do-not-run.yml"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text("not part of a runtime adoption")
    revision = commit(source)
    assert sync.update(root, source, revision) == 1
    assert sync.check(root)["revision"] == revision
    assert owned.read_text() == "local product UI"
    assert not (root / sync.PACKAGE / ".github").exists()


def test_local_runtime_edits_stop_adoption(repositories):
    source, root, revision = repositories
    owned = root / sync.PACKAGE / "src/stridetorch/__init__.py"
    owned.write_text("unfinished local fix")
    with pytest.raises(ValueError, match="Local runtime changes"):
        sync.update(root, source, revision)
    assert owned.read_text() == "unfinished local fix"


def test_symlinks_and_mutable_revisions_are_rejected(repositories):
    source, root, _ = repositories
    (source / "src/stridetorch/escape.py").symlink_to("../../LICENSE")
    revision = commit(source)
    with pytest.raises(ValueError, match="regular non-executable"):
        sync.update(root, source, revision)
    with pytest.raises(ValueError, match="immutable"):
        sync.snapshot(source, "main")
    for name in ["../secret", "src/../secret", "src/..\\secret", "src/.git/config", "/src/secret"]:
        assert not sync.managed(name)


def test_package_rename_requires_product_integration(repositories):
    source, root, previous = repositories
    package = source / "src/stridetorch"
    package.rename(source / "src/torchwasm")
    revision = commit(source)
    with pytest.raises(ValueError, match="coordinated product integration"):
        sync.update(root, source, revision)
    assert sync.check(root)["revision"] == previous
    assert (root / sync.PACKAGE / "src/stridetorch/__init__.py").is_file()
    assert not (root / sync.PACKAGE / "src/torchwasm").exists()
