"""Batch setup reads one live catalog without changing identifier resolution."""
import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from paperleet.judge import runner


@pytest.fixture
def batch_repo(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPERLEET_ROOT", str(tmp_path))
    scans = []
    catalog = runner.problems

    def scan():
        scans.append(runner.root())
        return catalog()

    def forbidden_lookup(*args):
        raise AssertionError("Batch setup must not reload the catalog per identifier")

    calls = []

    def pytest_process(args, **kwargs):
        plan_path = Path(args[args.index("--paperleet-plan") + 1])
        result_path = Path(args[args.index("--paperleet-result") + 1])
        calls.append({"args": args, "options": kwargs, "plan": json.loads(plan_path.read_text())})
        result_path.write_text(json.dumps({"tests": []}))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runner, "problems", scan)
    monkeypatch.setattr(runner, "problem", forbidden_lookup)
    monkeypatch.setattr(runner.subprocess, "run", pytest_process)
    return SimpleNamespace(root=tmp_path, scans=scans, calls=calls)


def package(root, identifier, number, **metadata):
    directory = root / "problems" / identifier
    (directory / "tests").mkdir(parents=True)
    (directory / "problem.yaml").write_text(yaml.safe_dump({
        "id": identifier, "number": number, **metadata,
    }))
    (directory / "solution.py").write_text("def value(): return 7\n")
    return directory


def variant(directory):
    return [{"path": str(directory / "solution.py"), "variant": "reference", "repeat": 0}]


def test_batch_reads_catalog_once_and_preserves_alias_paths_and_plan(batch_repo):
    a = package(batch_repo.root, "demo/a", 1)
    b = package(batch_repo.root, "demo/b", 2, short_id="B")
    c = package(batch_repo.root, "demo/c", 3)
    plan = {"demo/a": variant(a), "B": variant(b), "3": variant(c)}
    result = runner.run_batch(plan, layers="T2", timeout=17)
    assert result == {"tests": [], "exit_code": 0, "output": ""}
    assert batch_repo.scans == [batch_repo.root]
    call, = batch_repo.calls
    assert call["args"][3:6] == [str(p / "tests") for p in (a, b, c)]
    assert call["args"][-2:] == ["-m", "t2"]
    assert call["options"]["timeout"] == 17
    assert call["options"]["cwd"] == batch_repo.root
    assert list(call["plan"]) == list(plan), "Resolution must not rewrite implementation plan keys"
    for key, directory in zip(plan, (a, b, c)):
        frozen, = call["plan"][key]
        assert base64.b64decode(frozen["source_b64"]) == (directory / "solution.py").read_bytes()
        assert frozen["variant"] == "reference" and frozen["repeat"] == 0


def test_batch_keeps_catalog_first_match_when_aliases_collide(batch_repo):
    first = package(batch_repo.root, "z/first", 1, short_id="a/second")
    package(batch_repo.root, "a/second", 2, short_id="1")
    runner.run_batch({"a/second": variant(first), "1": variant(first)})
    assert batch_repo.scans == [batch_repo.root]
    assert batch_repo.calls[0]["args"][3:5] == [str(first / "tests")] * 2


@pytest.mark.parametrize("identifier", ["missing", "../demo/a", "99"])
def test_unknown_batch_identifier_rejected_before_sources_or_pytest(batch_repo, identifier):
    package(batch_repo.root, "demo/a", 1)
    with pytest.raises(ValueError) as caught:
        runner.run_batch({identifier: [{"path": "missing-implementation.py"}]})
    assert str(caught.value) == f"Unknown problem {identifier!r}. Run paperleet list."
    assert batch_repo.scans == [batch_repo.root]
    assert batch_repo.calls == []


@pytest.mark.parametrize("identifier", ["community/candidate", "C", "7"])
def test_community_alias_rejected_before_sources_or_pytest(batch_repo, identifier):
    package(batch_repo.root, "community/candidate", 7, short_id="C")
    with pytest.raises(ValueError, match="submission sandbox"):
        runner.run_batch({identifier: [{"path": "missing-implementation.py"}]})
    assert batch_repo.scans == [batch_repo.root]
    assert batch_repo.calls == []


@pytest.mark.parametrize("identifiers, message", [
    (["7", "missing"], "submission sandbox"),
    (["missing", "7"], "Unknown problem 'missing'"),
])
def test_rejections_follow_plan_order(batch_repo, identifiers, message):
    package(batch_repo.root, "community/candidate", 7)
    with pytest.raises(ValueError, match=message):
        runner.run_batch({identifier: [] for identifier in identifiers})
    assert batch_repo.calls == []


def test_later_batch_sees_new_packages_and_changed_aliases(batch_repo):
    first = package(batch_repo.root, "demo/first", 1, short_id="OLD")
    runner.run_batch({"OLD": variant(first)})
    second = package(batch_repo.root, "demo/second", 2, short_id="NEW")
    (first / "problem.yaml").write_text(yaml.safe_dump({
        "id": "demo/first", "number": 1, "short_id": "RENAMED",
    }))
    runner.run_batch({"NEW": variant(second), "RENAMED": variant(first)})
    assert batch_repo.scans == [batch_repo.root] * 2
    assert batch_repo.calls[1]["args"][3:5] == [str(p / "tests") for p in (second, first)]
    with pytest.raises(ValueError, match="Unknown problem 'OLD'"):
        runner.run_batch({"OLD": variant(first)})
    assert batch_repo.scans == [batch_repo.root] * 3
    assert len(batch_repo.calls) == 2


def test_batch_snapshot_follows_changed_root(batch_repo, tmp_path, monkeypatch):
    first = package(batch_repo.root, "demo/a", 1)
    runner.run_batch({"demo/a": variant(first)})
    other_root = tmp_path / "other"
    second = package(other_root, "demo/a", 1)
    monkeypatch.setenv("PAPERLEET_ROOT", str(other_root))
    runner.run_batch({"demo/a": variant(second)})
    assert batch_repo.scans == [batch_repo.root, other_root]
    assert batch_repo.calls[1]["args"][3] == str(second / "tests")
    assert batch_repo.calls[1]["options"]["cwd"] == other_root


def test_empty_plan_keeps_existing_no_catalog_lookup_behavior(batch_repo):
    runner.run_batch({})
    assert batch_repo.scans == []
    assert batch_repo.calls[0]["plan"] == {}
