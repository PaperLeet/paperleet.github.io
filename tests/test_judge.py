"""Framework integration tests use a temporary standalone problem registry."""
import json
from pathlib import Path
import pytest
import yaml

from paperleet.catalog import problem, problems
from paperleet.judge.runner import normalize_layers, run_problem
from paperleet.judge.mutation import mutate
from paperleet.judge.loader import load_implementation


@pytest.fixture
def mini_repo(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPERLEET_ROOT", str(tmp_path))
    folder = tmp_path / "problems" / "demo" / "addition"
    (folder / "tests").mkdir(parents=True)
    (folder / "mutants").mkdir()
    (folder / "viz").mkdir()
    (tmp_path / "conftest.py").write_text("from paperleet.judge.pytest_plugin import *\n")
    (tmp_path / "pytest.ini").write_text("[pytest]\nmarkers =\n    t0: contract\n    t2: property\n")
    (folder / "problem.yaml").write_text(yaml.safe_dump({"id": "demo/addition", "number": 1, "short_id": "X1", "difficulty": "E", "title": "Addition", "mutants": [{"id": "wrong_sign", "bug": "subtracts"}], "tests": {"T0": 1, "T2": 1}}))
    (folder / "solution.py").write_text("def add(a, b):\n    return a + b\n")
    (folder / "mutants" / "wrong_sign.py").write_text("# Bug: subtracts the second operand\ndef add(a, b):\n    return a - b\n")
    (folder / "starter.py").write_text("def add(a, b):\n    raise NotImplementedError\n")
    (folder / "notes.md").write_text("Check the sign.")
    (folder / "README.md").write_text("Add two scalars.")
    (folder / "viz" / "gen.py").write_text("def generate():\n    return {}\n")
    (folder / "tests" / "test_add.py").write_text('''import pytest
@pytest.mark.t0
def test_zero(impl):
    """Adding zero preserves the value."""
    assert impl.add(2, 0) == 2
@pytest.mark.t2
def test_positive(impl, artifact):
    """Adding positive operands increases the total."""
    result = impl.add(2, 3)
    artifact("comparison", {"expected": 5, "got": result})
    assert result == 5, f"at scalar: expected 5, got {result}"
''')
    return folder


def test_path_discovery_and_traversal_rejection(mini_repo):
    assert len(problems()) == 1
    assert problem("X1")["id"] == "demo/addition"
    with pytest.raises(ValueError, match="Unknown problem"):
        problem("../../etc/passwd")


def test_judge_records_real_failure_and_artifacts(mini_repo, tmp_path):
    destination = tmp_path / "report.json"
    report = run_problem("X1", mini_repo / "mutants/wrong_sign.py", report_path=destination)
    assert not report["passed"]
    assert report["summary"]["T0"] == [1, 1]
    assert report["summary"]["T2"] == [0, 1]
    failure = next(test for test in report["tests"] if not test["passed"])
    assert failure["why"] == "Adding positive operands increases the total."
    assert "expected 5, got -1" in failure["message"]
    assert failure["artifacts"]["comparison"] == {"expected": 5, "got": -1}
    assert json.loads(destination.read_text()) == report


def test_layer_selection(mini_repo):
    report = run_problem("X1", mini_repo / "mutants/wrong_sign.py", layers=["T0"])
    assert report["passed"] and report["summary"]["T2"] == [0, 0]
    assert report["layers"] == ["T0"]
    with pytest.raises(ValueError):
        normalize_layers("T0,T9")


def test_native_report_preserves_structured_numerical_and_shape_failures(mini_repo):
    (mini_repo / 'tests/test_add.py').write_text('''import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
@pytest.mark.t0
def test_shape(impl):
    close(torch.zeros(2), torch.zeros(2,1))
@pytest.mark.t2
def test_values(impl):
    close(impl.add(torch.tensor([2.,3.]),torch.ones(2)),torch.tensor([3.,4.]))
''')
    report=run_problem('X1',mini_repo/'mutants/wrong_sign.py')
    assert not report['passed']
    assert report['tests'][0]['comparison']=={'kind':'shape','expected_shape':[2,1],'actual_shape':[2]}
    assert report['tests'][1]['comparison']['index']==[0]
    assert report['tests'][1]['comparison']['expected']==3
    assert report['tests'][1]['comparison']['actual']==1
    assert 'max abs diff' in report['tests'][1]['message']


def test_mutation_checks_same_suite(mini_repo):
    report = mutate(problem("X1"))
    assert report["passed"], report
    assert report["mutation"] == {"wrong_sign": ["test_positive"]}


def test_collection_failure_cannot_pass(mini_repo):
    (mini_repo / "tests/test_add.py").write_text("raise RuntimeError('broken collection')\n")
    report = run_problem("X1", mini_repo / "solution.py")
    assert not report["passed"]
    assert "broken collection" in report["error"]


def test_invalid_implementation_does_not_count_as_mutant_kill(mini_repo):
    (mini_repo / "mutants/wrong_sign.py").write_text("# Bug: invalid module\nraise RuntimeError('cannot import')\n")
    report = mutate(problem("X1"))
    assert not report["passed"]
    assert report["mutation"] == {"wrong_sign": []}


def test_same_second_equal_length_edits_always_load_current_source(tmp_path):
    import os
    path = tmp_path / "learner.py"
    path.write_text("def value(): return 1\n")
    os.utime(path, (1800000000.1, 1800000000.1))
    assert load_implementation(path).value() == 1
    path.write_text("def value(): return 2\n")
    os.utime(path, (1800000000.2, 1800000000.2))
    assert load_implementation(path).value() == 2


def test_one_run_uses_a_frozen_source_snapshot(mini_repo):
    source = mini_repo / "solution.py"
    (mini_repo / "tests/test_add.py").write_text(f'''import pytest
from pathlib import Path
@pytest.mark.t0
def test_before_edit(impl):
    assert impl.add(2, 3) == 5
    Path({str(source)!r}).write_text("def add(a, b): return a - b\\n")
@pytest.mark.t2
def test_after_edit(impl):
    assert impl.add(2, 3) == 5
''')
    report = run_problem("X1", source)
    assert report["passed"], report
    assert len(report["source_sha256"]) == 64
    assert load_implementation(source).add(2, 3) == -1


def test_process_crash_keeps_completed_tests_and_artifacts(mini_repo):
    (mini_repo / "tests/test_add.py").write_text('''import os
import pytest
@pytest.mark.t0
def test_completed(impl, artifact):
    artifact("got", impl.add(2, 3))
    assert impl.add(2, 3) == 5
@pytest.mark.t2
def test_native_crash(impl):
    os._exit(3)
''')
    report = run_problem("X1", mini_repo / "solution.py")
    assert not report["passed"]
    assert "exited with code 3" in report["error"]
    assert report["summary"]["T0"] == [1, 1]
    assert report["tests"][0]["artifacts"] == {"got": 5}
