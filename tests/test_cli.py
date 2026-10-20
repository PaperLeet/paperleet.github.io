"""The CLI keeps listing, workspace setup and argument errors predictable."""
import json

import pytest
import yaml

from paperleet.cli import main


def package(tmp_path, canonical, number, **metadata):
    directory = tmp_path / "problems" / canonical
    directory.mkdir(parents=True)
    (directory / "problem.yaml").write_text(yaml.safe_dump({
        "id": canonical, "number": number, "title": "题目", "difficulty": "E", "tests": {"T0": 1, "T1": 2}, **metadata,
    }))
    (directory / "starter.py").write_text("def f():\n    raise NotImplementedError\n")
    return directory


def report(tmp_path, canonical, content):
    path = tmp_path / "reports" / (canonical.replace("/", "-") + ".json")
    path.parent.mkdir(exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content if isinstance(content, str) else json.dumps(content))


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPERLEET_ROOT", str(tmp_path))
    return tmp_path


def listing(capsys):
    assert main(["list"]) == 0
    return capsys.readouterr().out.splitlines()


def test_list_header_names_the_printed_columns_and_aligns_long_ids(catalog, capsys):
    package(catalog, "demo/a", 1)
    package(catalog, "architecture/a_problem_identifier_longer_than_thirty", 2)
    header, *rows = listing(capsys)
    assert header.split()[:2] == ["No.", "Problem"]
    level = header.index("Level")
    assert [row[level:].split()[0] for row in rows] == ["E", "E"], "Level stays in one column after the longest id"
    assert [row.split()[:2] for row in rows] == [["1", "demo/a"], ["2", "architecture/a_problem_identifier_longer_than_thirty"]]


def test_list_status_follows_local_reports(catalog, capsys):
    for number, name in enumerate(["todo", "failed", "partial", "accepted"], 1):
        package(catalog, f"demo/{name}", number)
    report(catalog, "demo/failed", {"passed": False, "summary": {"T0": [0, 1], "T1": [0, 2]}})
    report(catalog, "demo/partial", {"passed": True, "summary": {"T0": [1, 1]}})
    report(catalog, "demo/accepted", {"passed": True, "summary": {"T0": [1, 1], "T1": [2, 2]}})
    statuses = [row.split()[-1] for row in listing(capsys)[1:]]
    assert statuses == ["Todo", "Attempted", "Attempted", "Accepted"], "Layer subsets never count as accepted"


def test_list_accepts_reports_without_layers_that_require_no_tests(catalog, capsys):
    package(catalog, "demo/a", 1, tests={"T0": 1, "T3": 0})
    report(catalog, "demo/a", {"passed": True, "summary": {"T0": [1, 1]}})
    assert listing(capsys)[1].split()[-1] == "Accepted"


@pytest.mark.parametrize("content", [
    "{not json", "[]", "", '"text"', b"\xff\xfe{}",
    '{"passed": true, "summary": []}', '{"passed": true, "summary": {"T0": 5, "T1": [2, 2]}}', '{"passed": true, "summary": {"T0": [], "T1": [2, 2]}}',
])
def test_list_survives_unreadable_reports(catalog, capsys, content):
    package(catalog, "demo/broken", 1)
    package(catalog, "demo/after", 2)
    report(catalog, "demo/broken", content)
    statuses = [row.split()[-1] for row in listing(capsys)[1:]]
    assert statuses == ["Attempted", "Todo"], "One damaged report must not hide the rest of the catalog"


def test_init_copies_starter_and_requires_force_to_replace(catalog, capsys):
    package(catalog, "demo/a", 1)
    destination = catalog / "workspace" / "demo" / "a.py"
    assert main(["init", "1"]) == 0
    assert capsys.readouterr().out.strip() == str(destination)
    destination.write_text("my draft\n")
    assert main(["init", "demo/a"]) == 2
    assert "Use --force" in capsys.readouterr().err
    assert destination.read_text() == "my draft\n", "Existing work is kept without --force"
    assert main(["init", "demo/a", "--force"]) == 0
    assert "NotImplementedError" in destination.read_text()


@pytest.mark.parametrize("command", [["init", "missing"], ["test", "missing"], ["mutate", "missing"], ["ci", "--problem", "missing"]])
def test_unknown_problem_exits_with_usage_error(catalog, capsys, command):
    package(catalog, "demo/a", 1)
    assert main(command) == 2
    assert "Unknown problem 'missing'" in capsys.readouterr().err


@pytest.mark.parametrize("arguments", [["--repeats", "0"], ["--budget", "0"], ["--budget", "-1"]])
def test_ci_rejects_non_positive_limits(catalog, capsys, arguments):
    assert main(["ci", *arguments]) == 2
    assert "must be positive" in capsys.readouterr().err
