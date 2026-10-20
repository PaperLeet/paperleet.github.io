"""Public numbering survives filesystem order and keeps old workspace aliases usable."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from paperleet.catalog import problem, problems
from paperleet.export import export_catalog, vizdata, web_readme
from paperleet.judge.runner import run_batch
from paperleet.paper_catalog import paper_catalog, module_scope, source_key


def package(tmp_path, canonical, number, **metadata):
    directory = tmp_path / "problems" / canonical
    directory.mkdir(parents=True)
    (directory / "problem.yaml").write_text(yaml.safe_dump({
        "id": canonical, "number": number, "title": "题目", **metadata,
    }))
    return directory


def test_numbering_is_explicit_and_legacy_aliases_still_resolve(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPERLEET_ROOT", str(tmp_path))
    package(tmp_path, "rl/z", 1, short_id="R1")
    package(tmp_path, "diffusion/a", 12, short_id="D1")
    assert [p["number"] for p in problems()] == [1, 12]
    assert problem("1")["id"] == problem("R1")["id"] == "rl/z"
    assert problem("12")["id"] == problem("diffusion/a")["id"]
    package(tmp_path, "community/a", 13)
    assert [p["number"] for p in problems()] == [1, 12, 13]


@pytest.mark.parametrize("number", [True, 0, -1, "2", 2.5, None])
def test_invalid_public_number_is_rejected(tmp_path, monkeypatch, number):
    monkeypatch.setenv("PAPERLEET_ROOT", str(tmp_path))
    package(tmp_path, "demo/a", number)
    with pytest.raises(ValueError, match="positive integer"):
        problems()


def test_duplicate_number_cannot_enter_catalog(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPERLEET_ROOT", str(tmp_path))
    package(tmp_path, "demo/a", 2)
    package(tmp_path, "demo/b", 2)
    with pytest.raises(ValueError, match="duplicate problem number"):
        export_catalog()


def test_export_includes_translations_and_browser_run_instructions(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPERLEET_ROOT", str(tmp_path))
    folder = package(tmp_path, "rl/a", 1, short_id="R1", title_en="Example")
    (folder / "README.md").write_text("R1 的约定。论文 R1-Zero。\n\n## 6. 怎么运行\n\n```bash\npaperleet init rl/a\n```\n\n## 7. 延伸\n保持公式 $x^2$。\n")
    (folder / "README.en.md").write_text("English $x^2$ statement.\n\n## 6. Run\nOld CLI instructions.\n\n## 7. Further reading\nReferences.\n")
    (folder / "notes.en.md").write_text("Explain the invariant.")
    path = export_catalog()
    exported = json.loads(path.read_text())["problems"][0]
    assert exported["number"] == 1
    assert "1 的约定" in exported["readme"]
    assert "R1-Zero" in exported["readme"]
    assert "paperleet init" not in exported["readme"]
    assert "保持公式 $x^2$" in exported["readme"]
    assert exported["translations"]["en"]["title"] == "Example"
    assert "Run tests" in exported["translations"]["en"]["readme"]
    assert exported["translations"]["en"]["notes"] == "Explain the invariant."
    assert "paperleet init" in (folder / "README.md").read_text()


def test_catalog_exports_without_loading_judge_dependencies(tmp_path):
    package(tmp_path, "demo/a", 1, title_en="Example")
    # Run outside pytest: its own plugin imports the judge dependencies already.
    result = subprocess.run(
        [sys.executable, "-c", """
import importlib.abc
import json
import sys

class NoJudgeImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'pytest'} or fullname == 'paperleet.judge.pytest_plugin':
            raise AssertionError(f'Catalog export must not import {fullname}')

sys.meta_path.insert(0, NoJudgeImports())
from paperleet.export import export_catalog
catalog = json.loads(export_catalog().read_text())
assert catalog['problems'][0]['number'] == 1
assert catalog['problems'][0]['translations']['en']['title'] == 'Example'
assert not {'torch', 'numpy', 'pytest'} & sys.modules.keys()
"""],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PAPERLEET_ROOT": str(tmp_path)},
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_nonstandard_statement_is_preserved():
    statement = "# My paper\n\n## Experiments\nNo numbered run section."
    assert web_readme(statement) == statement


def test_split_index_omits_heavy_content_and_links_immutable_details(tmp_path, monkeypatch):
    import hashlib
    monkeypatch.setenv("PAPERLEET_ROOT", str(tmp_path))
    paper = {"title": "Example paper", "authors": "Researcher", "anchors": ["Source anchor " * 1000]}
    functions = [{"name": "solve", "signature": "solve(" + "parameter: Tensor, " * 1000 + ") -> Tensor"}]
    folder = package(tmp_path, "rl/demo", 1, title_en="Example", paper=paper, functions=functions)
    (folder / "README.md").write_text("Full statement " * 2000)
    (folder / "starter.py").write_text("def solve(): pass\n")
    (folder / "solution.py").write_text("reference source")
    (folder / "notes.md").write_text("notes")
    full_path = export_catalog()
    index_path = full_path.parent / "index.json"
    index = json.loads(index_path.read_text())
    assert index["schema_version"] == 2
    summary = index["problems"][0]
    assert summary["starter"] == "def solve(): pass\n"
    assert not summary["readme"] and not summary["notes"]
    assert "solution" not in summary and "vizData" not in summary
    assert summary["paper"] == {"title": ""} and summary["functions"] == []
    assert index["papers"][0]["title"] == paper["title"], "Paper cards retain their actual metadata"
    detail_path = full_path.parent / summary["detail_url"]
    detail = json.loads(detail_path.read_text())
    assert detail == json.loads(full_path.read_text())["problems"][0]
    assert detail["paper"] == paper and detail["functions"] == functions
    assert hashlib.sha256(detail_path.read_bytes()).hexdigest()[:20] in detail_path.name
    assert index_path.stat().st_size < full_path.stat().st_size / 5
    import base64
    import gzip
    compact_path = full_path.parent / "index-v3.json"
    compact = json.loads(compact_path.read_text())
    sources = json.loads(gzip.decompress(base64.b64decode(compact["starters"]["data"])))
    assert compact["schema_version"] == 3
    assert compact["starters"]["encoding"] == "gzip-base64"
    assert compact["starters"]["bytes"] == len(gzip.decompress(base64.b64decode(compact["starters"]["data"])))
    assert sources == {item["id"]: item["starter"] for item in index["problems"]}
    restored = {**compact, "schema_version": 2, "problems": [
        {**item, "starter": sources[item["id"]]} for item in compact["problems"]]}
    del restored["starters"]
    assert restored == index, "Transport compaction must not modify source, paper or grading metadata"
    latest_path = full_path.parent / "index-v4.json"
    latest = json.loads(latest_path.read_text())
    assert latest["schema_version"] == 4
    packed = json.loads(gzip.decompress(base64.b64decode(latest["payload"]["data"])))
    def unpack(value):
        if isinstance(value, list):
            return [unpack(item) for item in value]
        if isinstance(value, dict):
            shape, items = next(iter(value.items()))
            return dict(zip(packed["shapes"][int(shape)], map(unpack, items)))
        return value
    assert unpack(packed["tree"]) == index
    latest_before = latest_path.read_bytes()
    before = compact_path.read_bytes()
    export_catalog()
    assert compact_path.read_bytes() == before, "The source transport is reproducible"
    assert latest_path.read_bytes() == latest_before, "The complete compact transport is reproducible"
    (folder / "solution.py").write_text("updated reference source")
    export_catalog()
    updated = json.loads(index_path.read_text())["problems"][0]
    assert updated["detail_url"] != summary["detail_url"]
    assert detail_path.exists(), "Previously served manifests must not lose their immutable detail assets"


def test_generated_packages_never_fall_back_to_host_execution(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPERLEET_ROOT", str(tmp_path))
    folder = package(tmp_path, "community/candidate", 11)
    sentinel = tmp_path / "executed-on-host"
    (folder / "viz").mkdir()
    (folder / "viz/gen.py").write_text(f"from pathlib import Path\nPath({str(sentinel)!r}).touch()\n")
    with pytest.raises(ValueError, match="submission sandbox"):
        run_batch({"11": []})
    with pytest.raises(ValueError, match="on the host"):
        vizdata(problem("11"))
    assert not sentinel.exists()


def test_paper_source_deduplication_and_explicit_module_order(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPERLEET_ROOT", str(tmp_path))
    package(tmp_path, "diffusion/forward", 6, paper={"title": "DDPM", "arxiv": "2006.11239"})
    package(tmp_path, "diffusion/reverse", 8, paper={"title": "DDPM", "url": "https://arxiv.org/pdf/2006.11239v2.pdf"})
    package(tmp_path, "community/new", 11, paper={"title": "Same paper", "url": "https://arxiv.org/abs/2006.11239v3"})
    (tmp_path / "papers.yaml").write_text(yaml.safe_dump({"papers": [{
        "id": "ddpm", "number": 1, "short_title": "DDPM",
        "module_ids": ["diffusion/forward", "diffusion/reverse"],
    }]}))
    exported = json.loads(export_catalog().read_text())
    assert len(exported["papers"]) == 1
    assert exported["papers"][0]["module_ids"] == ["diffusion/forward", "diffusion/reverse", "community/new"]
    assert {module["paper_id"] for module in exported["problems"]} == {"ddpm"}
    assert [module["number"] for module in exported["problems"]] == [6, 8, 11]
    assert [module["module"]["optional"] for module in exported["problems"]] == [False, False, True]


@pytest.mark.parametrize("definitions, message", [
    ([{"id": "p", "number": 1, "module_ids": ["demo/missing"]}], "unknown module"),
    ([{"id": "p", "number": 1, "module_ids": ["demo/a", "demo/a"]}], "more than one paper"),
    ([{"id": "p", "number": True, "module_ids": ["demo/a"]}], "positive integers"),
    ([{"id": "p", "number": 1, "module_ids": ["demo/a"]},
      {"id": "q", "number": 1, "module_ids": ["demo/b"]}], "positive integers"),
    ([{"id": "p", "number": 1, "module_ids": ["demo/a", "demo/b"]}], "different primary papers"),
])
def test_bad_paper_membership_cannot_export(tmp_path, monkeypatch, definitions, message):
    monkeypatch.setenv("PAPERLEET_ROOT", str(tmp_path))
    package(tmp_path, "demo/a", 1, paper={"arxiv": "2006.11239"})
    package(tmp_path, "demo/b", 2, paper={"arxiv": "2010.02502"})
    (tmp_path / "papers.yaml").write_text(yaml.safe_dump({"papers": definitions}))
    with pytest.raises(ValueError, match=message):
        export_catalog()


def test_legacy_packages_still_get_paper_cards(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPERLEET_ROOT", str(tmp_path))
    package(tmp_path, "demo/a", 1)
    package(tmp_path, "demo/b", 2)
    papers, membership = paper_catalog(problems())
    assert len(papers) == 2
    assert membership == {"demo/a": "paper-1", "demo/b": "paper-2"}


def test_required_and_provided_functions_cannot_overlap():
    entry = {"id": "demo/a", "title": "Example", "functions": [{"name": "core"}, {"name": "helper"}]}
    module = module_scope(entry)
    assert module["required_functions"] == ["core", "helper"]
    entry["module"] = {**module, "provided_functions": ["helper"]}
    with pytest.raises(ValueError, match="partition"):
        module_scope(entry)


def test_maintained_catalog_can_export_complete_bilingual_packages(tmp_path, monkeypatch):
    """Exercise real package metadata without changing the dev server's exported catalog."""
    import shutil
    source_root = Path(__file__).resolve().parents[1]
    shutil.copyfile(source_root / "papers.yaml", tmp_path / "papers.yaml")
    (tmp_path / "problems").symlink_to(source_root / "problems", target_is_directory=True)
    monkeypatch.setenv("PAPERLEET_ROOT", str(tmp_path))
    exported = json.loads(export_catalog().read_text())
    for entry in exported["problems"]:
        if entry["id"].startswith("community/"):
            continue
        assert entry["readme"] and entry["starter"] and entry["solution"] and entry["notes"]
        assert entry["translations"]["en"]["readme"] and entry["translations"]["en"]["notes"]
    assert {module for card in exported["papers"] for module in card["curated_module_ids"]} <= {entry["id"] for entry in exported["problems"]}


def test_arxiv_source_versions_share_identity():
    assert source_key({"arxiv": "2006.11239"}) == source_key({"url": "https://arxiv.org/html/2006.11239v2"})


def test_categories_export_bilingual_labels_and_reject_unknown_ids(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPERLEET_ROOT", str(tmp_path))
    package(tmp_path, "rl/demo", 1)
    definition = {"id": "demo", "number": 1, "module_ids": ["rl/demo"], "category": "reinforcement-learning"}
    manifest = tmp_path / "papers.yaml"
    manifest.write_text(yaml.safe_dump({"papers": [definition]}))
    exported = json.loads(export_catalog().read_text())
    assert exported["categories"] == [{"id": "reinforcement-learning", "name": "强化学习", "name_en": "Reinforcement Learning"}]
    assert exported["papers"][0]["category"] == "reinforcement-learning"
    definition["category"] = "typo"
    manifest.write_text(yaml.safe_dump({"papers": [definition]}))
    with pytest.raises(ValueError, match="unknown paper category"):
        export_catalog()
