"""Produce the web's one source of truth from problem packages."""
import json
import hashlib
import re
from pathlib import Path

from paperleet.catalog import problems, root
from paperleet.paper_catalog import CATEGORIES, paper_catalog, module_scope
from paperleet.judge.loader import load_implementation


def web_readme(markdown: str, language: str = "zh") -> str:
    """Present the browser workflow while retaining maintainer docs on disk."""
    instructions = (
        "## 6. Run your implementation\n\n"
        "Complete the functions in the Python editor on the right, then click **Run tests**. "
        "Inspect each failed check, update your implementation, and run again. "
        "Passing every required check marks the problem as solved and unlocks the notes.\n\n"
        if language == "en" else
        "## 6. 在网页中运行\n\n"
        "在右侧 Python 编辑器中补全实现，点击 **运行测试**。查看失败测试的原因、期望值与实际值，"
        "修改代码后再次运行。所有必要测试通过后，题目会标记为已完成并解锁坑点讲解。\n\n"
    )
    return re.sub(r"^## 6\. [^\n]*\n.*?(?=^## 7\.|\Z)", instructions,
                  markdown, flags=re.MULTILINE | re.DOTALL)


def vizdata(entry):
    if entry["id"].startswith("community/"):
        raise ValueError("Generated community packages do not execute visualization generators on the host. Their paper, code, and test results are available in the web editor.")
    import random
    import numpy as np
    import torch
    from paperleet.judge.pytest_plugin import jsonable
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    module = load_implementation(entry["directory"] / "viz" / "gen.py")
    data = jsonable(module.generate())
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    if len(encoded.encode()) > 500 * 1024:
        raise ValueError(f"{entry['id']}: visualization data exceeds 500 KB ({len(encoded.encode())} bytes)")
    destination = root() / "web" / "public" / "data" / (entry["id"].replace("/", "-") + ".json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(encoded, encoding="utf-8")
    return destination


def export_catalog():
    catalog = []
    entries = problems()
    papers, membership = paper_catalog(entries)
    papers_by_id = {paper["id"]: paper for paper in papers}
    aliases = {entry["short_id"]: str(entry["number"]) for entry in entries if entry.get("short_id")}
    alias_pattern = re.compile(r"(?<![\w-])(" + "|".join(re.escape(key) for key in aliases) + r")(?![\w-])") if aliases else None

    def public_numbers(text):
        return alias_pattern.sub(lambda match: aliases[match[0]], text) if alias_pattern else text

    for entry in entries:
        directory = entry["directory"]
        data = {key: value for key, value in entry.items() if key != "directory"}
        data["paper_id"] = membership[entry["id"]]
        data["module"] = module_scope(entry)
        curated = papers_by_id[data["paper_id"]].get("curated_module_ids")
        if curated is not None and entry["id"] not in curated:
            data["module"] = {**data["module"], "optional": True}
        for field, filename in (("readme", "README.md"), ("starter", "starter.py"), ("solution", "solution.py"), ("notes", "notes.md")):
            path = directory / filename
            data[field] = path.read_text(encoding="utf-8") if path.exists() else ""
        data["readme"] = web_readme(data["readme"])
        for field in ("readme", "notes"):
            data[field] = public_numbers(data[field])
        english = {"title": entry.get("title_en", entry["title"])}
        for field, filename in (("readme", "README.en.md"), ("notes", "notes.en.md")):
            path = directory / filename
            if path.exists():
                english[field] = public_numbers(path.read_text(encoding="utf-8"))
        if "readme" in english:
            english["readme"] = web_readme(english["readme"], "en")
        data["translations"] = {"en": english}
        vizpath = root() / "web" / "public" / "data" / (entry["id"].replace("/", "-") + ".json")
        data["vizData"] = json.loads(vizpath.read_text()) if vizpath.exists() else None
        from paperleet.browser_judge import export_browser_bundle
        browser = export_browser_bundle(entry, vizpath.parent)
        if browser:
            data["browser_judge"] = browser
        catalog.append(data)
    destination = root() / "web" / "public" / "data" / "catalog.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    categories = [category for category in CATEGORIES if any(paper["category"] == category["id"] for paper in papers)]
    temporary.write_text(json.dumps({"categories": categories, "papers": papers, "problems": catalog}, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    temporary.replace(destination)
    # Keep catalog.json for CLI consumers and older clients. The website reads
    # the small index and requests only the selected module's immutable detail.
    def write_json(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        temp = path.with_suffix(".json.tmp")
        temp.write_text(raw, encoding="utf-8")
        temp.replace(path)

    fields = {"id", "number", "paper_id", "short_id", "title", "difficulty", "type", "parent",
              "module", "estimated_minutes", "tests", "starter", "browser_judge"}
    index = []
    for data in catalog:
        raw = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()[:20]
        relative = f"modules/{data['number']}-{digest}.json"
        write_json(destination.parent / relative, data)
        summary = {key: value for key, value in data.items() if key in fields}
        # Paper cards already use the paper-level record. Full author/source
        # metadata and function signatures belong to the selected module detail,
        # like its statement. Keep the lightweight Problem shape for old clients.
        # Starters remain exact: progress, random selection and backup review
        # compare against them even before a module has been opened.
        summary.update(readme="", notes="", paper={"title": ""}, functions=[], detail_url=relative,
                       translations={"en": {"title": data["translations"]["en"]["title"], "readme": "", "notes": ""}})
        index.append(summary)
    # Publish the manifest last so every referenced detail already exists.
    # Do not delete previous hashes: a browser may still hold the old manifest.
    write_json(destination.parent / "index.json",
               {"schema_version": 2, "categories": categories, "papers": papers, "problems": index})
    return destination
