"""Paper-level curation with stable module identities and source deduplication."""
import re
from urllib.parse import urlsplit, urlunsplit

import yaml

from paperleet.catalog import root

CATEGORIES = (
    {"id": "foundations", "name": "ML 基础", "name_en": "ML Foundations"},
    {"id": "reinforcement-learning", "name": "强化学习", "name_en": "Reinforcement Learning"},
    {"id": "post-training", "name": "后训练与微调", "name_en": "Post-training & Fine-tuning"},
    {"id": "generative-models", "name": "扩散与生成模型", "name_en": "Diffusion & Generative Models"},
    {"id": "multimodal", "name": "多模态", "name_en": "Multimodal"},
    {"id": "robotics", "name": "机器人学习", "name_en": "Robotics"},
    {"id": "ai-infra", "name": "AI 基础设施", "name_en": "AI Infra"},
    {"id": "diffusion-rl", "name": "扩散强化学习", "name_en": "Diffusion RL"},
    {"id": "architecture", "name": "架构与推理", "name_en": "Architecture & Inference"},
    {"id": "optimization", "name": "优化算法", "name_en": "Optimization"},
    {"id": "representation", "name": "表征学习", "name_en": "Representation Learning"},
    {"id": "world-models", "name": "世界模型", "name_en": "World Models"},
    {"id": "retrieval", "name": "检索与推荐", "name_en": "Retrieval & Recommendation"},
    {"id": "community", "name": "社区精选", "name_en": "Community Selections"},
)


def source_key(paper: dict) -> str:
    """Recognize arXiv abs/PDF/version links as the same paper."""
    arxiv = str(paper.get("arxiv") or "").strip()
    url = str(paper.get("url") or "").strip()
    parsed = urlsplit(url)
    if not arxiv and parsed.hostname in {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}:
        arxiv = re.sub(r"^/(abs|pdf|html)/", "", parsed.path).removesuffix(".pdf")
    if arxiv:
        return "arxiv:" + re.sub(r"v\d+$", "", arxiv)
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path.rstrip("/"), parsed.query, ""))


def paper_catalog(entries: list[dict]) -> tuple[list[dict], dict[str, str]]:
    """Return paper cards and membership; uncurated reviewed modules still appear."""
    path = root() / "papers.yaml"
    definitions = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    definitions = definitions.get("papers", []) if isinstance(definitions, dict) else None
    if not isinstance(definitions, list):
        raise ValueError("papers.yaml: papers must be a list")
    by_module = {entry["id"]: entry for entry in entries}
    papers, membership, by_source, by_id, numbers = [], {}, {}, {}, set()

    def card(definition, entry):
        metadata = entry.get("paper", {})
        return {
            "id": definition["id"], "number": definition["number"],
            "title": metadata.get("title", entry["title"]),
            "short_title": metadata.get("title", entry["title"]),
            "authors": metadata.get("authors", ""), "venue": metadata.get("venue", ""),
            "url": metadata.get("url") or ("https://arxiv.org/abs/" + str(metadata["arxiv"]) if metadata.get("arxiv") else ""),
            "arxiv": metadata.get("arxiv"), "tags": entry.get("tags", []),
            "category": "community",
            "summary": "实现这篇论文选定的核心模块。",
            "summary_en": "Implement selected core modules from this paper.",
            "scope": "完成下列核心模块即可完成本篇练习。",
            "scope_en": "Complete the core modules below to finish this paper's practice.",
            **definition,
            "module_ids": list(definition.get("module_ids", [])),
        }

    for definition in definitions:
        if not isinstance(definition, dict):
            raise ValueError("papers.yaml: each paper must be an object")
        identifier, number = definition.get("id"), definition.get("number")
        if definition.get("category", "community") not in {item["id"] for item in CATEGORIES}:
            raise ValueError(f"{identifier}: unknown paper category")
        if not isinstance(identifier, str) or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", identifier) or identifier in by_id:
            raise ValueError("papers.yaml: paper ids must be unique lowercase slugs")
        if type(number) is not int or number < 1 or number in numbers:
            raise ValueError("papers.yaml: paper numbers must be unique positive integers")
        modules = definition.get("module_ids")
        if not isinstance(modules, list) or not modules or not all(isinstance(m, str) for m in modules):
            raise ValueError(f"{identifier}: module_ids must be a nonempty list")
        for module in modules:
            if module not in by_module:
                raise ValueError(f"{identifier}: unknown module {module}")
            if module in membership:
                raise ValueError(f"{module}: module belongs to more than one paper")
            declared = by_module[module].get("paper_id")
            if declared and declared != identifier:
                raise ValueError(f"{module}: paper_id does not match papers.yaml")
            membership[module] = identifier
        item = card(definition, by_module[modules[0]])
        # Newly published modules can join a paper without expanding its
        # explicitly selected required practice scope.
        item["curated_module_ids"] = list(modules)
        keys = {source_key(by_module[m].get("paper", {})) for m in modules} - {""}
        if len(keys) > 1:
            raise ValueError(f"{identifier}: modules reference different primary papers")
        for key in keys:
            if key in by_source:
                raise ValueError(f"{identifier}: duplicate primary paper {key}")
            by_source[key] = item
        papers.append(item)
        by_id[identifier] = item
        numbers.add(number)

    for entry in entries:
        if entry["id"] in membership:
            continue
        key = source_key(entry.get("paper", {}))
        declared = entry.get("paper_id")
        if declared and declared not in by_id:
            raise ValueError(f"{entry['id']}: unknown paper_id {declared}")
        item = by_id.get(declared) if declared else by_source.get(key)
        if item and declared and key and source_key(item) != key:
            raise ValueError(f"{entry['id']}: paper_id has a different primary source")
        if item is None:
            number = entry["number"]
            if number in numbers:
                number = max(numbers, default=0) + 1
            identifier = "paper-" + str(number)
            while identifier in by_id:
                number = max(numbers | {number}) + 1
                identifier = "paper-" + str(number)
            item = card({"id": identifier, "number": number}, entry)
            papers.append(item)
            by_id[identifier] = item
            numbers.add(number)
            if key:
                by_source[key] = item
        item["module_ids"].append(entry["id"])
        membership[entry["id"]] = item["id"]
    return sorted(papers, key=lambda item: item["number"]), membership


def module_scope(entry: dict) -> dict:
    functions = [function["name"] for function in entry.get("functions", [])]
    module = entry.get("module") or {
        "goal": entry["title"], "goal_en": entry.get("title_en", entry["title"]),
        "scope": "实现下列函数，并通过接口、公式与行为验证。",
        "scope_en": "Implement the functions below and validate their contracts, formulas, and behavior.",
        "required_functions": functions, "provided_functions": [], "optional": False,
    }
    for field in ("goal", "goal_en", "scope", "scope_en"):
        if not isinstance(module.get(field), str) or not module[field].strip():
            raise ValueError(f"{entry['id']}: module.{field} must be nonempty text")
    required, provided = module.get("required_functions"), module.get("provided_functions")
    if not isinstance(required, list) or not required or not isinstance(provided, list):
        # Minimal legacy packages can have no declared functions.
        if functions or entry.get("module"):
            raise ValueError(f"{entry['id']}: module function lists are invalid")
    combined = (required or []) + (provided or [])
    if not all(isinstance(name, str) for name in combined) or len(set(combined)) != len(combined) or set(combined) != set(functions):
        raise ValueError(f"{entry['id']}: required/provided functions must partition public functions")
    if type(module.get("optional")) is not bool:
        raise ValueError(f"{entry['id']}: module.optional must be boolean")
    return module
