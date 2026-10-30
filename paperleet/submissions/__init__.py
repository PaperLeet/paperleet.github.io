"""Durable candidate workflow with an explicit hash-bound human publication gate."""
from datetime import datetime, timezone
import copy
import json
from pathlib import Path
import re
import shutil
import threading
import uuid

import yaml

from paperleet.catalog import problems, root
from paperleet.config import data_directory
from .checks import https_url, package_hash, static_checks
from .model import ModelUnavailable, OpenAIProvider, create_provider
from .sandbox import Sandbox

ACTIVE = {"generating", "validating"}
REVIEW_CHECKS = ("paper_fidelity", "scope_clarity", "test_independence", "license_checked")


def now():
    return datetime.now(timezone.utc).isoformat()


class Conflict(ValueError):
    pass


class SubmissionStore:
    def __init__(self, base=None, provider=None, sandbox=None, directory=None):
        self.base = Path(base or root())
        self.directory = Path(directory) if directory is not None else self.base / ".paperleet" / "submissions"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.provider = provider if provider is not None else create_provider()
        self.sandbox = sandbox or Sandbox()
        self.lock = threading.RLock()
        self.jobs = set()
        # A server restart cannot turn a partial run into a successful validation.
        for path in self.directory.glob("*.json"):
            data = json.loads(path.read_text())
            if data.get("status") in ACTIVE:
                data.update(status="generation_failed" if data["status"] == "generating" else "validation_failed",
                            error="The server stopped before this job completed. Retry the operation.")
                self._save(data)

    def capabilities(self):
        return {"generation": self.provider.capability(), "sandbox": self.sandbox.capability(), "human_review_required": True}

    def _path(self, identifier):
        if not isinstance(identifier, str) or not re.fullmatch(r"[0-9a-f]{32}", identifier):
            raise ValueError("Invalid submission identifier")
        return self.directory / (identifier + ".json")

    def _load(self, identifier):
        path = self._path(identifier)
        if not path.is_file():
            raise KeyError("Unknown submission")
        return json.loads(path.read_text(encoding="utf-8"))

    def _save(self, data):
        data["updated_at"] = now()
        target = self._path(data["id"])
        temporary = target.with_suffix("." + uuid.uuid4().hex + ".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(target)

    def get(self, identifier):
        with self.lock:
            return self._load(identifier)

    def list(self):
        with self.lock:
            rows = [json.loads(path.read_text()) for path in self.directory.glob("*.json")]
        # Keep the list cheap; detail retains source, complete code and all validation runs.
        return sorted([{k: v for k, v in row.items() if k not in ("candidate", "provenance", "excerpt", "validation")} for row in rows], key=lambda r: r["created_at"], reverse=True)

    def create(self, payload, owner_id=None):
        if not isinstance(payload, dict):
            raise ValueError("Expected a submission object")
        data = {"id": uuid.uuid4().hex, "status": "submitted", "created_at": now(), "reviews": []}
        if owner_id is not None:
            data["owner_id"] = owner_id
        for key, limit in (("title", 240), ("excerpt", 80000), ("focus", 4000)):
            value = payload.get(key, "")
            if not isinstance(value, str) or len(value) > limit:
                raise ValueError(f"{key} must be text under {limit} characters")
            data[key] = value.strip()
        if not data["title"]:
            raise ValueError("Paper title is required")
        data["paper_url"] = https_url(payload.get("paper_url"), required=True)
        data["code_url"] = https_url(payload.get("code_url", ""))
        with self.lock:
            self._save(data)
        return data

    def start(self, identifier, operation):
        if operation not in ("generate", "validate"):
            raise ValueError("Unknown submission operation")
        with self.lock:
            data = self._load(identifier)
            if data["status"] in ACTIVE or identifier in self.jobs:
                raise Conflict("A job is already running for this submission")
            if data["status"] in ("published", "rejected"):
                raise Conflict("Published or rejected submissions cannot be regenerated")
            if operation == "generate":
                capability = self.provider.capability()
                if not capability["ready"]:
                    raise ModelUnavailable(capability["reason"])
                # Keep hash-bound review history; a new revision still needs fresh validation and approval.
                if data.get("candidate"):
                    revisions = self.directory / (identifier + "-revisions")
                    revisions.mkdir(exist_ok=True)
                    prior_hash = package_hash(data["candidate"])
                    archived = revisions / (prior_hash + ".json")
                    if not archived.exists():
                        archived.write_text(json.dumps({key: data.get(key) for key in ("candidate", "validation", "provenance", "reviews")}, ensure_ascii=False, indent=2), encoding="utf-8")
                    data.setdefault("revisions", []).append({"package_hash": prior_hash, "archived_at": now()})
                for key in ("candidate", "package_hash", "validation", "provenance", "error"):
                    data.pop(key, None)
                data["status"] = "generating"
            else:
                if not data.get("candidate"):
                    raise Conflict("Generate a candidate before validation")
                data["status"] = "validating"
                data.pop("error", None)
            self._save(data)
            self.jobs.add(identifier)
            thread = threading.Thread(target=self._work, args=(identifier, operation), daemon=True, name="paperleet-" + operation)
            thread.start()
            return copy.deepcopy(data)

    def _work(self, identifier, operation):
        try:
            with self.lock:
                data = self._load(identifier)
            if operation == "generate":
                candidate, provenance = self.provider.generate(data)
                checks = static_checks(candidate)
                data.update(candidate=candidate, provenance=provenance, package_hash=package_hash(candidate),
                            validation={"status": "pending", "passed": False, "package_hash": package_hash(candidate), "checks": checks},
                            status="validation_pending" if all(row["passed"] for row in checks) else "validation_failed")
                if data["status"] == "validation_failed":
                    data["error"] = "Generated candidate failed structural checks. Regenerate with a narrower objective."
            else:
                validation = self.sandbox.validate(data["candidate"])
                data["validation"] = validation
                data["status"] = "needs_review" if validation["passed"] else ("validation_pending" if validation["status"] == "unavailable" else "validation_failed")
                if not validation["passed"]:
                    data["error"] = validation.get("output") or "Candidate did not pass all validation gates."
            with self.lock:
                self._save(data)
        except Exception as exc:
            with self.lock:
                data = self._load(identifier)
                data.update(status="generation_failed" if operation == "generate" else "validation_failed",
                            error=str(exc)[:1200] if isinstance(exc, (ValueError, ModelUnavailable)) else "The job failed. Check the server setup and retry.")
                self._save(data)
        finally:
            with self.lock:
                self.jobs.discard(identifier)

    def review(self, identifier, payload):
        if not isinstance(payload, dict):
            raise ValueError("Expected a human review object")
        with self.lock:
            data = self._load(identifier)
            if data["status"] in ACTIVE or data["status"] in ("published", "rejected"):
                raise Conflict("This submission cannot be reviewed in its current state")
            decision = payload.get("decision")
            if decision not in ("approve", "changes_requested", "reject"):
                raise ValueError("Choose approve, changes_requested or reject")
            for key in ("reviewer", "notes"):
                if not isinstance(payload.get(key), str) or not payload[key].strip() or len(payload[key]) > 8000:
                    raise ValueError("Human reviewer name and review notes are required")
            digest = package_hash(data["candidate"]) if data.get("candidate") else ""
            if not digest or payload.get("package_hash") != digest or data.get("package_hash") != digest:
                raise Conflict("Candidate changed or is missing; reload and review the current revision")
            checks = payload.get("checks", {})
            if not isinstance(checks, dict):
                raise ValueError("Review checks must be an object")
            if decision == "approve":
                validation = data.get("validation", {})
                if data["status"] != "needs_review" or not validation.get("passed") or validation.get("package_hash") != digest:
                    raise Conflict("This exact candidate must pass isolated validation before approval")
                if not all(checks.get(key) is True for key in REVIEW_CHECKS):
                    raise ValueError("A human must confirm paper fidelity, scope, independent tests, and source licenses")
                if not all(c["passed"] for c in static_checks(data["candidate"])):
                    raise Conflict("Candidate no longer satisfies structural checks")
            review = {"decision": decision, "reviewer": payload["reviewer"].strip(), "notes": payload["notes"].strip(),
                      "checks": {key: checks.get(key) is True for key in REVIEW_CHECKS}, "package_hash": digest, "created_at": now()}
            if decision == "approve":
                published = self._publish(data, review)
                data.update(status="published", published_problem=published)
            else:
                data["status"] = "changes_requested" if decision == "changes_requested" else "rejected"
            data["reviews"].append(review)
            data.pop("error", None)
            self._save(data)
            return data

    def _publish(self, data, review):
        candidate = data["candidate"]
        # Publication is the sole point at which a candidate enters the public registry.
        number = max((int(entry["number"]) for entry in problems()), default=0) + 1
        slug = "paper_" + data["id"][:12]
        identifier = "community/" + slug
        target = self.base / "problems" / identifier
        if target.exists():
            raise Conflict("This submission already has a published package")
        staging = self.directory / ("publish-" + data["id"])
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir()
        try:
            metadata = {"id": identifier, "number": number, "short_id": str(number), "title": candidate["title"], "title_en": candidate["title_en"],
                        "paper": {"title": data["title"], "url": data["paper_url"], "arxiv": "", "authors": "", "venue": "", "anchors": [r["locator"] for r in candidate["evidence"]]},
                        "type": "implement", "parent": None, "difficulty": candidate["difficulty"], "functions": candidate["functions"], "conventions": {},
                        "tests": {layer: sum(t["layer"] == layer for t in candidate["tests"]) for layer in ("T0", "T1", "T2", "T3")},
                        "t3_budget_s": 15, "mutants": [{"id": m["name"], "bug": m["bug"]} for m in candidate["mutants"]],
                        "viz": [], "estimated_minutes": 30, "formula": "", "tags": ["Community"],
                        "submission_id": data["id"], "package_hash": data["package_hash"], "execution": "sandbox", "review": review}
            (staging / "problem.yaml").write_text(yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False), encoding="utf-8")
            for field, filename in {"readme": "README.md", "readme_en": "README.en.md", "notes": "notes.md", "notes_en": "notes.en.md", "starter": "starter.py", "solution": "solution.py"}.items():
                (staging / filename).write_text(candidate[field], encoding="utf-8")
            for group in ("tests", "mutants"):
                (staging / group).mkdir()
                for row in candidate[group]:
                    (staging / group / (row["name"] + ".py")).write_text(row["code"], encoding="utf-8")
            (staging / "candidate.json").write_text(json.dumps(candidate, ensure_ascii=False, indent=2), encoding="utf-8")
            (staging / "provenance.json").write_text(json.dumps({"submission": data["id"], "provenance": data.get("provenance"), "validation": data["validation"], "review": review}, ensure_ascii=False, indent=2), encoding="utf-8")
            target.parent.mkdir(parents=True, exist_ok=True)
            staging.replace(target)
            from paperleet.export import export_catalog
            try:
                export_catalog()
            except Exception:
                shutil.rmtree(target)
                raise
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return {"id": identifier, "number": number}


_stores = {}
_stores_lock = threading.Lock()


def store():
    location = (root(), data_directory())
    with _stores_lock:
        if location not in _stores:
            _stores[location] = SubmissionStore(location[0], directory=location[1] / "submissions")
        return _stores[location]
