"""Link-to-candidate orchestration. Only a human publication gate can publish.

The hosted queue owns durable jobs, duplicate detection, quotas and retries.
This adapter consumes one job and returns an artifact; it never changes the
public catalog, executes generated Python on the host, or accepts model-issued
approval. All callbacks and return values are for trusted worker persistence.
"""
import copy
from urllib.parse import urlsplit

from .arxiv import arxiv_identity, canonical_arxiv_url, load_arxiv_metadata, normalize_metadata
from .checks import package_hash, static_checks
from .model import ModelUnavailable, OpenAIProvider, normalize_suitability, retrieved_urls
from .sandbox import Sandbox


def _paper_evidence(url, expected_id):
    """Search citations must include this paper, not unrelated pages or prompt URLs."""
    try:
        parsed = urlsplit(url)
        if parsed.path.startswith("/html/"):
            url = url.replace("/html/", "/abs/", 1)
        return arxiv_identity(url) == expected_id
    except (ValueError, TypeError):
        return False


def return_contract_check(candidate):
    functions = candidate.get("functions", []) if isinstance(candidate, dict) else []
    valid = bool(functions) and all(
        isinstance(row, dict) and isinstance(row.get("returns"), str)
        and bool(row["returns"].strip()) and len(row["returns"]) <= 4000 for row in functions)
    return {"name": "explicit_return_contracts", "passed": valid,
            "detail": "Every function declares its return format." if valid else
                      "Every function must explicitly declare return type, shape and tuple/dictionary structure."}


class PaperHarness:
    """Inject a provider, sandbox and metadata loader to test without paid calls.

    ``run({arxiv_url|paper_url, prompt|focus, revision_feedback?}, on_stage)`` calls
    ``on_stage(stage, state)`` for resolving/assessing/generating/validating.
    Results always contain metadata and suitability once assessed, and contain
    candidate/package_hash/provenance/validation once generated. Final status:

    * pool_ready: the exact candidate passed all isolated validation gates;
    * suitability_review: insufficient/conflicting evidence, no generation;
    * rejected: unsuitable with source-backed, explicit reasons;
    * validation_failed: a candidate failed structural or isolated validation;
    * validation_pending: sandbox unavailable; candidate remains private.

    SourceUnavailable/ModelUnavailable propagate for queue-level retry. Neither
    missing infrastructure nor a failed candidate means the paper is unsuitable.
    """
    def __init__(self, provider=None, sandbox=None, source_loader=None):
        self.provider = provider if provider is not None else OpenAIProvider()
        self.sandbox = sandbox if sandbox is not None else Sandbox()
        self.source_loader = source_loader if source_loader is not None else load_arxiv_metadata

    def run(self, submission, on_stage=None):
        if not isinstance(submission, dict):
            raise ValueError("Expected a paper proposal")
        url = canonical_arxiv_url(submission.get("arxiv_url", submission.get("paper_url")))
        prompt = submission.get("prompt", submission.get("focus", ""))
        if not isinstance(prompt, str) or len(prompt) > 4000:
            raise ValueError("Learning preferences must be text under 4000 characters")
        state = {"paper_url": url, "arxiv_id": arxiv_identity(url)}
        def stage(name):
            if on_stage is not None:
                on_stage(name, copy.deepcopy(state))
        stage("resolving")
        metadata = normalize_metadata(self.source_loader(url), url)
        state["metadata"] = metadata
        # Title, authors, source URL and abstract come from the fixed arXiv
        # resolver. A client's extra metadata, excerpt, code or approval fields
        # never replace them or reach the model.
        prepared = {"title": metadata["title"], "paper_url": metadata["resolved_url"],
                    "metadata": metadata, "focus": prompt.strip()}
        # Only the durable worker supplies this field, from an authenticated
        # maintainer revision decision. The public proposal API must not accept
        # it from submitters. It remains untrusted content for model purposes.
        feedback = submission.get("revision_feedback", "")
        if not isinstance(feedback, str) or len(feedback) > 8000:
            raise ValueError("Maintainer revision feedback must be text under 8000 characters")
        if feedback.strip():
            prepared["reviews"] = [{"decision": "changes_requested", "notes": feedback.strip()}]
        capability = self.provider.capability()
        if not capability.get("ready"):
            raise ModelUnavailable(capability.get("reason") or "The paper-generation provider is unavailable.")
        stage("assessing")
        assessment = self.provider.assess(prepared)
        if not isinstance(assessment, dict) or not isinstance(assessment.get("research"), dict):
            raise ModelUnavailable("The suitability assessment did not retain its retrieved evidence.")
        research = assessment["research"]
        suitability = normalize_suitability(assessment, retrieved_urls(research))
        paper_rows = [row for row in suitability["evidence"] if _paper_evidence(row["source_url"], metadata["arxiv_id"])]
        if not paper_rows or all(row["locator"].strip().lower() in {"abstract", "title", "summary", "摘要", "标题"} for row in paper_rows):
            suitability.update(decision="needs_review", full_paper_evidence=False,
                               reason="Evidence from the full submitted paper is not verified; a maintainer must review suitability.")
        state["suitability"] = suitability
        state["assessment_provenance"] = {"provider": capability.get("provider", ""), "model": capability.get("model", ""),
                                          "response_id": assessment.get("response_id", ""),
                                          "research_response_id": research.get("response_id", ""),
                                          "sources": research.get("sources", [])}
        if suitability["decision"] != "suitable":
            return {**state, "status": "rejected" if suitability["decision"] == "unsuitable" else "suitability_review"}
        stage("generating")
        candidate, provenance = self.provider.generate(prepared, research=research)
        try:
            checks = static_checks(candidate)
            checks.append(return_contract_check(candidate))
            digest = package_hash(candidate)
        except (TypeError, ValueError, AttributeError, RecursionError):
            raise ModelUnavailable("The model returned a malformed candidate package.") from None
        state.update(candidate=candidate, provenance=provenance, package_hash=digest)
        if not all(row["passed"] for row in checks):
            return {**state, "status": "validation_failed",
                    "validation": {"status": "failed", "passed": False, "package_hash": digest, "checks": checks}}
        stage("validating")
        validation = self.sandbox.validate(copy.deepcopy(candidate))
        if (not isinstance(validation, dict) or validation.get("package_hash") != digest
                or package_hash(candidate) != digest):
            raise ModelUnavailable("Validation did not match the generated candidate revision.")
        # Do not let an adapter's passed flag override an incomplete gate report.
        gates = validation.get("checks", [])
        if not isinstance(gates, list):
            raise ModelUnavailable("Validation returned an invalid gate report.")
        required = {row["name"] for row in checks if row["name"] != "explicit_return_contracts"}
        required.update({"reference", "reference_repeat", "starter"})
        required.update("mutant:" + row["name"] for row in candidate["mutants"])
        valid_gates = (all(isinstance(row, dict) and row.get("passed") is True for row in gates)
                       and required <= {row.get("name") for row in gates})
        passed = validation.get("passed") is True and validation.get("status") == "passed" and valid_gates
        validation = {**validation, "passed": passed, "checks": [*gates, return_contract_check(candidate)]}
        if passed:
            status = "pool_ready"
        elif validation.get("status") == "unavailable":
            status = "validation_pending"
        else:
            validation["status"] = "failed"
            status = "validation_failed"
        return {**state, "status": status, "validation": validation}
