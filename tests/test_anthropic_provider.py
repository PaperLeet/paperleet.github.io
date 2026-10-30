"""Offline transport fixtures: no credentials, external API or sandbox needed."""
import io
import json
from http.client import RemoteDisconnected
from urllib.error import HTTPError, URLError

import pytest

from paperleet.submissions import SubmissionStore
from paperleet.submissions.harness import PaperHarness
from paperleet.submissions import model
from paperleet.submissions.model import AnthropicProvider, ModelUnavailable, OpenAIProvider, create_provider


PAPER = "https://arxiv.org/html/2401.00001v1"
CODE = "https://github.com/example/paper"
SECRET = "test-only-credential-never-return"
SCHEMA = model.object_schema({"ok": {"type": "boolean"}})


def message(content=None, stop="end_turn", **extra):
    return {"type": "message", "role": "assistant", "id": "msg_fixture", "stop_reason": stop,
            "content": content if content is not None else [{"type": "text", "text": '{"ok":true}'}], **extra}


def search_blocks():
    return [
        {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "paper equation"}},
        {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": [
            {"type": "web_search_result", "url": PAPER, "title": "Paper", "encrypted_content": "opaque-search-content"},
            {"type": "web_search_result", "url": "http://example.com/insecure", "title": "ignored"},
            {"type": "web_search_result", "url": "https://user:password@example.com/private", "title": "ignored"},
        ]},
        {"type": "text", "text": "Full paper equation 2 and official code are accessible.", "citations": [
            {"type": "web_search_result_location", "url": PAPER, "title": "duplicate", "encrypted_index": "opaque-index"},
            {"type": "web_search_result_location", "url": CODE, "title": "Code", "encrypted_index": "opaque-code-index"},
            {"type": "char_location", "url": "https://example.com/unverified", "title": "Not search evidence"},
        ]},
    ]


def fetch_block(url=PAPER, source=None):
    return {"type": "web_fetch_tool_result", "tool_use_id": "srvtoolu_fetch", "content": {
        "type": "web_fetch_result", "url": url, "retrieved_at": "2026-10-29T12:00:00Z",
        "content": {"type": "document", "title": "Fetched full paper", "citations": {"enabled": True},
                    "source": source if source is not None else {"type": "text", "media_type": "text/plain", "data": "Full paper. Section 2: y = a + b."}}}}


@pytest.fixture(autouse=True)
def isolated_configuration(monkeypatch):
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_WORKSPACE_ID", "PAPERLEET_MODEL_PROVIDER", "PAPERLEET_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    monkeypatch.setenv("PAPERLEET_MODEL", "test-claude")
    # Accidental transport calls fail locally rather than reaching a provider.
    def no_network(*args, **kwargs):
        pytest.fail("Unexpected external network access")
    monkeypatch.setattr(model, "urlopen", no_network)
    monkeypatch.setattr(model, "build_opener", no_network)


def transport(monkeypatch, responses):
    captured = []
    responses = iter(responses)

    class Opener:
        def open(self, request, timeout):
            captured.append((request, json.loads(request.data), timeout))
            result = next(responses)
            if isinstance(result, Exception):
                raise result
            return io.BytesIO(result if isinstance(result, bytes) else json.dumps(result).encode())

    def factory(*handlers):
        assert len(handlers) == 1 and isinstance(handlers[0], model._NoProviderRedirect)
        assert handlers[0].redirect_request(None, None, 302, "redirect", {}, "https://example.com/leak") is None
        return Opener()

    monkeypatch.setattr(model, "build_opener", factory)
    return captured


def test_factory_defaults_and_explicit_selection_preserve_legacy_imports(monkeypatch, tmp_path):
    assert type(create_provider()) is OpenAIProvider
    assert type(PaperHarness().provider) is OpenAIProvider
    assert type(SubmissionStore(directory=tmp_path / "default").provider) is OpenAIProvider
    monkeypatch.setenv("PAPERLEET_MODEL_PROVIDER", "  ")
    assert type(create_provider()) is OpenAIProvider
    monkeypatch.setenv("PAPERLEET_MODEL_PROVIDER", "anthropic")
    assert type(create_provider()) is AnthropicProvider
    assert type(PaperHarness().provider) is AnthropicProvider
    assert type(SubmissionStore(directory=tmp_path / "claude").provider) is AnthropicProvider
    # Explicit dependency injection must not be overridden, even when falsey.
    class Injected:
        def __bool__(self):
            return False
    provider = Injected()
    monkeypatch.setenv("PAPERLEET_MODEL_PROVIDER", "unsupported-" + SECRET)
    assert PaperHarness(provider=provider).provider is provider
    assert SubmissionStore(provider=provider, directory=tmp_path / "injected").provider is provider
    with pytest.raises(ModelUnavailable, match="must be openai or anthropic") as error:
        create_provider()
    assert SECRET not in str(error.value)


@pytest.mark.parametrize("missing", ["ANTHROPIC_API_KEY", "PAPERLEET_MODEL"])
def test_missing_configuration_does_not_use_openai_key(monkeypatch, missing):
    monkeypatch.delenv(missing)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-only-test")
    provider = AnthropicProvider()
    assert provider.capability()["ready"] is False
    with pytest.raises(ModelUnavailable, match="ANTHROPIC_API_KEY and PAPERLEET_MODEL"):
        provider.request("paper")


def test_messages_json_schema_headers_and_workspace_are_not_exposed(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_test_only")
    captured = transport(monkeypatch, [message()])
    provider = AnthropicProvider()
    result = provider.request("paper contract", SCHEMA)
    request, body, timeout = captured[0]
    assert request.full_url == "https://api.anthropic.com/v1/messages"
    headers = {name.lower(): value for name, value in request.header_items()}
    assert headers["x-api-key"] == SECRET
    assert headers["anthropic-version"] == "2026-10-21"
    assert headers["anthropic-workspace-id"] == "wrkspc_test_only"
    assert "authorization" not in headers
    assert body == {"model": "test-claude", "system": model.BOUNDARY, "max_tokens": 16000,
                    "messages": [{"role": "user", "content": "paper contract"}],
                    "output_config": {"format": {"type": "json_schema", "schema": SCHEMA}}}
    assert 0 < timeout <= 240
    assert result == {"value": {"ok": True}, "sources": [], "response_id": "msg_fixture", "response_ids": ["msg_fixture"],
                      "fetched_urls": [], "fetch_failures": 0}
    public = json.dumps([provider.capability(), result])
    assert SECRET not in public and "wrkspc_test_only" not in public


def test_research_uses_only_typed_search_results_and_citations(monkeypatch):
    captured = transport(monkeypatch, [message(search_blocks())])
    research = AnthropicProvider().research({"paper_url": PAPER, "title": "Paper"})
    assert research["sources"] == [{"url": PAPER, "title": "Paper"}, {"url": CODE, "title": "Code"}]
    body = captured[0][1]
    assert body["tools"] == [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5},
                             {"type": "web_fetch_20250910", "name": "web_fetch", "max_uses": 3,
                              "max_content_tokens": 20000, "citations": {"enabled": True}}]
    assert "tool_choice" not in body and "output_config" not in body
    assert "Anthropic-workspace-id" not in dict(captured[0][0].header_items())
    assert "full paper" in body["messages"][0]["content"]
    assert "Use web_fetch" in body["messages"][0]["content"]
    assert PAPER in body["messages"][0]["content"] and PAPER.replace("/html/", "/pdf/") in body["messages"][0]["content"]
    assert research["fetched_urls"] == []


def test_text_url_without_retrieval_cannot_approve_or_generate(monkeypatch):
    captured = transport(monkeypatch, [message([{"type": "text", "text": "I read " + PAPER}])])
    provider = AnthropicProvider()
    assessment = provider.assess({"paper_url": PAPER})
    assert assessment["decision"] == "needs_review" and not assessment["full_paper_evidence"]
    assert assessment["research"]["sources"] == [] and len(captured) == 1
    with pytest.raises(ModelUnavailable, match="No accessible"):
        provider.generate({"paper_url": PAPER}, research=assessment["research"])
    assert len(captured) == 1


def test_pause_continuation_preserves_all_opaque_blocks_and_sources(monkeypatch):
    blocks = search_blocks() + [{"type": "thinking", "thinking": "fixture", "signature": "opaque-signature"}]
    paused = message(blocks, "pause_turn", id="msg_pause")
    captured = transport(monkeypatch, [paused, message([{"type": "text", "text": "Final dossier."}], id="msg_done")])
    result = AnthropicProvider().request("paper", search=True)
    assert len(captured) == 2
    assert captured[1][1]["messages"] == [*captured[0][1]["messages"], {"role": "assistant", "content": blocks}]
    assert captured[1][1]["tools"] == captured[0][1]["tools"]
    assert result["response_ids"] == ["msg_pause", "msg_done"] and result["response_id"] == "msg_done"
    assert result["sources"] == [{"url": PAPER, "title": "Paper"}, {"url": CODE, "title": "Code"}]
    assert result["value"].endswith("Final dossier.")


def test_continuations_are_bounded_without_automatic_network_retries(monkeypatch):
    captured = transport(monkeypatch, [message(search_blocks(), "pause_turn")] * 4)
    with pytest.raises(ModelUnavailable, match="continuation limit"):
        AnthropicProvider().request("paper", search=True)
    assert len(captured) == 3


@pytest.mark.parametrize("reply,search,match", [
    (message(stop="max_tokens"), False, "did not complete"),
    (message(stop="tool_use"), False, "did not complete"),
    (message(stop="stop_sequence"), False, "did not complete"),
    (message(stop=[]), False, "did not complete"),
    (message(stop="refusal"), False, "declined"),
    (message(stop_details={"type": "refusal", "explanation": SECRET}), False, "declined"),
    (message([{"type": "refusal", "refusal": SECRET}]), False, "declined"),
    (message(stop="pause_turn"), False, "continuation limit"),
    (message([], "pause_turn"), True, "continuation limit"),
    (message([]), False, "no candidate content"),
    (message([{"type": "text", "text": "   "}]), False, "no candidate content"),
    (message([{"type": "text", "text": 12}]), False, "invalid text"),
    (message([{"type": "text", "text": "ok", "citations": "bad"}]), True, "invalid text"),
    (message([None]), False, "invalid message content"),
    (message(content={"text": SECRET}), False, "invalid message content"),
    (message(role="user"), False, "invalid message response"),
    (message(id={"secret": SECRET}), False, "invalid response identifier"),
    ({"type": "error", "error": {"message": SECRET}}, False, "invalid message response"),
    ([], False, "invalid message response"),
    (message([{"type": "tool_use", "name": "execute", "input": SECRET}]), False, "unsupported client tool"),
    (message([{"type": "web_search_tool_result", "content": {"type": "web_search_tool_result_error", "error_code": SECRET}}]), True, "web search failed"),
    (message([{"type": "web_search_tool_result", "content": "malformed"}]), True, "invalid search results"),
])
def test_incomplete_refused_and_invalid_messages_fail_closed(monkeypatch, reply, search, match):
    captured = transport(monkeypatch, [reply])
    with pytest.raises(ModelUnavailable, match=match) as error:
        AnthropicProvider().request("paper", search=search)
    assert SECRET not in str(error.value) and len(captured) == 1


@pytest.mark.parametrize("raw", ["not JSON " + SECRET, "[]", "null", '"a string"'])
def test_invalid_structured_content_is_sanitized(monkeypatch, raw):
    transport(monkeypatch, [message([{"type": "text", "text": raw}])])
    with pytest.raises(ModelUnavailable, match="invalid structured output") as error:
        AnthropicProvider().request("paper", SCHEMA)
    assert SECRET not in str(error.value)


@pytest.mark.parametrize("failure", [
    HTTPError("https://example.com/" + SECRET, 401, SECRET, {"secret": SECRET}, None),
    HTTPError("https://example.com/" + SECRET, 302, SECRET, {"Location": "https://example.com/" + SECRET}, None),
    URLError(SECRET), TimeoutError(SECRET), OSError(SECRET), RemoteDisconnected(SECRET), b"malformed " + SECRET.encode(), b"\xff",
])
def test_transport_failures_are_redacted_and_not_retried(monkeypatch, failure):
    captured = transport(monkeypatch, [failure])
    with pytest.raises(ModelUnavailable, match="Anthropic request failed") as error:
        AnthropicProvider().request("paper")
    assert SECRET not in str(error.value) and "example.com" not in str(error.value)
    assert len(captured) == 1


def test_bounded_total_response_size_and_deadline(monkeypatch):
    provider = AnthropicProvider()
    provider.max_response_bytes = 8192
    big = message([{"type": "text", "text": "x" * 4400}], "pause_turn")
    captured = transport(monkeypatch, [big, big])
    with pytest.raises(ModelUnavailable, match="bounded response size"):
        provider.request("paper", search=True)
    assert len(captured) == 2
    captured = transport(monkeypatch, [message()])
    clock = iter([0, 0, 241])
    monkeypatch.setattr(model.time, "monotonic", lambda: next(clock))
    with pytest.raises(ModelUnavailable, match="timed out"):
        provider.request("paper")
    assert len(captured) == 1


@pytest.mark.parametrize("name", ["ANTHROPIC_API_KEY", "ANTHROPIC_WORKSPACE_ID"])
def test_header_injection_fails_before_network(monkeypatch, name):
    monkeypatch.setenv(name, SECRET + "\r\nInjected: value")
    with pytest.raises(ModelUnavailable, match="Invalid Anthropic credential") as error:
        AnthropicProvider().request("paper")
    assert SECRET not in str(error.value)


def test_citations_and_structured_outputs_cannot_be_combined():
    with pytest.raises(ModelUnavailable, match="separate requests"):
        AnthropicProvider().request("paper", SCHEMA, search=True)


def assessment_value(source=PAPER):
    return {"decision": "suitable", "reason": "Equation 2 defines a CPU-testable sum.", "suggested_focus": "addition",
            "full_paper_evidence": True, "criteria": {key: "yes" for key in model.SUITABILITY_CRITERIA},
            "evidence": [{"claim": "scalar sum", "source_url": source, "locator": "Equation 2"}]}


def json_message(value, identifier):
    return message([{"type": "text", "text": json.dumps(value)}], id=identifier)


def test_assess_generate_reuses_dossier_and_independent_tests_with_anthropic_provenance(monkeypatch):
    candidate = {key: "Teaching addition" for key in ("title", "title_en", "summary", "readme", "readme_en", "notes", "notes_en")}
    candidate.update(starter="def add(a, b): raise NotImplementedError\n", solution="def add(a, b): return a + b\n",
                     difficulty="E", functions=[{"name": "add", "signature": "add(a, b)", "returns": "scalar sum"}],
                     evidence=assessment_value()["evidence"],
                     toy={"kind": "teaching_adaptation", "source_url": PAPER, "locator": "Equation 2", "adaptation": "Scalar teaching inputs"})
    tests = {"tests": [{"name": "test_sum", "layer": "T1", "code": "def test_sum(impl): assert impl.add(2, 3) == 5"}],
             "mutants": [{"name": "subtract", "bug": "wrong sign", "code": "def add(a, b): return a - b"}]}
    captured = transport(monkeypatch, [message(search_blocks() + [fetch_block()], "pause_turn", id="research_pause"),
                                      message([{"type": "text", "text": "Equation 2 dossier."}], id="research"),
                                      json_message(assessment_value(), "assessment"),
                                      json_message(candidate, "implementation"), json_message(tests, "tests")])
    provider = AnthropicProvider()
    submission = {"paper_url": PAPER, "title": "Paper"}
    assessment = provider.assess(submission)
    assert assessment["decision"] == "suitable" and assessment["full_paper_evidence"]
    generated, provenance = provider.generate(submission, research=assessment["research"])
    assert generated == {**candidate, **tests}
    assert provenance["provider"] == "anthropic" and provenance["model"] == "test-claude"
    assert provenance["response_ids"] == ["research_pause", "research", "implementation", "tests"]
    assert provenance["independent_test_generation"] is True
    assert provenance["fetched_urls"] == [PAPER]
    assert len(captured) == 5
    assert [entry[1]["output_config"]["format"]["schema"] for entry in captured[2:]] == [model.SUITABILITY_SCHEMA, model.IMPLEMENTATION_SCHEMA, model.TEST_SCHEMA]
    prompt = captured[-1][1]["messages"][0]["content"]
    assert candidate["solution"] not in prompt and '"solution"' not in prompt and '"notes"' not in prompt
    assert all("tools" not in entry[1] for entry in captured[2:])


def test_uncited_assessment_is_downgraded_and_uncited_candidate_rejected(monkeypatch):
    captured = transport(monkeypatch, [message(search_blocks() + [fetch_block()]), json_message(assessment_value("https://example.com/invented"), "assessment"),
                                      json_message({"evidence": assessment_value("https://example.com/invented")["evidence"]}, "implementation")])
    provider = AnthropicProvider()
    assessment = provider.assess({"paper_url": PAPER})
    assert assessment["decision"] == "needs_review" and not assessment["full_paper_evidence"]
    with pytest.raises(ModelUnavailable, match="did not cite"):
        provider.generate({"paper_url": PAPER}, research=assessment["research"])
    assert len(captured) == 3


@pytest.mark.parametrize("source", [
    {"type": "text", "media_type": "text/plain", "data": "Full paper equation 2."},
    {"type": "base64", "media_type": "application/pdf", "data": "JVBERi0xLjQK"},
])
def test_successful_fetch_is_evidence_and_survives_pause(monkeypatch, source):
    block = fetch_block(source=source)
    captured = transport(monkeypatch, [message([block], "pause_turn"), message([{"type": "text", "text": "Fetched equation 2."}])])
    result = AnthropicProvider().request("read " + PAPER, search=True)
    assert result["sources"] == [{"url": PAPER, "title": "Fetched full paper"}]
    assert result["fetched_urls"] == [PAPER] and result["fetch_failures"] == 0
    assert captured[1][1]["messages"][-1]["content"] == [block]


@pytest.mark.parametrize("blocks", [
    [],
    [fetch_block(CODE)],
    [fetch_block("https://arxiv.org/html/2402.00001v1")],
    [fetch_block("https://arxiv.org/html/2401.00001v2")],
    [fetch_block("https://arxiv.org/abs/2401.00001v1")],
    [{"type": "web_fetch_tool_result", "content": {"type": "web_fetch_tool_result_error", "error_code": "url_not_accessible", "url": PAPER}}],
])
def test_search_claims_cannot_replace_fetch_of_submitted_full_paper(monkeypatch, blocks):
    captured = transport(monkeypatch, [message(search_blocks() + blocks), json_message(assessment_value(), "assessment")])
    provider = AnthropicProvider()
    assessment = provider.assess({"paper_url": PAPER})
    assert assessment["decision"] == "needs_review" and not assessment["full_paper_evidence"]
    with pytest.raises(ModelUnavailable, match="No accessible full text"):
        provider.generate({"paper_url": PAPER}, research=assessment["research"])
    assert len(captured) == 2


def test_failed_html_then_successful_pdf_fetch_has_only_successful_source(monkeypatch):
    pdf = PAPER.replace("/html/", "/pdf/")
    blocks = [{"type": "web_fetch_tool_result", "content": {"type": "web_fetch_tool_result_error", "error_code": SECRET, "url": PAPER}},
              fetch_block(pdf), {"type": "text", "text": "PDF equation 2 accessible; HTML unavailable."}]
    transport(monkeypatch, [message(blocks), json_message(assessment_value(pdf), "assessment")])
    result = AnthropicProvider().assess({"paper_url": "https://arxiv.org/abs/2401.00001"})
    assert result["decision"] == "suitable" and result["full_paper_evidence"]
    assert result["research"]["fetched_urls"] == [pdf]
    assert result["research"]["sources"] == [{"url": pdf, "title": "Fetched full paper"}]
    assert result["research"]["fetch_failures"] == 1 and SECRET not in json.dumps(result)


@pytest.mark.parametrize("source", [None, {}, {"type": "text", "media_type": "text/plain", "data": ""},
                                        {"type": "url", "url": PAPER}, {"type": "base64", "media_type": "application/executable", "data": "opaque"}])
def test_empty_or_invalid_fetch_is_not_evidence(monkeypatch, source):
    block = fetch_block()
    block["content"]["content"]["source"] = source
    transport(monkeypatch, [message([block, {"type": "text", "text": "I read the paper."}])])
    with pytest.raises(ModelUnavailable, match="invalid or empty fetched content"):
        AnthropicProvider().request("paper", search=True)
