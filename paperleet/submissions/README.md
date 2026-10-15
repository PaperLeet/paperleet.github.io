# Paper generation adapter

`PaperHarness` turns a durable proposal into a private, validated candidate. The
hosted service owns authentication, verified affiliations, deduplication, quotas,
job leases, persisted artifacts, community votes and final human publication.

```python
from paperleet.submissions.harness import PaperHarness

result = PaperHarness().run(
    {"arxiv_url": "https://arxiv.org/abs/2401.01234", "prompt": "Focus on the core loss."},
    on_stage=lambda stage, state: persist_job_progress(stage, state),
)
```

Users supply only an arXiv abstract/PDF URL and optional learning preferences
(at most 4,000 characters). The loader obtains title, authors, abstract and paper
version from the official arXiv API. `arxiv_identity(url)` strips version numbers
for duplicate checks, while the model receives the resolved version URL for
evidence fidelity. No arbitrary client URL is fetched; metadata responses are
size bounded and redirects are refused.

The provider researches the exact paper, screens four criteria (algorithmic
content, an explicit contract, small CPU execution and objective tests), generates
an exercise, and separately generates tests without seeing the reference
implementation. The same research dossier is reused across these steps. Paper
text, submitter prompts and review feedback are data, never system instructions.
An unavailable full text, unsupported citation, uncertain criterion or conflicting
assessment leads to `suitability_review`; missing infrastructure is retryable and
does not reject the paper.

An administrator's revision request may be passed by the trusted job worker as
`revision_feedback` (up to 8,000 characters). The public proposal endpoint must
not accept that field from users. It becomes `changes_requested` feedback for the
provider; it cannot authorize publication or bypass evidence/validation rules.

The optional callback receives `resolving`, `assessing`, `generating` and
`validating` with an independent copy of the state. The result's `status` is:

| Status | Meaning |
| --- | --- |
| `pool_ready` | All structural, return-contract and isolated execution gates passed for this exact package hash. |
| `suitability_review` | A maintainer must resolve missing or conflicting scientific evidence. |
| `rejected` | Specific paper-backed evidence establishes that the proposed exercise is unsuitable. |
| `validation_failed` | The generated exercise needs revision; this is not a rejection of the paper. |
| `validation_pending` | The isolated validator is unavailable; keep the candidate private. |

`metadata` and `suitability` accompany assessed results. Generated results also
include `candidate`, `package_hash`, `provenance` and `validation`. A pool must
require `validation.passed is True` and
`validation.package_hash == package_hash(candidate) == result.package_hash`.
`SourceUnavailable` and `ModelUnavailable` propagate for the durable worker to
record an infrastructure failure and retry according to its bounded retry policy.

Candidates contain public `title`, `title_en`, `readme`, `readme_en`, `starter`,
`difficulty` and `functions` (`name`, `signature`, `returns`). Keep reference
`solution`, `notes`, `notes_en`, `tests`, `mutants`, model provenance and validation
runs private. A pool-ready result cannot publish; the existing human review and
exact-revision publication gate still applies.

`Sandbox.try_candidate(candidate, code)` runs one pool attempt in the same
offline, resource-limited Docker worker used for generated-package validation.
It returns `package_hash`, `source_sha256`, `passed`, `tests`, `status`, `error`
and `execution_completed`. Only a complete test execution sets
`execution_completed` true, including genuine learner failures. Infrastructure
failures and incomplete reports must not unlock voting. The result omits raw
stdout, private test tracebacks and reference source.

Set `OPENAI_API_KEY` and `PAPERLEET_MODEL` only on the worker. Build the existing
submission sandbox image and set `PAPERLEET_SANDBOX_IMAGE` as needed. When the
worker runs inside a container controlling the host Docker daemon, configure
`PAPERLEET_SANDBOX_WORK_DIR` to an absolute existing directory bind-mounted at the
same path on both host and worker; candidate containers read that path read-only.
Temporary candidate directories are deleted after each run. On a standalone
host worker, leave the variable unset to use the normal temporary directory.

Tests inject provider, sandbox and source-loader fixtures; they never make paid
model calls or execute generated code on the host. Real Docker validation is an
explicitly enabled integration test in `test_submissions.py`.
