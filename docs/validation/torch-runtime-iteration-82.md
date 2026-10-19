# Iteration 82: 0.78 application integration checkpoint

This is an ongoing development checkpoint. Main adoption, GitHub CI, tags,
publication and deployment have not been performed by this iteration. The
independent-library efficiency and core-API goal remains active. BrowserTorch
is a proposed name; the package remains `stridetorch` pending a naming decision.

## Source reconciliation

The two qualified 0.78 commits were rebased onto main
`077fa5bc6adbea0ed9c84a73fc9272dc09a0cd26`. The application execution head is
`4cde6dd1045d2097206b18189d65e1c300b0ad5b`. The incoming changes affect UI and
documentation; the numerical application inputs and release policies are
unchanged. All 63 runtime-resource hashes and all 80 SDK-test hashes still
match [iteration 81](torch-runtime-iteration-81.md). The original source commits
remain reachable through retained local refs; their old numerical evidence has
not been relabeled as execution against the rebased commit.

## Completed checks

| Check | Result |
| --- | --- |
| Full application and SDK CPU suite | 45,624 passed, one skipped, 34 warnings |
| Native maintained catalog | 89 modules, 12,360 test records across three repetitions; passed |
| Native catalog wall budget | 247.61 seconds within the unchanged 600-second budget |
| Regenerated browser exports | 180 new, reachable immutable assets; 183 live asset/index identities |
| Prior immutable exports | All 15,499 retained with matching hashes |

The CPU suite uses a task-owned Python 3.12.11 environment with NumPy 2.5.4,
Torch 2.14.1 and an editable installation of this worktree's PaperLeet package.
Its private Torch namespace resolves to the same qualified source directory.
The first attempt stopped during collection because that editable mapping had
not been installed in the library-only environment. Its failed logs are
preserved separately; the complete successful run used a separate application
environment. The primary checkout and its environments were not modified.

These are correctness and absolute-budget observations. Their wall times are
not comparative speed measurements. The native execution log, summary, all 89
reference reports, source/test/application/policy hashes and export identities
are retained under the task's `torch-iteration-82` evaluation directory.

## Work in progress

The sequential application runner has started all-catalog Pyodide/WASM checks.
Whole-catalog completion and budget compliance remain unproven at this
checkpoint. The runner subsequently checks the frontend, both default and
Pages build profiles, immutable build assets, exact worker regeneration and a
fresh real Chrome nested problem page. The page probe adds `addmm` backward,
`addmv` and biased Linear assertions to the existing attention solution and
matrix-entry-point checks.

A separate performance program has been prepared for 98 existing matrix,
attention and GQA flows plus 26 affine/Linear flows. It shares operands,
tangents and gradient seeds across identical baseline A0/A1 implementations
and the candidate, checks numerical agreement, and retains interleaved paired
rounds. Its Python and JavaScript syntax checks pass; numerical validation and
timing have not run yet. Performance jobs must wait until the application
runner's numerical jobs terminate. No broader CPU or browser speedup is claimed.

Before main adoption, finish these gates, broader CPU/WASM performance and
lifetime qualification, the verified evidence archive, and the established
fresh release-chain guard. Further core-API coverage and evidence-based backend
selection remain part of the continuing goal.
