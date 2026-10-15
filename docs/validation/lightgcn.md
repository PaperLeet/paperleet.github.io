# LightGCN browser and native lesson validation

Validated 2026-10-15 against the reviewed public source snapshot based on `feaf351b35cbefd5c7211ae4472a8fa12b6694a1`. No application, image or production configuration was deployed.

## Scope and provenance

Paper 54 / module 69, `retrieval/lightgcn`, joins Retrieval immediately after BPR. The independent implementation follows [LightGCN](https://arxiv.org/abs/2002.02126), Sections 3.1.1–3.1.3 / Equations 3–8 (graph and uniform aggregation), Section 3.2.3 / Equations 13–14 (two-hop influence) and Section 3.3 / Equation 15 (initial-table prior).

The teaching adaptation uses a dense graph with 4 synthetic users, 6 items and an overlapping user history. User 3 and item 5 are isolated. Only 30 initial embedding entries are learned, using all 24 observed/unobserved triples, two propagation steps, uniform layers 0–2 and 160 full-batch Adam updates. Mean likelihood and a single .001 initial-table prior replace the paper's summed-data coefficient convention; this is not a large sparse system, real dataset, held-out evaluation or reproduction of paper benchmark numbers.

## Native and compatibility tests

- Native catalog CI: **69 modules, 8,829 test executions, 3 consistent runs, 257.49 seconds; PASS**. Original 300-second catalog budget and per-lesson T3 limits are unchanged. All 295 mutants trigger genuine test failures, with at most 10 changed lines per mutant.
- Full public platform suite: **1,324 passed, 1 skipped**, 331.36 seconds. One existing TorchScript deprecation warning; the optional check remains skipped.
- New lesson exported NumPy calibration: **1 passed**, with the same 10 maintained tests and all 8 mutants caught.
- Independent scalar loops cover graph normalization and multi-hop propagation. Finite differences check both initial-table and graph-operator derivatives. Other properties cover unbalanced endpoint degrees, zero edges/axes, zero layers, isolated nodes, linearity, user/item relabeling, dtype, strided inputs and input protection.
- No runtime operations, dependencies, test tolerances or training budgets were changed. Existing 68 module records and 53 paper records remain identical in the reviewed catalog export. All **2,393** preceding data files are retained; only the new lesson manifest/detail/visualization and catalog/index additions are published.

## Actual browser flow

The standalone production build was served on an isolated local port. Navigation followed Retrieval → LightGCN → module 69. A learner implementation constructed its graph with `cat`/`diag` and aggregated `stack(...).mean(0)`, independently of the reference's block assignment and accumulated sum.

- Correct learner: **10/10**; calibration **10 reference checks and 8 mutants**. Initial observed phases: preparation 4.35 s, calibration 3.80 s, full learner suite 0.29 s; T3 training 0.17 s. A restored run also passed 10/10 (full suite 0.23 s, T3 0.15 s). These are measurements from this browser/machine, not cross-device or Oracle server comparisons.
- Returning the final layer alone: **4/10**, with numerical errors, two-hop coefficient, gradients, isolated-node and training-objective checks detecting the defect.
- Adding native-only matrix 1-norm: an explicit browser-unsupported message; previous report marked stale; all **2,540 source characters** downloaded exactly as submitted. Restoring the correct source returned to 10/10.
- Lab displayed **My run results**, the actual 160-update loss curve and learned scores. Final loss **0.0490651056** from **0.6914770553**; training pair accuracy and Recall@2 both **1**. Graph scalar error at most **1.12e-16**; isolated-node and scalar-objective errors zero. The item-only upper bound **17/24** is counted, not a measured baseline.
- Both statement languages render all 3 display formulas without KaTeX errors. At 390 CSS pixels, document width and scroll width both equal 390. Browser warning/error log empty; 118 QA HTTP requests were GET-only, with no API requests or failed application assets.

Ignored local evidence: `reports/lightgcn-{ci.json,native.txt,tests.txt,numpy.txt,web.txt,build.txt,hosted.txt,assets.json,browser-pass.txt,browser-wrong.txt,browser-unsupported.txt,browser-final.txt,browser-english.txt,browser-mobile.png,browser-chart-full.png}` and exact learner/native-only sources. Temporary test pages and the user tab are separate.

## Frontend builds

Full frontend checks pass, including 433 compile-only Python sources, all 69 bundles, catalog/search/practice scope, browser runtime, reports, queueing, accounts and community navigation.

Standalone initial graph: **401,058 raw / 129,987 gzip bytes**. Hosted initial graph: **406,434 raw / 131,825 gzip bytes**. Original **400 KiB raw / 130 KiB gzip** caps are unchanged. Rich text, editor, accounts, community and Lab remain deferred. New training data is loaded through the catalog/detail path rather than imported into the initial JavaScript graph.

The private repository must pin this reviewed public commit, explicitly allowlist `retrieval/lightgcn`, include only its lesson in the worker image COPY list, and validate reference/fault and native-only execution through the actual packaged worker report boundary before release.
