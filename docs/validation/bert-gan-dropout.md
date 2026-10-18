# BERT, GAN and Dropout

Validated on 2026-10-17. Three previously absent representative papers add complete bilingual exercises, 31 learner tests and 20 localized mutants. The resulting catalog contains 67 papers plus one introductory course, 89 modules, 721 learner tests and 406 mutants.

Paper numbers 68–70 and module numbers 101–103 follow public main `44e5ccc362b64393e908074cf74063bcf5142052`. Export preserves all 86 existing module records and all 65 existing paper/course records exactly. Existing immutable assets are retained. The shared runtime remains StrideTorch 0.68.0.

## Sources and exercise scope

| Paper | Verified primary source | Required implementation |
| --- | --- | --- |
| BERT, NAACL 2019 | [Section 3.1, Task #1 and Appendix C.2](https://arxiv.org/html/1810.04805v2#S3.SS1) | Externally selected 80/10/10 corruption, original same-position labels and stable cross entropy averaged over selected tokens. |
| Generative Adversarial Nets, NeurIPS 2014 | [Section 3, Equation (1), Algorithm 1 and the non-saturating paragraph](https://arxiv.org/html/1406.2661v1#S3) | Two independently averaged discriminator expectations and the stable non-saturating generator objective. |
| Dropout, JMLR 2014 | [Published paper](https://jmlr.org/papers/v15/srivastava14a.html), [Sections 4 and 10, page 1951](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf) | Explicit Boolean keep masks, inverted training scaling, identity inference and both probability endpoints. |

BERT does not shift labels as an autoregressive objective would. GAN's minimax generator loss is a legitimate alternative in the source paper, but is outside this exercise's explicitly non-saturating contract; the notes explain that mutant. Dropout's primary identity uses the published JMLR URL, rather than attributing the differently titled 2012 predecessor to the 2014 article.

Each package includes seven-section Chinese/English statements, a fully specified starter, reference implementation, bilingual notes, independent T0–T3 tests, full-file single-bug mutants and a visualization generator. Independent scalar derivations, analytic or finite-difference gradients, reductions, extreme logits, endpoints, empty cases where permitted, dtype and strided-input checks supplement reference agreement. Tests execute learner functions in real training, and charts distinguish learner artifacts from precomputed reference observations.

The ordered `objectives-regularization` preparation path contains all three modules. Nine new bilingual prompts trace to their source statements and required functions. Registry checks cover all three new mentions, the JMLR URL, whole-token boundaries and the existing ColBERT identity.

## Numerical and execution checks

- New-module native gate: three repeats, 714 executions, all 20 mutants rejected by genuine assertions.
- Complete native gate: `paperleet ci --budget 600`, 89 modules, three repeats, **12,360 executions**, all 406 mutants caught, no issues, 300.32 seconds. The CLI aggregate allowance is explicit; the existing individual test and training budgets are unchanged. An earlier run under concurrent heavy validation exceeded the old Flow Matching T3 budget by 0.60 seconds. The complete quieter rerun passed without changing tests, tolerances, runtime, training or deadlines.
- Complete actual WASM catalog: **89/89 modules** passed reference, all-mutant calibration and learner suites, using Pyodide 0.29.4 / NumPy 2.2.5 and the pinned 0.68 runtime. The old Flow Matching module recorded a browser phase-budget warning on this loaded host. Correctness completion does not establish browser deadline compliance. The three new modules also passed separately with no phase-budget overruns.
- Actual production worker: Linux ARM64, Python 3.12, PyTorch 2.6.0, exact private `Dockerfile.judge` allowlisted sources. Three references passed and all 20 mutants produced real assertion failures. The existing non-root worker ran without network, with a read-only root, dropped capabilities, no new privileges, two CPUs, 1 GiB memory and bounded temporary storage. No application account code or credentials were copied into the image.
- Complete frontend check passed (321 PASS observations), including 89 maintained guides and 267 bilingual prompts. Hosted, connected-public, standalone-public and local production builds all pass using Node 22.23.3. Their initial JavaScript graphs remain inside the unchanged 400 KiB raw / 130 KiB gzip limits; hosted uses 409,215 raw / 133,032 gzip bytes.
- Complete public infrastructure regression: `python -m pytest tests -q`, **33,628 passed, one skipped**, 30 existing native-Torch warnings, 333.03 seconds. The unchanged standalone runtime package suite was qualified upstream in iteration 68 and is not counted as rerun here.
- Actual Chrome production-build workspace: independent BERT implementation **11/11** and GAN implementation **10/10**. Both display successful learner-executed training curves and current-source metrics in Lab. Their source links, formulas and experiment descriptions render correctly. Dropout's complete actual WASM run is covered above.

Detailed reports, blind implementations and the validation Docker image remain local development artifacts. Timings are observations on this Mac, not CPU/WASM/server performance comparisons.

## Calibrated teaching experiments

All experiments use CPU float64, two Torch threads, batch size 64 and fixed seeds. Three native reference repetitions established the recorded thresholds.

| Module | Actual experiment | Reference observation | Acceptance |
| --- | --- | --- | --- |
| BERT | 100 Adam updates, lr 0.08, seed 163; 16 left/right context pairs, 15% eligible-center selection, supplied conditional table | NLL 1.386294 → 0.028433; center accuracy 100% | NLL <0.1 and <10% of initial; accuracy ≥95% |
| GAN | 240 alternating D/G Adam update pairs, lr 0.03, seed 179; unit-variance Gaussian location game | Generator location 0 → 2.023909 for target 2; error 0.023909 | Final location error <0.5 |
| Dropout | 100 Adam updates, lr 0.06, seed 193; four-feature linear regression, drop probability 0.25; 8,192 realized masks | Inference MSE 2.184033 → 0.201856; maximum mean/variance errors 0.011068/0.014634 | MSE <20% of initial; mean error <0.06; variance error <0.08 |

These are labeled teaching adaptations. BERT does not implement a Transformer, NSP or a language benchmark; GAN does not reproduce image quality or claim general convergence; Dropout does not claim a measured generalization improvement.

## Independent blind implementation

As requested by CONTRIBUTING §5, an independent agent received only the three starters and bilingual statements, without tests, references, mutants, notes or website access. Its first implementations passed all suites: **11/11 BERT, 10/10 GAN, 10/10 Dropout**. No blocking contract ambiguity was reported. BERT was returned about 20 seconds after the reading checkpoint; GAN and Dropout shared an approximately 11-second implementation window. These agent observations do not estimate human difficulty.

## Release coordination

The companion [private PR #26](https://github.com/PaperLeet/backend/pull/26) extends the judge allowlist/COPY rules and pins the public exercise commit, with native protocol and exact copied-worker checks. Review and publish the public change first, then update the private gitlink to the exact published SHA if squash/rebase changes its identity. Follow each repository's release guide. This contribution does not merge main, enable workflows, tag a release or deploy.
