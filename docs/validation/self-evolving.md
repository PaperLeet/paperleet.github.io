# Self-evolving collection and walkthroughs

Added 2026-10-20. Papers 74–77 / exercises 107–113 were allocated after the published paper/module maxima 73/106. Canonical arXiv IDs were absent from the registry. The four-paper collection has seven required modules, bilingual statements, notes and preparation guides, complete native/browser suites and reference experiments.

## Sources and exercise boundaries

- [R-Zero](https://arxiv.org/html/2508.05004v4), §2.2–2.4 and Figure 2: triangular uncertainty, batch repetition, valid-format gating, inclusive [0.25,0.75] curriculum and frozen majority pseudo-labels. Smallest-ID vote ties, pre-normalized answers and caller-supplied clusters are explicit teaching conventions. Consistency is not correctness.
- [G-Zero](https://arxiv.org/html/2605.09959), §3.1–3.3, Eqs. 3/5 and Figure 2; [official Phase 2](https://github.com/Chengsong-Huang/G-Zero/blob/main/g_zero/phase2.py): score the same unassisted tokens under both contexts. Use base-minus-hinted mean log probabilities. Proposer rewards retain negative values; length penalties count characters. DPO curation keeps delta <= the full-pool linear median before structural checks, including ties. Prompts exclude hints; chosen is assisted, rejected unassisted. The signal is a proxy, not measured quality improvement or KL.
- [VisPlay](https://arxiv.org/html/2511.15661), §2.3, Eqs. 5–8 and Figure 2; [official repository](https://github.com/Bruno686/VisPlay): rewards and repetition groups are image-local. External unlabeled images remain required. Group denominators include malformed proposals under the explicit exercise contract; cluster IDs can repeat across images.
- [MM-Zero](https://arxiv.org/html/2603.09206), §2.4–2.6, Eqs. 4–12 and Figure 1; [official Coder reward](https://github.com/zli12321/MM-Zero/blob/main/MM-zero_final/reward_function/codegen_reward.py): failed renders contribute zero but remain in the proposer denominator. Its easy-score cap is 0.5. Diversity combines caption/easy/hard cluster shares. Easy-hard and content-type adjustments are supplied upstream. Coder uses uncapped easy scores plus continuous min(c,1-c), matching Eq.7 and the code despite the binary range typo near Eq.11. Solver agreement and format are additive. Normalization, code execution, rendering, clustering and full GRPO/DPO training are outside these tensor APIs.

## Original interactive diagrams

Each paper has a separately authored React/SVG flow linked to the verified original figure anchor. No paper image or third-party diagram asset is copied. Steps distinguish frozen inference/data collection from the one role being updated, explain feedback ownership, and finish one round without implying measured multi-round improvements. G-Zero explicitly describes its frozen round-start DPO reference. MM-Zero separately illustrates proposer, coder and solver phases.

Controls recompute numeric examples for uncertainty/repetition, Hint-delta/length/median filtering, or successful renders/easy score/consistency. Examples disclose fixed assumptions. The original diagram is a teaching abstraction; actual submitted-function toy traces remain separately labeled. Playback starts only by explicit action, stops at the round end, pauses when hidden, cleans timers on unmount, and suppresses moving edges for reduced-motion preferences. Source links are external links; controls do not submit data, change learner code or record progress.

## Calibration

All seven suites have T0=1, T1=1, T2=3, T3=1 and five complete single-fault mutants each. Scalar counting/arithmetic oracles, endpoints, invariances and gradient ownership provide independent checks. Fixed toy feedback drives a categorical policy through 60 gradient steps at learning rate 0.5, CPU float64, uniform initial logits, deterministic seed 42/two PyTorch threads. There is no model-generated dataset or paper benchmark claim. Acceptance requires target probability >0.72 and improvement >0.3.

| Exercise | Initial target probability | Reference after 60 steps |
| --- | --- | --- |
| gzero_hint | 0.25 | 0.960967684 |
| gzero_pairs | 0.50 | 0.981613974 |
| mmzero_feedback | 0.25 | 0.990647380 |
| mmzero_proposer | 0.25 | 0.951846641 |
| rzero_curriculum | 0.50 | 0.981613974 |
| rzero_rewards | 0.25 | 0.951427373 |
| visplay_rewards | 0.25 | 0.915578951 |

## Catalog delivery

The exact expanded homepage model is preserved, including every starter byte and progress comparison. Index v4 shares repeated JSON object shapes and uses build-time Zopfli to emit standard gzip. The browser uses its native bounded gzip decoder, with nesting/node/shape limits and full round-trip checks. Legacy v2/v3 manifests stay available for existing clients. The existing 256 KiB raw / 96 KiB gzip index limits remain unchanged; no source or contract was shortened. The new UI stays inside the deferred Lab chunk.

## Validation record

- New-module native calibration: 756 test cases across three repeats passed in 21.43 seconds; all 35 mutants were assertion-caught. The initial one-repeat run also passed.
- Actual Chrome inspection confirmed the collection and explanation route, 390 px mobile layout without flow overflow, and readable Chinese/light and English/dark diagrams.
- Frontend interaction checks verify all four flows, one-role update ownership, numeric boundaries and ties, actual controls, visibility/reduced-motion behavior, locale focus retention and cleanup with zero requests.
- Complete catalog/build, full native and WASM, infrastructure and private hosting checks remain release requirements. A local full-native attempt was interrupted by the existing per-test five-second timeout while formatting a mutation failure; this attempt is not recorded as passing.

The complete real Pyodide 0.29.4 / NumPy 2.2.5 run passed all 99 modules, including each new reference and its five mutants. Production path routing generated and checked 184 HTTP-200 sitemap routes (75 papers, 99 exercises, nine collections, homepage), with valid math and unique metadata. The full local infrastructure attempt reported 47,663 passed, one skipped and nine failures: eight existing macOS signed-zero differences and one wheel-metadata expectation made stale by the upstream addition of optional test extras. The metadata check now distinguishes required NumPy from optional pytest/torch; both isolated-wheel cases and catalog tests pass (26 tests). The platform-specific signed-zero cases are not skipped or changed; the complete Linux gate remains required.
