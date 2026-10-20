# ES, Visual Domain Prompt and Neural Thickets

Added on 2026-10-20. Stable paper IDs 71–73 and module IDs 104–106 extend the latest public registry; arXiv identities 2509.24372, 2212.04145 and 2603.12228 were absent before this addition. Each paper has one required module, bilingual statements/notes, a starter containing only unfinished required functions, a reference, five single-fault mutants, seven T0–T3 tests and an actual synthetic experiment.

## Sources and scope

- ES at Scale: [paper](https://arxiv.org/abs/2509.24372), Section 3, Algorithms 1–2, and [official code](https://github.com/VsonicV/es-at-scale). Exercise 104 implements population reward standardization and one ascent update. It explicitly chooses population standard deviation, a positive standard-deviation floor, and detached optimizer results. Alpha absorbs inverse sigma; the supplied noise is epsilon. It does not train an LLM or implement the distributed seed protocol.
- Visual Domain Prompt: [paper](https://arxiv.org/abs/2212.04145), Method Eqs. (1), (3)–(4), and [AAAI proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/25922). Exercise 105 adds full-canvas prompts, sums supplied per-domain importance ratios and constrains the current domain-agnostic prompt. The source's prompt-symbol inconsistency is disclosed; semantic names follow the prose. Teacher EMA, confidence-based domain detection and importance-path estimation are excluded. The penalty's absence of a one-half is intentional.
- Neural Thickets: [paper](https://arxiv.org/abs/2603.12228), Section 4, Eqs. (3)–(5), Algorithm 1; [author overview](https://thickets.mit.edu/) and [official code](https://github.com/sunrainyg/RandOpt). Exercise 106 independently perturbs a common base, selects by selection-set scores and votes on discrete test answers. Original-index score ties and smallest-class vote ties are explicit exercise conventions. No weight averaging, iterative ES update or thicket-scaling reproduction is claimed.

## Independent checks and observed experiments

Each suite has one T0, one T1, four T2 and one T3 test. Scalar Python computations, mathematical invariants and analytic derivatives/optima supply independent checks. Tests never import the reference solution. Each T3 invokes the learner's required functions throughout the actual experiment.

| Module | Experiment | Native reference observation | Acceptance |
| --- | --- | --- | --- |
| ES at Scale | 100 updates, 30 directions/update, sigma 0.1, alpha 0.05; two-coordinate squared distance; seed 42 | Distance 25 -> 0.01253819984 | Final distance < 0.08 |
| Visual Domain Prompt | Frozen two-pixel readout, supplied teacher/history, 120 SGD updates at lr 0.1, strengths 0/1 | Prompts approach [1,-1] and [1/9,-1]; regularized data loss 0.3950617284 | Coordinates within 1e-4 of independent optima; regularized loss in (0.39,0.40) |
| Neural Thickets | 120 linear candidates, scales 0.25/0.5/1, 41 selection samples, 9 experts, 80 distinct test samples; seed 42 | Test accuracy 0.5 -> 0.9 | Nine distinct experts and accuracy >= 0.9 |

All experiments use CPU float64 with two PyTorch threads. VDP's squared feature consistency is a teaching adaptation with an analytic optimum, not the paper's classifier consistency loss. Neural Thickets test labels are used only for final evaluation, never selection. None of the measurements are paper benchmark results.

Initial native calibration passes all 21 reference checks and assertion-catches all 15 mutants. Exported NumPy compatibility checks pass for all three complete suites. Real pinned Pyodide 0.29.4 / NumPy 2.2.5 independently passes each reference, all five mutants and the learner suite; no tests or training steps are removed. The normal catalog-wide quality/release gate remains required.

The local macOS infrastructure run reports 47,657 passed, one skipped, and eight signed-zero failures in the unchanged `packages/torchwasm/tests/test_matmul_wasm.py`. The failing comparison is NumPy's native matrix product versus the WASM-oriented product on zero-valued operands; neither that runtime nor its tests is modified here. This is not recorded as a passing infrastructure run. The complete Linux release gate must pass before publication.

The frontend adds the ES/RandOpt papers to LLM Frontier and all three to Post-training, and provides bilingual explain/implement/verify prompts in the existing study paths. Source identities feed the existing duplicate-paper registry automatically.
