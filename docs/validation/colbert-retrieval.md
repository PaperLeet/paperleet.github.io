# ColBERT late interaction teaching module

Validated locally on 2026-10-14. Paper 51 adds module 66,
`retrieval/late_interaction`: 10 maintained checks and six assertion-caught fault
variants. The catalog contains 51 papers, 66 CPU/WASM modules, 517 checks and 272
fault variants. The retrieval collection now connects CLIP, ColBERT, Rotation
Trick and TIGER: four papers / five modules.

## Source and scope

The [original paper](https://arxiv.org/abs/2004.12832) and its
[author-hosted PDF](https://people.eecs.berkeley.edu/~matei/papers/2020/sigir_colbert.pdf)
establish normalized token representations in Section 3.2, cosine MaxSim in
Section 3.3 / Equation 3, and positive/negative pairwise softmax training. Original
query-augmentation `[MASK]` positions participate in scoring; this exercise's
Boolean query mask excludes absent batching positions, not those augmentation
vectors.

The learner receives projected token vectors and implements all-pairs MaxSim and
stable pairwise loss. Supporting input validators are provided. BERT, tokenizer
markers, projection training, ANN indexing and MS MARCO evaluation are outside
the exercise. All-masked document scores, empty queries, epsilon normalization
and first-position tie gradients are explicit teaching conventions. No official
implementation is copied or claimed.

Ten tests cover float32/float64 and strided inputs, independent scalar values,
54-coordinate central finite differences for both encodings, ties, permutations,
positive scaling, duplicate evidence, negative matches, padding, zero-vector
normalization gradients, extreme pairwise margins and both score gradients.
Six small faults target document/query means, zero padding, missing normalization,
reversed ranking and detached documents. An independently written loop learner
passes the same complete native and NumPy-adapter suites.

## Actual learner experiment

Four synthetic two-token queries search eight fixed documents. Each positive has
two separate concept vectors; its negative has a single blended vector. Their
normalized pooled directions coincide, illustrating information lost by this
specific pooling fixture. This is not a general claim about retrieval quality.

The submitted scorer and loss perform all 120 Adam updates on 32 query-vector
parameters. Loss falls from **1.0193437 to 0.1958832**, minimum positive/negative
margin rises from about **-0.571756 to 1.530697**, and Recall@1 on the four supplied
synthetic labels rises from **0 to 1**. Maximum independent scalar error and pooled
direction gap are zero. The chart includes the real final point at update 120.
These are teaching diagnostics, not paper benchmark reproduction or BERT tuning.

## Verification

- Full catalog CI: **66 modules / 8,022 cases / three repeats**, 98.44 seconds;
  all 272 faults caught. The final ColBERT-only CI, including the added zero-query
  gradient assertion and final plot point, passes **210 cases / three repeats**,
  2.77 seconds, all six faults caught.
- Public regression in an independent public-only environment: **1,092 passed,
  one skipped**, 203.32 seconds. The shared development environment contains the
  private editable backend and correctly fails the public-edition isolation
  assertion, so it is not used as evidence of a clean public installation.
  Final focused alternative-learner, export-fidelity and adapter-calibration
  checks pass **3 tests**, with 66 unrelated parameterizations deselected.
- Every preexisting module's browser metadata and immutable detail URL is
  unchanged. The index is **252,396 bytes**, below the original 256 KiB cap.
  The shared adapter, harness and worker implementations are unchanged.
- Full frontend checks and three production builds pass. Initial JavaScript is
  **399,380 / 129,083 bytes** raw/gzip for Pages, **403,100 / 130,279** for split
  API hosting, and **402,870 / 130,197** for explicitly enabled same-origin hosted
  edition. Original 400 KiB raw / 130 KiB gzip limits remain unchanged.
- Real browser: reference **10/10**, all **six** faults calibrated. Final observed
  phases: preparation **1.39 s**, calibration **0.90 s**, complete learner suite
  **0.09 s**. These are localhost single observations, not internet cold-start or
  Oracle performance measurements. An earlier explicit zero-padding fault passes
  only **7/10**; restoring the reference returns 10/10.
- The real learner's Lab displays its own 120-step curve and the synthetic
  metrics above, separately from reference charts. A 390 px iframe reruns 10/10;
  its viewport, document and main scroll widths all measure **390 px**. A later
  edit in another tab correctly marks the earlier report as older source.
- Unsupported `torch.cdist(p=1)` produces an incompatibility message and **zero
  judge POSTs** in Browser only. In the explicitly hosted build, a disposable
  account starts with its own starter and no guest acceptance. Auto submits the
  exact reviewed unsupported source once, producing **one POST / one native
  execution**, matching its SHA-256, and **Accepted 10/10**. The QA native adapter
  allows only exact reviewed fixtures; it is not a production Docker benchmark.

Private worker packaging and pinned-core integration are recorded in the private
repository. No independent-reader blind study, public push or deployment is
claimed. Browser feedback remains local until separately confirmed by the server.
