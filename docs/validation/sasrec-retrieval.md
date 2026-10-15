# SASRec causal recommendation module

Paper 52 / module 67, `retrieval/sasrec`, adds ten maintained checks and seven
fault variants. The catalog now has 52 papers, 67 CPU/WASM modules, 527 checks
and 279 fault variants. This local review took place on 2026-10-14 Pacific.

## Source and contract

The [paper v1](https://arxiv.org/abs/1808.09781v1), Sections III-B and III-E,
and author [model.py](https://github.com/kang205/SASRec/blob/e3738967fddab206d6eeb4fda433e7a7034dd8b1/model.py)
and [modules.py](https://github.com/kang205/SASRec/blob/e3738967fddab206d6eeb4fda433e7a7034dd8b1/modules.py)
were reviewed. The functions implement causal query/key padding visibility and
the sum of positive and negative binary terms, averaged over valid target IDs.
The diagonal remains visible. Numerically stable logits replace the author's
sigmoid/log-with-small-constant expression. Interior/right padding, empty inputs
and ignored nonfinite logits are explicit teaching contracts.

Independent scalar loops, central finite differences, extreme logits, both
float precisions, input preservation, empty dimensions, zero ignored gradients
and future-state/embedding isolation support the ten maintained tests. Seven
faults cover future leakage, excluding self, visible padding keys, wrong mean
denominator, reversed negative sign, detached positive gradients and substituting
a pairwise objective.

## Actual history-dependent teaching experiment

Two histories `[0,1,3,4]` and `[0,2,3,4]` require different final targets, 5 and 6;
an all-padding row checks empty attention. The supplied encoder has fixed item
vectors, positional offsets and identity K/V. Adam trains a query projection
and readout: 72 parameters for 160 real updates through the learner's functions.

Any predictor using only the current item and position has an exact training
accuracy ceiling of 5/6, computed by majority labels within identical input
groups. This is an analytic bound, not another measured model. Final training
accuracy is 1; loss falls from 1.3862943611 to 0.0233127490. Native independent
scalar error is about 3.47e-18 (browser 6.94e-18); changing future items changes
past states by zero and the empty-row attention mass is zero.

An initial cyclic fixture could be solved from the current item alone and was
replaced before publication. The final history-dependent fixture requires older
context. This remains a small synthetic teaching adaptation: fixed embeddings,
no FFN/dropout/full SASRec stack, no real dataset or held-out Recall/NDCG.
Negatives rotate through non-target items and may occur in the history, unlike
the author's history-excluding sampler. Accuracy does not reproduce paper results.

## Verification

- Final full catalog CI: 67 modules, 8,262 cases over three repeats, 101.94
  seconds; all 279 faults caught.
- Final focused native CI: 240 cases over three repeats, 3.43 seconds; all seven
  faults caught. Final NumPy-adapter calibration test passes (1.89 seconds).
- Public-only regression: 1,119 passed, one optional Docker check skipped,
  124.18 seconds. The existing TorchScript deprecation warning remains.
- Full frontend checks pass on a snapshot containing only this change and the
  committed baseline; concurrent uncommitted UI edits are excluded. Standalone
  initial JS is 398,110 / 127,963 bytes raw/gzip; split API hosted build is
  403,773 / 129,888. The original 400 KiB raw / 130 KiB gzip caps remain.
- All 66 existing modules retain identical browser metadata and immutable detail
  URLs. The index is 256,934 bytes, below the original 256 KiB cap. Previously
  committed immutable assets are retained; the adapter/harness/worker is unchanged.
- Real browser CPU/WASM: reference 10/10 and seven calibrated faults. First
  observed phases are preparation 1.67 s, calibration 1.51 s, complete tests
  0.17 s. An explicitly submitted pairwise-loss fault scores 6/10; restoring
  the reference returns 10/10, reusing only page-local calibration.
- The learner's Lab shows its actual 160-step curve, attention bars and metrics
  separately from reference data. A 390 px iframe also executes 10/10; viewport,
  document and main widths are all 390 px. Desktop warning/error logs are empty.
  The iframe helper reports a MutationObserver observe/non-Node error on initial
  loading; its source is not established, so the iframe is not claimed error-free.
  It did not prevent editing, execution or result display.
- The local static QA server observed no API requests or POSTs. Timings are
  single localhost observations, not internet cold-start or Oracle benchmarks.

Private worker packaging and its pinned public core are verified separately.
No public push, production application/image deployment or official server score
is claimed. Browser feedback remains local until independently confirmed by the
server. The separately approved Nginx transfer repair is already deployed and
is documented in the private repository.
