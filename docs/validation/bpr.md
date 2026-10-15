# BPR-MF lesson

Paper 53 / module 68 adds BPR's observed/unobserved preference construction and
matrix-factorization ranking objective. Primary sources are the
[UAI publication record](https://proceedings.mlr.press/r7/rendle09a.html),
[original paper](https://auai.org/uai2009/papers/UAI2009_0139_48141db02b9f0b02bc7158819ebfa2c7.pdf)
and [arXiv archive](https://arxiv.org/abs/1205.2618), checked for Sections 3.2,
4.1 and 4.3.1. Statements and notes are bilingual. No published benchmark score
or full LearnBPR reproduction is claimed.

## Learning contract

`eligible_negatives` excludes all items observed by each selected user, without
using other users' histories. `bpr_mf_loss` gathers user/positive/unobserved rows,
minimizes mean stable negative log-sigmoid of the dot-score difference, and adds
one full-table Gaussian-prior penalty. The declared coefficient multiplies the
sum of all squared factor entries, without one half; it is at the normalized
likelihood scale. Empty triples retain the prior and both gradient graphs.

## Zero-penalty numerical regression

The initial implementation evaluates table norms even when `l2=0`. A finite,
unused float32 row of magnitude `1e20` makes that norm overflow and turns a
finite pair loss into NaN through `0*Inf`. The correction skips the disabled
prior. A dedicated T2 check covers both float32 (`1e20`) and float64 (`1e200`)
unused rows, a directly computed scalar likelihood and both complete gradient
tables. The old implementation fails this check at 10/11. The corrected implementation and an independent log-sigmoid learner pass all eleven checks in actual Pyodide CPU/WASM.

Eleven checks include shape/precision/strides/empty axes, supplied validation,
hand-computed values, independent scalar likelihoods and central differences
through both factor tables, repeated rows, exact prior gradients on unselected
rows, batch replication, item translation and extreme margins. They import
learner implementations through `impl`, never the reference directly.

Eight small executable faults cover reversed eligibility, global history,
reversed ranking, detached positives, selected-row regularization, averaging
features, independent binary terms and summing rather than averaging triples.
All eight are caught in three stable native repeats: **297 cases in 4.50 s**.
The complete exported NumPy calibration also passes all eleven checks and rejects
all eight faults. Its shared runtime is unchanged.

## Genuine personalized training

Three users prefer disjoint two-item groups in a six-item catalog. The learner's
mask supplies all 24 observed/unobserved training triples, and its loss trains
27 factor entries for 120 full-batch Adam updates. A directional count bounds
any single global item-score order at 1/2 strict pair accuracy. The reference
reaches 1.0 pair accuracy and training Recall@2, with loss 0.6942681754 to
0.03493172158 and minimum final margin 5.1590. Scalar objective error is zero in
the recorded native run. Losses and per-user item scores are genuine artifacts
from these updates.

This is an explicitly labelled teaching adaptation: full-batch Adam and one
global prior, without LearnBPR bootstrap SGD, per-role regularizers, real users,
public datasets or held-out evaluation. The paper's summed likelihood and this
exercise's normalization/coefficient relationship are explained in both
languages.

## Final regression and actual browser

- Full native catalog CI passes **8,559 test instances over three repeats** for
  all 68 modules in **113.86 s**, within the original 300-second budget. It
  preserves all previous lessons and catches all 287 maintained faults.
- Final public regression: **1,214 passed, one optional Docker skip**, in **179.01 s**, using `python -m pytest -q tests`. The initial unscoped pytest invocation also collected 538 exercise cases without learner files and failed fixture setup; lesson correctness is instead covered by the passing full catalog CI. Concurrent UI/account changes are excluded from the isolated source snapshot. All 67 pre-existing module entries, 52 paper records and previously committed immutable assets remain intact; shared runtime bytes are unchanged.
- Real Pyodide CPU/WASM executes a separate correct learner using log-sigmoid
  and separate positive/negative dot products: **11/11**, all eight faults
  calibrated and all 120 actual training updates. Its Lab displays the generated
  objective curve and the three users' learned item scores, with training pair
  accuracy and Recall@2 both 1.0.
- The old zero-prior implementation fails exactly the new numerical check at **10/11**, showing expected `0.3132616877555847` versus NaN in the browser. The corrected independent implementation passes **11/11**, including its 120 actual training updates. The initial lesson verification also rejected the wrong independent-binary objective and preserved source on a native-only matrix 1-norm probe; those observations preceded this numerical correction.
- The initial lesson verification covered the discovery route: homepage → Retrieval collection → paper 53 →
  module 68. The paper's arXiv-labelled link targets its arXiv archive. A completed
  paper remains 1/1 after refresh; revisiting and downloading the implementation
  returns the exact saved 3,008-character alternate source. The editor virtualizes
  its DOM, so full-source fidelity is checked through the download feature.
- Final corrected localhost phases: preparation **1.57 s**, complete reference
  and mutation calibration **1.11 s**, learner tests **0.11 s**. These are not
  internet cold-start or Oracle hardware comparisons. No new live hosted
  submission, account credit or Docker image is claimed.

- The complete frontend checks pass in their standalone test mode, followed by
  separate standalone and hosted production builds. Final graphs are **399,891
  raw / 129,233 gzip bytes** standalone and **405,267 raw / 131,049 gzip bytes**
  hosted (Node 24), within unchanged 400 KiB / 130 KiB limits. Concurrent account
  and UI changes are excluded from the verified source snapshot and commit.
- Initial English layout verification has document/main width 390 at a 390-pixel viewport. The corrected bilingual statements now use separated display-math blocks; actual Chinese and English pages show the complete objective and all subsequent headings. The main
  browser warning/error log is empty, and the static QA server records zero API
  requests or POSTs. Temporary tabs, downloaded verification file, server and
  build snapshot are cleaned up; ignored evidence is retained.

Existing suites, tolerances, training budgets and immutable assets remain intact.
No push or production deployment is performed. Private pinned-core and native
worker validation are recorded separately.
