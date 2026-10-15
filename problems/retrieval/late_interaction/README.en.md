# ColBERT: Token-Level MaxSim and Pairwise Ranking

## 1. Problem

A single pooled vector can collapse distinct token bags into the same direction. ColBERT keeps separate query/document token vectors and delays their interaction until scoring. Implement its cosine MaxSim core and paired ranking objective on supplied projected vectors. No encoder download, BERT training or GPU is required.

## 2. Core equations

### Return requirements

#### `maxsim_scores` (required)

Return one floating `Tensor` `[B,N]`, all pairs of B queries and N documents, preserving their shared dtype/device. Max over valid document tokens, then sum retained query-token evidence; do not average query tokens or return aligned-pair scores. All-masked documents score -inf; empty/fully masked queries score zero against valid documents. Keep both encoding gradients, with exact max ties routed to the first valid document token and masked gradients zero; do not mutate inputs.

#### `pairwise_loss` (required)

Return one scalar floating `Tensor`, shape `()`, of the score dtype/device, averaging the negative log-softmax probability of the positive over B nonempty score pairs. Retain gradients to both `[B]` score inputs. Do not return a per-pair vector, detach, convert to a Python float or mutate inputs.

#### `_checked_masks` (provided)

Provided helper: return `(query_mask, document_mask)`, two Boolean `Tensor`s of shapes `[B,Q]` and `[N,L]` on the encoding device. A supplied valid mask is returned unchanged; each None becomes an all-True mask. Reject invalid dimensions/dtype/device with `ValueError`. No input mutation or differentiable output.

#### `_check_pairs` (provided)

Provided validator: return Python `None` when positive and negative are matching, nonempty 1D float32/float64 score tensors on one device. Raise `ValueError` when a checked condition fails; do not mutate inputs or return transformed tensors.

### Formula and conventions

Normalize each vector with `eps=1e-12`, then match every kept query position to its best kept document position:

$$\widehat{x}=\frac{x}{\max(\|x\|_2,\epsilon)},\qquad S_{b,n}=\sum_{i:\,m^q_{b,i}}\max_{j:\,m^d_{n,j}}\widehat{q}_{b,i}^{\mathsf T}\widehat{d}_{n,j}.$$

For a finite positive/negative score pair, the negative log-softmax of the positive simplifies to a stable scalar loss:

$$\mathcal L=\frac1B\sum_b\log\left(1+\exp(S_b^- - S_b^+)\right).$$

These correspond to normalized encodings in §3.2 and cosine scoring / pairwise training in §3.3, equation 3 of the [original paper](https://arxiv.org/abs/2004.12832).

## 3. Equations to code

`maxsim_scores(query, documents, query_mask=None, document_mask=None)` returns all query/document combinations, not aligned pairs. For two query axis vectors, a document containing both axes scores 2; one blended vector scores $\sqrt2$. Normalize every token separately, reduce the **document token** dimension with max, then sum the **query token** dimension.

`pairwise_loss(positive, negative)` returns the mean loss of the supplied `[B]` pairs. Avoid directly exponentiating a large margin. `_checked_masks` and `_check_pairs` are supplied validation helpers; the two core functions are yours to implement.

## 4. Contract and paper boundaries

- Finite CPU float32/float64 inputs `query[B,Q,D]`, `documents[N,L,D]` share dtype/device. Return `[B,N]`, preserve floating dtype/device, support noncontiguous inputs and retain gradients to **both** token encodings. Do not mutate any input.
- Masks are Boolean `[B,Q]` and `[N,L]`; **True means keep**. Omitted masks keep every position. A document's padded positions must never win a maximum, including when all real similarities are negative.
- The paper's `[MASK]` **query augmentation** positions participate in scoring. `query_mask=False` here means absent batching positions, not all BERT `[MASK]` positions. There is no tokenizer or marker filtering in this task; callers supply encoded positions and masks.
- B/N/Q may be zero; physical L and D are positive. A valid document gets zero from an empty/fully masked query. An all-masked document always scores `-inf`, including with an empty query. This invalid-document convention is a teaching extension. Its excluded gradients must be finite zeros.
- `max(dim)` routes an exact tie to the first valid document token, matching PyTorch; this gradient convention is explicit rather than paper-specified. Scaling invariance applies when norms remain above epsilon.
- Pair scores are finite, nonempty, same-dtype/device `[B]`; return a scalar. Gradients must reach both positive and negative scores. The supplied validators reject malformed shapes/masks and incompatible dtypes.
- This implements the cosine variant. It does not implement the paper's squared-L2 alternative, BERT, projection training, query augmentation generation, punctuation filtering, ANN indexing or MS MARCO evaluation.

## 5. Interactive experiment

Lab contains four synthetic two-token queries and eight fixed documents. Each labeled positive keeps two separate axis vectors; its negative has their blended vector. These document pairs have exactly the same normalized mean direction, making the pooling limitation explicit without claiming a general benchmark advantage.

**My run results** calls your scorer and loss for all 120 Adam updates to the query vectors. It plots loss and positive-minus-negative margins, checks an independent Python scalar cosine oracle, and reports Recall@1 against the supplied four toy labels. Initial retrieval selects the blended negatives; the maintained target is all four relevant documents after optimization. This is query-vector optimization on synthetic data, not fine-tuning BERT or reproducing paper metrics. Reference plots remain separate from your actual T3 artifacts.

## 6. Run

```bash
paperleet test retrieval/late_interaction --impl problems/retrieval/late_interaction/solution.py
```

Browser only runs the unchanged full suite on CPU/WASM. T0 checks the contract, T1 a hand example, T2 scalar values, finite-difference gradients, masks, ties and stable ranking loss; T3 trains with the actual learner functions.

## 7. Further reading

Compare CLIP's single-vector contrastive matching, ColBERT's multi-vector late interaction and TIGER's generated semantic IDs in the Retrieval collection. These are different retrieval mechanisms; only the selected teaching cores are covered here.
