# TIGER: Generate IDs and Retrieve Valid Items

## 1. Problem

Module 64 encodes items into unique semantic IDs. Here a frozen `next_logits(prefix)` callback supplies conditional scores. Implement autoregressive beam search, then map complete IDs back to catalog items. This suite is independent; completing module 64 first is helpful but not required.

## 2. Core equation

$$s(c_{1:L})=\sum_{d=1}^{L}\log\operatorname{softmax}(f(c_{<d}))_{c_d}.$$

At each step expand every retained prefix over all V tokens and keep the global best b candidates. After L steps discard unknown full IDs and duplicate items, then take k. Keep original model log-scores without renormalization.

## 3. Equation to code

Implement `beam_search(next_logits, length, beam_width)` and `lookup_candidates(beams, catalog_ids, k)`. Start at `[((),0.0)]`. The callback accepts a token tuple and returns 1D logits. Normalize over the **entire vocabulary** at every prefix.

Hand example: first-step probabilities A=.6 and B=.4; next probabilities A→0=.51 and B→0=.9. Then A0=.306 and B0=.36. Width two ranks B0 first. Expanding only the first parent or keeping only the final token score is wrong.

## 4. Contract

- Integer length>=0, beam_width>=1. Callback always returns finite frozen CPU float64 `[V]`, with fixed V>=1. Length zero returns the initial beam without invoking it. Do not mutate logits.
- Return `list[(tuple[int,...], float)]`, sorted by descending cumulative log-score, then lexicographic complete token tuple for exact ties. Keep all candidates when fewer than beam_width exist. Fixed length, no EOS, length penalty or sampling.
- Lookup receives ranked beams, possibly repeated, and unique CPU int64 catalog rows `[N,L]`; integer k>=0. Remove unknown IDs and duplicate items **before** taking k. Return `(catalog row index, original score)` preserving order. Zero k, empty catalog or no valid IDs gives `[]`. Do not mutate inputs.
- Inputs satisfy the contract; validation is not required. Tokens are interpreted by position in this toy, unlike the paper's level-specific vocabulary. A fixed callback replaces the full user-history encoder-decoder: no T5 training or relevance evaluation.
- Section 4.5 suggests increasing beam size then filtering invalid IDs; prefix matching is future work. This exercise has **no trie constraint** or catalog-only normalization. A finite beam is approximate; arbitrary models need not improve monotonically with wider beams.

## 5. Interactive experiment

The same 8 catalog IDs as module 64, five tokens and four positions give 625 possible IDs. Independent scalar enumeration ranks this **fixed model** exactly. Compare widths 1/2/4/8/16/625; the last covers the whole tree. Top-3 overlap measures approximation to model ranking, not user relevance Recall/NDCG. Charts use your actual generation and lookup functions.

## 6. Run

```bash
paperleet test retrieval/generative_retrieval --impl problems/retrieval/generative_retrieval/solution.py
```

Browser only runs on CPU/WASM. T0 checks identity and empty inputs; T1 is hand-computed; T2 checks scalar oracles, prefix shifts, cumulative scores, global pruning, ties and filter order; T3 runs the beam-width experiment with your submission.

## 7. Further reading

Read [TIGER Sections 3.2 and 4.5](https://arxiv.org/html/2305.05065). Trie constraints, temperature sampling and learned logits are separate extensions that change decoding rules. Real recommendation evaluation needs user histories, train/validation separation and relevance labels; exact toy-model ranking cannot replace it.
