# TIGER: Residual Quantization and Semantic IDs

## 1. Problem

TIGER represents an item as a semantic ID, then generates IDs to retrieve items. Implement **frozen residual encoding** and collision suffixes. Latents and codebooks are supplied: no RQ-VAE training, downloads or GPU. Continue with module 65 to generate these IDs and map them to catalog items.

## 2. Core equations

Start with $r_0=z$. Each level has a distinct codebook $C_d$:

$$c_d=\arg\min_j\|r_d-C_d[j]\|_2^2,\quad r_{d+1}=r_d-C_d[c_d],\quad \hat z=\sum_d C_d[c_d].$$

Section 3.1 describes residual quantization; Section 4 appends a fourth codeword to distinguish items sharing the first three. This exercise generalizes to any positive number of levels and explicitly fixes suffix assignment to input occurrence order.

## 3. Equations to code

Implement `residual_quantize(latents, codebooks)` returning indices and reconstruction, and `resolve_collisions(codes)` appending per-tuple ordinals. Example: `[[7,1,4],[8,2,0],[7,1,4]]` becomes `[[7,1,4,0],[8,2,0,0],[7,1,4,1]]`.

For `z=4.8`, book `[0,4,-4]` selects 4, leaving .8; next book `[0,.75,-.75]` selects .75. Reconstruction is 4.75 with codes `[1,1]`. Later books quantize the residual, not the original latent.

## 4. Contract

- Finite CPU float32/float64 `latents[B,D]` and `codebooks[L,K,D]` share dtype. B may be zero; L/K/D are positive. Noncontiguous inputs are supported.
- Choose each level's nearest code; exact ties select its smallest index. Return int64 codes `[B,L]` and frozen reconstruction `[B,D]` preserving floating dtype/device. Neither output requires gradients; do not mutate inputs.
- Collision input is nonnegative CPU int64 `[B,L]`; output is int64 `[B,L+1]` on the same device. Count previous identical **complete tuples**, giving 0,1,2,... per group; singletons also get 0. Do not mutate input.
- Inputs satisfy this contract; validation is not required. Occurrence ordering is an explicit teaching convention, not a paper-specified item order. A production catalog should persist mappings rather than silently reassigning suffixes after edits.
- The paper trains encoder, decoder and codebooks, using three levels of 256 vectors of dimension 32. This task covers frozen encoding only, without straight-through gradients, reconstruction training or paper metrics.

## 5. Interactive experiment

Lab uses 8 synthetic two-dimensional items and three five-entry books. **My run results** invokes your two functions; reference results remain separate. Independent scalar distances check selected codes and reconstruction; suffixes produce 8 distinct full IDs. Because each supplied book contains zero, this example's residual error cannot increase with depth. That is not guaranteed for arbitrary books.

## 6. Run

```bash
paperleet test retrieval/semantic_ids --impl problems/retrieval/semantic_ids/solution.py
```

Browser only runs the full suite on CPU/WASM. T0 covers shapes, precision, frozen outputs and empty batches; T1 is hand-computed; T2 checks scalar oracles, ties, reconstruction and collision groups; T3 calls your functions for the catalog experiment.

## 7. Further reading

Read [TIGER Sections 3.1 and 4](https://arxiv.org/html/2305.05065), then compare Rotation Trick's training gradients. Training an RQ-VAE requires a separate dataset and objective; this toy is not an Amazon recommendation benchmark.
