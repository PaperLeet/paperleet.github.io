# SASRec: Causal Histories and Sampled Next-Item Loss

Predict the next item from an observed history. During training the whole sequence is available, so each position must be prevented from reading its future target. This exercise selects the causal constraint and training objective from [SASRec Sections III-B and III-E](https://arxiv.org/abs/1808.09781v1); the encoder and optimization loop are provided.

## 1. History visibility

`causal_item_mask(item_ids)` maps nonnegative int64 `[B,T]` IDs to a Boolean `[B,T,T]` mask. ID 0 means padding. Query i can read key j exactly when both IDs are nonzero and `j <= i`:

$$M_{b,i,j}=[s_{b,i}\ne0][s_{b,j}\ne0][j\le i].$$

Keep the diagonal: the current item is already observed; the target is the next item. Mask both padding queries and keys. Left, right, interior and all-padding histories are supported. For `[0,4,2]`, the rows are `[0,0,0]`, `[0,1,0]`, `[0,1,1]`.

## 2. Position-wise binary objective

`sampled_next_item_loss(positive, negative, target_ids)` takes equal `[B,T]` shapes. Positive and negative values are unnormalized logits with matching float32/float64 precision; target IDs are int64. The harness already selected the positive/negative scores. Let $V=\{(b,t):o_{b,t}\ne0\}$:

$$L=\frac{\sum_{(b,t)\in V}[\operatorname{softplus}(-r^+_{b,t})+\operatorname{softplus}(r^-_{b,t})]}{\max(1,|V|)}.$$

Use two binary terms, rather than BPR or pairwise softmax. Divide by valid target positions, excluding padding, without an additional factor of two. The paper writes a summed objective; this exercise follows the author implementation's valid-position mean with stable logits, avoiding the extreme-value saturation of sigmoid/log plus small constants.

The loss mask comes from **target IDs**, not the history length. Valid-position logits must be finite. Ignored positions may contain NaN/Inf and must be masked before loss arithmetic. Empty dimensions or no valid targets return a differentiable zero with zero gradients for both inputs. Preserve dtype/device and do not mutate inputs. `_check_ids` and `_check_loss` are supplied; retain the starter's validation calls.

## 3. Implement and verify

Broadcast nonpadding IDs over both attention axes and combine with `j <= i`. Mask ignored logits, compute stable binary terms and reduce over valid positions. Preserve both logit gradients.

T0 checks contracts, precision, empty inputs and mutation. T1 is hand-computable. T2 uses scalar Python oracles, finite differences, element-wise mask checks, extreme logits and future isolation. T3 calls the learner functions during the entire teaching experiment.

## 4. Teaching experiment

Two histories `[0,1,3,4]` and `[0,2,3,4]` end in the same item 4 but require different next items, 5 and 6. An empty row checks padding. Fixed item vectors, identity K/V, residual LayerNorm and fixed position offsets support learned query/readout projections with 72 trainable parameters. Your mask controls attention and your loss supplies gradients for **160 actual Adam updates**.

Charts show training loss and attention over four keys for each query of the first history. Negatives rotate, so the loss curve need not decrease at every checkpoint. Any predictor using only the current item and position has accuracy at most 5/6 here. The experiment must exceed this exact majority-vote bound using earlier history; it is not a measured model baseline. Metrics independently verify the scalar objective, unchanged past states after future perturbation, and zero attention in the empty row.

This is a **PaperLeet teaching adaptation**. Rotating negatives exclude only the positive target and may occur in history, unlike the author's outside-history random sampling. There is no FFN, dropout, fully learned embedding/projection stack or real user dataset. Accuracy describes these training positions, not held-out Recall/NDCG or the original recommendation benchmarks.

## 5. Prerequisites

The Attention and LoRA modules provide useful matrix/gradient practice before this history-visibility and training-objective exercise.

## 6. Run in the browser

Complete the two functions in the Python editor and run all tests. CPU/WASM needs no GPU, account, model download or external API. Unsupported custom Python/PyTorch operations retain the server path.

## 7. Sources and next steps

See the [paper](https://arxiv.org/abs/1808.09781v1), pinned author [training implementation](https://github.com/kang205/SASRec/blob/e3738967fddab206d6eeb4fda433e7a7034dd8b1/model.py) and [mask implementation](https://github.com/kang205/SASRec/blob/e3738967fddab206d6eeb4fda433e7a7034dd8b1/modules.py). Continue with TIGER semantic IDs and generative retrieval to compare direct item scoring with ID generation.
