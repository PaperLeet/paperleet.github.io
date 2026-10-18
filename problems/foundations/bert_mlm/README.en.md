# BERT: Corruption and Masked Language Modeling

## 1. What problem does the paper solve?

BERT recovers selected original tokens using bidirectional context. This exercise implements the corruption and objective in Section 3.1, rather than the complete encoder. Selection and random draws come from the caller, making your functions deterministic and independently inspectable.

[原始来源 / Original source](https://arxiv.org/html/1810.04805v2#S3.SS1) · Section 3.1, Task #1: Masked LM; Appendix C.2: masking procedures.

## 2. Core equations and return contract

$$C_{b,t}=\begin{cases}\mathrm{MASK}&S_{b,t}\land u_{b,t}<0.8\\R_{b,t}&S_{b,t}\land0.8\le u_{b,t}<0.9\\X_{b,t}&\mathrm{otherwise}\end{cases}\qquad L=-\frac{1}{|S|}\sum_{(b,t)\in S}\log\operatorname{softmax}(Z_{b,t})_{X_{b,t}}.$$

Required functions: `make_mlm_inputs`, `masked_lm_loss`.

## 3. From equations to code

Build disjoint Boolean conditions for MASK and random replacement. Labels always come from the original tokens. Gather stable log-softmax values and divide by the number of non-ignored targets.

## 4. Exercise conventions

For `make_mlm_inputs`, all four arrays have shape `[B,L]`: nonnegative int64 tokens/random_ids, bool selected, and finite float32/float64 draws in `[0,1)`. Return independent int64 `(corrupted,labels)` arrays without modifying inputs. At selected sites use mask_token_id for `u<0.8`, random_ids for `0.8<=u<0.9`, and the original token otherwise. Unselected tokens stay unchanged and receive the negative integer ignore_index (default -100). Selected labels always hold the original token; **do not shift targets**. The nonnegative integer mask_token_id may lie outside the prediction vocabulary. The caller handles 15% selection, special-token exclusion and replacement sampling.

`masked_lm_loss` takes finite float32/float64 logits `[B,L,V]`, `V>0`, and int64 labels `[B,L]` on the same device. Targets must be ignore_index or in `[0,V)`. Return a scalar in the logit dtype, averaged over non-ignored sites. With no targets, including an empty batch/sequence, return graph-connected zero with zero logit gradients. Invalid shapes, dtypes, draws or target IDs raise ValueError.

## 5. Experiment

The supplied 64 three-token sequences cover 16 ordered left/right pairs; the center is `(2*left+right)%4`. Only centers are eligible, selected at rate 15%. Your corruption function supplies inputs/labels. A 16×4 conditional table reads only the corrupted left/right tokens and trains through your MLM loss for 100 Adam updates (lr=0.08, seed=163, float64). The chart evaluates all centers. Acceptance requires final NLL<0.1 and <10% of initial, with context accuracy>=95%. This is a synthetic conditional-table experiment, without a Transformer or natural-language generalization claim.

## 6. Run your implementation

Complete the starter and run all tests in the workspace to unlock Notes. Maintainers can also run:

```bash
paperleet test foundations/bert_mlm --impl problems/foundations/bert_mlm/solution.py
```

## 7. Further exploration

Derive the selected-site softmax gradient and compare empty supervision with a single label. Read Appendix C.2 before experimenting with corruption variants. Reproducing BERT also requires its encoder, NSP and corpus pipeline.
