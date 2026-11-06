# BEiT: masked discrete visual-token prediction

## 1. Problem

[BEiT](https://arxiv.org/abs/2106.08254) turns masked image modeling into classification of discrete visual tokens. An image tokenizer supplies target IDs; a masked image encoder predicts those IDs at removed patch positions. This exercise starts from supplied IDs and patch logits, isolating the prediction objective.

Implement `masked_token_cross_entropy(logits, token_ids, mask)`. Inputs are finite `[B,N,K]` vocabulary scores, int64 `[B,N]` targets and a Boolean `[B,N]` mask. `True` selects a masked position for supervision. Return one scalar mean loss.

## 2. Core equations

For selected patch $(b,n)$, score $z_{bnk}$ predicts target ID $y_{bn}$:

$$
p_{bnk}=\frac{\exp z_{bnk}}{\sum_j\exp z_{bnj}},\qquad
\ell_{bn}=\log\sum_k\exp z_{bnk}-z_{bn,y_{bn}}.
$$

With $M=\sum_{bn}m_{bn}$, the objective is

$$
\mathcal L=\begin{cases}\frac1M\sum_{bn:m_{bn}=1}\ell_{bn},&M>0,\\0,&M=0.\end{cases}
$$

Use a stable log-sum-exp or log-softmax. For a selected position, the logit gradient is $(p_{bnk}-\mathbf1[k=y_{bn}])/M$. Visible positions have zero gradient. Uniform logits give loss `log(K)`; a one-class vocabulary gives zero.

The [official training engine](https://github.com/microsoft/unilm/blob/master/beit/engine_for_pretraining.py) obtains IDs under `no_grad`, selects masked labels, and applies mean `CrossEntropyLoss` to masked logits. Returning differentiable zero for an empty mask and accepting sentinels in visible target slots are explicit exercise extensions.

## 3. From equations to code

1. Keep the supplied validation helpers: only masked IDs must be valid vocabulary indices.
2. Select the masked rows of logits and corresponding IDs. For an empty selection, return a scalar zero connected to the input logits.
3. Compute each selected target's score and stable vocabulary log partition.
4. Average the negative log probabilities over all selected tokens.

Do not feed probabilities into a logits-based cross entropy. Do not divide by batch size or average per-image means. Keep gradients through selected scores and their normalization.

## 4. Exercise conventions

- `B,N,K>=1`; logits have finite float32/float64 values. IDs are int64 and mask is Boolean. Their device and `[B,N]` axes match logits. Preserve dtype/device and inputs; support noncontiguous views.
- Masked IDs are in `[0,K)`. IDs at unselected positions may be arbitrary sentinel values and are never gathered.
- The mask marks positions to **predict**, not positions to ignore. Visible scores do not affect the loss or its gradient.
- Empty mask returns differentiable scalar zero; `K=1` is valid. Discrete IDs and mask do not receive gradients.
- Adding a separate constant to every patch's vocabulary scores preserves the loss. Consistent vocabulary relabeling and duplicated data also preserve it.

Independent tests include scalar log-sum-exp, exact softmax gradients and finite differences, unequal masked counts, sentinels, score shifts, vocabulary reversal, empty/full masks and scores of magnitude 1000.

## 5. Interactive charts and CPU experiment

The Lab trains a **shared four-class linear predictor** with 16 parameters. Twelve synthetic one-hot context vectors form three images; six positions are masked, covering all four labels, with mask counts `1,3,2`. Adam runs **100 actual updates at learning rate `.08`** using your function. The independent diagnostic recomputes CE and masked accuracy from Python scalar scores every ten steps.

Reference CPU CE falls from `log(4)≈1.3863` to about `.01292`, with masked accuracy `1.0`. T3 requires final CE below `.03` and below `.03` of baseline, accuracy `1.0`, and submitted/oracle agreement within `1e-12`. Charts display actual optimization and training-set accuracy. One-hot context makes this a bounded, learnable objective demonstration; it does not train a tokenizer, an image encoder or BEiT's full pretraining system.

## 6. Running

Complete the function and choose **Run tests**. All numerical, gradient and training checks support CPU/WASM, without external models or datasets.

```bash
paperleet test representation/beit --impl problems/representation/beit/solution.py
paperleet mutate representation/beit
paperleet ci --problem representation/beit
paperleet vizdata representation/beit
```

## 7. Further reading

Read BEiT Section 2 and the official `engine_for_pretraining.py`. How do the visual tokenizer's targets differ from raw pixels? Why are target IDs computed from the uncorrupted image? What changes when a tokenizer groups several visually different patches into one class?

Compare [MAE](https://arxiv.org/abs/2111.06377), which reconstructs pixel values, and [iBOT](https://arxiv.org/abs/2111.07832), which replaces hard tokenizer labels with centered soft teacher targets. Explain why iBOT's supplied patch-branch reduction differs from this global token mean.
