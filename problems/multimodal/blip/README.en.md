# BLIP momentum soft-target mixing

## 1. Problem

Implement `momentum_target_loss(logits: Tensor, teacher_logits: Tensor, positives: Tensor, alpha: float) -> Tensor`.

Implement BLIP momentum soft-target distillation for one retrieval direction with rectangular negative queues; omit queue updates, momentum encoders, and CapFilt data generation.

## 2. Core equations

$$
q_{ij}=\alpha\,\operatorname{stopgrad}(\operatorname{softmax}(s^m_i)_j)+(1-\alpha)\mathbf 1[j=p_i],\quad L=-\frac1B\sum_{i,j}q_{ij}\log\operatorname{softmax}(s_i)_j.
$$

## 3. From equations to code

Normalize teacher logits across all queue columns, then mix the soft distribution with one-hot positives. Each target row still sums to one, giving student gradient (softmax(logits)-targets)/B. Alpha zero recovers hard labels; alpha one uses only distillation. Momentum updates belong outside this loss, so teacher gradients must stop.

## 4. Conventions

logits and teacher_logits are [B,N], where N may include queued negatives. positives is int64 [B] of positive columns. alpha is in [0,1]; targets mix alpha*teacher_softmax with (1-alpha)*one_hot. Stop teacher gradients and average over B, not B*N. Inputs are already temperature-scaled.

CPU float32/float64; no input mutation. Finite inputs only. Tests use scalar oracles, boundary cases, appropriate gradient checks, and a submitted-function experiment.

## 5. Interactive figure

The deterministic CPU experiment calls your submitted function. The chart displays measured optimization, retrieval, or grouping behavior; inspect its metrics alongside the trace. Synthetic experiments do not reproduce paper benchmark scores.

## 6. Run

```bash
paperleet test multimodal/blip --impl problems/multimodal/blip/solution.py
paperleet mutate multimodal/blip
paperleet vizdata multimodal/blip
```

## 7. Extensions

Read the cited mechanism, then extend the isolated kernel to the paper’s complete architecture and data pipeline. The exercise fixes a narrow, testable contract; its teaching simplifications are stated above.

Section 3.1: image-text contrastive learning with momentum encoders and soft targets

- [https://arxiv.org/abs/2201.12086](https://arxiv.org/abs/2201.12086)
