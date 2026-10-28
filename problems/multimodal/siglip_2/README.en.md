# SigLIP 2 masked patch self-distillation

## 1. Problem

Implement `masked_patch_distillation(student: Tensor, teacher_prob: Tensor, mask: Tensor, temperature: float) -> Tensor`.

Implement the probability-distillation kernel of SigLIP 2 masked prediction with supplied teacher targets; exclude EMA encoders, augmentation, projection heads, and the full training recipe.

## 2. Core equations

$$
L=-\frac{\sum_{b,p}m_{bp}\sum_k\operatorname{stopgrad}(q_{bpk})\log\operatorname{softmax}(z_{bp}/\tau)_k}{\sum_{b,p}m_{bp}}.
$$

## 3. From equations to code

Teacher inputs are already probabilities; applying another softmax changes their meaning. Stop their gradients, compute cross-entropy per patch, and divide the masked sum by the number of selected patches. Student gradients are zero outside the mask and (softmax(student/tau)-teacher)/(M*tau) inside it. This kernel isolates one auxiliary objective.

## 4. Conventions

student and teacher_prob have shape [B,P,K]. Teacher probabilities are nonnegative and sum to one over K; teacher projection and normalization are already done. mask is boolean [B,P] with at least one selected patch. temperature is a positive Python float. Average cross-entropy only over masked patches, stop teacher gradients, and preserve student gradients.

CPU float32/float64; no input mutation. Finite inputs only. Tests use scalar oracles, boundary cases, appropriate gradient checks, and a submitted-function experiment.

## 5. Interactive figure

The deterministic CPU experiment calls your submitted function. The chart displays measured optimization, retrieval, or grouping behavior; inspect its metrics alongside the trace. Synthetic experiments do not reproduce paper benchmark scores.

## 6. Run

```bash
paperleet test multimodal/siglip_2 --impl problems/multimodal/siglip_2/solution.py
paperleet mutate multimodal/siglip_2
paperleet vizdata multimodal/siglip_2
```

## 7. Extensions

Read the cited mechanism, then extend the isolated kernel to the paper’s complete architecture and data pipeline. The exercise fixes a narrow, testable contract; its teaching simplifications are stated above.

Section 2.3: masked prediction uses stopped teacher targets at masked patch locations

- [https://arxiv.org/abs/2502.14786](https://arxiv.org/abs/2502.14786)
