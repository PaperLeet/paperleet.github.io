# 105. VDP: Visual Prompts and Homeostatic Regularization

## Problem

[Visual Domain Prompt](https://arxiv.org/abs/2212.04145) adapts image inputs while freezing the source model. Implement prompt addition, historical homeostatic factors, and the domain-agnostic prompt penalty. Teacher inference, domain detection, importance estimation and teacher EMA are outside this module.

## Core equations

The numerical cores of Method Eqs. (1), (3) and (4) are:

$$
x'=x+p_{specific}+p_{agnostic},\qquad
\Lambda_i=\sum_{v=1}^{T}\frac{\eta_{v,i}}{\delta_{v,i}^2+\xi},
$$

$$
P(p)=\alpha\sum_i\Lambda_i(p_i-a_i)^2.
$$

`importance` supplies $\eta$, `displacement` supplies $\delta$, and `anchor` supplies $a$. Each completed domain contributes separately. There is no one-half in this penalty. The paper's prose and prompt-symbol labels around Eqs. (4)–(6) are inconsistent; this exercise uses semantic names and applies the penalty to the domain-agnostic prompt, following the method description.

## From equations to code

`apply_prompts(images, specific, agnostic)` broadcasts two `[C,H,W]` canvases over `[B,C,H,W]`. Prompt placement is already done: regions outside a prompt are zero.

`homeostatic_factor(importance, displacement, xi=0.01)` maps `[T,*S]` records to a detached tensor `S`. For one coordinate with importance `[2,3]`, displacement `[1,0]`, and xi `0.5`, the factor is `2/1.5 + 3/0.5`.

`homeostatic_penalty(prompt, anchor, factor, strength)` returns a scalar. Only current `prompt` is trainable.

## Exercise conventions

- Finite, same-dtype CPU float32/float64 inputs and finite intermediate arithmetic; no input validation or mutation. Preserve dtype/device and support noncontiguous inputs.
- Images have `B>=0` and positive `C,H,W`. Add without clipping. Preserve gradients to images and both current prompts.
- History has `T>=0`, with nonempty parameter shape `S`, nonnegative importance and `xi>0`. An empty history yields zeros; detach both historical records.
- Penalty tensors have identical shape `S`, nonnegative factor and `strength>=0`. Sum coordinates. Freeze anchor and factor even if they arrive with gradients enabled. Zero strength/factors produce differentiable zero.

## Interactive figures and experiment

A frozen identity readout sees two prompted pixels. The supplied teacher feature target is `[1,-1]`; history gives factors `[4,0]`. Only the domain-agnostic prompt trains; the specific prompt remains zero. Two arms run 120 SGD steps at learning rate `0.1`, strengths 0 and 1, using float64. Squared feature consistency `0.5*||p-target||²` deliberately replaces the paper's classification loss, giving independent optima `[1,-1]` and `[1/9,-1]`.

The Lab shows the adaptation/retention tradeoff. This controlled experiment excludes teacher EMA, corruption datasets and online domain detection; it does not reproduce full CTTA or claim guaranteed prevention of forgetting.

## Running the exercise

Complete the three functions and select **Run tests**. Tests cover independent scalar values, broadcasting gradients, domain additivity, zero history, frozen records, and actual prompt optimization against analytic optima.

```bash
paperleet test finetuning/visual_domain_prompt --impl problems/finetuning/visual_domain_prompt/solution.py
paperleet mutate finetuning/visual_domain_prompt
```

## Further questions

Why should a high-importance coordinate with small historical movement be constrained strongly? How does this coefficient differ from a supplied Fisher diagonal in EWC? Which additional components are needed for unlabeled online adaptation?
