# 106. Neural Thickets: Random Experts and Majority Voting

## Problem

Implement the core of [RandOpt](https://arxiv.org/abs/2603.12228): create independent parameter perturbations, select the best experts, then ensemble their discrete answers. This module separates selection-set scores from test predictions. It does not perform iterative ES updates or average model weights.

## Core equations

Section 4, Eqs. (3)–(5) and Algorithm 1 define:

$$
\theta_i=\theta+\sigma_i\epsilon_i,\qquad
I=\operatorname{TopK}(s_1,\ldots,s_N),
$$

$$
\hat y_b=\operatorname{mode}\{y_{i,b}:i\in I\}.
$$

The caller supplies noise, per-candidate scales, selection scores and parsed class IDs. The voting unit is each expert's final answer, not tokens, probabilities or parameter coordinates.

## From equations to code

- `perturb_candidates(theta, noise, scales)`: `theta` is shape `S`, `noise` is `[N,*S]`, and `scales` is `[N]`. Return `[N,*S]`, starting every candidate from the same base.
- `select_experts(scores, k)`: return long indices `[k]`, ordered by descending score.
- `majority_vote(predictions, selected, num_classes)`: predictions are long `[N,B]`; return one long class ID per sample, shape `[B]`.

For scores `[0.9,0.1,0.8]` and k=2, select `[0,2]`. If those experts answer `[2,0]` and `[2,1]`, output `[2,0]`: the second sample's tie selects the smaller class ID.

## Exercise conventions

- Finite CPU float32/float64 parameter/noise/scale inputs share dtype/device, with finite intermediate arithmetic. Parameter shape `S` is nonempty or scalar; noncontiguous views are allowed. Return detached candidates without mutating any input.
- `N>=1`, scales are nonnegative, and `1<=k<=N`. Score ties preserve lower original indices. This deterministic tie rule is an exercise convention.
- Predictions use integer classes in `[0,num_classes)`; `num_classes>=1`. Selected indices are distinct and nonempty. Every selected expert gets one vote. Tied counts choose the smallest class ID; this too is an explicit exercise convention.
- `B=0` returns an empty long vector. No prediction decoding, string normalization or input validation is required. Test labels must never influence expert selection.

## Interactive figures and experiment

The supplied experiment draws 120 two-parameter linear classifiers around base `[0.2,0.8]`, with scales 0.25/0.5/1 and seed 42. Scores come from 41 points in `[-2,2]`; the top 9 vote on 80 different points in `[-1.975,1.975]`. The target is the sign of x. In CPU float64 the reference improves test accuracy from 0.5 to 0.9. Tests require at least 0.9.

These actual candidate evaluations illustrate the pipeline, not the paper's large-model results or the existence of dense experts around arbitrary pretrained models. The example does not establish that an ensemble always beats the strongest individual expert.

## Running the exercise

Complete all three functions and choose **Run tests**. Checks cover independent per-candidate scales, stable selection, an independent Python vote counter, rejected-expert isolation, ties and empty test batches. T3 uses all three submitted functions.

```bash
paperleet test finetuning/neural_thickets --impl problems/finetuning/neural_thickets/solution.py
paperleet mutate finetuning/neural_thickets
```

## Further questions

What inference cost does retaining K experts introduce? Why is voting different from weight averaging? Why should expert selection use a separate labeled set? Compare this procedure with ES, which combines reward-weighted directions into one updated parameter vector.
