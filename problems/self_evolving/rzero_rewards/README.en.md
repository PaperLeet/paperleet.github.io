# 107. R-Zero Challenger Rewards

## 1. Problem

Train a challenger toward uncertain solver responses while penalizing repeated question templates. Answer normalization, clustering and format checks are supplied.

Implement `uncertainty_reward`, `challenger_rewards`.

## 2. Core equations

$$
r_i=v_i\max(0,1-2|p_i-1/2|-\lambda |C_i|/B).
$$

## 3. From equations to code

Compute triangular uncertainty, then the cluster share over the whole batch, then floor at zero and gate by format. With p=0.5 and two duplicates in a batch of four, reward is 0.5; invalid format returns zero.

## 4. Exercise conventions

Agreement is not accuracy: unanimous wrong answers still give p=1. Invalid rows remain in this exercise’s clustering denominator. Observed rewards are detached.

Finite CPU inputs, no mutation; float32/float64 scores, long IDs and bool masks. The APIs below define all shape, tie and gradient conventions.

~~~python
def uncertainty_reward(consistency: Tensor) -> Tensor:
    """consistency: float [B] in [0,1]. Return detached same-dtype [B]: 1-2*abs(p-.5)."""
~~~

~~~python
def challenger_rewards(consistency: Tensor, clusters: Tensor, valid: Tensor, strength: float = 1.) -> Tensor:
    """Float consistency [B], long cluster IDs [B], bool valid [B]; B>=1, strength>=0.
    Count clusters across ALL B proposals, including invalid ones. Return detached
    float [B]: valid * max(0, uncertainty - strength*cluster_size/B). No mutation.
    Clustering/format parsing is supplied upstream; cluster IDs need not be contiguous."""
~~~

## 5. Interactive figures and experiment

Open Explanation to step through the paper’s co-evolution loop: active model, frozen evaluator, feedback and filtering. Controls recompute the displayed teaching scores. The Lab trains a small categorical policy for 60 gradient steps at learning rate 0.5, starting from uniform logits, using your functions. It uses fixed feedback records and CPU float64; target probability must exceed 0.72 and improve by more than 0.3. This checks the local mechanism, not full LLM/VLM self-evolution or benchmark gains.

## 6. Running the exercise

Complete the starter and use Run tests. T0 checks interfaces; T1 uses independent scalar/counting oracles; T2 checks boundaries, invariances and gradient ownership; T3 runs your implementation in the toy learning loop.

~~~bash
paperleet test self_evolving/rzero_rewards --impl problems/self_evolving/rzero_rewards/solution.py
paperleet mutate self_evolving/rzero_rewards
~~~

## 7. Further questions

Why is the hardest question not necessarily the most useful training question?

Source: [R-Zero: Self-Evolving Reasoning LLM from Zero Data](https://arxiv.org/abs/2508.05004), Section 2.2: uncertainty, repetition and format gating.
