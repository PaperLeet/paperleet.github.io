# 109. G-Zero Hint-Delta

## 1. Problem

Score the same unassisted answer under the original and hint-augmented contexts, then combine its token probability shift with hint-length and repetition penalties.

Implement `hint_delta`, `proposer_rewards`.

## 2. Core equations

$$
\delta=\frac{\sum_t m_t(\log p_t-\log p_t^h)}{\max(1,\sum_t m_t)},\quad r=\delta-0.03\max(L-200,0)/100-s.
$$

## 3. From equations to code

Align both scoring passes to the same response tokens; the hint itself is excluded. Token shifts [2,1] yield delta=1.5. Empty rows return zero. Invalid format replaces delta and length terms with -1, while retaining the cluster penalty.

## 4. Exercise conventions

Delta is a probability-shift proxy, not measured quality gain or a full KL divergence. Do not compare different responses. Length is in characters. Keep negative rewards. Hint-delta is differentiable; final rewards are fixed.

Finite CPU inputs, no mutation; float32/float64 scores, long IDs and bool masks. The APIs below define all shape, tie and gradient conventions.

~~~python
def hint_delta(base_logp: Tensor, hinted_logp: Tensor, mask: Tensor) -> Tensor:
    """Float log-probabilities [B,T], bool mask [B,T]; same UNASSISTED answer tokens
    scored under two aligned contexts. Return float [B], mean(base-hinted) over
    valid tokens. All-masked rows return 0. Finite inputs, no mutation.
    Differentiable in both log-prob tensors; masked positions have zero gradient."""
~~~

~~~python
def proposer_rewards(delta: Tensor, hint_lengths: Tensor, cluster_share: Tensor, valid: Tensor) -> Tensor:
    """Float delta/share [B], integer hint character lengths [B], bool valid [B].
    Valid reward: delta - .03*max(length-200,0)/100 - cluster_share.
    Invalid reward: -1 - cluster_share, ignoring delta and length. Return detached
    same-dtype [B]; negative rewards are retained. Clustering is supplied upstream."""
~~~

## 5. Interactive figures and experiment

Open Explanation to step through the paper’s co-evolution loop: active model, frozen evaluator, feedback and filtering. Controls recompute the displayed teaching scores. The Lab trains a small categorical policy for 60 gradient steps at learning rate 0.5, starting from uniform logits, using your functions. It uses fixed feedback records and CPU float64; target probability must exceed 0.72 and improve by more than 0.3. This checks the local mechanism, not full LLM/VLM self-evolution or benchmark gains.

## 6. Running the exercise

Complete the starter and use Run tests. T0 checks interfaces; T1 uses independent scalar/counting oracles; T2 checks boundaries, invariances and gradient ownership; T3 runs your implementation in the toy learning loop.

~~~bash
paperleet test self_evolving/gzero_hint --impl problems/self_evolving/gzero_hint/solution.py
paperleet mutate self_evolving/gzero_hint
~~~

## 7. Further questions

Why might a large delta caused by answer leakage mislead training?

Source: [G-Zero: Self-Play for Open-Ended Generation from Zero Data](https://arxiv.org/abs/2605.09959), Section 3.1, Eq. 3; Section 3.2, Eq. 5.
