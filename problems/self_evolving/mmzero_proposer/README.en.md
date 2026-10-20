# 112. MM-Zero Visual Curriculum Rewards

## 1. Problem

Compute proposer feedback for a generated visual curriculum: rendering gates the score, easy questions provide solvability feedback, hard questions provide consistency, and diversity adjusts the task mixture.

Implement `diversity_adjustment`, `proposal_rewards`.

## 2. Core equations

$$
r=\frac1N\sum_j e_j[\min(s_j,0.5)+\min(c_j,1-c_j)]+\sum_k a_k.
$$

## 3. From equations to code

Average over every render attempt: failures contribute zero and remain in the denominator. With one success in two attempts, easy=1 and c=0.5 yield base reward 0.5. Invalid proposal format overrides everything with -1.

## 4. Exercise conventions

Easy-question contribution caps at 0.5, as does difficulty. Diversity uses caption/easy/hard weights 0.45/0.20/0.35. M is the supplied valid proposal population, not necessarily this tensor’s row count. The other two adjustments are supplied.

Finite CPU inputs, no mutation; float32/float64 scores, long IDs and bool masks. The APIs below define all shape, tie and gradient conventions.

~~~python
def diversity_adjustment(shares: Tensor, valid_count: int, strength: float = .5) -> Tensor:
    """Float [B,3] caption/easy/hard cluster shares, valid_count>=1, strength>=0.
    Return detached [B]: -clip(dot(shares-1/M,[.45,.20,.35])*M*strength,
    -strength,+strength). M=valid_count is supplied, not necessarily B.
    Clustering and the valid population count are provided upstream."""
~~~

~~~python
def proposal_rewards(executed: Tensor, easy_accuracy: Tensor, consistency: Tensor, adjustments: Tensor, valid: Tensor) -> Tensor:
    """bool executed [B,N], float easy_accuracy/consistency [B,N] in [0,1],
    float adjustments [B,3] = easy-hard penalty, type penalty, diversity adjustment,
    bool format valid [B]. B,N>=1. Return detached same-dtype [B].
    Mean over ALL N render attempts of executed*(min(easy,.5)+min(c,1-c)),
    then add all three adjustments. Failed renders contribute zero but remain
    in the denominator. Invalid format overrides the ENTIRE result with -1.
    Adjustment construction (except diversity_adjustment) is supplied upstream."""
~~~

## 5. Interactive figures and experiment

Open Explanation to step through the paper’s co-evolution loop: active model, frozen evaluator, feedback and filtering. Controls recompute the displayed teaching scores. The Lab trains a small categorical policy for 60 gradient steps at learning rate 0.5, starting from uniform logits, using your functions. It uses fixed feedback records and CPU float64; target probability must exceed 0.72 and improve by more than 0.3. This checks the local mechanism, not full LLM/VLM self-evolution or benchmark gains.

## 6. Running the exercise

Complete the starter and use Run tests. T0 checks interfaces; T1 uses independent scalar/counting oracles; T2 checks boundaries, invariances and gradient ownership; T3 runs your implementation in the toy learning loop.

~~~bash
paperleet test self_evolving/mmzero_proposer --impl problems/self_evolving/mmzero_proposer/solution.py
paperleet mutate self_evolving/mmzero_proposer
~~~

## 7. Further questions

Why would averaging only successful renders conceal coder failures?

Source: [MM-Zero: Self-Evolving Multi-Model Vision Language Models From Zero Data](https://arxiv.org/abs/2603.09206), Section 2.4, Eqs. 4–10: execution gating, score balance and diversity.
