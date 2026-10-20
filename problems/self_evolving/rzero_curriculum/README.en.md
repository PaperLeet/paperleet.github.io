# 108. R-Zero Pseudo-Labels and Curriculum

## 1. Problem

Turn historical answer votes into frozen pseudo-labels, retain an informative consistency band, and reward fresh answers against those labels.

Implement `consensus`, `curriculum_mask`, `pseudo_rewards`.

## 2. Core equations

$$
\tilde y_i=\arg\max_c n_{ic},\quad p_i=\max_c n_{ic}/K,\quad m_i=\mathbf1[\ell\le p_i\le u],\quad r_{ij}=\mathbf1[a_{ij}=\tilde y_i].
$$

## 3. From equations to code

Count along each question’s rollout axis. Votes [1,1,0,0] tie, so this exercise chooses 0 with consistency 0.5. Against frozen label 1, fresh answers [2,2,1] earn [0,0,1].

## 4. Exercise conventions

Smallest-ID tie breaking is a deterministic teaching convention. Include both 0.25 and 0.75. Do not refresh the label from fresh answers: that changes the training target. Answer equivalence is provided upstream.

Finite CPU inputs, no mutation; float32/float64 scores, long IDs and bool masks. The APIs below define all shape, tie and gradient conventions.

~~~python
def consensus(answers: Tensor, num_answers: int) -> tuple[Tensor, Tensor]:
    """answers: long [B,K], B,K>=1, IDs in [0,num_answers). Answer equivalence is supplied.
    Return (labels: long [B], consistency: float64 [B]) on the input device.
    Ties choose the smallest answer ID. Count ALL K answers. No mutation."""
~~~

~~~python
def curriculum_mask(consistency: Tensor, low: float = .25, high: float = .75) -> Tensor:
    """Float [B], 0<=low<=high<=1. Return bool [B], retaining BOTH boundary values."""
~~~

~~~python
def pseudo_rewards(answers: Tensor, labels: Tensor) -> Tensor:
    """New long answers [B,K], frozen pseudo-labels [B]. Return float64 [B,K].
    Match against the supplied OLD labels, never recompute a vote from new answers.
    These are agreement rewards, not independent correctness verification."""
~~~

## 5. Interactive figures and experiment

Open Explanation to step through the paper’s co-evolution loop: active model, frozen evaluator, feedback and filtering. Controls recompute the displayed teaching scores. The Lab trains a small categorical policy for 60 gradient steps at learning rate 0.5, starting from uniform logits, using your functions. It uses fixed feedback records and CPU float64; target probability must exceed 0.72 and improve by more than 0.3. This checks the local mechanism, not full LLM/VLM self-evolution or benchmark gains.

## 6. Running the exercise

Complete the starter and use Run tests. T0 checks interfaces; T1 uses independent scalar/counting oracles; T2 checks boundaries, invariances and gradient ownership; T3 runs your implementation in the toy learning loop.

~~~bash
paperleet test self_evolving/rzero_curriculum --impl problems/self_evolving/rzero_curriculum/solution.py
paperleet mutate self_evolving/rzero_curriculum
~~~

## 7. Further questions

How could you detect a highly consistent but incorrect pseudo-label?

Source: [R-Zero: Self-Evolving Reasoning LLM from Zero Data](https://arxiv.org/abs/2508.05004), Sections 2.3–2.4: majority-vote labels, informative band and solver rewards.
