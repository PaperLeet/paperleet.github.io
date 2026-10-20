# 113. MM-Zero Coder and Solver Feedback

## 1. Problem

Compute coder execution/visual feedback and solver agreement with silver labels. These roles have different reward components and boundaries.

Implement `coder_rewards`, `solver_rewards`.

## 2. Core equations

$$
r_C=\mathbf1_e[1+s+\min(c,1-c)]-0.1\mathbf1_{\neg e}-0.05\mathbf1_{syntax},\quad r_S=\alpha\mathbf1[a=\tilde y]+(1-\alpha)v.
$$

## 3. From equations to code

Successful coder output with easy=0.8 and c=0.5 earns 2.3; both render and syntax failure earn -0.15. For solver alpha=0.9, agreement with invalid format earns 0.9; disagreement with valid format earns 0.1.

## 4. Exercise conventions

The coder does not inherit the proposer’s easy-score cap. Failed renders have no visual scores; error penalties can add. The binary difficulty range annotation near Eq.11 conflicts with Eq.7; this exercise uses the continuous min(c,1-c) from Eq.7. Solver format is additive, not a gate.

Finite CPU inputs, no mutation; float32/float64 scores, long IDs and bool masks. The APIs below define all shape, tie and gradient conventions.

~~~python
def coder_rewards(executed: Tensor, easy_accuracy: Tensor, consistency: Tensor, syntax_error: Tensor) -> Tensor:
    """bool executed/syntax_error [B], float easy_accuracy/consistency [B] in [0,1].
    Successful score=1+easy+min(c,1-c); failed score=-.1. Additionally subtract
    .05 for syntax_error (both error penalties may apply). Unrendered visual
    scores are ignored. Return detached same-dtype [B]. Uses Eq.7's continuous
    difficulty score; Eq.11's later binary range annotation is inconsistent."""
~~~

~~~python
def solver_rewards(answers: Tensor, format_valid: Tensor, num_answers: int, alpha: float = .9) -> Tensor:
    """Long answer IDs [B,K], bool format_valid [B,K], B,K>=1, 0<=alpha<=1.
    Answer IDs are normalized upstream and lie in [0,num_answers). Majority
    includes ALL K answers; tie chooses smallest ID. Return float64 [B,K]:
    alpha*agreement + (1-alpha)*format_valid. Format is an ADDITIVE component,
    not a gate. Consensus is a silver label, not an independent gold answer."""
~~~

## 5. Interactive figures and experiment

Open Explanation to step through the paper’s co-evolution loop: active model, frozen evaluator, feedback and filtering. Controls recompute the displayed teaching scores. The Lab trains a small categorical policy for 60 gradient steps at learning rate 0.5, starting from uniform logits, using your functions. It uses fixed feedback records and CPU float64; target probability must exceed 0.72 and improve by more than 0.3. This checks the local mechanism, not full LLM/VLM self-evolution or benchmark gains.

## 6. Running the exercise

Complete the starter and use Run tests. T0 checks interfaces; T1 uses independent scalar/counting oracles; T2 checks boundaries, invariances and gradient ownership; T3 runs your implementation in the toy learning loop.

~~~bash
paperleet test self_evolving/mmzero_feedback --impl problems/self_evolving/mmzero_feedback/solution.py
paperleet mutate self_evolving/mmzero_feedback
~~~

## 7. Further questions

What reward exploitation might appear if all three roles shared one reward?

Source: [MM-Zero: Self-Evolving Multi-Model Vision Language Models From Zero Data](https://arxiv.org/abs/2603.09206), Sections 2.5–2.6, Eqs. 11–14: execution feedback and silver-answer reward.
