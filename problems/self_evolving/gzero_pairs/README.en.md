# 110. G-Zero Preference Curation

## 1. Problem

Curate question-only preference pairs so the generator learns hint-assisted responses without receiving the hint in its training prompt.

Implement `preference_mask`, `build_preferences`.

## 2. Core equations

$$
m_i=\mathbf1[\delta_i\le\operatorname{median}(\delta)]\land q_i,\quad (x,y_w,y_l)=(question,assisted,unassisted).
$$

## 3. From equations to code

Compute the linearly interpolated median over the full pool before applying quality checks, preserving input order. For deltas [-3,-2,10,11], if the first two fail quality, retain nothing; do not refill from the upper half.

## 4. Exercise conventions

Include median ties, so more than half may survive. Negative deltas are allowed. Do not leak hints into prompts or reverse chosen/rejected. Structural quality evaluation is supplied upstream.

Finite CPU inputs, no mutation; float32/float64 scores, long IDs and bool masks. The APIs below define all shape, tie and gradient conventions.

~~~python
def preference_mask(delta: Tensor, quality: Tensor) -> Tensor:
    """Finite float delta [B], bool quality [B], B>=1. Compute the LINEAR median
    of the full pool first; retain delta<=median AND quality. Include ties and
    negative deltas; a singleton is retained iff quality. Return bool [B].
    Quality checks are upstream. The median is NOT recomputed after filtering."""
~~~

~~~python
def build_preferences(records: list[dict], delta: Tensor, quality: Tensor) -> list[dict]:
    """B records have question, hint, unassisted, assisted string fields.
    Return NEW dictionaries in original order for preference_mask-selected rows:
    exactly prompt=question, chosen=assisted, rejected=unassisted. Never put the
    hint into prompt, mutate inputs or include delta/hint in output dictionaries."""
~~~

## 5. Interactive figures and experiment

Open Explanation to step through the paper’s co-evolution loop: active model, frozen evaluator, feedback and filtering. Controls recompute the displayed teaching scores. The Lab trains a small categorical policy for 60 gradient steps at learning rate 0.5, starting from uniform logits, using your functions. It uses fixed feedback records and CPU float64; target probability must exceed 0.72 and improve by more than 0.3. This checks the local mechanism, not full LLM/VLM self-evolution or benchmark gains.

## 6. Running the exercise

Complete the starter and use Run tests. T0 checks interfaces; T1 uses independent scalar/counting oracles; T2 checks boundaries, invariances and gradient ownership; T3 runs your implementation in the toy learning loop.

~~~bash
paperleet test self_evolving/gzero_pairs --impl problems/self_evolving/gzero_pairs/solution.py
paperleet mutate self_evolving/gzero_pairs
~~~

## 7. Further questions

Why does proposer training favor high delta while preference curation retains lower delta?

Source: [G-Zero: Self-Play for Open-Ended Generation from Zero Data](https://arxiv.org/abs/2605.09959), Section 3.3: lower-half delta filter and question-only DPO; official phase2.py.
