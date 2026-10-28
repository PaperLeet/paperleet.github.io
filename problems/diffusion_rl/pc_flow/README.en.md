# PC-Flow Adjacent-Time Classifier Preference Ratios

## 1. Problem

Build a reference-free preference objective from classifier logits at adjacent sampling times.

Implement the adjacent-time classifier objective; the base flow and guided sampler are outside the exercise.

## 2. Core equations

$$
d_w=\log\sigma(a^w_{next})-\log\sigma(a^w_{now}),\quad d_l=\log\sigma(a^l_{next})-\log\sigma(a^l_{now}),\quad L=\operatorname{mean}\operatorname{softplus}(-\beta T(d_w-d_l)).
$$

## 3. From equations to code

```python
pcflow_loss(winner_next: Tensor, winner_now: Tensor, loser_next: Tensor, loser_now: Tensor, beta: float, total_steps: int) -> Tensor
```

Convert logits to stable log-sigmoid probabilities. Subtract adjacent log probabilities on each trajectory, multiply the winner-minus-loser margin by beta times T, then apply logistic loss.

## 4. Contract

`next` is the state at $t-1$ after one denoising transition; `now` is the state at $t$ before that transition.

winner_next/winner_now/loser_next/loser_now are equal (B,) classifier logits. beta>0, total_steps is positive integer. Classifier probabilities are sigmoid(logit), not raw logits.

All four classifier evaluations differentiate because both adjacent states contribute to each log probability ratio.

## 5. Visualization and experiment

The experiment calls your submitted functions; plotted points are measured results.

A linear classifier learns opposite next-state changes for winner and loser transitions over 50 actual updates.

## 6. Run

```bash
python -m paperleet.cli test diffusion_rl/pc_flow --impl problems/diffusion_rl/pc_flow/solution.py
python -m paperleet.cli mutate diffusion_rl/pc_flow
```

## 7. Extensions

Use classifier gradients to implement the paper’s guided sampling step.

[PC-Flow: Preference Alignment in Flow Matching via Classifier](https://ojs.aaai.org/index.php/AAAI/article/view/37971) · Preference Classifier Training Objective, Eq. (10)
