# D3PO Per-Transition Preference Optimization

## 1. Problem

Propagate terminal preferences to denoising transitions and compute reference-corrected preference losses per step.

Implement stepwise D3PO from trajectory log probabilities, without collapsing them into one trajectory-level likelihood.

## 2. Core equations

$$
z_{bt}=\beta[(\ell^w_{bt}-\ell^{w,ref}_{bt})-(\ell^l_{bt}-\ell^{l,ref}_{bt})],\qquad L=\frac1{BT}\sum_{bt}\operatorname{softplus}(-z_{bt}).
$$

## 3. From equations to code

```python
d3po_loss(preferred: Tensor, rejected: Tensor, ref_preferred: Tensor, ref_rejected: Tensor, beta: float) -> Tensor
```

Compute winner and loser log ratios relative to their references, then subtract at matching timesteps. Each transition has its own logistic objective; average over batch and time only afterward.

## 4. Contract

preferred/rejected/ref_preferred/ref_rejected: equal (B,T) finite log probabilities. beta>0. Return a scalar mean over B and T after the per-step softplus.

Both current preferred and rejected log probabilities differentiate; both reference tensors are detached.

## 5. Visualization and experiment

The experiment calls your submitted functions; plotted points are measured results.

A three-step preference model learns independently signed transition margins through 60 gradient updates.

## 6. Run

```bash
python -m paperleet.cli test diffusion_rl/d3po --impl problems/diffusion_rl/d3po/solution.py
python -m paperleet.cli mutate diffusion_rl/d3po
```

## 7. Extensions

Compare gradients against a classifier on summed trajectory log ratios.

[Using Human Feedback to Fine-tune Diffusion Models without Any Reward Model](https://arxiv.org/abs/2311.13231) · Section 4.3 Eq. (14): a preference objective for every state-action pair
