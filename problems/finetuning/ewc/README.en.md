# 95. EWC: Penalize Changes to Important Parameters

## Problem

After learning task A, a model learns task B. Updates for B can move parameters on which A depends. EWC penalizes displacement from the old solution according to each parameter's importance to A.

Implement `ewc_penalty(theta, anchor, fisher_diag, strength)`. `anchor` stores the parameters learned on A, and `fisher_diag` is an **already supplied** nonnegative Fisher diagonal. You do not estimate Fisher information or load a language model.

## Core equation

[Kirkpatrick et al., Section 2, Eq. (3)](https://arxiv.org/html/1612.00796v2#S2.E3) adds an old-task penalty to the new-task objective:

\[
L(\theta)=L_B(\theta)+P(\theta),\qquad
P(\theta)=\frac{\lambda}{2}\sum_i F_i(\theta_i-\theta^*_{A,i})^2.
\]

Return only \(P\); the caller adds \(L_B\). `strength` is \(\lambda\). Larger \(F_i\) makes the same displacement more expensive. **Sum over parameters**, without averaging or multiplying by the old dataset size.

## From equation to code

```python
ewc_penalty(theta: Tensor, anchor: Tensor,
            fisher_diag: Tensor, strength: float) -> Tensor
```

For `theta=[2,-1]`, `anchor=[1,1]`, `fisher_diag=[2,0.5]` and `strength=3`, return scalar `6`. The gradient to current parameters is `[6,-3]`; neither stored snapshot receives gradients.

Identify the trainable parameter and the fixed records before computing elementwise displacements and reducing them. Detaching snapshots must not detach `theta`.

## Exercise conventions

- The three tensors have identical shapes, dtypes and devices. Inputs are finite CPU float32/float64; Fisher weights are nonnegative. Inputs satisfy the contract; validation code is not required.
- Support arbitrary matched parameter blocks, including scalars, empty tensors and noncontiguous slices. Do not broadcast. Return a zero-dimensional tensor in theta's dtype/device without mutating inputs.
- Only `theta` retains gradients. Freeze `anchor` and `fisher_diag` even if their input tensors have `requires_grad=True`.
- `strength` is a finite nonnegative Python number. Zero strength, zero Fisher coordinates and empty inputs contribute differentiable zero. Large finite values on zero-Fisher coordinates must not produce NaN through `0 * inf`; exclude their displacement before squaring. Active arithmetic is within dtype range.
- New-task loss, Fisher estimation, consolidation across several historical tasks and replay buffers are outside the function.

## Interactive figures and experiment

**My run results** uses your penalty in every new-task update. First SGD trains a two-parameter linear model on A. Two copies of that old solution then learn B with strengths 0 and 1.

A has four inputs `[-2,0],[-1,0],[1,0],[2,0]`, with labels `2*x0`. B has `[-1,0],[1,0],[0,-1],[0,1]`, with labels `-2*x0+2*x1`. Both predict `x @ theta` and use `MSE/2` as the data loss. A unit-variance Gaussian observation model supplies A's Fisher diagonal `E[x_i²]=[2.5,0]`; no sampled estimator is implemented. A receives 60 SGD updates at learning rate 0.2; each B arm receives 160 updates at 0.1. Data are deterministic and arithmetic uses CPU float64.

The charts show MSE on both tasks: in this example EWC retains more of A at the cost of some fit to B. The second parameter has zero Fisher weight and remains free to learn B. These are synthetic task measurements, not held-out generalization, the paper's MNIST/Atari benchmarks or an LLM result. Retention is not guaranteed for arbitrary tasks. Reference figures and learner-run results are shown separately.

## Running the exercise

Complete the function, select **Run tests**, then inspect the Lab. To validate the official reference locally:

```bash
paperleet test finetuning/ewc --impl problems/finetuning/ewc/solution.py
paperleet mutate finetuning/ewc
```

T0 checks scalar/dtype behavior; T1 uses an independent Python scalar sum; T2 checks analytic gradients, frozen snapshots, zero boundaries, additivity and input preservation. T3 actually trains and compares with an independently solved quadratic optimum.

## Further questions

Why does a uniform L2 coefficient fail to express unequal parameter importance? How does increasing `strength` trade old-task error against new-task error? Fisher estimation, task boundaries and combining several historical tasks belong to later continual-learning exercises.
