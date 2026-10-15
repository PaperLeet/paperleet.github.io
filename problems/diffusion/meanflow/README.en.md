# 16. MeanFlow: Average Velocity and JVP Targets

## 1. Exercise goal

Implement the total-derivative average-velocity target and its frozen-target loss, then sample in one call.

## 2. Interfaces and returns

### Return requirements

#### `meanflow_target` (required)

Return the tuple (prediction, target), in that order, as Tensors with z shape [B,...] and floating dtype/device; B counts samples and trailing axes are coordinates. prediction is the model prediction with training gradients retained; target is the fully detached JVP-corrected target. Do not return the JVP derivative itself.

#### `meanflow_loss` (required)

Return a floating scalar Tensor of shape (), matching clean/noise and model-output dtype/device. Average squared prediction-target error over the batch and all coordinates, without adaptive weighting. Preserve model gradients through prediction and stop target gradients.

#### `one_step` (required)

Return the data-endpoint Tensor from one model call, matching noise shape [B,...] and floating dtype/device; B counts samples and trailing axes are coordinates. Return only samples, not a trajectory. The supplied no_grad decorator makes the output gradient-free; noise is unchanged.

#### Provided local helpers

- `_time`: Provided: return t reshaped from [B] to [B,1,...,1], a Tensor with the same rank as x; B counts samples and every non-batch axis is singleton. Preserve t dtype/device (not x dtype/device) and gradients. Do not mutate inputs; the result may share storage with t.

### Equations and interface notes

### Contract

- `meanflow_target(model, z: Tensor, r: Tensor, t: Tensor, velocity: Tensor) -> tuple[Tensor, Tensor]`
- `meanflow_loss(model, clean: Tensor, noise: Tensor, r: Tensor, t: Tensor) -> Tensor`
- `one_step(model, noise: Tensor) -> Tensor`

Use paper argument order `(z,r,t)`: JVP tangents `(v,0,1)` hold r fixed. Noise increases with t. The target is `stopgrad(v-(t-r) du/dt)`. We use elementwise mean MSE, omitting the optional adaptive weighting in the official image implementation. At r=t only the target becomes conditional flow matching; the untrained network need not predict v.

## Runnable experiment

**Provenance: PaperLeet teaching adaptation.** An analytic point-data family learns its center using the learner JVP target. This is a mechanism demonstration, not a low-dimensional experiment claimed by the paper.

The T3 test runs your submitted implementation, saves points/curves/metrics, and uses fixed seed 42. `toy.py` contains the provided experiment. It does not download models or datasets. These exercises reproduce selected mechanisms, not full image benchmark results.

## Sources

- https://arxiv.org/abs/2505.13447
- https://github.com/Gsunshine/py-meanflow/blob/1f6d72d94247c8fdeb89489acca1a8007a6baf6c/meanflow/models/meanflow.py

Anchors: arXiv v1 §3 MeanFlow identity and stop-gradient training objective; Official forward_with_loss and sample.
