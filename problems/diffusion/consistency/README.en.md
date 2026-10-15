# 21. Consistency Models: Boundary and Teacher Targets

## 1. Exercise goal

Implement a boundary-preserving consistency function, stop-gradient pair loss, and EMA target update.

## 2. Interfaces and returns

### Return requirements

#### `consistency_function` (required)

Return the consistency prediction Tensor with the same [B, ...] shape as noisy; B is batch size and trailing axes are sample coordinates. Preserve floating dtype/device and model gradients. Samples at sigma=sigma_min must equal the input. Do not detach the prediction.

#### `consistency_loss` (required)

Return a floating scalar Tensor of shape (), matching prediction dtype/device. Average squared error over the batch and all coordinates, not a per-example vector or Python float. Keep prediction gradients and stop target_prediction gradients inside the function.

#### `update_ema` (required)

Return Python None. Update target parameters in place under torch.no_grad(), retaining each parameter shape and dtype/device. Leave online parameters unchanged. This interface updates parameters only, not buffers, and does not return a replacement model.

#### `teacher_euler` (provided)

Return one teacher sample Tensor at next_sigma, with noisy shape [B, ...] and floating dtype/device. B is batch size and trailing axes are sample dimensions. The supplied no_grad decorator makes the output gradient-free. Do not mutate inputs or return a trajectory.

#### Provided local helpers

- `_time`: Provided: return t reshaped from [B] to [B,1,...,1], a Tensor with the same rank as x; B counts samples and every non-batch axis is singleton. Preserve t dtype/device (not x dtype/device) and gradients. Do not mutate inputs; the result may share storage with t.

### Equations and interface notes

### Contract

- `consistency_function(model, noisy: Tensor, sigma: Tensor, sigma_min: float = .002, sigma_data: float = .5) -> Tensor`
- `consistency_loss(prediction: Tensor, target_prediction: Tensor) -> Tensor`
- `update_ema(online, target, decay: float = .95) -> None`
- `teacher_euler(noisy: Tensor, denoised: Tensor, sigma: Tensor, next_sigma: Tensor) -> Tensor` (provided)

At sigma_min the function must be exactly the identity, regardless of network output. The model receives c_in*x and the original sigma; this exercise supplies raw sigma conditioning instead of the official embedding. The skip/output coefficients follow the official boundary parameterization. The lower-noise teacher/EMA branch is frozen. The supplied Euler helper uses decreasing positive noise scales. This is consistency distillation with a teacher, not the separate teacher-free consistency-training algorithm.

## Runnable experiment

**Provenance: PaperLeet teaching adaptation.** A point-data teacher makes its probability-flow Euler step exact. Small MLP distillation exercises the boundary and target routing without pretrained models. Half the training pairs use a positive adjacent noise level sigma/2, so the EMA network affects the target; the other half anchor the exact sigma_min boundary.

The T3 test runs your submitted implementation, saves points/curves/metrics, and uses fixed seed 42. `toy.py` contains the provided experiment. It does not download models or datasets. These exercises reproduce selected mechanisms, not full image benchmark results.

## Sources

- https://arxiv.org/abs/2303.01469
- https://github.com/openai/consistency_models/blob/e32b69ee436d518377db86fb2127a3972d0d8716/cm/karras_diffusion.py

Anchors: arXiv v1 §3 consistency function boundary parameterization; §4 consistency distillation and EMA target; Official get_scalings_for_boundary_condition.
