# Causal-JEPA Object Trajectory Masking

[Causal-JEPA](https://arxiv.org/html/2602.11389v2) masks selected object trajectories across a history window. The first observed frame remains visible as an identity anchor. A predictor recovers masked history and predicts every future object state from the remaining context.

Implement the object-only input assembly and learning objective. Frozen object slots and all projection/embedding parameters are supplied; no image encoder or Transformer is required.

| Function | Inputs | Outputs |
|---|---|---|
| `object_mask_inputs(history, masked_slots, anchor_projection, time_embeddings, mask_token)` | `(B,H,S,D)`, bool `(B,S)`, `(D,D)`, `(T,D)`, `(D,)` | Predictor inputs `(B,T,S,D)` and bool target mask `(B,T,S)` |
| `masked_prediction_loss(predictions, targets, history_length, masked_slots)` | Matching `(B,T,S,D)`, integer H, bool `(B,S)` | `(total, history_loss, future_loss)` |

Require `T>H>=1`. `masked_slots=True` selects an object whose post-anchor history is hidden. Batch-specific masks are a deterministic extension of the official code's shared selection.

## Input construction

### Return requirements

#### `object_mask_inputs` (required)

Return `(inputs, target_mask)`, exactly two Tensors: floating inputs `[B,T,S,D]` and Boolean target_mask `[B,T,S]`. B is batch size, T total history+future frames, S object slots and D latent width; H history frames satisfy T>H>=1. True marks prediction targets: selected non-anchor history slots and every future slot; frame zero is never a target. Preserve history dtype/device and gradients through visible history, anchors and parameters; the mask has no gradient. Do not mutate inputs.

#### `masked_prediction_loss` (required)

Return `(total, history_loss, future_loss)`, three scalar floating `Tensor`s of shape `()` with prediction dtype/device. History and future each use their own all-selected-element MSE mean; total is their sum, not their average. If no non-anchor history is selected, history_loss is a differentiable zero. Keep prediction gradients only at supervised positions and detach targets internally. Leave inputs unchanged.

#### `_float_tensor` (provided)

Provided validator: return Python `None` when x is a finite, nonempty float32/float64 Tensor with rank ndim. Raise `ValueError` when a checked condition fails; do not mutate inputs or return transformed tensors.

#### `_mask` (provided)

Provided validator: return Python `None` when masked_slots is a Boolean Tensor of exactly `[batch,slots]` on the requested device. Raise `ValueError` when a checked condition fails; do not mutate inputs or return transformed tensors.

### Formula and conventions

At time zero, **every object remains visible**. During times `1...H-1`, only selected objects are masked. Every object is masked at times `H...T-1`.

Visible token `(b,t,s)` is `history[b,t,s] + time_embeddings[t]`. A masked token is the identity query

$$q_{b,t,s}=history_{b,0,s}W^T+time_t+mask\_token.$$

Use the first frame, never the object's hidden current value. The official `id_projector` is affine; its shared bias is absorbed into the supplied `mask_token` here, so the query is algebraically equivalent. Temporal embeddings are shared across objects. There is no object-index positional embedding.

Return a bool target mask even when a query happens numerically to equal a visible value. Mask membership is a structural decision, not a tensor inequality. Inputs must not be modified, and input assembly retains gradients through visible history, anchors and supplied parameters. Freezing the image encoder is handled by the caller.

## Joint masked objective

Detach `targets` inside the loss. Follow the paper's masked-token indicator in Eq. (5), with separately normalized history/future components:

1. Mean squared error across the selected history tokens at times `1...H-1`, including all their feature elements.
2. Mean squared error across **all** future tokens at times `H...T-1`.

Return their sum and both components. Do not average them together or use one common denominator. If no history tokens are selected, return a differentiable zero for that component. The anchor and visible history must not contribute to history loss. Auxiliary action/proprioception tokens are outside this object-only interface.

The released CLEVRER training loop also supervises selected slots at the visible anchor frame. This exercise follows the paper's masked-token indicator and excludes that frame; it is not an exact reproduction of that training loop.

All floating tensors are finite float32/float64 with matching dtype/device. Validators are supplied. For `H=3,T=5` and one of two objects selected, the target mask for the selected object is `[0,1,1,1,1]`; the other object's mask is `[0,0,0,1,1]`.

## Toy: reconstruct a coupled object

**Teaching adaptation.** Sixteen synthetic two-object sequences expose one object's motion while hiding the other object's post-anchor trajectory. Your input assembler feeds a supplied linear probe, and your loss trains it to recover history and predict future. Charts show the two loss components and the reconstructed trajectory. A separate perturbation adds 100 to hidden history and verifies that predictor inputs remain unchanged.

This isolates observability and objective design. It does not reproduce CLEVRER/Push-T, train a vision model, or establish causal identification. T3 uses the submitted functions; the reference preview uses the solution.
