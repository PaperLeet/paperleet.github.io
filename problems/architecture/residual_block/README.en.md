# ResNet: Residual Branches and Projection Shortcuts

## 1. What problem does the paper solve?

Stacking layers can make optimization harder even when the deeper model can represent the shallower solution. ResNet learns a correction to a shortcut instead of asking every block to reconstruct its entire input-to-output mapping. This exercise implements the matrix formulation in [Section 3.2, Equations (1)–(2) and Figure 2](https://arxiv.org/html/1512.03385v1#S3.SS2).

The selected scope is one two-layer, **post-activation** block. Convolutions, BatchNorm, bottlenecks and complete ResNet architectures are outside scope. The supplied vector experiment is a teaching adaptation.

## 2. Core equations and return contract

For right-multiplied weights:

$$h=\operatorname{ReLU}(XW_1),\quad r=hW_2,\quad s=X\text{ or }XW_s,\quad Y=\operatorname{ReLU}(r+s).$$

Implement `residual_block(x, w1, w2, projection=None) -> Tensor`:

- `x`: `[B,D_in]`; `w1`: `[D_in,H]`; `w2`: `[H,D_out]`.
- Return `[B,D_out]`, preserving the matching float32/float64 dtype and device.
- `projection=None` uses identity and requires `D_in == D_out`. Otherwise use the supplied `[D_in,D_out]` projection, including a square one.
- Preserve gradients to input, both branch weights and projection. Do not modify inputs.
- Feature widths are positive. Empty batches are valid. Incompatible shapes raise `ValueError`; broadcasting a shortcut is forbidden.

## 3. From equations to code

Compute the hidden activation, then the second linear map. Construct the shortcut separately, add the two paths, and apply the final activation. The second branch output may be negative: rectifying it before addition changes the model. There are no biases or normalization layers to implement here.

## 4. Exercise conventions

The original block places an activation **after** addition. A zero residual therefore returns `ReLU(x)`, not identity for arbitrary signed inputs. Positive inputs recover identity exactly. Pre-activation ResNet is a different design and is not this exercise's contract.

Weights use `x @ w`, unlike `nn.Linear.weight` storage. A projection is trainable, rather than a detached shape repair. During backward propagation the input receives gradients from both paths. Tests choose points away from ReLU kinks when checking derivatives.

## 5. Experiment

Four blocks learn `target = x + 0.2*x²` on 32 positive three-dimensional examples, using two-layer branches of width 8 and 80 Adam updates at learning rate 0.02. Zero second-layer initialization begins at the positive-input identity. The plotted loss is computed using your function in every block; the reference chart is separately labeled.

This is synthetic regression, not CIFAR/ImageNet training or evidence of a universal speed or accuracy advantage. Fixed seed 73 and float64 make the experiment small and inspectable. The checker requires final MSE below 0.003, a 95% loss reduction and a learned nonzero residual.

## 6. Run your implementation

Complete `residual_block` in the editor and select **Run tests**. Locally:

```bash
paperleet test architecture/residual_block --impl problems/architecture/residual_block/solution.py
```

T0 checks interfaces, T1 uses independent scalar dot products, T2 checks activation order, projection, gradients and layouts, and T3 runs actual training.

## 7. Further exploration

Compare identity and projection shortcuts at the same width. Derive the Jacobian when both activations are active, then identify where the derivative changes. Read the paper's convolutional architecture and normalization details before attempting a full reproduction; this block does not implement them.
