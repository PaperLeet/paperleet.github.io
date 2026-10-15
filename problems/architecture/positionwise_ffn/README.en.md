# Position-wise Feed-Forward Network

Attention mixes information across positions. The Transformer then applies the same nonlinear network independently to every token. Implement [Attention Is All You Need, §3.3, Eq. (2)](https://arxiv.org/html/1706.03762v7#S3.SS3).

## Your core module

### Return requirements

#### `positionwise_ffn` — Required

Return one floating Tensor `[...,D]`, matching x's complete shape: D is the model feature width and leading batch/token axes are unchanged. Return the second linear layer's output, not hidden activations of width H. Preserve dtype/device and gradients to x, w1, b1, w2 and b2. Each token stays independent and inputs are not modified.

### Formula and conventions

Implement `positionwise_ffn(x, w1, b1, w2, b2)`:

$$
\mathrm{FFN}(x)=\operatorname{ReLU}(xW_1^\top+b_1)W_2^\top+b_2.
$$

This exercise uses PyTorch's `[out_features, in_features]` weight layout, so the written matrices are transposed relative to the paper's row-vector convention. The two affine layers both include biases. Use the original ReLU activation, with no final activation.

| Argument/output | Convention |
|---|---|
| `x` | `[..., D]`; any leading batch/sequence axes, including a single `[D]` vector |
| `w1`, `b1` | `[H,D]`, `[H]` |
| `w2`, `b2` | `[D,H]`, `[D]` |
| Output | Same shape, dtype and device as `x` |
| Scope | Compatible finite floating tensors; preserve inputs and gradients for all five arguments |
| Independence | Share parameters across positions; never sum or average over token axes |

The original base model uses `D=512`, `H=2048`; the exercise uses small non-square widths. Residual connections, LayerNorm and dropout are separate components. This function performs only the two affine maps and ReLU.

For hidden preactivations `[-1, 2, -3]`, ReLU produces `[0,2,0]`. The second projection then maps that vector back to the model dimension and adds `b2`.

## Toy experiment

The provided experiment trains a width-16 FFN on 128 synthetic two-dimensional tokens to learn:

$$
y_1=|x_1|+0.3x_2,\qquad y_2=\max(0,x_2-0.2)-0.4x_1.
$$

Labels come directly from these equations. The training loop calls your function and updates both matrices and both biases. Evaluation uses 96 independently drawn held-out tokens. The lab displays the training curve, held-out MSE and the learned absolute-value slice at `x₂=0`. A composition of two linear layers without ReLU cannot represent this target.

**Provenance:** a PaperLeet teaching adaptation of §3.3, not a toy reported in the paper or a reproduction of translation results. The data, training loop and visualization are supplied. T3 uses your implementation; the reference preview is separate.

## Validation

T0 checks leading dimensions, dtype and input preservation. T1 uses a hand-computed non-square example. T2 checks token permutation and independence, all five gradients against an independent expression, closed ReLU gates and strided inputs. T3 trains and evaluates the nonlinear regression experiment. The mutant suite covers missing ReLU, missing first-layer bias, detached hidden activations and accidental token mixing.
