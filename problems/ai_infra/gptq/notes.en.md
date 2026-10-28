# GPTQ Cholesky Error-Compensated Quantization — Solution notes

## Derivation and implementation

Clone working weights, save each quantized column, then update the remaining columns with the normalized residual. Divide by the Cholesky diagonal, not the Hessian diagonal; subtract the correction.

$$
H^{-1}=U^TU,\qquad q_j=\Delta\,\operatorname{clip}(\operatorname{round}(w_j/\Delta),q_{min},q_{max}),\qquad e_j=(w_j-q_j)/U_{jj},\qquad W_{:,j:}\leftarrow W_{:,j:}-e_jU_{j,j:}.
$$

## Boundary and gradient ownership

weight:(O,D), upper:(D,D), scales:(O,). upper has zeros below its positive diagonal and represents H^{-1}=U^T U. Use ties-to-even rounding and inclusive integer bounds. Quantize left to right and compensate the current and all later columns. Return dequantized floating weights without mutating inputs; offline PTQ has no gradients.

## Independent checks

T0 checks the callable contract and invalid boundaries. T1 uses hand calculations or an independent scalar formulation. T2 checks invariances, state conservation, gradients when meaningful, or atomic state changes. T3 invokes the supplied implementation in the deterministic experiment; it does not compare two imports of the same reference.

## Behavioral experiment

Use two correlated calibration features and two weight rows, sweep fixed quantization steps, and compare output-reconstruction MSE against independent round-to-nearest.

The single-block GPTQ core with a supplied upper Cholesky factor and fixed quantization grid; Hessian estimation, damping and factorization are supplied externally.

### Reference calibration

Deterministic CPU fixture; Python 3.14.2, PyTorch 2.14.1.

| Metric | Reference value |
| --- | ---: |
| compensated_mse | 0.09065 |
| rtn_mse | 0.28265 |

## Common mistakes

- m1_no_compensation: Independent rounding ignores correlated calibration channels.
- m2_wrong_error_sign: Adding quantization residuals amplifies instead of compensating error.

[GPTQ](https://arxiv.org/abs/2210.17323) · Section 4, Algorithm 1; official gptq.py fasterquant inner column loop.
