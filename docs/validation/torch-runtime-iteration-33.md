# StrideTorch 0.33 Cholesky solve candidate

The candidate adds `torch.cholesky_solve(input, input2, upper=False, *, out=None)`
and `Tensor.cholesky_solve(input2, upper=False)`. Matrix right-hand sides and
factor batches broadcast; results use native column-major final axes.

The contract is checked against the [official API](https://docs.pytorch.org/docs/2.14/generated/torch.cholesky_solve.html),
the installed native2.14.1 signature, and the
[primary differentiation rules](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/csrc/autograd/FunctionsManual.cpp#L4719).
The documented B/L names describe the mathematics; actual accepted keyword
arguments are `input` and `input2`, verified with native probes.

Primal values read only the selected factor triangle. Native VJP/JVP extend the
full factor and its direction, including entries outside that triangle. This is
preserved rather than silently projecting the factor gradient. Native guards
RHS,factor and output mutations even when just one input needs gradients;
StrideTorch preserves all three guards while avoiding retention of unneeded RHS
arrays. Float32/64 support analytical reverse gradients,JVP and full reverse of
both channels. Complex64/128 support inference;half,complex AD and `out` remain
explicit unsupported cases. No native Torch or SciPy is needed at runtime.

The first implementation passes215 new native/independent tests. Additional
Cholesky complex/empty/singleton/ignored-NaN/failed-factor/infinite-diagonal
boundaries bring that focused run to278. Narrow-factor-gradient association and
wide/narrow constructed systems bring the current focused run to310 passes.
The original implementation and source hashes are frozen before that refinement.
Full core,wheel,Node/Chrome and actual application gates remain pending.

## Portable factor failure status

The0.32 source passes native catalog and exported WASM remote jobs,but its
infrastructure job fails four NaN exception assertions. The corresponding
Pages build/deploy jobs are skipped. The local native assumption that NaN
always raises is not portable:PyTorch explicitly
[does not guarantee nonfinite linalg behavior](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html#non-finite-values).
Finite singular/non-positive-definite cases retain strict native comparisons;
the adapter's NaN rejection is tested as its own consistent boundary.

New real Pyodide probes show NumPy2.2.5 returns NaN Cholesky factors for finite
non-positive-definite matrices,without raising,for both single/batched inputs.
It also returns NaNs for singular solve/inverse and silently accepts empty RHS
solves with singular coefficients. A positive returned Cholesky diagonal is
now required on all runtimes;the same check drives first-batch/leading-minor
diagnostics. This handles the silent NumPy failure without reading ignored
input entries. Positive infinite diagonals retain the previously tested path.

An import-time capability probe checks whether NumPy raises floating-point
status errors and restores its caller's error mode. General solve/inverse on
quiet runtimes check suspicious NaN results with `slogdet`,whose zero sign
explicitly represents singular pivots. Empty RHS solves check that status even
on CPU;normal CPU solves avoid extra output scans/factorizations. This caches
only a runtime capability boolean,never tensor data or numerical solutions.
Extremal non-singular results are not mislabeled as singular.

The triangular API's
[zero-pivot behavior](https://docs.pytorch.org/docs/2.14/generated/torch.linalg.solve_triangular.html)
permits nonfinite results. New comparisons require that behavior rather than
one vendor-specific NaN/Inf pattern;no extra per-call triangular scan is added.
All prior finite value/gradient tests remain intact. Explicit quiet-backend
injection tests cover Cholesky,solve,inverse and the triangular distinction.

The initial0.33 full CPU run passes31337 with one optional skip;Node and Chrome
each expose12 real invalid-factor failures with1203 passes. Those logs/source
remain frozen. The expanded corrected CPU status/factor focus passes740 before
the final zero-pivot assertions are scoped to the documented native contract.
Refined complete gates and performance remain required before main adoption.

The VJP shares the adjoint positive-definite solve between both active inputs.
For finite factors with at least64 rows and RHS width at most one quarter of the
factor size,the factor gradient associates products through the narrow RHS.
This computes the same full-factor native expression in O(n*n*k),avoiding its
O(n**3) dense product. Wider/nonfinite inputs retain the original evaluation
order. No tensor values or factorizations are cached. Performance evidence is
required before adoption; algebraic complexity alone is not a speedup claim.
