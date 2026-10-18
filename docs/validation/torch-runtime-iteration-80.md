# Iteration 80: affine products and forward-AD development checkpoint

This is an unfinished 0.78 development checkpoint. The 0.77 runtime was adopted
at `3999719df6549ab850c91bacb12372e939f0a61a`. Remote main has since advanced
to `4842b5ae2ca459a47ea5579bdb2db3514afd73be`; standalone sources, tests and
package metadata are unchanged between those refs. No GitHub CI,
tag, package publication or deployment was requested or performed. The project
continues to use StrideTorch while the proposed browser-oriented name is undecided.

## Implementation

- Add `torch.addmm` / `torch.addmv` and Tensor methods. The private NumPy
  affine output has one analytical reverse node, real float32/float64 reverse
  AD, JVP and functional HVP. Native Torch 2.6 is the pinned compatibility
  reference; newer native versions have different addmv dtype acceptance.
- Distinguish checked BLAS coefficient conversion from ordinary gradient
  multiplication. Preserve native beta-zero bias handling and tested half
  nonfinite behavior. Reverse-based JVP/HVP place scalar multiplication before
  the appropriate contractions, including empty/nonfinite coefficient cases.
- Preserve wider direct-forward tangents through addition and softmax. Retain
  direct scalar JVP's differentiable zero edge without evaluating `primal*0`.
- Handle empty softmax/log-softmax inputs, empty reverse-based mean JVP and
  cat's ignored one-dimensional empty-operand Jacobian. Preserve cat's native
  all-ignored integer-axis behavior, with checked integer overflow.
- Reuse the private empty mv result and shape-only zero pullbacks. Linear
  inference avoids unused pullback closures and adds bias into its private output.
- Allow an operator implementation to validate selected unsupported options
  while preserving capability metadata. Affine `out=None` is accepted; a Tensor
  output buffer is unsupported and invalid output types raise `TypeError`.

## Evidence collected so far

| Check | Observed result | Scope |
| --- | --- | --- |
| Full canonical CPU SDK | 11,964 passed, one warning | Includes 88 new tests and the existing standalone/CLI tests |
| Node / Pyodide 0.29.4 | 88 passed | Focused new tests, including large WASM contractions |
| Real Chrome 155.0.8059.82 / Pyodide | 88 passed | Same frozen source/test hashes |
| Native inference controls | 22,004 matching serialized contracts | Acceptance, exception class, dtype, shape, values, strides; no separate sign-bit/NaN-payload claim |
| Native basic AD controls | 5,520 mathematical contracts matched | Values, dtype, shape, flags and exception class; empty derivative layouts retained separately |
| Native coefficient/AD controls | 6,202 matched; 198 existing complex-autograd boundaries | 6,400 cases including extreme coefficients and empty products |
| Precision/boundary controls | 2,976 cases; no binding, metadata or nonfinite mismatch | 158 finite kernel-rounding rows retained as observations |
| Native empty cat/stack AD controls | 960 mathematical/AD-flag contracts matched | Empty layouts retained as separate observations |
| Native primitive scalar AD controls | 600 mathematical/AD-flag contracts matched | Includes zero, nonfinite and empty cases; empty layouts retained separately |

The finite precision observations include up to two half ULPs, maximum real
float32 absolute difference `1.33514404296875e-05`, and maximum complex64
component difference `1.9073486328125e-05`. These observations do not establish
bitwise equality or accuracy for arbitrary conditioning.

The new tests use independent polynomial VJP/JVP/Hessian formulas, explicit
native edge controls, mutation checks and repeated option-value validation.
Browser tests include contiguous, Fortran and sliced inputs with contractions
large enough to enter the WASM numerical path.

## Correcting the native observation tool

Native forward/reverse transforms can return an unallocated `ZeroTensor`.
Exporting its storage directly to NumPy can expose undefined values. The
encoder now clones such native tensors to materialize their mathematical zero
before export. All seven native/candidate suites were regenerated. Earlier
unmaterialized observations remain in a superseded attempt directory and are
not qualification evidence. Older immutable iteration records were not edited;
the older core native controls still need a fresh materialization audit.

## Open compatibility work

The exploratory 2,736-case composition/mutation suite is not an unconditional
pass. Its strict comparison currently records 2,010 matches and 726 differences:

- 360 modified-saved-graph cases retain the existing explicit server fallback
  instead of native `RuntimeError`.
- 102 valid native nonleaf updates retain the existing unsupported boundary.
- 72 cases exercise the missing `torch.linalg.vector_norm` API.
- 162 differ only in empty allocated-output layouts. A separate native layout
  probe found 44 nonstandard layouts in 253 cases, so globally normalizing every
  empty result to contiguous strides would be incorrect.
- 30 nonlinear finite differences pass native Torch 2.6's own
  `torch.testing.assert_close` default tolerances, with exact dtype, shape and
  reverse flags.

Six previously differing empty `cat` functional JVP cases with
`create_graph=True` now match reverse flags. Their remaining differences are
included in the empty-layout category. The 960-case expansion also distinguishes
ignored one-dimensional empties from empty matrices and the stack operation.

The 600-case primitive scalar forward/reverse probe also retains its empty-layout
observations; it does not declare exact empty-layout equivalence.

## Remaining adoption gates

The final source was measured over 26 CPU flows, three fresh processes and
three rotated batches per process, with separately loaded identical A/A controls.
Every measured batch exceeded 73 ms. New addmm/addmv flows compare against the
equivalent affine composition in 0.77; Linear and mv compare the same public APIs.
Small Linear inference median elapsed-time ratios were 0.802 (float32) and
0.836 (float64); empty mv backward ratios were 0.822 and 0.824. However, six of
the 78 A/A flow/process controls drifted by more than 5%, with a maximum 43.4%
drift. These results are retained as exploratory observations, including every
sample and control outlier. They do not qualify application-wide or browser
performance and are not a release performance claim.

Before main adoption, expand the relevant mixed-dtype AD and storage controls,
refresh older native controls, and complete the
full Node/Chrome SDK, minimum-environment/wheel checks, balanced CPU/WASM
performance comparisons and lifetime probes. The full application CPU suite,
three native catalog runs, whole-catalog WASM budgets, frontend checks, both
build profiles, worker/asset checks and a fresh nested-route browser run are
also pending. Reconcile any intervening main changes, archive and verify the
evidence, and use the existing guarded main-push procedure only after those
gates pass. WebGPU remains a later routing decision informed by the complete
measurements.

Raw checkpoint receipts and reproducible programs are currently in the task's
`paperleet-wasm-evaluation-20261015/torch-iteration-80` directory. They have not
yet been bundled into a final release-validation archive. This checkpoint does
not mark the continuing library-improvement goal complete.
