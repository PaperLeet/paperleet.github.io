# Terminal HVP traversal: values-only pilot

This is an isolated experiment on the qualified 0.81 source, not an enabled
runtime change or a new release. The public HVP API already rejects
`create_graph=True` and returns detached results. The private Tensor-based norm
HVP uses three reverse traversals; its last traversal can compute values under
`no_grad()` while the first two continue constructing their derivative graphs.
The ordinary non-norm HVP route remains unchanged.

The candidate changes only that final call in `_higher_order.Context.hvp`.
It retains validation, parent order, numerical expressions, mutation checks,
strict-mode handling and caller gradients. A scoped gradient context restores
state on exceptions; there is no cross-call cache, graph pruning or alternate
norm formula. Version literals stay at 0.81 in the isolated copies so this pilot
cannot be mistaken for a qualified 0.82 distribution.

## Existing contracts and failure behavior

- The initial focused norm/HVP/JVP suites pass 622 tests.
- The full native SDK passes 12,225 tests on NumPy 2.5.4 and 12,222 with three
  existing skips on NumPy 1.26.4. Imports explicitly use the isolated candidate.
- Actual Node/Pyodide 0.29.4/NumPy 2.2.5 passes 12,212 SDK tests, with zero
  skips/failures. The 13 native-process-only tests retain the existing WASM
  exclusion; their native runs are included above.
- Eight success/injected-terminal-failure controls cover both real dtypes and
  both caller gradient modes. They retain the caller's exact gradient buffer,
  restore gradient/NumPy policy and context state, clear all context rules,
  pending descriptors and owners, and permit subsequent numerical backward.

## Exploratory CPU timing

One process measures 14 full HVP calls including owned output copies. Twelve
norm workloads cover 64/1,024 elements, float32/64, norm-square, norm-sine and a
nonlinear fractional norm. Two ordinary non-norm controls use the unchanged
route. Independent A0/A1 load the same frozen baseline; B loads the candidate.
All results match baseline exactly before timing. All 12 balanced permutation
orders and their raw samples are retained, with shared per-flow batch counts,
60 ms calibration and a 30 ms target.

The norm B/A0 median ratios are **0.872–0.910** (about 9–13% lower time). The two
ordinary controls are **1.010–1.012**. All 14 A/A medians stay within 5%.
A separate 100-call norm-square profile records 527,801 baseline versus
470,101 candidate function calls. Profiling times are not used as benchmark
samples. These are one-process CPU observations, with baseline equality rather
than a separate native oracle in the timing harness; they are not release
performance qualification or WASM/GPU speedup claims.

Next steps are an independent native oracle corpus, repeated CPU and actual
WASM measurements, allocation/lifetime checks, actual Chrome SDK and final
distribution/application/export checks. The previously documented large-finite
float32 norm/sine phase discrepancy remains unresolved. Broader core API and
GPU backend work also remain active.

[Machine-readable pilot](../../packages/stridetorch/benchmarks/results/hvp-terminal-values-pilot.json)
and [raw harness](../../packages/stridetorch/benchmarks/results/hvp-terminal-values-pilot-harness.tar.gz):
315 individually verified files, 1,001,825 bytes, SHA-256
`9d35f9bcfe3e633f1dacdfaae457e97374b5888cd81bc0a0e69a588a63df2c48`.
The archive contains isolated sources, scripts, existing-test identities,
native/real-WASM logs, failure controls, all timing samples and profile dumps.
Replay requires external Python/NumPy/Torch/Node/Pyodide dependencies and path
adaptation. Full native SDK test sources are identified by Git/source hashes;
the archive's WASM contract copy contains the 89 browser-compatible suites.
