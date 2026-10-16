# Cholesky NaN failure portability

The existing main quality run 38948582658 failed four unchanged native comparison cases: a used off-diagonal NaN or diagonal NaN could produce a returned factor instead of the required positive-definite error on Linux. The same inputs raise locally. LAPACK status for NaN inputs differs by build.

The factor wrapper now rejects NaN factor diagonals explicitly. This inspects the computed factor, preserving ignored input triangles and supported infinite diagonals. The same wrapper is used by the exceptional-path batch and leading-minor search. No test, budget, repetition, training step or mutant is removed or relaxed.

Four independent regression cases simulate successful NaN factorization status for upper/lower and batched/unbatched inputs. Each verifies ignored NaNs still succeed, used and diagonal NaNs produce the right leading minor, and batch attribution remains correct. Run the full existing Cholesky comparison and independent suites, full infrastructure gate and all exported WASM contracts before release.
