# Pitfalls and diagnostics

- **Swapped axes.** Final performance is the last row. Diagonal entries capture different just-learned stages, so their average is not final ACC (mutant m1).
- **Reversed BWT sign.** Subtract the just-learned score from the final score; positive values mean improvement (m2).
- **Already-trained scores in FWT.** Read `R[j-1,j]`, not `R[j,j]` or an average over the upper triangle (m3).
- **Forgetting replaced with -BWT.** Their baselines are historical peaks and diagonal scores respectively. BWT and peak forgetting can both be positive. This exercise includes the final stage in the peak, so forgetting is nonnegative (m4).
- **Unused-cell leakage.** Matrix cells with `j>i+1` and baseline[0] may be NaN. Task j's historical peak starts at stage j; its untrained scores do not count.
- **Wrong denominators.** ACC averages T tasks; the other metrics average T-1 relevant tasks. For T=1, return zeros for those three rather than dividing by zero.
- **Overstated scope.** ACC/BWT/FWT come from GEM. Nonnegative peak forgetting is an explicitly labeled extension. The measured experiment evaluates real training but does not implement GEM gradient projection.

The reference scans task histories in O(T²) time with O(T) extra temporary storage. Vectorized implementations are also valid if unused cells cannot affect results. Autograd is not required. Do not use stage-wise test scores to select training hyperparameters.
