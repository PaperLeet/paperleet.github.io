# 96. Continual Learning: Evaluation Matrices and Forgetting

## 1. Problem

Current-task accuracy alone cannot show whether sequential training preserves previous abilities. GEM evaluates tasks after each training stage and organizes the results into an evaluation matrix. Read [Section 2, Equations (2)–(4)](https://papers.nips.cc/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf).

Implement one evaluation function using supplied scores and initialization baselines. No GEM optimizer, gradient projection or prerequisite exercise is needed. The provided experiment genuinely trains a small classifier sequentially and calls your function after each stage.

```python
continual_metrics(scores: Tensor, baseline: Tensor) -> dict[str, Tensor]
```

## 2. Core equations

Use **zero-based** indexing. Train `T>=1` tasks in order `0,1,...,T-1`.

- `scores[i,j] = R[i,j]` is accuracy on **task j's test set after training task i**, shape `[T,T]`. Rows are completed training stages; columns are evaluated tasks.
- `baseline[j] = b[j]` is task j's accuracy **before training any task**, shape `[T]`. Do not reset this baseline at each stage.

For `T>1`:

\[
\mathrm{ACC}=\frac1T\sum_{j=0}^{T-1}R_{T-1,j},\qquad
\mathrm{BWT}=\frac1{T-1}\sum_{j=0}^{T-2}(R_{T-1,j}-R_{j,j}),
\]
\[
\mathrm{FWT}=\frac1{T-1}\sum_{j=1}^{T-1}(R_{j-1,j}-b_j).
\]

These three metrics follow GEM. Positive BWT means later training improved earlier tasks relative to their just-learned scores. Positive FWT means earlier training improved a not-yet-trained task over initialization.

This exercise additionally defines **nonnegative peak forgetting** for comparison; it is not one of GEM's Equations (2)–(4):

\[
\mathrm{forgetting}=\frac1{T-1}\sum_{j=0}^{T-2}
\left(\max_{j\le i\le T-1}R_{i,j}-R_{T-1,j}\right).
\]

The peak begins when task j has been learned and **includes the final stage**, making the drop nonnegative. The final task has no subsequent training and contributes to neither BWT nor forgetting.

Return exactly the keys `acc`, `bwt`, `fwt`, `forgetting`, each containing a scalar Tensor of shape `()`. For `T=1`, return `R[0,0]` for `acc` and zero for the other three metrics.

## 3. From equations to code

Identify the relevant cells first: ACC uses the final row; BWT compares old-task diagonal cells with that row; FWT uses the first superdiagonal; forgetting uses each old task's history starting at its diagonal. Compute per-task differences before taking the specified equal-task averages.

For example:

```text
R = [[0.5, 0.2, NaN],
     [0.9, 0.6, 0.3],
     [0.7, 0.8, 0.9]]
b = [NaN, 0.1, 0.15]
```

The results are `acc=0.8, bwt=0.2, fwt=0.125, forgetting=0.1`. Task 0 improves from its just-learned score of 0.5 to 0.7, but also falls by 0.2 from an intermediate peak of 0.9. Positive BWT can therefore coexist with positive peak forgetting.

## 4. Conventions

| Item | Contract |
| --- | --- |
| Inputs | Matching CPU float32/float64 tensors; `T>=1` |
| Used values | Finite accuracies and baselines in `[0,1]` |
| Unused cells | `R[i,j]` where `j>i+1`, plus `b[0]`, may be NaN |
| Future-task evaluations | FWT needs `R[j-1,j]`; earlier future-task cells must not affect metrics |
| Reduction | Equal task weighting, not example-count weighting or a whole-matrix average |
| Outputs | Four scalar Tensors preserving input dtype/device |
| Side effects | Preserve inputs and keep no state across calls; no gradient requirement |

Inputs satisfy these shape/value rules; general validation is not required. Comparing only peaks strictly before the final stage and allowing negative drops is another possible convention, but differs from this exercise's explicit nonnegative definition.

In real projects, separate training, validation and test data. Report stage-wise test performance without using test scores to select hyperparameters. Turns from the same conversation or related source examples should not leak across train/test splits. These are evaluation-design reminders, not attributes this function must infer.

## 5. Interactive view and measured experiment

[Explore sequential evaluation](#viz-sequential_evaluation). A two-weight logistic classifier learns three synthetic binary tasks whose label directions differ, deliberately creating interference. Each task has 64 training inputs and 80 separate test inputs. Training uses zero initialization and 36 full-batch SGD updates per task at learning rate 0.8. Circle grids are deterministic; no RNG is used.

Updates read only training data. After every stage, predictions on the held-out grid produce a row of R, and your `continual_metrics` computes the displayed metrics. The untrained model provides the baseline; independent scalar formulas verify the results. T3 requires current-task accuracy of at least 0.9 and final peak forgetting above 0.25 for this deliberately conflicting stream. These thresholds apply only to this tiny experiment.

This demonstrates evaluation, not GEM optimization, replay, LLM training or paper-benchmark performance. Passing does not establish that a model has solved forgetting.

## 6. Run

Open problem **96**, implement the function, and select **Run tests**. T0 checks the interface; T1 uses hand calculations and an independent scalar oracle; T2 checks the single-task endpoint, unused cells, offsets, noncontiguous inputs and preservation; T3 runs the sequential-training experiment.

```bash
paperleet test foundations/continual_metrics --impl workspace/foundations/continual_metrics.py
```

## 7. Further exploration

Explain how models with similar final accuracy can have different BWT/FWT, and why peak forgetting is not simply negative BWT. Then read GEM Section 3's memory-gradient constraints and compare them with EWC parameter constraints and replay. These are methods to explore, not assumptions about your own project.
