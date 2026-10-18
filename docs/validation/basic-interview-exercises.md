# Basic sampling, multi-turn SFT and continual-learning exercises

Four maintained exercises extend existing numerical foundations. They are original
teaching tasks based on primary sources, not claims about company interview questions.

| Number | Module | Required function | Tests | Mutants |
| --- | --- | --- | --- | --- |
| 93 | foundations/top_p_draw | sample_top_p | 7 | 5 |
| 94 | foundations/multiturn_sft_mask | assistant_target_mask | 8 | 4 |
| 95 | finetuning/ewc | ewc_penalty | 7 | 4 |
| 96 | foundations/continual_metrics | continual_metrics | 8 | 4 |

Each includes bilingual statements and notes, starter/reference implementations,
independent oracles, properties, boundary cases, single-bug mutants, real synthetic
experiments and visualization data. The preparation handbook supplies three bilingual
prompts per module; the new preparation path connects these modules to existing
Top-p/Top-k, next-token loss, SFT mask, attention and PPO exercises.

## Sources and implementation scope

- Holtzman et al., *The Curious Case of Neural Text Degeneration*, ICLR 2020,
  https://arxiv.org/abs/1904.09751, §3.1 Eqs. (2–3). The supplied helper forms nucleus
  probabilities. Learners implement inverse-CDF token sampling. One float64 uniform
  per row and a caller-owned RNG are explicit teaching conventions. Tests distinguish
  original vocabulary IDs, exact interval endpoints and a CDF tail-rounding gap.
- TRL SFT documentation, https://huggingface.co/docs/trl/sft_trainer, and packing
  documentation, https://huggingface.co/docs/trl/reducing_memory_usage. Role IDs,
  EOS ownership and boundary rules are explicit exercise conventions. The target
  mask does not provide attention isolation between packed conversations.
- Kirkpatrick et al., *Overcoming catastrophic forgetting in neural networks*,
  PNAS 2017, https://arxiv.org/abs/1612.00796v2, §2 Eq. (3). Fisher weights are provided;
  the exercise implements the summed penalty and theta-only gradient, not Fisher
  estimation. Two-task optimization demonstrates a retention/adaptation tradeoff.
- Lopez-Paz and Ranzato, *Gradient Episodic Memory for Continual Learning*,
  NeurIPS 2017, https://arxiv.org/abs/1706.08840, §2 Eqs. (2–4). The exercise implements
  ACC/BWT/FWT; nonnegative peak forgetting is an explicitly labelled extension.
  GEM gradient projection is outside scope. Test data only evaluates the synthetic
  sequential learner; it is not used to train or choose model parameters.

## Validation

Every module passed its original native suite, all mutants, and three repetitions
of the per-module quality gate. Independent implementations written from only the
statement and starter passed each suite. Cross-review added a missing numerical
endpoint regression to the sampling task and a fifth mutant for that error.

All four modules passed their complete exported reference/mutant/learner suites
under the actual pinned Pyodide 0.29.4 / NumPy 2.2.5 runtime. Runtime sources and
published test suites are unchanged between browser and native execution; no
pretrained model, external dataset, CUDA device or new adapter operator is needed.

The complete native catalog gate passed three repetitions across all 82 exercises
(10,668 cases before the added sampling mutant); the final sampling package then
passed all five mutants and 126 cases over three repetitions. Infrastructure and
StrideTorch checks passed with 41,600 tests and one skipped test. The hosted
production build and dependency-graph check passed. Backend protocol checks also
ran each new reference through the real worker subprocess and validated its full
T0–T3 report; the hosted-judge/deployment suites passed 85 tests (86 opt-in skips).
