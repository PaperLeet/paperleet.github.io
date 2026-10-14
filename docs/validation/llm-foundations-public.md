# LLM foundations in the standalone public core

Validated on 2026-10-14 after the repository split. The public checkout now includes the reviewed RMSNorm (paper 47 / module 61), GQA (48 / 62) and RoFormer/RoPE (49 / 63) packages. They retain stable IDs, bilingual statements/notes, independent numerical and gradient checks, mutants and submitted-code experiments. LLM Basics contains 7 papers / 11 modules.

The sources and exercise boundaries are recorded in each package:

- [RMSNorm](../../problems/architecture/rms_norm/problem.yaml): full/first-k normalization, prefix-statistic bias and cross-feature gradients.
- [GQA](../../problems/architecture/gqa/problem.yaml): contiguous shared KV heads, mean-pooled projection conversion, accumulated gradients and persistent cache element counts.
- [RoPE](../../problems/architecture/rope/problem.yaml): adjacent-pair rotations, explicit positions, relative logits, analytical feature/position gradients and chunked-cache offsets. Continuous positions are a teaching extension; half-split layouts and context scaling are outside this contract.

The positive scalar-base power operator needed by conventional RoPE code retains analytical VJPs and forward AD, including native scalar-precision behavior. Unsupported bases, tensor/tensor powers and output buffers retain native execution. No private server implementation, production deployment file or credential was copied into this public change.

## Public-checkout verification

- `paperleet ci`: all **63 modules / 7,548 cases / three repeats**, **96.14 seconds**, passed. The three new packages have **9/10/10 checks** and **4/5/6 mutants** respectively; each mutant satisfies the ten-line bound and is caught by assertions.
- Focused native differential/forward-AD checks and browser bundle calibration: **99 passed**. The public package and catalog paths were verified before export. All frontend checks passed; a cold Vite dependency scan emitted nonfatal teardown warnings before the final successful exit.
- Local and Pages production builds passed TypeScript and the unchanged initial JavaScript budget. Local: **370,730 raw / 118,784 gzip bytes**. Pages with its actual workflow settings: **367,031 / 117,661 bytes**.
- Real static Pages-style loopback site, with no API server: RoPE reference **10/10**, calibration **10 reference tests / 6 mutants**; rich editor, next-step card and Notes worked. Reload preserved completion and unlocked Notes. The Server option correctly remained disabled because this build has no configured account backend.
- The maintained source had already passed real CPU/WASM failure checks in the combined checkout: starter **0/10**, detached-feature mutant **8/10** (analytical-gradient and Jacobian failures), restored reference **10/10**. Its actual submitted-run cache plot distinguishes correct offsets from resetting new keys to zero. An exact trusted native-only alternative also passed **10/10** through the previous local queue/protocol validator. Those earlier checks did not exercise the production Docker sandbox.

The catalog has **49 papers / 63 modules**, **491 maintained tests** and **257 mutants**. All maintained modules are browser-enabled. Local development retains native judging; public Pages stays browser-only until a separate backend is configured. Private account/hosted integration is validated in the private repository against a pinned public commit. No independent-reader blind study, push or deployment was performed.
