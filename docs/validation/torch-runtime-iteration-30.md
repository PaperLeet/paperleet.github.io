# StrideTorch0.30 checkpoint data

StrideTorch now provides `torch.save/load` and `torch.serialization.save/load` for
CPU checkpoint data in native PyTorch's modern ZIP format. This extends the
existing in-memory model/optimizer/scheduler lifecycle. Native PyTorch remains a
reference-test dependency; the independent package needs only NumPy.

The writer emits persistent storage descriptors and fixed native tensor rebuilders
without importing native Torch or changing `sys.modules`. The reader resolves
only known data globals. It retains shared storage, repeated tensor identity,
shape/stride/offset, empty/expanded views, OrderedDict metadata, Parameter identity
and requires-grad flags. Loaded tensors are fresh leaves without gradients/tapes.
Bool, signed/unsigned8/16/32/64, float16/32/64 and complex64/128 inference are
covered. Wider unsigned types use untyped storage and explicit dtype metadata.
Borrowed padded/offset NumPy buffers use a bounded accessible physical span rather
than flattening their logical values; mixed-dtype buffers and negative tensor
strides remain explicit unsupported cases.

Default protocol2 exchanges weights-only data with native Torch. Our reader also
accepts protocols3–5; native weights-only readers reject some higher-protocol
opcodes, so native higher-protocol tests use only trusted generated fixtures with
that reader's executable mode. StrideTorch itself rejects `weights_only=False`,
whole modules/arbitrary classes, custom tensor attributes, executable hooks,
legacy/non-ZIP/TorchScript formats, custom pickle modules, mmap and callable
storage mapping. CUDA-tagged data requires explicit CPU mapping. Typed storage
converts byte order; non-native-endian untyped storage is unsupported.

The format/arguments follow [Torch serialization notes](https://docs.pytorch.org/docs/stable/notes/serialization.html)
and [`torch.load`](https://docs.pytorch.org/docs/stable/generated/torch.load.html),
with the narrower subset above. No external implementation was copied.

## Final compatibility proof

The final hardened source has44 Python modules and two license resources.

- Complete root/independent suite:29960 passed, one optional Docker skip,
  34 warnings in184.26s. Focused serialization/dispatch:114 passed.
- Independent wheel/sdist source and README metadata match. A fresh Python3.10.19,
  NumPy1.26.4 environment with no native Torch passes960 tests.
- Pyodide0.29.4/NumPy2.2.5 under Node and real Chrome155 Worker each pass948
  contracts across30 independently frozen suites. All46 resource/source hashes
  match the final wheel and current checkout.
- Native2.14.1 fixtures cover14 dtypes and both directions of interoperability,
  including resumed AdamW/StepLR updates. An embedded native golden fixture and
  analytical SGD/scheduler continuation also work without native Torch in WASM.
- Public/private wheels have disjoint installed paths, execute ordinary Torch
  aliases and checkpoint/resume workflows, and survive uninstalling either owner.
  Private wheel/sdist match current application server/catalog bytes.

Initial prototypes exposed two concrete pickle boundary failures: cached EXT
opcodes bypassed a C unpickler's global check, and BUILD could mutate a host dtype
object. Both red tests are retained. The final stdlib data reader rejects all
extensions, including host-cached ones, and permits BUILD only for OrderedDict
attributes. Tensor rebuild state, bounds, storage length and byte order are
validated. This is a bounded data loader, not a resource-isolation mechanism.

An initial coinstall probe incorrectly equated the fresh namespace returned by
`as_torch()` with `m.torch`; the corrected probe checks the yielded namespace and
passes without a product change. A focused invocation with a nonexistent test
path failed before tests and was corrected. Earlier947/941-contract trials and
all failed probes stay separate from the final948-contract proof.

## Measured cost

Two CPU and two actual Pyodide processes retain every sample. Numerical controls
use nine ABBA/BAAB blocks,200 warmups and100 calls/sample with GC enabled; float64 values
and gradients match0.29. Unary/LayerNorm/attention backward, JVP and
inference medians span about−2.14% to+1.04% latency change. This provides no basis
for a broad computation-speedup claim.

Sorted concatenated source/license bytes grow470669→489518 raw and
110287→115361 gzip (+5074 bytes). These are source-footprint measurements, not
network transfer sizes. Four fresh CPU processes per variant/preload condition
show namespace import medians77.42→79.50ms including NumPy, or55.26→57.01ms with
NumPy preloaded. `zipfile` is still absent after ordinary namespace import.

Checkpoint benchmarks use nine alternating blocks, five warmups and three/five
calls per sample. BytesIO/archive work is included; browser download/upload and
persistent disk I/O are excluded. The CPU native context uses the same shapes and
values but supports a broader format. Absolute median ranges across two processes:

| Flow | StrideTorch CPU | Pyodide/WASM |
| --- | --- | --- |
| Save/load one1MiB tensor |110–111 /103–104µs |367–368 /340–342µs |
| Save/load128 tensors of16 values |2.884–2.885 /1.994–1.995ms |6.483–6.578 /4.463–4.487ms |
| Save/load64 shared views |1.047–1.075 /0.565–0.581ms |2.248–2.261 /1.272–1.273ms |
| Save/load/rebuild/restore/real AdamW step |0.730–0.748ms |1.441–1.467ms |

Native CPU saves128 small tensors in1.040–1.057ms and shared views in
0.468–0.474ms; StrideTorch's Python metadata writer remains slower there. Native
loads one1MiB tensor in65–66µs and the continued-training flow takes0.555–0.598ms.
Those regressions remain future work. Source-owner lookup is O(1) per tensor and
storage bytes are deduplicated within one call; no cross-call value cache exists.
Timing processes run sequentially after compatibility gates on a shared host;
actual BLAS thread counts remain unverified. No CPU/WASM cross-runtime throughput
comparison or general native-Torch performance claim is made.

[Core evidence](../../packages/stridetorch/benchmarks/results/iteration-30.json)
and its receipt archive retain source variants, native fixtures, red/green
boundary tests, final CPU/WASM/browser/package reports, all20 timing/import
children and exact recipes. Full application integration and main/remote
adoption remain separate pending gates.


## Complete local application integration

The original native catalog passes78 modules/10218 executions/342 mutants with
three repeats in140.09s. Exported WASM passes78/342 in194947ms at concurrency2,
including all30 independent suites948 contracts with exact frozen hashes. No
original browser budget is exceeded. These durations are budget gates and are
not controlled performance comparisons with prior releases.

All15 incoming upstream UI/style/check/docs files through9d35c936 are preserved
byte-for-byte. They change no Python/package/problem/judge/catalog/lockfile input,
so the complete final29960-test core remains exact. All7546 previously committed
immutable assets remain unchanged;158 new reachable payloads plus three mutable
indexes are frozen. Full runtime source is507015 raw/118223 estimated gzip bytes;
the small schedule profile is37343/11550. Both profiles retain Apache-2.0/NOTICE.

The complete frontend check passes57 bilingual guides/171 prompts at acce1f3a.
The later9d35 change removes affiliation badge logos and adjusts related CSS,
profile-settings assertions and documentation. The targeted profile-settings
check and both builds pass again, using the revalidated actual Pages variable.
Latest default and hosted Pages-profile builds each have three initial JS assets:
403778 raw/130431 gzip and405183 raw/130858 gzip bytes, within original budgets.
The condensed React review preserves existing guards, unconditional hooks,
localized accessible labels and deferred pages;no manual visual QA is claimed.

Public/private wheel ownership, independent uninstall and checkpoint-resume
proofs remain in the core record. The application wheel/sdist are checked again
against all46 core/license hashes and current server/catalog bytes after UI
integration. [Application evidence](../../packages/stridetorch/benchmarks/results/iteration-30-integration.json)
and its219-file archive retain exports, real VM results, dependencies, initial
frontend context and targeted latest-main follow-up. Main adoption uses a
non-force push;remote Quality, actual Pages deployment and live-byte checks remain
separate gates. The last positively verified site version is0.29 through033dc1bc.
