# WebGPU whole-call comparison against StrideTorch 0.80

The experimental GPU matrix service merits further work for substantial dense
float32 workloads. Small matrices remain faster on the current CPU/WASM path.
These measurements do not enable a production GPU backend or establish a
device-independent dispatch threshold.

The baseline is the actual **0.80.0 StrideTorch CPU implementation in Pyodide**
at `13fcfb6033b7ffab7749c7059f13371dcb225fdb`, rather than native PyTorch or an
older raw NumPy expression. All 65 loaded runtime/license resources match that
checkout. Native PyTorch 2.6 and 2.14 provide the previously qualified numerical
oracles; they are not the timed baseline.

Each measured call computes `a @ b`, both input gradients, and returns three
independent CPU array snapshots. CPU uses `autograd.grad`; the GPU prototype
uses explicit analytical VJP programs. Both include their wrappers and output
copies. Cold mode uploads inputs on every GPU call; resident mode uploads them
at construction and reuses handles. Construction is measured separately. Browser,
Pyodide and service startup, and first-call shader compilation, are outside these
steady-state timings.

## Results

Three fresh headless Chrome **155.0.8059.82** processes use Pyodide **0.29.4**,
NumPy **2.2.5**, and the same real Apple **Metal3** adapter (`fallback=false`).
Times below are medians of the three process medians. The final column is the
range of each process's median paired GPU/CPU ratio; lower is better for GPU.
Shapes denote `(m,k,n)` for `(m,k) @ (k,n)` and the corresponding two VJPs.

| Shape | Inputs | CPU/WASM ms | GPU ms | GPU / CPU across processes |
| --- | --- | ---: | ---: | ---: |
| 37,53,29 | Cold | 0.124 | 1.062 | 8.43–9.94 |
| 37,53,29 | Resident | 0.097 | 0.994 | 9.13–10.51 |
| 64,64,64 | Cold | 0.375 | 1.162 | 2.60–3.46 |
| 64,64,64 | Resident | 0.375 | 1.075 | 2.53–3.01 |
| 256,256,256 | Cold | 13.317 | 3.781 | 0.260–0.284 |
| 256,256,256 | Resident | 13.538 | 3.244 | 0.227–0.240 |
| 512,512,512 | Cold | 99.100 | 20.400 | 0.134–0.204 |
| 512,512,512 | Resident | 100.650 | 15.750 | 0.138–0.158 |

For these measured workloads, GPU is about **3.5–4.4× faster at 256** and
**4.9–7.5× faster at 512**, including the three CPU readbacks. Resident inputs
do not consistently improve the 512 case in every process. No benefit for an
arbitrary shape, dtype, device or end-user workload follows from this table.

The unchanged program MAC limit forces the complete 512-cube workload into two
GPU programs. Cold calls consequently upload **5 MiB** and read back **3 MiB**;
resident input setup uploads **3 MiB**, followed by zero input uploads and
**3 MiB** of readback per call. Partitioning and repeated cold uploads are included.

## Controls and limits

Each process measures all eight shape/input flows. Two independently constructed
identical CPU operations, A0 and A1, share the exact same read-only operands with
the GPU operation. Every flow uses all six implementation orders followed by
their reverse: 12 measured rounds. Flow order rotates across processes.
All **24** median paired A1/A0 controls pass the preregistered **±5%** range.
No timed samples or process results are excluded or replaced.

Calibration targets 80 ms; measured batches target 40 ms, with an explicit
8192-call CPU cap and eight-call GPU cap. Several GPU batches remain shorter
than the target: observed minima range from **5.8 to 24.6 ms** across the eight
flows. These are exploratory measurements with visible raw batch lengths,
rather than a precision claim about small differences. Large process-to-process
variation is preserved in the paired ratio ranges above.

Every implementation passes both native numerical oracles before timing. Each
calibration and measured batch's last output is checked against its preflight
result outside the clock. Input fingerprints, array ownership, no accumulated
caller gradients, NumPy error state, GC restoration and zero live GPU buffers
after each flow are checked. All three processes dispose the host device with
zero tracked handles, bytes and allocations. Queue completion counts include
upload fences and are not counts of compute programs.

The opaque-origin learner Worker still exposes no `navigator.gpu`. The trusted
host owns the GPU through the fixed MessagePort protocol. All eleven default
limits remain unchanged: 32 MiB tracked bytes, 4 MiB/matrix, dimension 1024,
268435456 MAC/program, 24 handles, eight pending, 4096 requests, a 120-second
lease, and eight program inputs/nodes/outputs. Processes send **1006, 826, 830**
requests and peak at **7,340,096** tracked bytes. The original 100-second fixture
deadline also remains in force.

The first harness attempt (`torch-iteration-337`) aborted before any timed case:
its exact quota assertion accidentally omitted the three existing program-count
fields. The fresh frozen `338` fixture includes all eleven defaults. The failed
attempt and all three successful process outputs are retained.

## Follow-up

Continue a bounded dense-float32 GPU prototype, while improving the CPU/WASM
fallback independently. Automatic torch device/autograd integration still needs
storage, mutation, graph and lifetime semantics, more shapes/layouts, exceptional
values, additional operators and cross-device evaluation. These explicit VJP
programs are not a substitute for that work.

Machine-readable samples, source identities and process controls:
[`webgpu-whole-call-0.80.json`](../../packages/torchwasm/benchmarks/results/webgpu-whole-call-0.80.json).
The [raw harness archive](../../packages/torchwasm/benchmarks/results/webgpu-whole-call-0.80-harness.tar.gz)
contains 167 files, 718,014 bytes, SHA-256
`42468a14da58ec5b451120eefb87ae545f33591eb15d9a973e10d2a176b66fa3`.
Every payload was verified against the included SHA-256 manifest. Original
workspace paths and external Chrome/Playwright/Pyodide dependencies must be
adapted for replay. This work changes benchmark evidence only; it does not
publish a package, deploy the application, trigger CI, or alter the supported
CPU/WASM runtime.
