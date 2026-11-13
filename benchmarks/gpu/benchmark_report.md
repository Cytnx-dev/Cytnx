# GPU Tensor::get Float64 L2 — PR #1230 experiment

**Status: Kaggle CUDA Release build attempted and FAILED after ~12m49s. No validated latency data
has been produced or retrieved.** Do not interpret any value in this document
as a benchmark measurement.

- Baseline: PR #1230 head `8de452fa550e0fa8fe3a4d0c2b8d28afa037bffe`.
- Experimental branch: `bench/1230-gpu-get-optimization`.
- Kaggle: [cytnx-1230-gpu-get-benchmark-v2](https://www.kaggle.com/code/ivanagyro/cytnx-1230-gpu-get-benchmark-v2)
- GPU selected: NVIDIA T4 x2; script sets `CUDA_VISIBLE_DEVICES=0`.
- CUDA Toolkit observed during execution: 13.0; GCC 13 host toolchain.
- Initial Kaggle Notebook saved execution failed in cell 3 with
  `CalledProcessError` from `bash benchmarks/gpu/kaggle_run.sh`.
  Compilation stopped before `pilot.json` existed. Only a truncated compiler
  diagnostic referencing `include/UniTensor.hpp` was accessible; root cause
  not yet verified. A browser tool suggested a `const st` typo at line 107,
  but the exact GitHub source has valid `const std::vector` there, so that
  speculation is **rejected**. Preserve the full first compiler `error:` line
  from the Kaggle log before making source changes.
- Build flags: Release; CUDA ON; Benchmarks ON; Tests, Python,
  cuTensor, cuQuantum, HPTT, IPO, ASan and coverage OFF.
- Cases: A 12 + B 13 + C 4 = 29 unique workloads, 5 strategies =
  **145 benchmark registrations**, each with 5 repetitions.
- Correctness gate: full Double output bitwise compared to independent
  CPU reference before timing, including random, reversed, strided indexing.

## Pending measurements

No numeric latency, speedup or effective S4 thresholds should be reported
until a successful rebuilt `full.json` and corresponding `pilot.json` are available.
The adaptive S4 policy defaults to `0223` if pilot cases cannot all be
measured; this default is a **hypothesis**, not a performance-derived
recommendation.

The Kaggle runner writes:

- `results/gpu_get/pilot.json` and `dispatch.txt`: pilot and mapping.
- `results/gpu_get/full.json`: complete Google Benchmark data.
- `results/gpu_get/gpu_get_summary.csv`: all S0–S4 case latencies and speedups.
- `results/gpu_get/gpu_get_report.md`: complete Markdown tables and win/loss counts.
- `results/gpu_get/environment.txt` and `CMakeCache.txt`: reproducibility metadata.

The `gpu_get_report.md` is generated **only after** the full run completes.
No recommendation for merging S1, S2, S3 or S4 can be justified until
correctness, complete data, and noise are reviewed.

## Analysis plan

Compare `speedup = latency(S0) / latency(candidate)` across A, B and C,
especially the small-`Nunit` versus large-`Nunit` transition. Because
the full L2 function allocates and transfers metadata on every invocation,
small kernel-level improvements may disappear into host-to-device and
allocator costs. This benchmark deliberately times that total overhead,
but cannot separately isolate its exact fraction without kernel-only
instrumentation (out of scope).
