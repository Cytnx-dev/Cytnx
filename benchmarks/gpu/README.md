# GPU Float64 `get` L2 experiment (PR #1230)

Source baseline: [Cytnx #1230](https://github.com/Cytnx-dev/Cytnx/pull/1230), commit
`8de452fa550e0fa8fe3a4d0c2b8d28afa037bffe`.
This experimental branch does **not** change the production `Tensor::get` dispatch.

## Status

The CUDA S1–S4 entrypoint, correctness-gated Google Benchmarks, report processor,
and executable Kaggle notebook are committed. **No GPU benchmark result or
performance recommendation should be inferred from a successful source commit.**
Only publish performance tables after the notebook completes Release compilation
and produces `full.json`.

Kaggle notebook source: `benchmarks/gpu/cytnx_gpu_get_kaggle.ipynb`.

## Methods

- S0 calls the original `cuGetElems_contiguous_gpu_d()` **unchanged**.
- S1 one thread per scalar; recomputes `Loc` per scalar.
- S2 one warp per unit, one lane computes `Loc`, broadcasts with `__shfl_sync`,
  each lane copies offsets spaced by 32.
- S3 2-D block `(32, 8)`, 32-element tile per unit, flattening tile and unit
  group into the 1-D grid to avoid `grid.y` limits. Large units span blocks.
- S4 dispatches to S0–S3 using four `Nunit` buckets: 1–4, 5–32,
  33–1024, 1025+. The notebook measures representative A/B cases and
  chooses the fastest successful S0–S3 in each bucket, **then** measures all
  S0–S4 on the full suite. Override using the four-digit environment
  variable `CYTNX_BENCH_GPU_GET_DISPATCH` (`0`.. `3`; initial fallback
  is `0223`). These are pilot-trained coarse thresholds, not claims of
  general optimality.

All variants include full per-call L2 metadata allocation, H2D transfers,
launch and cleanup. `cudaDeviceSynchronize()` is inside the Google
Benchmark timed loop, and `UseRealTime()` provides wall-clock timing.
Every case is verified against an independently computed CPU reference
**before** measuring; invalid output fails with `SkipWithError`.
The input buffer and output buffer are allocated and filled outside timing.

### Workloads

A (fixed output size): 
- 262144 elements: `(1,262144), (32,8192), (256,1024), (4096,64), (65536,4), (262144,1)`.
- 1048576 elements: corresponding `TotalElem` values multiplied by four
  while retaining `Nunit`.

B (vary TotalElem):
- `Nunit=1`: `TotalElem=256,2048,32768`.
- `Nunit=32`: `TotalElem=256,2048,32768`.
- `Nunit=1024`: `TotalElem=1,32,256,1024`.
- `Nunit=65536`: `TotalElem=1,4,16`.

C: `TotalElem=4096, Nunit=64`: identity, strided (2× larger source
buffer and locator `2*i`), reversed, seeded random permutation.

All S0–S4 kernels use the **same input, output and metadata semantics**
in each case.

### Run on Kaggle

1. Create a Kaggle Notebook, enable an NVIDIA GPU and Internet, select
   Python, and use `cytnx_gpu_get_kaggle.ipynb` as the notebook source.
2. The notebook clones the branch and prints the exact commit SHA.
3. It installs native C++ development dependencies (C++20 `std::format`,
   OpenBLAS, LAPACKE, ARPACK, Google Benchmark, CUDA Toolkit from Kaggle),
   uses detected GPU compute capability, configures `CMAKE_BUILD_TYPE=Release`,
   `USE_CUDA=ON`, `RUN_BENCHMARKS=ON`, `BUILD_PYTHON=OFF`,
   `RUN_TESTS=OFF`, `USE_CUTENSOR=OFF`, `USE_CUQUANTUM=OFF`,
   `USE_HPTT=OFF`, and builds **only** `benchmarks_main`.
4. Pilot `pilot.json` derives S4 mapping. Full `full.json` and
   `gpu_get_summary.csv`/`gpu_get_report.md` summarize all cases.
5. Review `environment.txt`, `CMakeCache.txt`, and logs for exact GPU
   and compiler. Build tree in `build/kaggle-gpu-get` is reused inside a
   continuing Kaggle session.

If run manually after cloning the branch:

```bash
bash benchmarks/gpu/kaggle_run.sh
```

The notebook/script must **not** be run concurrently with another
benchmark or build in the same GPU session. Use the generated full report
for any speedup claim (`speedup = S0 latency / candidate latency`).

## Build strategy decision

**Selected A (native Kaggle build)**. Option B (GitHub Actions prebuild)
adds Linux ABI and CUDA toolkit/library version compatibility, artifact
packaging and download complexity; Cytnx GPU-test Actions and CUDA wheel
artifacts do not provide a directly runnable `benchmarks_main`. Native
build is simpler for one experimental branch, supports the actual GPU's SM,
and caches objects within the same Kaggle session. A cross-session cache
would require persisting Kaggle Dataset/Notebook artifacts explicitly.

## Reporting cautions

The L2 timing is **not** kernel-only. Metadata allocation and transfer can
dominate small workloads; this suite cannot isolate their percentage without
a separately instrumented profiling experiment. Avoid interpreting a
single-GPU margin of a few percent as robust acceleration. Preserve every
slow or failed strategy rather than pruning unfavourable results.
