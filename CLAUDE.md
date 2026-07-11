# Cytnx — guide for coding agents

Cytnx is a C++ tensor-network library with CUDA acceleration and pybind11 Python
bindings. This file is the operational contract for coding agents (Claude Code,
Codex). `AGENTS.md` is a pointer to this file — edit only this file. The Gemini review
bot reads `GEMINI.md` (review-specific reminders); keep it in sync when review
guidance changes. For *what* to work on, see issue #759; this file is only about
working *correctly*.

## Build

Configure with a preset (see `CMakePresets.json`), then build. Artifacts land in
`build/<preset>/`.

```bash
git submodule update --init --recursive          # first time (hptt, morse_cmake), else configure fails
cmake --preset openblas-cpu                       # or: mkl-cpu, openblas-cuda, mkl-cuda, openblas-apple
cmake --build --preset openblas-cpu
```

Key toggles (set by presets): `USE_CUDA`, `USE_MKL`, `BUILD_PYTHON` (ON by
default), `USE_HPTT`. CPU presets iterate fastest; only use a `*-cuda` preset
when a change actually touches GPU code — but when it does, build the cuda preset
to compile-check it **even without a GPU** (the CUDA toolkit installs on GPU-less
machines, including cloud agents; only *running* GPU tests needs real hardware).

## Test

C++ tests need `RUN_TESTS=ON` — use a `debug-*` preset:

```bash
cmake --preset debug-openblas-cpu
cmake --build --preset debug-openblas-cpu --target test_main
ctest --preset cpu-only --output-on-failure
# single suite: build/debug-openblas-cpu/tests/test_main --gtest_filter='Storage.*'
```

Debug presets build with AddressSanitizer (`USE_DEBUG=ON` → `-fsanitize=address`).
If a debug test binary aborts inside ASAN on startup — common under CUDA — run it
with `ASAN_OPTIONS='protect_shadow_gap=0:replace_intrin=0:detect_leaks=0'`.

GPU tests — target `gpu_test_main` (binary under `tests/gpu/`), NOT `test_main`:

```bash
cmake --preset debug-openblas-cuda                # add -DCMAKE_CUDA_ARCHITECTURES=<arch> if autodetect fails
cmake --build --preset debug-openblas-cuda --target gpu_test_main test_main
ctest --preset cpu-and-cuda --output-on-failure   # full GPU arith suite is slow (~10 min)
```

Python — CI installs editable and runs pytest (pyproject pins `--preset=openblas-cpu`):

```bash
pip install --editable '.[dev]' --config-settings=build-dir=build \
  --config-settings=cmake.define.CMAKE_BUILD_TYPE=Debug \
  --config-settings=cmake.define.RUN_TESTS=ON
pytest pytests/ --doctest-modules
```

Two agent skills carry the day-to-day mechanics; invoke them instead of
re-deriving commands:

- **`build-test-workflow`** — incremental builds, running ctest/pytest without
  rebuilding, per-suite `--gtest_filter` iteration, standalone harnesses
  against `libcytnx.a`.
- **`cross-revision-benchmark`** — fair before/after performance comparisons
  across git revisions.

## Format — must match CI exactly

- clang-format is **pinned to v14** (`.clang-format`: Google base, column 100,
  C++20). A newer clang-format reflows differently and **fails CI** — do not
  hand-format, and do not use a different version.
- Run the hooks instead of formatting by hand:
  ```bash
  pre-commit run --all-files          # or: pre-commit run --files <changed>
  ```
- The clang-format hook rewrites files in place and **aborts the commit** if it
  changed anything → `git add` the reformatted files and commit again.

## Coding style

New code follows the **Google style guides** — the project baseline
(`.clang-format` is `BasedOnStyle: Google`). Two agent skills carry the full
rules plus the Cytnx-specific overrides; invoke them when writing or reviewing
code:

- **`google-cpp-style`** — C++/CUDA (`.cpp`/`.hpp`/`.cu`).
- **`google-python-style`** — Python (`cytnx/`, `pytests/`, `tools/`).

The rules agents most often get wrong:

- **No leading underscore in new code** (strict). `_foo` was the old
  private-member convention and is being retired (#836); new members/locals must
  not start with `_`.
- **snake_case for new identifiers**; type names stay `PascalCase`. Existing code
  is mixed — match a file's convention when it is consistent, don't propagate
  messy naming, and don't reformat old files as churn.
- **Trailing underscore = in-place mutator returning `*this`** (`Add_`,
  `contiguous_`) — a Cytnx convention, *not* Google's data-member meaning.
- **Pass scalars by value, not `const&`.** Built-ins, enums, `Scalar`, and
  `complex<double>` are never faster by reference in Cytnx (a value rides in
  registers; `const&` forces a stack spill + indirection). Reserve `const&` for
  large objects (`Tensor`, `Storage`, `UniTensor`, containers).
- **C++20 / CUDA 20 are the standard** — use them; never add C++17-compat
  workarounds.

## Before you push — the CI gates

Green locally before opening a PR. CI enforces:

- **clang-format-check** (v14) — formatting clean.
- **ci-cmake_tests** — `ctest` (`test_main`) *and* `pytest pytests/` both pass; Codecov.
- **ci-downstream-find-package** — installed `find_package(Cytnx)` still works;
  don't break exported targets or public headers.
- **version-consistency** — if you change the minimum Python version, update every
  site listed in `CONTRIBUTING.md`.

## Commits & PRs

- Branch off `master`; never push straight to `master`.
- Small, single-purpose commits following the
  [Conventional Commits](https://www.conventionalcommits.org) standard:
  `fix(linalg): …`, `refactor(Storage): …`, `test(Type): …`.
- Attribute agent-authored commits with a `Co-Authored-By:` trailer naming the agent.
- PR body: **problem → fix → testing**, and link the issue it closes.

## Guardrails (per #759)

- **Keep diffs scoped and reviewable.** Do not refactor broadly in one PR on a
  codebase this size; prefer stacked, single-purpose PRs.
- **Physics / numerical-correctness changes require human review.** An agent must
  not alter algorithmic or mathematical behavior unprompted — flag it explicitly.
- **Heuristic tie-breaks are observable behavior.** Changing *which* of several
  equal-cost candidates a greedy/heuristic planner picks (e.g. the contraction
  order from `OptimalTreeSolver`) changes user-visible results even though every
  choice is individually valid — treat it like the physics guardrail: flag it and
  get sign-off. Verify a reviewer's counterexample by reproducing it on both
  revisions before accepting or disputing it.
- **Call out any mixed-dtype or type-promotion change** — it is easy to get subtly
  wrong (see gotchas below).

## Domain gotchas (these bite agents)

- **dtype enum ordering: lower index = higher precision** (ComplexDouble=1 <
  ComplexFloat=2 < Double=3 < Float=4 < Int64=5 < …). Do *not* assume a larger
  enum means a wider type.
- **Type promotion goes through `Type.type_promote(a, b)`**, which promotes across
  the real/complex boundary by precision (#858, #982). Never hand-roll a
  "min enum index" rule — always fold with `type_promote`.
- **GPU in-place arithmetic has kernel gaps.** `cuMul`/`cuDiv` lack the
  non-contiguous tensor⊗tensor kernels that `cuAdd`/`cuSub` have — contiguous-ize
  first or results are silently wrong; a narrow LHS can OOB-write; a length-1
  scalar RHS must stay CPU-resident (#988).
- **`cuKron` GPU compile is currently broken** under CUDA 13 (#999) — the CUDA
  build is not green for that path.
- **Raise errors via `cytnx_error_msg(cond, "fmt", …)`** (throws `cytnx::error`,
  surfaced in Python as `cytnx.CytnxError`); the function name comes from
  `CYTNX_FUNC_NAME`.
- **Dispatch direction:** the codebase is migrating from function-lookup tables to
  `std::variant`-based dispatch (#650, #938). Follow that pattern for new dispatch;
  do not add lookup tables.

## Repo map

- `src/` — C++ implementation: `src/backend/` (storage, element kernels; CPU +
  `*_gpu` CUDA), `src/linalg/` (linear algebra).
- `include/` — public headers (`cytnx.hpp` is the umbrella).
- `pybind/` — pybind11 bindings (one `*_py.cpp` per submodule; `cytnx.cpp` = root).
- `cytnx/` — Python package wrapping the compiled `.so`.
- `tests/` — C++ gtest (`tests/gpu/` = CUDA suite); `pytests/` — Python tests.
- `CMakePresets.json` — every build configuration.

## See also

- `CONTRIBUTING.md` — metadata housekeeping (min Python version lives in 3 places).
- `RELEASING.md` — tagged-release process (maintainers).
- `Readme.md` — user-facing build/install.
