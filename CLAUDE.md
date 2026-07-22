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

C++ tests need `RUN_TESTS=ON` (a `debug-*` preset sets this already). Two agent
skills carry the full build/test/benchmark mechanics — invoke them instead of
re-deriving commands:

- **`build-test-workflow`** — invoke before any build or test run (any
  preset, C++ or Python, with or without a GPU).
- **`cross-revision-benchmark`** — invoke before making or checking any
  performance claim.

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

### Robust C++ in touched code

Treat surrounding Cytnx code as historical context, not as an automatic style
guide. When writing or modifying code:

- Pass cheap scalar values by value, not by `const&`. This includes `bool`,
  integral types, enum-like dtype/device values, floating-point values, `Scalar`,
  and complex scalar types.
- Avoid `cytnx_XXXX` typedefs for ordinary local programming types. Use standard
  C++ types for counters, sizes, flags, loop indices, tolerances, and local
  arithmetic unless the value is specifically a dtype-backed tensor/storage
  scalar.
- Do not add magic dtype integers. Use named dtype constants, type traits, or
  established dispatch helpers.
- Do not copy raw-memory idioms such as `&vec[0]`, vector-plus-`memcpy`, manual
  container copies, or untyped `void*` handling unless the code is genuinely
  doing representation-level storage or serialization work.
- Prefer `std::vector::data()`, iterator construction, `std::copy`, range
  algorithms, and typed loops when the operation is logically typed.
- Keep this as a touched-code cleanup policy. Do not mechanically churn unrelated
  files.

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
- **Tests must check independent expected values.** Do not compare one Cytnx
  implementation path against another path with the same possible bug. For
  arithmetic and dtype changes, check both the result dtype and the numeric value.
  Include values that expose common failures: fractional values, negative values,
  mixed signed/unsigned cases, complex cases, rank-0 cases, and nontrivial shapes.
- **Fix the relevant object families, or state the scope.** If a semantic fix
  applies to `Tensor`, `DenseUniTensor`, `BlockUniTensor`, and
  `BlockFermionicUniTensor`, either fix all relevant paths or state explicitly why
  the PR scope is narrower.

## Domain gotchas (these bite agents)

- **dtype enum ordering: lower index = higher precision** (ComplexDouble=1 <
  ComplexFloat=2 < Double=3 < Float=4 < Int64=5 < …). Do *not* assume a larger
  enum means a wider type.
- **Type promotion goes through `Type.type_promote(a, b)`**, which promotes across
  the real/complex boundary by precision (#858, #982). Never hand-roll a
  "min enum index" rule — always fold with `type_promote`.
- **Rank, scalar, and void states are semantic states.** Use `is_void()`,
  `is_scalar()`, and `rank()` instead of inferring state from `shape()[0]`,
  `shape().size()`, or direct storage access. Check rank before indexing
  `shape()[n]` or `bonds()[n]`.
- **Rank-0 scalar is not the same as shape `{1}`.** A rank-0 tensor has shape
  `{}` and one element. For plain dense `Tensor`, a single-element tensor may be
  convertible to a scalar as a convenience. For `UniTensor`, scalar semantics
  should mean rank 0; do not treat storage size 1 or all extents equal to 1 as
  a scalar because bonds, labels, quantum numbers, directions, and symmetry
  conventions are part of the object.
- **Symmetric `UniTensor` element access is coefficient access.** `ut.at(...)`
  exposes a raw stored coefficient in a chosen basis/sector. It is not generally
  a physical scalar observable. User-facing code should prefer tensor-network
  operations such as contraction, factorization, permutation, conjugation, and
  scalar extraction only after a mathematically scalar result has been produced.
- **GPU in-place arithmetic has kernel gaps.** `cuMul`/`cuDiv` lack the
  non-contiguous tensor⊗tensor kernels that `cuAdd`/`cuSub` have — contiguous-ize
  first or results are silently wrong; a narrow LHS can OOB-write; a length-1
  scalar RHS must stay CPU-resident (#988).
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
