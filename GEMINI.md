# GEMINI.md

Guidance for Gemini Code Assist reviews of this repository.

`CLAUDE.md` (symlinked as `AGENTS.md`) is the canonical guide for anyone — human
or agent — working in Cytnx. Read it first. This file is a short, focused list of
review-specific reminders: the recurring false positives to **stop raising**, and
the things actually worth flagging.

## Hard project constraints — do not suggest changing these

- **Cytnx targets C++20 and CUDA 20.** Both `CMAKE_CXX_STANDARD` and
  `CMAKE_CUDA_STANDARD` are set to `20`. Do **not** request C++17 (or earlier)
  compatibility changes, `#if __cplusplus` guards, or replacements for C++20
  features — concepts, `<bit>`, designated initializers, `constexpr` algorithms,
  and `cuda::std::*` are all intentional.
- **Formatting is owned by clang-format, pinned to v14** (`.clang-format`,
  `BasedOnStyle: Google`, column 100). Do not comment on whitespace, brace
  placement, column width, or include ordering (`SortIncludes: false` is
  deliberate) — CI's formatting check is the source of truth. A *newer*
  clang-format reflows differently and would fail CI, so do not propose
  reformatting either.
- **Complex scalars on GPU use `cuda::std::complex<T>`**, not `cuComplex` /
  `cuDoubleComplex` or host `std::complex` inside device code. This migration is
  deliberate. Do not suggest reverting to the CUDA complex C types or to
  `thrust::complex`.

## Cytnx house style — intentional, not a defect

- **Pass small/scalar types by value, not by `const&`.** For built-ins, enums,
  `cytnx_uint64`, `Scalar`, and even `complex<double>`, by-value is never slower
  in Cytnx and is often faster (values ride in registers). Do not suggest
  converting scalar parameters to `const&`.
- For new or touched code, avoid propagating legacy `cytnx_XXXX` typedefs as
  ordinary loop counters, sizes, flags, or local arithmetic types. Prefer standard
  C++ types unless the value is specifically a dtype-backed tensor/storage scalar.
- Do not recommend vector-plus-`memcpy`, `&vec[0]`, or raw `void*` idioms for
  typed operations. Those patterns are only appropriate for intentional
  representation-level storage or serialization code.
- **A trailing underscore on a method name (`Add_`, `contiguous_`, `Inv_`) marks
  an in-place mutator that returns `*this`.** This is Cytnx's convention and is
  *not* Google's "member variable" meaning — do not flag it or suggest renaming.
- **Errors are raised via `cytnx_error_msg(cond, "fmt", …)`** (with
  `CYTNX_FUNC_NAME`), not `throw` / `assert`. The `if (cond) cytnx_error_msg(...)`
  pattern and trailing `"\n"` are house style; leave them alone.

## What "Google style" means here

Cytnx has adopted the Google C++ / Python Style Guides as its baseline — that is
what `BasedOnStyle: Google` and the `google-cpp-style` / `google-python-style`
agent skills encode. For *review*, apply Google naming/design guidance to **new**
code only:

- New identifiers use snake_case; a **leading underscore is prohibited** in new
  code (it was the old private-member convention and is being retired, #836).
- Do **not** ask contributors to rename to match a file that is already
  inconsistent — much of the existing tree predates the convention. Flag Google
  deviations in *new* files, not churn in old ones.

## Review priorities

Focus reviews on what actually bites this codebase:

- **Numerical correctness**, especially dtype promotion across the real/complex
  boundary (`Type.type_promote`) and mixed-dtype paths.
- **GPU kernel correctness** — in-place `cuMul`/`cuDiv` non-contiguous gaps
  (#988), contiguity assumptions, and out-of-bounds writes on a narrow LHS.
- **Tie-break / selection-order changes in heuristic planners** (e.g. the
  contraction-order search) — observable behavior even when each choice is
  individually valid; they deserve an explicit call-out, not silent acceptance.
- **Rank/scalar/void correctness** — missing rank checks before `shape()[n]` or
  `bonds()[n]`, treating shape `{1}` as a rank-0 scalar, or using direct storage
  access where `item()`, `is_scalar()`, or `is_void()` is the intended API.
- **Symmetric `UniTensor` semantics** — raw `ut.at(...)` coefficient access should
  not be treated as producing a tensor-network scalar, and single-element
  symmetric tensors should not be used as generic scalar broadcasts.
- **Tests with independent expectations** — flag tests that compare one Cytnx
  implementation path against another path with the same suspected bug. Prefer
  tests that check independent expected values, and check both dtype and numeric
  value for arithmetic changes.
- Missing or weak tests; thread-safety and reentrancy; leaked resources.
- Public-API / exported-target breakage (downstream `find_package(Cytnx)`).
- Documentation drift.

Avoid style nits that a formatter or the constraints above already settle.
