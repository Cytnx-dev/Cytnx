---
name: google-cpp-style
description: >-
  Enforce the Google C++ Style Guide as adopted by Cytnx when writing, reviewing,
  or refactoring C++ or CUDA (.cpp/.hpp/.cu/.cuh) code. Use when creating new
  C++/CUDA files or symbols, naming types/functions/variables, choosing
  parameter-passing conventions, or reviewing a C++ diff for style. Covers
  Cytnx's naming rules (snake_case for new code, no leading underscore,
  trailing-underscore in-place mutators), pass-scalars-by-value, includes, error
  handling, and the clang-format v14 boundary.
---

# Google C++ style (as adopted by Cytnx)

Cytnx uses the **Google C++ Style Guide** as its baseline — `.clang-format` is
`BasedOnStyle: Google` and the maintainers deliberately picked Google style for
the project. The full guide is at
<https://google.github.io/styleguide/cppguide.html>. This skill distills the
rules that matter for review plus the **Cytnx-specific overrides that win where
they diverge from stock Google style**.

## Order of precedence

1. **The file you are editing.** If a file already uses a *consistent* local
   convention, match it — never mix two styles inside one file.
2. **Cytnx overrides** (below) — for new files, and for already-inconsistent
   ("messy") files, where these beat both the file and stock Google.
3. **Google C++ Style Guide** for everything the above do not cover.

Do **not** rewrite an existing file to "fix" its style as a side effect of an
unrelated change — that is churn. Apply these rules to *new* code; much of the
existing tree predates the convention.

## Cytnx overrides (take precedence over stock Google)

- **No leading underscore in new code — strict.** `_foo`, `_impl`, `_Load` are
  the *old* private-member / internal convention and are being retired (#836).
  New members, helpers, and locals must not start with `_`.
- **A trailing underscore marks an in-place mutator that returns `*this`** —
  `Add_`, `contiguous_`, `Inv_`, `Exp_`. This is Cytnx's meaning and **differs
  from Google**, where a trailing underscore denotes a data member. Do not give
  member variables a trailing underscore, and do not "correct" the existing
  mutators.
- **snake_case for new identifiers.** Functions, variables, and members in new
  code are `lower_snake_case` (#836). Type names stay `PascalCase` (`UniTensor`,
  `Storage`). Much existing code is `CamelCase`/mixed — match the file when
  editing it, use snake_case when adding new files.
- **Pass small/scalar types by value, not `const&`.** Built-ins, enums,
  `cytnx_uint64`, `cytnx_int64`, `Scalar`, and complex scalars (`complex<double>`
  / `cytnx_complex128`) go **by value**. For a Cytnx scalar a reference is *never*
  faster: `const&` forces a stack spill plus pointer indirection, whereas a value
  can ride in registers. Reserve `const&` for genuinely large objects — `Tensor`,
  `Storage`, `UniTensor`, `std::vector`, `std::string`, containers.
- **C++20 / CUDA 20 are available and expected** (`CMAKE_CXX_STANDARD 20`,
  `CMAKE_CUDA_STANDARD 20`). Use them; never add C++17 back-compat guards or
  workarounds (see `GEMINI.md`).

## Google naming reference (baseline)

- **Types** (class/struct/enum/alias/type template param): `PascalCase`, no
  underscores — `UniTensor`, `BlockStore`.
- **Constants** (`constexpr`/`const`, static storage): `kCamelCase` —
  `kDefaultBondDim`.
- **Macros**: `ALL_CAPS_WITH_UNDERSCORES` — but avoid macros; prefer `constexpr`.
- **Namespaces**: short, `lower_case` — `cytnx`, `cytnx::linalg`.
- **Files**: `lower_case` (underscores allowed); headers `.hpp`, sources `.cpp`,
  CUDA `.cu`/`.cuh`.
- Variables/functions: stock Google uses `CamelCase` functions and `snake_case`
  variables — **Cytnx overrides functions to snake_case for new code** (above).

## Design & correctness rules worth enforcing

- **Includes**: include what you use; forward-declare in headers instead of
  pulling heavy includes. `.clang-format` has `SortIncludes: false`, so **do not
  reorder existing include blocks** — only place new includes sensibly.
- **Const-correctness**: mark methods `const` when they do not mutate; mark
  single-argument constructors `explicit`.
- **Ownership**: prefer values and RAII; owning heap resources use smart
  pointers; a raw pointer/reference is non-owning.
- Use `nullptr` (not `NULL`/`0`); `override`/`final` on virtual overrides;
  `enum class` over unscoped enums for new code.
- **No `using namespace` in headers** — keep public headers self-contained.
- **Errors**: raise Cytnx errors with `cytnx_error_msg(cond, "fmt", …)` (throws
  `cytnx::error`, surfaced in Python as `cytnx.CytnxError`; the function name
  comes from `CYTNX_FUNC_NAME`) rather than a bare `throw` or `assert` — this is
  the project idiom.

## Formatting is not your job

`.clang-format` (**pinned to v14**, `BasedOnStyle: Google`, column 100,
`IndentWidth: 2`) owns every whitespace, brace, and wrapping decision. Never
hand-format and never comment on layout — run `pre-commit run --files <changed>`
and let the hook settle it (CI's `clang-format-check` uses v14; a *different*
version reflows and fails CI). Review naming, design, and correctness — the
things a formatter cannot check.

## Review checklist

- [ ] No new leading-underscore identifiers.
- [ ] Trailing underscore only on in-place mutators returning `*this`.
- [ ] Scalars / built-ins passed by value, not `const&`.
- [ ] New symbols snake_case (types `PascalCase`); existing file's style matched.
- [ ] No C++17-compat workarounds; C++20 / CUDA 20 features used freely.
- [ ] Errors via `cytnx_error_msg`; `explicit` single-arg ctors; `const` methods.
- [ ] No layout comments — clang-format v14 owns formatting.
