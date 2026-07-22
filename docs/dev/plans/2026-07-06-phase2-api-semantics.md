# Phase 2 API Semantics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Land the API-semantic decisions: shared-block permute corruption fix (#724), Bond immutability + signflip encapsulation (#846/#841), UniTensor elementwise removal (#934/#753/#675), `norm()->double` with `Norm()` deprecation (#676), underscore convention (#335/#336/#381 + #421/#422 merges), permute-syntax unification (#293).

**Decision record (maintainer = yingjerkao, 2026-07-06, via this session):**
1. UniTensor elementwise `+`/`-` (and other non-TN-meaningful elementwise UniTensor⊙UniTensor ops per #934's enumeration): **REMOVED from the public surface**; scalar⊙UniTensor stays. Python dunders raise TypeError with guidance (Contract/Kron/scalar ops).
2. Norm: **add `norm()` returning double** (snake_case per #836); `Norm()` deprecated for one release, still returns Tensor.
3. Underscore rule: **`_` suffix = in-place returning self, everywhere**; non-underscore = out-of-place. `set_name`/`set_label(s)`/`combineBonds` gain `_` variants as the in-place form; duplicate spellings collapse with `[[deprecated]]`/warning shims for one release.
4. permute/reshape syntax: **both variadic and list forms accepted on both Tensor and UniTensor** (additive).

**Consensus items from issue threads:** #724 fix = per-block non-mutating `Tensor.permute()/reshape()` (manuschneider + ianmccul); #846 = full public immutability of Bond, in-place mutators removed, SVD-truncate call sites build new Bonds (IvanaGyro's position prevailed); #841 = `_signflip` private, no public mutable accessor.

## Task/branch map (bases matter — 10 PRs are open)

| Task | Branch | Base | Scope |
|---|---|---|---|
| T1 | `fix/blockut-permute-shared-blocks` | origin/master (d02cb299) | #724, C++ + gtests |
| T2 | `refactor/bond-immutable` | `fix/pybind-inplace-return-self` (#986 tip a3367f99 — bond_py.cpp conflicts otherwise) | #846 + #841, C++ + gtests + pybind mutator removal |
| T3 | `refactor/unitensor-drop-elementwise` | `fix/pybind-collapse-operator-overloads` (ecb1fbd1, #991) | #934/#753/#675 removal, C++ + pybind + pytests |
| T4 | `feat/api-conventions` | stacked on T3's branch | norm() (#676) + underscore audit (#335/#336/#381/#421/#422) + permute syntax (#293); 3 commits, one PR |

Worktrees: `phase1-gil` (free; gets a `build_t` for C++ tests) for T1/T2; `phase1-binding` (free after #991) for T3/T4. `phase0-safety-fixes` may still be held by a chip session — avoid.

## Shared context

- C++ test build (create once in phase1-gil): `cmake -S . -B build_t -DRUN_TESTS=ON -DBUILD_PYTHON=OFF -DUSE_MKL=OFF -DUSE_HPTT=OFF -DUSE_CUDA=OFF -DCMAKE_CXX_FLAGS="-Wno-c++11-narrowing" -DCMAKE_EXE_LINKER_FLAGS="-L/Users/yjkao/miniconda3/lib" -DCMAKE_BUILD_RPATH=/Users/yjkao/miniconda3/lib` (the BUILD_RPATH is required — without it `gtest_discover_tests` can't run the binary post-link, "Error running test executable", and make deletes it). Baseline on new master (post-#982): establish before T1 and record (expect old 1063-baseline + #982's tests, same 4 known BkUt failures unless PR #977 merged — CHECK and record).
- Local shims may be needed again in fresh worktree state (see [[macos-test-build-environment]] memory / Phase-0 plan): tests/BlockUniTensor_test.h fill cast + tests/utils/getNconParameter.h headers — apply as uncommitted, never stage. (Skip if PR #972 merged.)
- pytest recipe: as Phase-1 plan (pypkg dirs, `python3 -P`). Gates per branch = that branch's base tally + new tests.
- Deprecation shim pattern (C++): `[[deprecated("use X_ instead")]]` on the old spelling delegating to the new one; python: `warnings.warn(..., DeprecationWarning)` where a python-level shim exists. One release cycle, per decision record.
- Every task: TDD, two-stage review, single commit per topic, never stage build dirs/logs/shims, verify branch attached.

## T1: #724 — permute_/reshape_ on shared blocks (consensus fix)

Bug: `BlockUniTensor::permute_` (and `reshape_` path on Dense) mutate per-block `Tensor`s in place; when blocks are shared between UniTensors (get_block_ views, cheap copies), the other holder's meta corrupts. Fix per issue consensus: build NEW per-block Tensor meta via the non-mutating `Tensor.permute()/reshape()` and assign into `_blocks[i]` (storage stays shared, meta detaches). Audit ALL Block*/Dense UniTensor in-place reshapers/permuters (`permute_`, `permute_nosignflip_`, `reshape_`, `combineBonds` path if it permutes in place) for the same pattern.

Tests (gtest, tests/BlockUniTensor_test.cpp or new file + register): construct two UniTensors sharing blocks (via get_block_ / the reproduction in issue #724 — read it), permute_ one, assert the other's shape/meta unchanged and data still correct; same for reshape_ on Dense; regression for normal non-shared behavior.

## T2: #846 + #841 — Bond immutability, signflip encapsulation

Per issue #846's enumeration: remove/privatize `Bond::set_type`, `redirect_`, `combineBond_`, `clear_type`, non-const `qnums()`, `syms()`, mutable `getDegeneracies()`; keep const accessors + return-new variants (`redirect()`, `combineBond()`, `retype()` if needed). UniTensor: `bonds()` non-const overload removed (const-only) — internal code migrates to `_bonds` directly or explicit internal setter. #841: remove public `signflip_()` (UniTensor_base virtual + BlockFermionicUniTensor), make `_signflip` private, friend-grant the Svd_truncate internals that legitimately rebuild it (read the issue's proposal). Migrate ALL internal call sites (grep `\.set_type\|\.redirect_\|combineBond_\|clear_type\|\.qnums()\|\.syms()\|signflip_` across src/ include/ pybind/) to the immutable forms — expect Svd/Gesvd truncate + BlockFermionicUniTensor to need new-Bond construction. Keep deprecated shims ONLY where a removal would break the python API silently (pybind: drop `redirect_`/mutators bound by #986 — that is why this branch bases on #986's tip). Behavior: gtest that Bond handed to a UniTensor cannot be mutated through any public path; existing suite must stay green (fixes to tests that used mutators are expected — list each).

## T3: drop UniTensor elementwise (+ decision 1)

Read #934's enumeration first. Remove from public C++ surface (UniTensor.hpp operators + Add/Sub member forms taking UniTensor where elementwise-only) and from pybind (unitensor_py.cpp operator groups just collapsed by #991 — hence stacking). Scalar⊙UniTensor and UniTensor⊙scalar remain. Python `ut1 + ut2` must raise TypeError with message naming Contract()/Kron()/scalar alternatives. C++ callers inside the library (if any legitimately used elementwise UniTensor addition — e.g. linalg Lanczos_Ut adds UniTensors!) — AUDIT: `grep -rn "operator+\|Add(" src/linalg/*Ut*`; if solvers rely on elementwise UniTensor addition as a *vector-space* operation (they do — Krylov needs axpy), the C++ INTERNAL capability must remain (move to a documented internal/explicit function e.g. `linalg::VectorAdd(UniTensor,...)` or keep C++ operators but remove ONLY python exposure — decide based on audit, report; the ruling targets the user-facing python surface first). This nuance MUST be resolved by the implementer with evidence and reported; if removal breaks solvers, implement as: python surface removed + C++ operators kept but documented internal-only — and say so in the PR.

Tests: pytest asserting `ut+ut` raises TypeError with the guidance message; scalar ops still work; solvers (Lanczos_Ut pytest path if exists) still green; full C++ suite green.

## T4: api-conventions (3 commits, one PR)

Commit A — `norm()`: C++ `Tensor::norm()`/`UniTensor::norm()` returning `double` (impl: existing Norm().item<double>() equivalent); `Norm()` marked `[[deprecated]]` (still Tensor). pybind binds `norm` (+ DeprecationWarning shim for `Norm` python-side if feasible via a wrapper binding that warns). linalg::Norm free function: add `linalg::norm` returning double, deprecate the old. Tests both layers.
Commit B — underscore audit: enumerate every mutating method lacking `_` and every `_`-less/`_` duplicate pair across UniTensor/Tensor/Bond/Storage public APIs (`set_name`→`set_name_`, `set_label(s)`→`set_label_`/relabel_ merge per #421 semantics, `combineBonds`→`combineBonds_` in-place + `combineBond`(s) out-of-place merge per #422, `set_rowrank` vs `set_rowrank_` collapse, `tag`→`tag_`? — build the table FIRST, present it in the report, implement with deprecated shims; python shims warn). All `_` methods return self (C++ ref / py object) — #986 already did python; C++ side: fix `void` returns to `Type&` where the table says.
Commit C — permute/reshape syntax: python bindings accept variadic ints on UniTensor (`permute(1,2,0)` and `permute("a","b")`? — variadic strings too for label form) and list form already exists on Tensor? (Tensor python takes *args today; add list acceptance) — make both classes accept both int-list and variadic (+ UniTensor string variadic), tests for each form on each class.

## Sequencing

T1 ∥ T2 forbidden in one worktree (both C++/gtest in phase1-gil) → T1 then T2 sequentially there. T3 then T4 sequentially in phase1-binding (pybind chain). T1/T2 may run parallel to T3/T4 (disjoint worktrees/files — CAUTION: T2 touches pybind/bond_py.cpp and T3/T4 touch unitensor/tensor_py — disjoint files, OK; T2 and T3 both touch UniTensor.hpp (bonds() removal vs operator removal — different regions, and DIFFERENT BRANCHES that will conflict at merge time; acceptable, resolve at merge; note in PRs).

Gates: C++ suite at its base's tally + new (T1/T2); pytest at base tally + new (T3/T4); no regressions beyond recorded known-failing sets.
