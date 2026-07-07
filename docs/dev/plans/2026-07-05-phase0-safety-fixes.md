# Phase 0 Safety Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix six silent-corruption / memory-safety class defects in Cytnx (issues #906, #965, #951, #858, plus two newly found bugs) as six independent branches, one PR each.

**Architecture:** Each task is a self-contained fix on its own branch cut from `master`, developed TDD-style in a single shared worktree with one test-enabled build directory (`build_t`). Tasks touch disjoint files. No task depends on another.

**Tech Stack:** C++20, CMake (Unix Makefiles), GoogleTest (`tests/test_main` via `RUN_TESTS=ON`), pybind11 ≥ 3.0 (Task 4 only).

---

## Shared context for every task

- Repo root (worktree): all paths below are relative to it.
- **Branch naming:** each task creates its branch with `git checkout -b <branch> master` (we are in a linked worktree — `git checkout master` itself would fail because master is checked out in the main repo). Never commit to `master` or to `worktree-phase0-safety-fixes`.
- **Build setup (once, on master, before Task 1):**

```bash
git submodule update --init --recursive   # fresh worktrees lack submodules (cmake_modules/morse_cmake)
cmake -S . -B build_t -DRUN_TESTS=ON -DBUILD_PYTHON=OFF -DUSE_MKL=OFF -DUSE_HPTT=OFF -DUSE_CUDA=OFF \
      -DCMAKE_CXX_FLAGS="-Wno-c++11-narrowing" \
      -DCMAKE_EXE_LINKER_FLAGS="-L/Users/yjkao/miniconda3/lib"   # gtest/gmock dylibs live in the conda env
cmake --build build_t -j10 --target test_main
```

  First build takes a while (fetches googletest, compiles whole library). Later builds are incremental. `build_t/` is untracked, so it survives branch switches; switching branches only rebuilds changed TUs.
  (`-Wno-c++11-narrowing` is a local-dev workaround: pre-existing narrowing conversions in `tests/BlockUniTensor_test.cpp` are hard errors under AppleClang. Tracked separately — do NOT fix those test files in any Phase 0 branch.)
- **Run tests:** `./build_t/tests/test_main --gtest_filter='<Filter>'` for fast iteration; `cmake --build build_t -j10 --target test_main && ./build_t/tests/test_main` for the full suite before each final commit.
- **Error macro convention:** `cytnx_error_msg(cond, "fmt%s", "")` — the trailing `"%s", ""` is required by the existing macro (pre-`__VA_OPT__` style). Match it in any new call.
- **Test naming:** GoogleTest names must not contain underscores (existing violations are issue #857 — do not add new ones). Use `TEST(SuiteName, CamelCaseName)`.
- **Errors throw `std::logic_error`** (until Task 6 introduces `cytnx::error` deriving from it). `EXPECT_THROW(expr, std::logic_error)` works before and after.
- **Known-failing baseline (pre-existing on unmodified master with this toolchain — do NOT try to fix, do NOT count as your regression):**
  `linalg_Test.BkUt_Svd_truncate_return_err_returns_discarded_values`,
  `linalg_Test.BkUt_Gesvd_truncate_return_err_returns_discarded_values`,
  `linalg_Test.BkUt_Svd_truncate_return_err_one_returns_first_discarded_value`,
  `linalg_Test.BkUt_Gesvd_truncate_return_err_one_returns_first_discarded_value`.
  Full-suite gate for every task = 1063 passing / 11 skipped / exactly these 4 failing.
- **Local uncommitted files:** `git status` shows modified `tests/BlockUniTensor_test.h`, `tests/utils/getNconParameter.h` (macOS build workarounds) and untracked `docs/superpowers/`, `build_t/`, logs. Leave them alone; `git add` ONLY the files your task lists.

---

### Task 1: Reject multiple `-1` dimensions in reshape

**Bug:** In `Tensor_impl::reshape_`, the duplicate-`-1` guard tests `new_shape[i] != -1` (always false inside the `< 0` branch) instead of `has_undetermine`, so `t.reshape({-1, -1})` is silently accepted and leaves a `-1` in `_shape`.

**Files:**
- Modify: `include/backend/Tensor_impl.hpp:269-281` (inside `reshape_`)
- Test: `tests/Tensor_test.cpp` (append)

- [ ] **Step 1: Branch**

```bash
git checkout -b fix/reshape-reject-multiple-unknown-dims master
```

- [ ] **Step 2: Write the failing test** — append to `tests/Tensor_test.cpp`:

```cpp
TEST(Tensor, ReshapeRejectsMultipleUnknownDims) {
  cytnx::Tensor t = cytnx::zeros({12}, cytnx::Type.Double);
  EXPECT_THROW(t.reshape({-1, -1}), std::logic_error);
  EXPECT_THROW(t.reshape_({-1, -1}), std::logic_error);
  EXPECT_THROW(t.reshape({-2, 6}), std::logic_error);
  // a single -1 must keep working
  cytnx::Tensor r = t.reshape({3, -1});
  EXPECT_EQ(r.shape(), (std::vector<cytnx::cytnx_uint64>{3, 4}));
}
```

(Match the include/namespace style at the top of the existing file — if the file has `using namespace cytnx;`, drop the `cytnx::` prefixes.)

- [ ] **Step 3: Run test to verify it fails**

```bash
cmake --build build_t -j10 --target test_main && ./build_t/tests/test_main --gtest_filter='Tensor.ReshapeRejectsMultipleUnknownDims'
```

Expected: FAIL — `t.reshape({-1,-1})` does not throw.

- [ ] **Step 4: Fix the guard** in `include/backend/Tensor_impl.hpp`. Replace:

```cpp
        if (new_shape[i] < 0) {
          if (new_shape[i] != -1)
            cytnx_error_msg(
              new_shape[i] != -1, "%s",
              "[ERROR] reshape can only have dimension > 0 and one undetermine rank specify as -1");
          if (has_undetermine)
            cytnx_error_msg(
              new_shape[i] != -1, "%s",
              "[ERROR] reshape can only have dimension > 0 and one undetermine rank specify as -1");
          Udet_id = i;
          has_undetermine = true;
        } else {
```

with:

```cpp
        if (new_shape[i] < 0) {
          cytnx_error_msg(
            new_shape[i] != -1, "%s",
            "[ERROR] reshape can only have dimension > 0 and one undetermine rank specify as -1");
          cytnx_error_msg(
            has_undetermine, "%s",
            "[ERROR] reshape can only have dimension > 0 and one undetermine rank specify as -1");
          Udet_id = i;
          has_undetermine = true;
        } else {
```

- [ ] **Step 5: Run test to verify it passes**, then run the full suite (`./build_t/tests/test_main`). Expected: all pass (this header is widely included — the rebuild is large; that's normal).

- [ ] **Step 6: Commit**

```bash
git add include/backend/Tensor_impl.hpp tests/Tensor_test.cpp
git commit -m "fix(Tensor): reject multiple -1 dimensions in reshape

The duplicate-'-1' guard tested new_shape[i] != -1 (always false inside
the negative branch) instead of has_undetermine, so reshape({-1,-1})
was silently accepted and left a -1 in the shape.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Unconditional dtype check in `Storage::at<T>` / `Tensor::item<T>` (issue #965)

**Bug:** Every `Storage_base::at<T>` specialization wraps its dtype check in `if (cytnx::User_debug)` (default `false`), so e.g. `Tensor({1}, Type.Float).item<double>()` reinterprets a float buffer as double and returns garbage. Note `data<T>()` specializations in the same file already check unconditionally — this makes `at<T>`/`back<T>` consistent with them.

**Files:**
- Modify: `src/backend/Storage_base.cpp` (all `at<T>` and `back<T>` specializations, ~line 650 onward)
- Test: `tests/Storage_test.cpp`, `tests/Tensor_test.cpp` (append)

- [ ] **Step 1: Branch**

```bash
git checkout -b fix/storage-at-unconditional-dtype-check master
```

- [ ] **Step 2: Audit internal callers first** (so the change doesn't break library code that abuses `at<T>`):

```bash
grep -rn "\.at<" src/ include/ | grep -v "//"
grep -rn "\.back<" src/ include/ | grep -v "//"
```

Read each hit and confirm the requested `T` matches the storage dtype at that site (most internal code uses `data<T>()`, which already checks). If a mismatched internal call exists, fix it in the same commit and mention it in the report.

- [ ] **Step 3: Write the failing tests** — append to `tests/Storage_test.cpp`:

```cpp
TEST(Storage, AtDtypeMismatchThrows) {
  cytnx::Storage s(4, cytnx::Type.Float);
  EXPECT_THROW(s.at<double>(0), std::logic_error);
  EXPECT_THROW(s.at<cytnx::cytnx_int64>(0), std::logic_error);
  EXPECT_NO_THROW(s.at<float>(0));
}
```

and to `tests/Tensor_test.cpp`:

```cpp
TEST(Tensor, ItemDtypeMismatchThrows) {
  cytnx::Tensor t = cytnx::zeros({1}, cytnx::Type.Float);
  EXPECT_THROW(t.item<double>(), std::logic_error);
  EXPECT_NO_THROW(t.item<float>());
}
```

(Verify the `Storage(size, dtype)` constructor signature in `include/backend/Storage.hpp`; if it differs, use `Storage s; s.Init(4, cytnx::Type.Float);`.)

- [ ] **Step 4: Run to verify they fail** (the mismatched `at<double>`/`item<double>` calls return garbage instead of throwing):

```bash
cmake --build build_t -j10 --target test_main && ./build_t/tests/test_main --gtest_filter='Storage.AtDtypeMismatchThrows:Tensor.ItemDtypeMismatchThrows'
```

- [ ] **Step 5: Remove the `User_debug` guard** in every `at<T>` and `back<T>` specialization in `src/backend/Storage_base.cpp`. Pattern — change:

```cpp
  template <>
  double &Storage_base::at<double>(const cytnx_uint64 &idx) const {
    if (cytnx::User_debug) {
      cytnx_error_msg(this->dtype() != Type.Double,
                      "[ERROR] type mismatch. try to get <double> type from raw data of type %s",
                      Type.getname(this->dtype()).c_str());
    }
```

to:

```cpp
  template <>
  double &Storage_base::at<double>(const cytnx_uint64 &idx) const {
    cytnx_error_msg(this->dtype() != Type.Double,
                    "[ERROR] type mismatch. try to get <double> type from raw data of type %s",
                    Type.getname(this->dtype()).c_str());
```

Apply to all specializations (float, double, complex<float>, complex<double>, all int widths, bool — both `at<T>` and `back<T>`; some guards are written without braces, e.g. the `std::complex` ones). Do NOT touch the bounds checks or other `User_debug` uses elsewhere in the file.

- [ ] **Step 6: Run the new tests, then the full suite.** If any existing test fails, it was relying on a type-punned read — fix that test/call site and list it in your report.

- [ ] **Step 7: Commit**

```bash
git add src/backend/Storage_base.cpp tests/Storage_test.cpp tests/Tensor_test.cpp
git commit -m "fix(Storage): check dtype in at<T>/back<T> unconditionally (#965)

The dtype-mismatch check was guarded by cytnx::User_debug (off by
default), so Tensor::item<T>() and Storage::at<T>() silently
reinterpreted storage as the wrong type in normal builds. data<T>()
already checked unconditionally; at<T>/back<T> now match.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Scalar in-place Tensor arithmetic must mutate storage, not replace it (issue #906)

**Bug:** `Tensor::operator+=<scalar>` (and `-=`, `*=`, `/=`; also the `Scalar` overloads) do `this->_impl->storage() = linalg::Add(*this, rc)._impl->storage();` (`src/Tensor.cpp:562-...`), replacing the storage pointer. Any view sharing the old storage silently detaches. The Tensor-RHS path (`iAdd`) mutates in place — the two paths disagree.

**Fix:** route scalar RHS through the same `iAdd/iSub/iMul/iDiv` kernels by wrapping the scalar in a shape-`{1}` Tensor (the kernels already broadcast a length-1 RHS, preserve LHS dtype, and reject `real ⊙= complex` — see `src/linalg/iAdd.cpp:16` and `src/linalg/iArithmetic_visit.hpp::ApplyInplaceArithmeticOp`).

**Deliberate behavior changes** (call out in commit message):
1. Views sharing storage stay attached (the bug fix).
2. dtype is preserved: `Float tensor += 1.0` stays `Float` (previously promoted to `Double`). This matches the existing `t += tensor_of_shape_{1}` behavior and numpy's scalar rules.
3. `real_tensor += complex_scalar` now throws (previously silently promoted the tensor to complex).

**Files:**
- Modify: `src/Tensor.cpp` (the `+=`, `-=`, `*=`, `/=` scalar specializations block, starting ~line 561)
- Test: `tests/Tensor_test.cpp` (append)

- [ ] **Step 1: Branch**

```bash
git checkout -b fix/scalar-inplace-ops-preserve-storage master
```

- [ ] **Step 2: Write the failing tests** — append to `tests/Tensor_test.cpp`:

```cpp
TEST(Tensor, ScalarInplaceAddKeepsStorageSharing) {
  cytnx::Tensor a = cytnx::zeros({4}, cytnx::Type.Double);
  cytnx::Tensor b = a;  // shares storage (reference semantics)
  a += 1.0;
  EXPECT_TRUE(cytnx::is(a.storage(), b.storage()));
  EXPECT_DOUBLE_EQ(b.storage().at<double>(0), 1.0);
}

TEST(Tensor, ScalarInplaceOpsPreserveDtype) {
  cytnx::Tensor a = cytnx::ones({2}, cytnx::Type.Float);
  a += 1.0;   // double scalar must not promote the tensor
  a -= 0.5;
  a *= 2.0;
  a /= 3.0;
  EXPECT_EQ(a.dtype(), cytnx::Type.Float);
  EXPECT_FLOAT_EQ(a.storage().at<float>(0), 1.0f);
}

TEST(Tensor, ScalarInplaceRealPlusComplexThrows) {
  cytnx::Tensor a = cytnx::zeros({2}, cytnx::Type.Double);
  EXPECT_THROW(a += cytnx::cytnx_complex128(0, 1), std::logic_error);
}

TEST(Tensor, ScalarInplaceSubMulDivKeepStorageSharing) {
  cytnx::Tensor a = cytnx::ones({3}, cytnx::Type.Double);
  cytnx::Tensor b = a;
  a -= 0.5;
  a *= 4.0;
  a /= 2.0;
  EXPECT_TRUE(cytnx::is(a.storage(), b.storage()));
  EXPECT_DOUBLE_EQ(b.storage().at<double>(2), 1.0);
}

TEST(Tensor, CytnxScalarInplaceAddKeepsStorageSharing) {
  cytnx::Tensor a = cytnx::zeros({2}, cytnx::Type.Double);
  cytnx::Tensor b = a;
  a += cytnx::Scalar(2.5);
  EXPECT_TRUE(cytnx::is(a.storage(), b.storage()));
  EXPECT_DOUBLE_EQ(b.storage().at<double>(1), 2.5);
}
```

`cytnx::is()` lives in `include/utils/is.hpp` (already included by Tensor.cpp; verify the test TU sees it — include `cytnx.hpp` covers it).

- [ ] **Step 3: Run to verify the sharing/dtype tests fail** (throw-test may already pass or fail — record which):

```bash
cmake --build build_t -j10 --target test_main && ./build_t/tests/test_main --gtest_filter='Tensor.Scalar*:Tensor.CytnxScalar*'
```

- [ ] **Step 4: Implement.** In `src/Tensor.cpp`, add a file-local helper above the `+=` block (inside `namespace cytnx`):

```cpp
  namespace {
    // Wrap a scalar as a shape-{1} Tensor on `device` so scalar in-place
    // arithmetic can reuse the iAdd/iSub/iMul/iDiv kernels (which broadcast a
    // length-1 RHS and mutate LHS storage in place). See #906.
    template <class T>
    Tensor _scalar_as_rank1_tensor(const T &rc, const int device) {
      Tensor s({1}, Type.cy_typeid(rc), Device.cpu);
      s.storage().at<T>(0) = rc;
      if (device != Device.cpu) s = s.to(device);
      return s;
    }
    Tensor _scalar_as_rank1_tensor(const Scalar &rc, const int device) {
      Tensor s({1}, rc.dtype(), Device.cpu);
      s.item() = rc;  // Sproxy assignment
      if (device != Device.cpu) s = s.to(device);
      return s;
    }
  }  // namespace
```

(If `s.item() = rc` does not compile, use `s.storage().at(0) = rc;` — both return a `Scalar::Sproxy`; check `include/backend/Scalar.hpp` for the supported assignment. Report which you used.)

Then rewrite every scalar specialization — 11 builtin types + `Scalar`, for each of the four ops (the `Tensor`/`Tproxy`/`Sproxy` specializations stay as they are). Pattern:

```cpp
  template <>
  Tensor &Tensor::operator+=<cytnx_double>(const cytnx_double &rc) {
    cytnx::linalg::iAdd(*this, _scalar_as_rank1_tensor(rc, this->device()));
    return *this;
  }
```

Same for `cytnx_complex128`, `cytnx_complex64`, `cytnx_float`, `cytnx_int64`, `cytnx_uint64`, `cytnx_int32`, `cytnx_uint32`, `cytnx_int16`, `cytnx_uint16`, `cytnx_bool`, and `Scalar` — with `iSub` for `-=`, `iMul` for `*=`, `iDiv` for `/=`.

- [ ] **Step 5: Run the new tests, then full suite.** Existing tests that assert dtype *promotion* after scalar in-place ops encode the old buggy behavior — update them to expect dtype preservation and list every such change in your report.

- [ ] **Step 6: Commit**

```bash
git add src/Tensor.cpp tests/Tensor_test.cpp
git commit -m "fix(Tensor): scalar in-place arithmetic mutates storage in place (#906)

t += 1.0 previously replaced t's Storage with a freshly computed one,
silently detaching every view sharing that storage, while t += tensor
mutated in place. Scalar RHS now routes through the same
iAdd/iSub/iMul/iDiv kernels via a shape-{1} tensor.

Behavior changes: dtype is preserved (Float += 1.0 stays Float, as the
tensor-RHS path already behaved), and real ⊙= complex now throws
instead of silently promoting the tensor.

GPU note: the iAdd GPU path still rebinds on promotion; that remains
tracked under #906.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Remove C `complex.h` from the pybind TU; rename macro-colliding template param (issue #951)

**Bug:** `pybind/cytnx.cpp:15` does `#include "complex.h"` — the C header that defines `#define I _Complex_I`. It only compiles because it comes after `cytnx.hpp`; any reordering (or user TU including `<complex.h>` before cytnx headers) breaks `template <std::size_t I, ...>` in `include/Type.hpp`.

**Files:**
- Modify: `pybind/cytnx.cpp` (delete the include)
- Modify: `include/Type.hpp` (rename template parameters named `I` to `Idx`)
- No new test (compile-time fix); verification is building the python module.

- [ ] **Step 1: Branch**

```bash
git checkout -b fix/remove-complex-h-macro-collision master
```

- [ ] **Step 2: Delete** the line `#include "complex.h"` from `pybind/cytnx.cpp`. Grep the pybind TUs to confirm nothing needs it:

```bash
grep -rn "_Complex_I\|creal\|cimag\| I(" pybind/ | head
```

- [ ] **Step 3: Rename template parameters named `I` in `include/Type.hpp`** to `Idx` (find with `grep -n "std::size_t I\b\|size_t I," include/Type.hpp`). This hardens the public header against any TU that includes `<complex.h>` first. Do NOT touch `src/backend/linalg_internal_cpu/Det_internal.cpp` (its `<complex.h>` use is TU-local and deliberately ordered).

- [ ] **Step 4: Verify the C++ library still builds and tests pass:**

```bash
cmake --build build_t -j10 --target test_main && ./build_t/tests/test_main --gtest_filter='Type*'
```

- [ ] **Step 5: Verify the python module compiles without the include:**

```bash
cmake -S . -B build_py -DBUILD_PYTHON=ON -DRUN_TESTS=OFF -DUSE_MKL=OFF -DUSE_HPTT=OFF -DUSE_CUDA=OFF
cmake --build build_py -j10
```

Expected: builds cleanly. (Import smoke-testing needs package assembly — compile success is the acceptance bar for this task.)

- [ ] **Step 6: Commit**

```bash
git add pybind/cytnx.cpp include/Type.hpp
git commit -m "fix: drop C complex.h from pybind TU and rename macro-colliding template param (#951)

complex.h defines the macro I, which collides with template
<std::size_t I> in Type.hpp and made the pybind TU compile only by
include-order luck. The include was vestigial. Type.hpp's index
parameters are renamed to Idx so user TUs that include <complex.h>
before cytnx headers no longer break.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: Type-promotion helpers + fix cross-kind promotion (issues #858, part of #692)

**Bugs:**
1. Nine-plus call sites use the magic expression `in.dtype() <= 2 ? in.dtype() + 2 : in.dtype()` to compute "the real counterpart" of a dtype (for singular-/eigen-value tensors), relying on enum ordering (`Void=0, ComplexDouble=1, ComplexFloat=2, Double=3, Float=4, ...`).
2. `Type_class::type_promote` (`include/Type.hpp:342-359`) picks the *lower index*, so `ComplexFloat + Double → ComplexFloat`, silently discarding double precision (numpy/torch: `ComplexDouble`).

**Files:**
- Modify: `include/Type.hpp` (add `to_real`/`to_complex`, fix `type_promote`)
- Modify: `src/linalg/Svd.cpp:27`, `src/linalg/Gesvd.cpp:29`, `src/linalg/Gesvd_truncate.cpp:56,71`, `src/linalg/Rsvd.cpp:83,503,518`, `src/linalg/Lstsq.cpp:53`, `src/linalg/Eigh.cpp:30`, `src/linalg/Tridiag.cpp:26` (and any further hits of the grep below)
- Test: `tests/Type_test.cpp` (append)

- [ ] **Step 1: Branch**

```bash
git checkout -b fix/type-promotion-cross-kind master
```

- [ ] **Step 2: Write the failing tests** — append to `tests/Type_test.cpp`:

```cpp
TEST(Type, ToRealMapsToRealCounterpart) {
  EXPECT_EQ(cytnx::Type.to_real(cytnx::Type.ComplexDouble), cytnx::Type.Double);
  EXPECT_EQ(cytnx::Type.to_real(cytnx::Type.ComplexFloat), cytnx::Type.Float);
  EXPECT_EQ(cytnx::Type.to_real(cytnx::Type.Double), cytnx::Type.Double);
  EXPECT_EQ(cytnx::Type.to_real(cytnx::Type.Int64), cytnx::Type.Int64);
}

TEST(Type, ToComplexMapsToComplexCounterpart) {
  EXPECT_EQ(cytnx::Type.to_complex(cytnx::Type.Double), cytnx::Type.ComplexDouble);
  EXPECT_EQ(cytnx::Type.to_complex(cytnx::Type.Float), cytnx::Type.ComplexFloat);
  EXPECT_EQ(cytnx::Type.to_complex(cytnx::Type.ComplexFloat), cytnx::Type.ComplexFloat);
  EXPECT_EQ(cytnx::Type.to_complex(cytnx::Type.Int64), cytnx::Type.ComplexDouble);
}

TEST(Type, PromoteMixedComplexRealUsesMaxPrecision) {
  EXPECT_EQ(cytnx::Type.type_promote(cytnx::Type.ComplexFloat, cytnx::Type.Double),
            cytnx::Type.ComplexDouble);
  EXPECT_EQ(cytnx::Type.type_promote(cytnx::Type.Double, cytnx::Type.ComplexFloat),
            cytnx::Type.ComplexDouble);
  EXPECT_EQ(cytnx::Type.type_promote(cytnx::Type.ComplexFloat, cytnx::Type.Float),
            cytnx::Type.ComplexFloat);
  EXPECT_EQ(cytnx::Type.type_promote(cytnx::Type.ComplexDouble, cytnx::Type.Float),
            cytnx::Type.ComplexDouble);
  // unchanged same-kind rules
  EXPECT_EQ(cytnx::Type.type_promote(cytnx::Type.Double, cytnx::Type.Float),
            cytnx::Type.Double);
  EXPECT_EQ(cytnx::Type.type_promote(cytnx::Type.ComplexFloat, cytnx::Type.Int64),
            cytnx::Type.ComplexFloat);
}

TEST(Type, TensorAddMixedComplexRealPromotes) {
  auto a = cytnx::zeros({2}, cytnx::Type.ComplexFloat);
  auto b = cytnx::zeros({2}, cytnx::Type.Double);
  EXPECT_EQ((a + b).dtype(), cytnx::Type.ComplexDouble);
}
```

- [ ] **Step 3: Run to verify failure** (the `to_real`/`to_complex` tests fail to *compile* — that's the red step for new API; comment them out to watch the promote tests fail at runtime if you want a clean red, then restore):

```bash
cmake --build build_t -j10 --target test_main
```

- [ ] **Step 4: Implement in `include/Type.hpp`.** Next to `type_promote` (line ~342), add (inside `Type_class`, matching the style of `is_complex(unsigned int)` at line 323 — note the enum constants are accessible unqualified inside the class):

```cpp
    // Real counterpart of a dtype: ComplexDouble -> Double, ComplexFloat -> Float,
    // anything else unchanged. Replaces the "dtype <= 2 ? dtype + 2 : dtype" idiom.
    static constexpr unsigned int to_real(unsigned int type_id) {
      check_type(type_id);
      return is_complex(type_id) ? type_id + 2 : type_id;
    }

    // Complex counterpart of a dtype: Double -> ComplexDouble, Float -> ComplexFloat,
    // complex types unchanged, integral/bool types -> ComplexDouble.
    static constexpr unsigned int to_complex(unsigned int type_id) {
      check_type(type_id);
      if (is_complex(type_id)) return type_id;
      if (type_id == Double) return ComplexDouble;
      if (type_id == Float) return ComplexFloat;
      if (type_id == Void) return Void;
      return ComplexDouble;
    }
```

(Adapt the constant spellings to how the enum is declared in this class — check how `type_promote` or `check_type` refer to them; if they use e.g. `Type_class::Double`, follow that.)

Then fix `type_promote` by inserting the mixed-kind branch at the top:

```cpp
    // Find a common type for typeL and typeR
    static constexpr unsigned int type_promote(unsigned int typeL, unsigned int typeR) {
      if (typeL == Void || typeR == Void) return Void;
      // Mixed complex/real: promote the real counterparts, then re-complexify.
      // Fixes ComplexFloat + Double -> ComplexDouble (was ComplexFloat, dropping
      // precision because the enum interleaves complexness and precision).
      if (is_complex(typeL) != is_complex(typeR)) {
        return to_complex(type_promote(to_real(typeL), to_real(typeR)));
      }
      if (typeL < typeR) {
        if (!is_unsigned(typeR) && is_unsigned(typeL)) {
          return typeL - 1;
        } else {
          return typeL;
        }
      } else {
        if (typeR == 0) return 0;
        if (!is_unsigned(typeL) && is_unsigned(typeR)) {
          return typeR - 1;
        } else {
          return typeR;
        }
      }
    }
```

(The old `if (typeL == 0) return 0;` early-outs are subsumed by the `Void` guard at the top; keep the body otherwise identical. `type_promote` is used at compile time via `type_promote_t` — it must stay `constexpr`.)

- [ ] **Step 5: Replace the ordering hacks.** Find all sites:

```bash
grep -rn "dtype() <= 2" src/
```

At each `S.Init(..., in.dtype() <= 2 ? in.dtype() + 2 : in.dtype(), ...)` site, substitute `Type.to_real(in.dtype())`. At `src/linalg/Tridiag.cpp:26`, replace the `Diag.dtype() <= 2 || Sub_diag.dtype() <= 2` condition with `Type.is_complex(Diag.dtype()) || Type.is_complex(Sub_diag.dtype())`. If the grep finds sites beyond the list above, convert them too and list them in your report.

- [ ] **Step 6: Run new tests, then the FULL suite** (`./build_t/tests/test_main`). The promote change alters result dtypes for mixed `ComplexFloat`⊙`Double` operations everywhere (Tensor ops, UniTensor contractions — `src/BlockUniTensor.cpp` uses `type_promote`). Any test asserting the old `ComplexFloat` result encodes the precision-loss bug — update it and list it in your report. If a *kernel dispatch* fails (missing instantiation for a promoted-type combination), STOP and report BLOCKED with the error — do not patch kernel tables ad hoc.

- [ ] **Step 7: Commit**

```bash
git add include/Type.hpp tests/Type_test.cpp src/linalg/
git commit -m "fix(Type): promote mixed complex/real dtypes by precision; add to_real/to_complex (#858)

type_promote picked the lower enum index, so ComplexFloat + Double
yielded ComplexFloat, silently discarding double precision. Mixed
complex/real pairs now promote their real counterparts and
re-complexify (ComplexFloat + Double -> ComplexDouble). Same-kind
promotion is unchanged.

Adds Type.to_real/Type.to_complex and replaces the enum-ordering hack
'dtype <= 2 ? dtype + 2 : dtype' at all linalg call sites.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: Rewrite `cytnx_error.hpp` (bounded formatting, namespaced, single-report)

**Bugs (all in `include/cytnx_error.hpp:27-83`):** unbounded `vsprintf`/`sprintf` into 512/1024-byte stack buffers; `static inline` free functions `error_msg`/`warning_msg` at global namespace in a public header; macro bodies are bare `{...}` blocks (dangling-else hazard); the condition is evaluated twice; errors print to stderr *and* throw; `#define __PRETTY_FUNCTION__ __FUNCTION__` defines a reserved identifier on MSVC.

**Constraints:**
- Call sites stay source-compatible: `cytnx_error_msg(cond, "fmt", args...)` with at least one vararg (the `"%s",""` idiom). C++20 `__VA_OPT__` is available (CMAKE_CXX_STANDARD 20) — use it so zero-vararg calls also work, but do not rewrite call sites.
- 225 test assertions expect `std::logic_error` — the new exception `cytnx::error` must derive from `std::logic_error`.
- Keep the `#include`s the old header provided (`<cstdio> <cstdlib> <cstring> <stdarg.h> <iostream> <stdexcept>` + the execinfo block) — other TUs may depend on them transitively.
- The `UNI_GPU` section (line 85 onward) is out of scope — do not modify it.
- Backtrace printing: keep it, but only when `cytnx::User_debug` is true (declared in `include/Type.hpp:440` — but do NOT include Type.hpp from cytnx_error.hpp, that would create an include cycle; declare `namespace cytnx { extern bool User_debug; }` locally).

**Files:**
- Modify: `include/cytnx_error.hpp:1-83`
- Test: create `tests/cytnx_error_test.cpp`, register in `tests/CMakeLists.txt`

- [ ] **Step 1: Branch**

```bash
git checkout -b fix/error-macro-rewrite master
```

- [ ] **Step 2: Check for direct (non-macro) callers of `error_msg`/`warning_msg`:**

```bash
grep -rn "error_msg(" src/ include/ pybind/ tests/ | grep -v "cytnx_error_msg\|cytnx_warning_msg\|_error_msg_impl" | head -20
```

If any TU calls `error_msg(...)`/`warning_msg(...)` directly, adapt those call sites in this branch (they are few or none).

- [ ] **Step 3: Write the failing test** — create `tests/cytnx_error_test.cpp`:

```cpp
#include <stdexcept>
#include <string>

#include "cytnx.hpp"
#include "gtest/gtest.h"

TEST(CytnxError, LongMessagesDoNotOverflow) {
  std::string big(5000, 'x');
  try {
    cytnx_error_msg(true, "%s", big.c_str());
    FAIL() << "expected throw";
  } catch (const std::logic_error &e) {
    EXPECT_NE(std::string(e.what()).find("xxxx"), std::string::npos);
    EXPECT_GE(std::string(e.what()).size(), big.size());
  }
}

TEST(CytnxError, ThrowsCytnxErrorType) {
  EXPECT_THROW(cytnx_error_msg(true, "boom%s", ""), cytnx::error);
  EXPECT_THROW(cytnx_error_msg(true, "boom%s", ""), std::logic_error);
}

TEST(CytnxError, FalseConditionDoesNotThrow) {
  EXPECT_NO_THROW(cytnx_error_msg(false, "never%s", ""));
}

TEST(CytnxError, MacroIsSafeInIfElse) {
  // dangling-else regression guard: must compile and take the else branch
  bool flag = false;
  if (flag)
    cytnx_error_msg(true, "unreachable%s", "");
  else
    SUCCEED();
}

TEST(CytnxError, ConditionEvaluatedOnce) {
  int n = 0;
  cytnx_error_msg((++n, false), "never%s", "");
  EXPECT_EQ(n, 1);
}
```

Register it in `tests/CMakeLists.txt` inside the `add_executable(test_main ...)` list (alphabetical placement near `Type_test.cpp` is fine; the list is one file per line).

- [ ] **Step 4: Run to verify failure.** `cytnx::error` doesn't exist yet → compile failure (red). The 5000-char test against the OLD header would smash the stack — do not bother running it under the old header; compile failure is the red step.

- [ ] **Step 5: Rewrite `include/cytnx_error.hpp` lines 1-83** (everything before the `#if defined(UNI_GPU)` block) as:

```cpp
#ifndef CYTNX_CYTNX_ERROR_H_
#define CYTNX_CYTNX_ERROR_H_

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdarg.h>

#include <iostream>
#include <stdexcept>
#include <string>

#if defined(__has_include)
  #if __has_include(<execinfo.h>)
    #include <execinfo.h>
    #define CYTNX_HAS_EXECINFO 1
  #endif
#endif

#ifndef CYTNX_HAS_EXECINFO
  #define CYTNX_HAS_EXECINFO 0
#endif

#ifdef _MSC_VER
  #define CYTNX_FUNC_NAME __FUNCSIG__
#else
  #define CYTNX_FUNC_NAME __PRETTY_FUNCTION__
#endif

namespace cytnx {

  extern bool User_debug;  // defined in src/Type.cpp

  // Exception thrown by cytnx_error_msg. Derives std::logic_error for
  // backward compatibility with existing catch/EXPECT_THROW sites.
  class error : public std::logic_error {
   public:
    using std::logic_error::logic_error;
  };

  namespace internal {

    inline std::string vformat_message(char const *format, va_list args) {
      va_list args_copy;
      va_copy(args_copy, args);
      const int n = std::vsnprintf(nullptr, 0, format, args_copy);
      va_end(args_copy);
      if (n <= 0) return std::string(format);
      std::string msg(static_cast<std::size_t>(n), '\0');
      std::vsnprintf(msg.data(), msg.size() + 1, format, args);
      return msg;
    }

    inline std::string compose_report(char const *kind, char const *func, char const *file,
                                      int line, const std::string &msg) {
      std::string out;
      out.reserve(msg.size() + 256);
      out += "\n# Cytnx ";
      out += kind;
      out += " occur at ";
      out += func;
      out += "\n# ";
      out += kind;
      out += ": ";
      out += msg;
      out += "\n# file : ";
      out += file;
      out += " (";
      out += std::to_string(line);
      out += ")";
      return out;
    }

    inline void print_backtrace_if_debug() {
      if (!cytnx::User_debug) return;
#if CYTNX_HAS_EXECINFO
      std::cerr << "Stack trace:" << std::endl;
      void *array[10];
      const int size = backtrace(array, 10);
      char **strings = backtrace_symbols(array, size);
      if (strings != nullptr) {
        for (int i = 0; i < size; i++) std::cerr << strings[i] << std::endl;
        free(strings);
      }
#else
      std::cerr << "Stack trace is unavailable on this platform/compiler." << std::endl;
#endif
    }

    [[noreturn]] inline void error_msg_impl(char const *func, char const *file, int line,
                                            char const *format, ...) {
      va_list args;
      va_start(args, format);
      const std::string msg = vformat_message(format, args);
      va_end(args);
      const std::string report = compose_report("error", func, file, line, msg);
      print_backtrace_if_debug();
      throw cytnx::error(report);
    }

    inline void warning_msg_impl(char const *func, char const *file, int line,
                                 char const *format, ...) {
      va_list args;
      va_start(args, format);
      const std::string msg = vformat_message(format, args);
      va_end(args);
      std::cerr << compose_report("warning", func, file, line, msg) << std::endl;
    }

  }  // namespace internal
}  // namespace cytnx

#define cytnx_error_msg(is_true, format, ...)                                        \
  do {                                                                               \
    if (is_true)                                                                     \
      cytnx::internal::error_msg_impl(CYTNX_FUNC_NAME, __FILE__, __LINE__, (format) \
                                        __VA_OPT__(, ) __VA_ARGS__);                 \
  } while (0)

#define cytnx_warning_msg(is_true, format, ...)                                        \
  do {                                                                                 \
    if (is_true)                                                                       \
      cytnx::internal::warning_msg_impl(CYTNX_FUNC_NAME, __FILE__, __LINE__, (format) \
                                          __VA_OPT__(, ) __VA_ARGS__);                 \
  } while (0)
```

Keep everything from `#if defined(UNI_GPU)` onward unchanged, EXCEPT: if the GPU section references `__PRETTY_FUNCTION__` via the old `#ifdef _MSC_VER` redefine, leave the old `#define __PRETTY_FUNCTION__ __FUNCTION__` for MSVC in place *only if* something in the retained section still uses `__PRETTY_FUNCTION__` (grep the remainder of the file; if nothing does, delete the redefine).

Behavior changes to note: errors no longer pre-print to stderr (the message is in `what()`; backtrace prints only under `User_debug`); exception type is now `cytnx::error` (still a `std::logic_error`).

- [ ] **Step 6: Full rebuild and full test suite.** This header is included everywhere — expect a near-full rebuild. Two classes of fallout to watch:
  - call sites written WITHOUT a trailing semicolon (`{...}` block tolerated it; `do{}while(0)` requires `;`) — fix any the compiler finds;
  - tests asserting on stderr output of errors — update them.

  List all fallout fixes in your report.

- [ ] **Step 7: Commit**

```bash
git add include/cytnx_error.hpp tests/cytnx_error_test.cpp tests/CMakeLists.txt
git commit -m "fix(error): bounded formatting, namespaced impl, single-report errors

Replaces unbounded vsprintf into 512/1024-byte stack buffers with
size-measured vsnprintf into std::string; moves error_msg/warning_msg
out of the global namespace; wraps macros in do{}while(0) (dangling-
else safety, single evaluation of the condition); introduces
cytnx::error : std::logic_error; stops double-reporting (no stderr
print before throw; backtrace only under User_debug); stops defining
the reserved identifier __PRETTY_FUNCTION__ on MSVC.

Call sites are source-compatible; __VA_OPT__ additionally allows
zero-vararg calls going forward.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Plan self-review notes

- Task 5 merges original items "to_real helper (#858)" and "type_promote fix" because the fix uses the helpers — shipping them separately would create a cross-branch dependency.
- Task 3's helper uses `at<T>` writes on a freshly created CPU tensor — dtype always matches, so it composes safely with Task 2 (independent branches).
- Task 6's test file must be registered in `tests/CMakeLists.txt`; all other tasks append to already-registered test files.
- All tasks: full-suite run before final commit; report any existing-test updates explicitly.
