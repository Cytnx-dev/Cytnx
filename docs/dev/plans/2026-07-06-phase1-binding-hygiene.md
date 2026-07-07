# Phase 1 Binding Hygiene Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean up the pybind11 layer: kill the `c__*` shadow API + monkey-patch delegation, collapse per-dtype operator overloads, fix `!=`/`bool()` semantics, register an exception translator, and release the GIL on expensive operations.

**Architecture:** Five tasks. T2 and T3 touch the same operator regions of `tensor_py.cpp`/`unitensor_py.cpp`, so **T3's branch stacks on T2's**. T4 stacks on `fix/error-macro-rewrite` (needs `cytnx::error`, PR #983). T5 is independent from master but must NOT touch the `ExpH`/`ExpM` bindings (rewritten by open PR #915). Tests are pytest-based (these changes are Python-visible); the C++ suite is unaffected.

**Tech Stack:** pybind11 ≥3.0 (FetchContent during build; `py::numpy_scalar<T>` available), scikit-build CMake build (`build_py`), pytest.

---

## Shared context for every task

- **Tasks 3-6 work in the SECOND worktree `/Users/yjkao/Cytnx/.claude/worktrees/phase1-binding`** (created at T2's commit; has its own `build_py/`). The original worktree `phase0-safety-fixes` is occupied by a concurrent session (truncate_ fix on the T2 branch) — do not touch it. Use pypkg dir `$SCRATCH/pypkg2` (not `pypkg`) for the same reason. Branch with `git checkout -b <branch> <base>` (never `git checkout master`; never commit to `master`/`worktree-*` utility branches).
- **T2-branch drift note:** `fix/pybind-inplace-return-self` may gain a truncate_-fix commit (+2 pytests → gate 37) from the concurrent session. T3 branches from whatever the branch TIP is at branch time; record the observed baseline tally in your report.
- **Python build:** `build_py/` exists (configured `BUILD_PYTHON=ON, RUN_TESTS=OFF, USE_MKL=OFF, USE_HPTT=OFF, USE_CUDA=OFF`). Rebuild with `cmake --build build_py -j10`. The module lands at `build_py/cytnx.cpython-313-darwin.so`.
- **Run pytests against the dev build** (never copy the .so into the repo's `cytnx/` dir):

```bash
SCRATCH=/private/tmp/claude-501/-Users-yjkao-Cytnx/ec89d9c5-55f7-474b-ad68-a79b3d8f7a37/scratchpad
mkdir -p $SCRATCH/pypkg/cytnx
cp cytnx/*.py cytnx/*.tmp $SCRATCH/pypkg/cytnx/   # glue + build-info files __init__.py reads at import
cp build_py/cytnx.cpython-313-darwin.so $SCRATCH/pypkg/cytnx/
# python3 -P is REQUIRED: without it, the repo's cytnx/ source dir (cwd) shadows the package
PYTHONPATH=$SCRATCH/pypkg python3 -P -m pytest pytests/ -q      # full suite (~7 s)
PYTHONPATH=$SCRATCH/pypkg python3 -P -m pytest pytests/<file> -q  # targeted
```

RE-COPY `cytnx/*.py` into `$SCRATCH/pypkg/cytnx/` after every edit to the python glue, and the `.so` after every rebuild — stale copies are the #1 confusing failure.

- **Baseline (established 2026-07-06 on master-equivalent base): 28 passed, 0 failed, 0 skipped, ~7 s.** Every task's gate = 28 + its new tests, nothing else changed.
- **Known landmines:** `numpy` is required by from_numpy paths; the DMRG/TDVP example tests are slow (minutes) — use `-x -q` and run the full suite only before commits.
- **Do not touch:** `pybind/linalg_py.cpp` ExpH/ExpM bindings (open PR #915 rewrites them); `tests/BlockUniTensor_test.h`, `tests/utils/getNconParameter.h` (local uncommitted shims); anything in `docs/`, `build_t/`, `build_py/`, logs.
- The stub pipeline (`tools/generate_stubs.py`, committed `.pyi`) is NOT on master (it's in PR #915) — do not attempt stub regeneration; instead ensure new bindings follow the one-overload-per-Python-type discipline so stubs come out clean once #915 lands.

---

### Task 1: Python test baseline (setup, no PR)

- [ ] On `worktree-phase0-safety-fixes` (base state = master): `cmake --build build_py -j10` (should be a no-op or small rebuild).
- [ ] Assemble `$SCRATCH/pypkg` per shared context; `PYTHONPATH=$SCRATCH/pypkg python3 -c "import cytnx; print(cytnx.__version__ if hasattr(cytnx,'__version__') else 'ok')"`.
- [ ] Run `PYTHONPATH=$SCRATCH/pypkg python3 -m pytest pytests/ -q 2>&1 | tail -5`; record pass/fail/skip counts as THE BASELINE. If tests fail at baseline, record which (pre-existing) — do not fix them.

---

### Task 2: Kill the shadow API (branch `fix/pybind-inplace-return-self`, base `master`)

**Problem (issues #779, #336):** In-place methods are bound as `c__iadd__`/`cConj_`/`c_relabel_`/… and re-exported by monkey-patching wrappers in `cytnx/*_conti.py` that call the shadow binding and `return self`. Three inconsistent return conventions exist; every in-place op pays a wasted Python-object copy; `help()`/stubs see the wrong API.

**Canonical replacement pattern** — bind the real dunder/method name directly with a lambda that takes and returns the SAME `py::object`:

```cpp
// in-place arithmetic: python `t += x` keeps object identity
.def("__iadd__", [](py::object self, const cytnx::Tensor &rhs) {
  self.cast<cytnx::Tensor &>().Add_(rhs);
  return self;
})
// in-place named method: `t.Conj_()` chains
.def("Conj_", [](py::object self) {
  self.cast<cytnx::Tensor &>().Conj_();
  return self;
})
```

This is identity-exact (returns the caller's own PyObject), needs no return_value_policy reasoning, and works for void- and ref-returning C++ methods alike.

**Complete inventory to convert (from recon — implementer must cover every row):**

- `tensor_py.cpp`: `c__iadd__`→`__iadd__` (Add_), `c__isub__`→`__isub__` (Sub_), `c__imul__`→`__imul__` (Mul_), `c__itruediv__`→`__itruediv__` (Div_), `c__ifloordiv__`→`__ifloordiv__` (FloorDiv_) — each currently ~24 overloads; keep the same overload set in this task (T3 collapses them). `c__ipow__`→`__ipow__` (linalg::Pow_), `c__imatmul__`→`__imatmul__` (self = Dot(self,rhs) — keep the rebind INSIDE C++ via `self.cast<Tensor&>() = Dot(...)`, return self). `cConj_/cExp_/cInvM_/cInv_/cAbs_/cPow_` → `Conj_/Exp_/InvM_/Inv_/Abs_/Pow_` returning self.
- `unitensor_py.cpp`: `cConj_/cTrace_(×2)/cTranspose_/cnormalize_/cDagger_/ctag/c__ipow__/cPow_/ctruncate_(×2)/c_set_name/c_set_label(×2)/c_set_labels/c_relabel_(×4)/c_relabels_(×2)/c_set_rowrank_/cfrom` → direct names (`Conj_`, `Trace_`, `Transpose_`, `normalize_`, `Dagger_`, `tag`, `__ipow__`, `Pow_`, `truncate_`, `set_name`, `set_label`, `set_labels`, `relabel_`, `relabels_`, `set_rowrank_`, `convert_from`), all returning self via the py::object pattern. Keep the deprecation warnings where the conti layer or binding had them.
- `bond_py.cpp`: `c_redirect_`→`redirect_` returning self. `c_getDegeneracy_refarg`(×2) and `c_group_duplicates_refarg` → bind `getDegeneracy(qnum, return_indices=False)` and `group_duplicates()` directly in C++ returning `py::tuple`/value (fold the Bond_conti.py logic into the lambda).
- `storage_py.cpp`: 11 `c_pylist_<T>` → one `pylist()` binding that switches on `self.dtype()` in C++ (returns the right `std::vector<T>` via `py::cast`).
- `Tensor_conti.py`/`UniTensor_conti.py`/`Storage_conti.py`: keep ONLY the genuinely-Python pieces: `__iter__` iterators, the `to/astype/contiguous` short-circuit wrappers (or move the short-circuit into the C++ lambda and delete — implementer's choice, but be consistent and report), `UniTensor.at` Hclass proxy wrapper, `Storage.pylist` (delete once C++ pylist exists). `Bond_conti.py`: delete the delegation parts. `Network_conti.py` (Diagram/graphviz) and `Symmetry_conti.py` (`Qs`) stay.
- `cytnx/__init__.py`: keep imports for whatever conti files remain.

**Steps:** (TDD; new pytest file first)

- [ ] Branch: `git checkout -b fix/pybind-inplace-return-self master`
- [ ] Write `pytests/binding_inplace_test.py` — identity + chaining + still-works assertions:

```python
import cytnx
from cytnx import Tensor, Type


def test_iadd_preserves_identity():
    t = cytnx.zeros([4])
    tid = id(t)
    t += 1.0
    assert id(t) == tid
    assert t[0].item() == 1.0


def test_inplace_named_methods_chain():
    t = cytnx.ones([2, 2])
    r = t.Conj_().Abs_()
    assert r is t


def test_unitensor_inplace_chain():
    ut = cytnx.UniTensor(cytnx.ones([2, 2]))
    r = ut.set_name("x").relabel_(["a", "b"])
    assert r is ut
    assert ut.name() == "x"
    assert ut.labels() == ["a", "b"]


def test_storage_pylist_roundtrip():
    s = cytnx.Storage(3, Type.Double)
    s.fill(1.5)
    assert s.pylist() == [1.5, 1.5, 1.5]


def test_bond_group_duplicates_tuple():
    b = cytnx.Bond(cytnx.BD_KET, [[0], [0], [1]], [1, 1, 2], [cytnx.Symmetry.U1()])
    nb, mapper = b.group_duplicates()
    assert isinstance(mapper, list)
```

(Adapt constructor signatures to what pytests/bond_test.py actually uses — read it first. `r is t` for the named methods is the new guarantee this task introduces.)

- [ ] Run against baseline build: identity/chaining tests FAIL today for `Conj_` etc. (`r is t` false — wrapper returns self but `t.Conj_()`... check: today's monkey-patch DOES return self, so those may PASS at baseline; the ones that fail are the direct-binding behaviors after conti deletion. Record which are red at baseline honestly; the real red is after deleting the conti layer with bindings not yet fixed — don't ship tests that were already green without noting it.)
- [ ] Convert the inventory above file by file; delete the corresponding conti wrappers as each class completes; rebuild `cmake --build build_py -j10` (tensor_py.cpp/unitensor_py.cpp are slow TUs — batch edits before rebuilding); re-copy `cytnx/*.py` into `$SCRATCH/pypkg` after each conti edit.
- [ ] Grep gates: `grep -rn '"c__\|"c_[a-z]' pybind/ | grep def` → zero shadow bindings left (careful of false positives: `clone`, `contiguous_`, `combineBond*`, `capacity`, `construct`); `grep -rn "c__\|cConj_\|cExp_\|cAbs_\|cPow_\|cInv\|cTrace_\|cTranspose_\|cnormalize_\|cDagger_\|ctag\|ctruncate_\|c_set_\|c_relabel\|c_redirect\|c_pylist\|cfrom" cytnx/ pytests/ example/` → zero references.
- [ ] Full pytest suite = baseline + new tests. Commit:

```bash
git add pybind/tensor_py.cpp pybind/unitensor_py.cpp pybind/storage_py.cpp pybind/bond_py.cpp cytnx/ pytests/binding_inplace_test.py
git commit -m "refactor(pybind): bind in-place methods directly, drop the c__* shadow API (#779)

In-place dunders and _-suffixed methods are now bound under their real
names with lambdas that take and return the same py::object, so python
object identity is preserved and chaining works without the
*_conti.py monkey-patch delegation layer. The c__iadd__/cConj_/...
shadow bindings and their python wrappers are removed; genuinely
python-side features (iterators, Network.Diagram, Qs, UniTensor.at
proxy) remain.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Collapse per-dtype operator overloads; fix `__ne__`/`__bool__`; close numpy gaps (branch `fix/pybind-collapse-operator-overloads`, base `fix/pybind-inplace-return-self`)

**Problem (#928/#916/#692):** every Tensor/UniTensor arithmetic op is bound ~24× (12 C++ scalar types + 12 numpy scalars); many collapse to identical Python signatures (→ ~5600 stub `overload-cannot-match` errors once stubs exist). `__pow__` takes only `cytnx_double`; Tensor `__setitem__` has no numpy-scalar overloads; `__ne__` is not bound (Python default negates the elementwise `__eq__` Tensor via `__len__` → `t1 != t2` is a single always-False bool for non-empty tensors — silent wrong answers); `__bool__` is not bound (truthiness falls through to `__len__`).

**The keep-set per binary operator** (the #915 discipline, adapted; reference implementation: `git show origin/claude/fix-stubtest-errors -- pybind/linalg_py.cpp`, helper `dispatch_pyint`):

1. `const Tensor &` (or UniTensor)
2. `cytnx_double` — absorbs Python `float` and `numpy.float64` (a `float` subclass)
3. `cytnx_complex128` — absorbs Python `complex` and `numpy.complex128`
4. `py::int_` — arbitrary-precision Python int, runtime int64/uint64 dispatch (copy `dispatch_pyint`, single-arg variant)
5. `py::numpy_scalar<float>`, `py::numpy_scalar<std::complex<float>>` — preserve 32-bit dtypes (#692)
6. `py::numpy_scalar<T>` for int64/uint64/int32/uint32/int16/uint16/bool — preserve integer dtypes
7. `const Scalar &` — cytnx Scalar objects

Delete the now-unreachable `cytnx_float/int64/uint64/int32/uint32/int16/uint16/bool` direct overloads and `numpy_scalar<double>`/`numpy_scalar<complex<double>>` duplicates. Apply to: `__add__/__radd__/__sub__/__rsub__/__mul__/__rmul__/__truediv__/__rtruediv__/__eq__` and the T2-renamed `__iadd__/__isub__/__imul__/__itruediv__/__ifloordiv__` in BOTH `tensor_py.cpp` and `unitensor_py.cpp` (UniTensor has no `__eq__`/in-place ops bound — leave that gap; do NOT add new operators to UniTensor in this task beyond what exists).

**Additional fixes in this task:**
- `__pow__` (Tensor + UniTensor): accept the keep-set numerics (double path suffices computationally — `linalg::Pow` takes double — but bind `py::int_` and `numpy_scalar<float>` wrappers so dtypes/ints don't fail).
- Tensor `__setitem__`: add the numpy-scalar keep-set (mirror UniTensor's existing coverage).
- `__ne__` (Tensor): explicit elementwise binding — `[](const Tensor &self, rhs) { return self != rhs; }`? Cytnx C++ has no operator!=; implement as `cytnx::linalg::Cpr(self, rhs)`-based inversion — check how `__eq__` is implemented (`self == rhs` → linalg::Cpr) and whether a `Neq`/logical-not kernel exists; if there is no cheap elementwise !=, bind `__ne__` to raise `TypeError` with a clear message ("elementwise != not implemented; use ==") — NEVER leave the silent default. Report which option the codebase supports.
- `__bool__` (Tensor): raise `ValueError` for size>1 ("truth value of a multi-element Tensor is ambiguous"), return `bool(item)` for exactly one element, raise for empty — numpy semantics. NOTE: this changes `if tensor:` behavior (previously used `__len__`); document in commit message.
- Verify pybind11 overload ORDER (first-registered wins in the no-convert pass): Tensor first, then exact numpy scalars, then py::int_, then double/complex, then Scalar last. Write the order down in a comment block above each operator group.

**Tests** — `pytests/binding_dtype_test.py`:

```python
import numpy as np
import cytnx
from cytnx import Type


def test_numpy_float32_preserves_dtype():
    t = cytnx.zeros([2], dtype=Type.Float)
    assert (t + np.float32(1.0)).dtype() == Type.Float
    assert (np.float32(1.0) + t).dtype() == Type.Float
    t += np.float32(1.0)
    assert t.dtype() == Type.Float


def test_python_int_works_and_large_int():
    t = cytnx.zeros([2], dtype=Type.Int64)
    assert (t + 1).dtype() == Type.Int64
    t2 = cytnx.zeros([2], dtype=Type.Uint64)
    _ = t2 + (2**63 + 1)  # > int64 max: must dispatch to uint64, not raise


def test_pow_accepts_int_and_numpy():
    t = cytnx.ones([2])
    assert (t**2)[0].item() == 1.0
    assert (t ** np.float32(2.0))[0].item() == 1.0


def test_setitem_numpy_scalar_preserves_dtype():
    t = cytnx.zeros([3], dtype=Type.Float)
    t[0] = np.float32(2.5)
    assert t.dtype() == Type.Float


def test_ne_is_not_silently_wrong():
    a = cytnx.ones([3])
    b = cytnx.zeros([3])
    r = a != b
    # either elementwise Tensor of ones, or TypeError — never a bare False
    assert not (r is False)


def test_bool_multielement_raises():
    import pytest
    t = cytnx.ones([3])
    with pytest.raises(ValueError):
        bool(t)
    assert bool(cytnx.ones([1]))
```

- [ ] Branch: `git checkout -b fix/pybind-collapse-operator-overloads fix/pybind-inplace-return-self`
- [ ] Write tests; run: dtype tests largely pass (numpy coverage exists), `test_ne_is_not_silently_wrong` and `test_bool_multielement_raises` FAIL (the red).
- [ ] Implement per operator group; rebuild; targeted pytest; count final overloads per op (report before/after, expect 24→~13).
- [ ] Full pytest = baseline + T2 + these. Commit with a body listing the keep-set, the `__ne__` decision, and the `__bool__` behavior change.

---

### Task 4: Exception translator (branch `fix/pybind-exception-translator`, base `fix/error-macro-rewrite`)

**Problem:** every cytnx error surfaces as bare `RuntimeError` (pybind's fallback for `std::logic_error`), indistinguishable from other failures.

- [ ] Branch: `git checkout -b fix/pybind-exception-translator fix/error-macro-rewrite`
- [ ] In `pybind/cytnx.cpp` inside `PYBIND11_MODULE` (top, before submodule bindings):

```cpp
  // cytnx::error (cytnx_error_msg) -> cytnx.CytnxError (subclass of RuntimeError,
  // so existing `except RuntimeError` code keeps working).
  static py::exception<cytnx::error> cytnx_error_exc(m, "CytnxError", PyExc_RuntimeError);
  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) std::rethrow_exception(p);
    } catch (const cytnx::error &e) {
      py::set_error(cytnx_error_exc, e.what());
    }
  });
```

(If `py::set_error` is unavailable in the fetched pybind11, use `cytnx_error_exc(e.what())` — check pybind11 3.x API; report which.)

- [ ] Tests — `pytests/binding_exception_test.py`:

```python
import pytest
import cytnx


def test_cytnx_error_type_exists_and_raises():
    with pytest.raises(cytnx.CytnxError):
        cytnx.zeros([2]).reshape(3, 3)  # shape mismatch -> cytnx_error_msg
    with pytest.raises(RuntimeError):   # subclass relationship preserved
        cytnx.zeros([2]).reshape(3, 3)


def test_message_content():
    with pytest.raises(cytnx.CytnxError, match="reshape"):
        cytnx.zeros([2]).reshape(3, 3)
```

- [ ] Build `build_py` on this branch (first build after branch switch rebuilds the error-header dependents — expected), assemble pypkg, run tests red→green (red = `cytnx.CytnxError` doesn't exist), full pytest = baseline + these.
- [ ] Commit; note in the PR body that this stacks on #983 and must merge after it.

---

### Task 5: GIL release on expensive operations (branch `fix/pybind-gil-release`, base `master`)

**Problem:** no binding releases the GIL; any long linalg call blocks all Python threads.

**Safe by recon:** the `PyLinOp` trampoline uses `PYBIND11_OVERLOAD`, which reacquires the GIL internally, so releasing around LinOp-taking solvers is safe.

- [ ] Branch: `git checkout -b fix/pybind-gil-release master`
- [ ] Add `py::call_guard<py::gil_scoped_release>()` to these bindings ONLY (do not touch ExpH/ExpM — PR #915 rewrites them):
  - `linalg_py.cpp`: Svd, Gesvd, Svd_truncate, Gesvd_truncate, Eigh, Eig, Qr, Qdr(if present), Lanczos (all variants), Lanczos_Exp, Lanczos_Gnd(if separate), Arnoldi (all), Tensordot, Gemm, Gemm_Batch(if present), Kron, Directsum, Det, Matmul, Dot (verify each exists; add to every overload of each).
  - `unitensor_py.cpp`: `Contract`/`Contracts` free functions + `.contract` method.
  - `network_py.cpp`: `Launch`.
  - NOTE where a binding already has `py::call_guard<py::scoped_ostream_redirect,...>` — combine: `py::call_guard<py::gil_scoped_release, py::scoped_ostream_redirect, py::scoped_estream_redirect>()`? CAUTION: ostream redirect must hold the GIL (it touches python sys.stdout)! For any binding with an ostream redirect guard, put gil_scoped_release INSIDE the lambda body around the compute call instead, or skip that binding and report. Investigate which applies; do not blindly combine.
- [ ] Tests — `pytests/binding_gil_test.py`:

```python
import threading
import time
import cytnx


def test_svd_releases_gil():
    a = cytnx.random.normal([400, 400], mean=0.0, std=1.0)
    ticks = []
    stop = threading.Event()

    def ticker():
        while not stop.is_set():
            ticks.append(time.monotonic())
            time.sleep(0.001)

    th = threading.Thread(target=ticker)
    th.start()
    for _ in range(3):
        cytnx.linalg.Svd(a)
    stop.set()
    th.join()
    gaps = [b - a for a, b in zip(ticks, ticks[1:])]
    assert max(gaps) < 0.2, f"GIL held too long: {max(gaps):.3f}s"


def test_lanczos_with_python_linop_still_works():
    # python-subclassed LinOp matvec must be callable while the solver
    # holds a released GIL (trampoline reacquires)
    class MyOp(cytnx.LinOp):
        def __init__(self):
            cytnx.LinOp.__init__(self, "mv", 4)

        def matvec(self, v):
            return v * 2.0

    op = MyOp()
    v = cytnx.ones([4])
    out = op.matvec(v)
    assert out.shape() == [4]
```

(Adapt LinOp constructor/random signatures to actual API from pytests/examples — read `example/` DMRG code for the LinOp subclass pattern, and use the real Lanczos entry point if cheap enough; otherwise the matvec smoke test plus Svd threading test suffice. The ticker-gap assertion can be flaky on loaded machines — use a generous 0.2 s threshold and mark `@pytest.mark.flaky`-free but rerun-once logic if needed; report your choice.)

- [ ] Red: `test_svd_releases_gil` fails on baseline (max gap ≈ full Svd duration). Green after. Full pytest = baseline + these.
- [ ] Commit; body lists every binding that got the guard and the ostream-redirect decision.

---

### Task 6 (optional, last): One-line docstrings for undocumented linalg bindings

Only if time permits after T2–T5 review cycles; scope = `linalg_py.cpp` functions with zero docstrings EXCEPT ExpH/ExpM (#915), one line each, format `"Svd(Tin, is_UvT=True) -> [S, U, Vt]: singular value decomposition of a rank-2 Tensor."`. No tests; build + import suffices. Separate branch `docs/pybind-linalg-docstrings` from master.

---

## Plan self-review notes

- T2 before T3 so the collapse applies to the final (direct) binding names; T3 stacks on T2's branch — its PR must say "merge after T2's PR".
- T4 stacks on #983's branch — PR notes the dependency.
- T5 from master avoids #915 (ExpH/ExpM) and #983 (error header) collisions; the only overlap risk is trivial adjacent-line conflicts in linalg_py.cpp.
- The `__bool__`/`__ne__` changes and Storage `pylist` C++-ification are user-visible; each commit message documents them.
- Stub regeneration deliberately out of scope until PR #915 lands its pipeline.
