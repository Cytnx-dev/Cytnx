#ifndef PYBIND_PYINT_DISPATCH_HPP_
#define PYBIND_PYINT_DISPATCH_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "Type.hpp"
#include "cytnx_error.hpp"

// =========================================================================
// KEEP-SET ORDERING (canonical rationale -- per-operator comment blocks in
// tensor_py.cpp / unitensor_py.cpp point here)
// =========================================================================
//
// Every scalar-accepting Tensor/UniTensor operator is bound once per
// Python-DISTINGUISHABLE dtype (the "keep-set"), not once per C++ dtype,
// and the overloads MUST be registered in this order:
//
//   1. Tensor / UniTensor (elementwise operand)
//   2. py::numpy_scalar<float>, py::numpy_scalar<std::complex<float>>
//   3. py::numpy_scalar<int64/uint64/int32/uint32/int16/uint16/bool>
//   4. py::int_ (arbitrary precision; dispatch_pyint below picks the
//      int64 or uint64 kernel by magnitude)
//   5. cytnx_double  (absorbs Python float and np.float64, a float subclass)
//   6. cytnx_complex128  (absorbs Python complex and np.complex128)
//   7. cytnx::Scalar  (widest implicit-conversion net, always last)
//
// WHY THE ORDER IS LOAD-BEARING (not just stub cosmetics): pybind11
// resolves an overloaded call in two passes -- every overload is tried
// WITHOUT implicit conversion first, then every overload is retried WITH
// conversion -- and within a pass the first-registered match wins. Two
// traps follow:
//
//   * The __index__ no-convert trap: pybind11's plain arithmetic
//     type_caster for integral C++ types accepts ANY object satisfying the
//     __index__ protocol even in the no-convert pass (pybind11 3.x cast.h,
//     arithmetic type_caster). Every numpy integer scalar implements
//     __index__, so a cytnx_int64-style overload registered before
//     numpy_scalar<int32_t> greedily consumes np.int32 in the FIRST pass
//     and silently rewrites its dtype to Int64; the numpy_scalar overload
//     becomes unreachable. numpy scalars must therefore precede any plain
//     integral (or py::int_-absorbing) overload.
//
//   * The __float__-fallback trap: Python's complex() builtin falls back
//     to __float__ when __complex__ is absent, so the cytnx_complex128
//     caster's convert pass accepts any float-like object (np.float32
//     included). A complex128 overload registered before the float/double
//     ones therefore captures real-valued np.float32 -- either silently
//     upcasting it or, for __setitem__ on a real-dtype container, raising
//     "cannot assign complex element to real container". np.float32 /
//     np.complex64 must precede cytnx_double / cytnx_complex128.
//
// WHEN ADDING A NEW SCALAR-ACCEPTING OPERATOR: copy the keep-set and this
// exact registration order; do NOT add per-C++-dtype overloads (they
// collapse to duplicate Python signatures and break stub generation, see
// issue #928), and do NOT register any plain integral/double/complex
// overload ahead of the numpy scalars.
// =========================================================================

namespace pybind_cytnx {

  // Python int is arbitrary precision while cytnx's fixed-width kernels are
  // not, so a single py::int_ overload dispatches on the operand's magnitude:
  // the signed int64 kernel when it fits (covering all negatives), otherwise
  // the unsigned uint64 kernel for non-negative values up to uint64 max.
  //
  // Single-argument variant of the dispatch_pyint helper introduced for
  // ExpH/ExpM in PR #915 (branch claude/fix-stubtest-errors,
  // pybind/linalg_py.cpp, two-argument form). Once #915 merges,
  // linalg_py.cpp's local copy should be folded into this header (e.g. by
  // adding the two-arg overload here) instead of keeping a third copy.
  template <class Fn>
  auto dispatch_pyint(const pybind11::int_ &a, Fn &&fn) {
    int overflow = 0;
    const long long ia = PyLong_AsLongLongAndOverflow(a.ptr(), &overflow);
    if (overflow == 0) return fn(static_cast<cytnx::cytnx_int64>(ia));
    const unsigned long long ua = PyLong_AsUnsignedLongLong(a.ptr());
    if (PyErr_Occurred()) {
      PyErr_Clear();
      cytnx_error_msg(true, "[ERROR] integer scalar out of the supported int64/uint64 range.%s",
                      "\n");
    }
    return fn(static_cast<cytnx::cytnx_uint64>(ua));
  }

  // =========================================================================
  // permute()/reshape() call-form unification (#293, ruling 4)
  // =========================================================================
  //
  // Both the variadic form (`t.permute(1, 2, 0)`) and the list form
  // (`t.permute([1, 2, 0])`) must be accepted on both Tensor and UniTensor.
  // A single `py::args args` binding sees a variadic call as an N-tuple of
  // scalars and a list call as a 1-tuple whose sole element is itself a
  // Python list -- so the two forms are disambiguated by inspecting args:
  // if len(args) == 1 and that element is a list/tuple, unwrap it; otherwise
  // treat every element of args as one entry of the mapper.
  //
  // parse_index_args<T> extracts a std::vector<T> (T = cytnx_int64 or
  // std::string) from a py::args tuple under either call form. It does not
  // itself decide between the int-mapper and string-mapper overloads of
  // permute()/reshape() -- callers that must support both label kinds (e.g.
  // UniTensor::permute) inspect the args once with is_string_args() first.
  template <class T>
  std::vector<T> parse_index_args(const pybind11::args &args) {
    if (args.size() == 1) {
      pybind11::object sole = args[0];
      if (pybind11::isinstance<pybind11::list>(sole) ||
          pybind11::isinstance<pybind11::tuple>(sole)) {
        return sole.cast<std::vector<T>>();
      }
    }
    return args.cast<std::vector<T>>();
  }

  // Returns true if the mapper (in either call form) is made of Python str
  // elements rather than integers, so the caller can pick the
  // std::vector<std::string> overload instead of std::vector<cytnx_int64>.
  // An empty mapper is treated as the int form (matches the pre-existing
  // list-only bindings' behavior of defaulting to the int overload).
  inline bool is_string_args(const pybind11::args &args) {
    if (args.size() == 0) return false;
    pybind11::object first = args[0];
    if (args.size() == 1 && (pybind11::isinstance<pybind11::list>(first) ||
                             pybind11::isinstance<pybind11::tuple>(first))) {
      pybind11::sequence seq = first;
      if (seq.size() == 0) return false;
      return pybind11::isinstance<pybind11::str>(seq[0]);
    }
    return pybind11::isinstance<pybind11::str>(first);
  }

}  // namespace pybind_cytnx

#endif  // PYBIND_PYINT_DISPATCH_HPP_
