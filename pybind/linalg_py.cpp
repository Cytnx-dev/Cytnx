#include <pybind11/buffer_info.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <map>
#include <random>
#include <vector>

#include "cytnx.hpp"
// #include "../include/cytnx_error.hpp"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace cytnx;

#ifdef BACKEND_TORCH
#else

namespace {
  // ExpH/ExpM accept a Python int through a single overload. Python int is
  // arbitrary precision while the kernels are fixed width, so dispatch on the
  // operands' magnitude: the signed int64 kernel when both values fit int64
  // (covering all negatives), otherwise the unsigned uint64 kernel for
  // non-negative values up to uint64 max. A value outside [int64_min, uint64_max]
  // is rejected. Both operands share one kernel, matching the per-dtype overloads
  // (which require `a` and `b` to have the same C++ type).
  template <class Fn>
  auto dispatch_pyint(const py::int_ &a, const py::int_ &b, Fn &&fn) {
    int oa = 0, ob = 0;
    const long long ia = PyLong_AsLongLongAndOverflow(a.ptr(), &oa);
    const long long ib = PyLong_AsLongLongAndOverflow(b.ptr(), &ob);
    if (oa == 0 && ob == 0) return fn(static_cast<cytnx_int64>(ia), static_cast<cytnx_int64>(ib));
    const unsigned long long ua = PyLong_AsUnsignedLongLong(a.ptr());
    const unsigned long long ub = PyLong_AsUnsignedLongLong(b.ptr());
    if (PyErr_Occurred()) {
      PyErr_Clear();
      cytnx_error_msg(true, "[ERROR] integer scalar out of the supported int64/uint64 range.%s",
                      "\n");
    }
    return fn(static_cast<cytnx_uint64>(ua), static_cast<cytnx_uint64>(ub));
  }

  // Register the binary-scalar ExpH/ExpM overloads for one tensor type, where
  // `call(Tin, x, y)` invokes the kernel for concrete C++ scalar types x and y.
  // Keeping one pybind overload per dtype preserves each kernel's numerics, while
  // pybind11-stubgen (against pybind11 3.0.4) renders each overload as a distinct
  // Python annotation: the numpy_scalar variants become the concrete numpy.*
  // types, py::int_ becomes `int`, cytnx_double becomes
  // `typing.SupportsFloat | typing.SupportsIndex`, and cytnx_complex128 becomes
  // `typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex`. The
  // cytnx_complex128 annotation is a supertype of the cytnx_double and integer
  // ones, but it is registered last, so every earlier (narrower) overload is
  // still reachable and mypy reports no overload-cannot-match.
  //
  // Redundant dtypes are intentionally omitted: a Python float/complex already
  // binds to cytnx_double/cytnx_complex128, np.float64/np.complex128 are builtin
  // subclasses caught by those same overloads, and the narrow integer widths all
  // accept the same Python int, so only one Python-int path (dispatched to the
  // int64/uint64 kernels) is kept.
  //
  // Two `b` defaults are passed through py::arg_v with an explicit repr so the
  // stub stays valid, instead of letting pybind11 format the default itself:
  //   - The numpy_scalar overloads use a `numpy.<type>(...)` repr. Left alone,
  //     pybind11 formats the default from the numpy scalar's own repr
  //     (`np.<type>(...)` under numpy 2.x), an unimported `np` alias that mypy
  //     rejects; naming the imported `numpy` module fixes it. stubgen evaluates
  //     that repr and emits an elided `...`.
  //   - The cytnx_complex128 overload uses a literal `...` repr. pybind11 3.0.4
  //     annotates the parameter `SupportsComplex | SupportsFloat | SupportsIndex`,
  //     which does not include the builtin `complex` (typeshed's `complex` has
  //     no `__complex__`), so a `0j` default is rejected as an incompatible
  //     default. A `numpy.complex128(...)` repr would elide too, but it would
  //     misdescribe the default since this overload accepts a Python `complex`,
  //     not a numpy scalar; `...` states "elided" without claiming a type.
  // Either way the default still exists at runtime, so mypy.stubtest is
  // unaffected and tools/generate_stubs.py needs no special-casing.
  template <class TensorT, class Call>
  void bind_exp_scalar(py::module &m, const char *name, Call call) {
    m.def(
      name,
      [call](const TensorT &Tin, const py::numpy_scalar<float> &a,
             const py::numpy_scalar<float> &b) {
        return call(Tin, static_cast<cytnx_float>(a), static_cast<cytnx_float>(b));
      },
      py::arg("Tin"), py::arg("a"),
      py::arg_v("b", py::make_scalar(static_cast<float>(0)), "numpy.float32(0.0)"));
    // #914: cytnx::linalg::Exp returns incorrect values for complex64 tensors and
    // both ExpH and ExpM rely on Exp, so route complex64 scalars through the
    // complex128 kernel. Drop this once #914 is fixed.
    m.def(
      name,
      [call](const TensorT &Tin, const py::numpy_scalar<std::complex<float>> &a,
             const py::numpy_scalar<std::complex<float>> &b) {
        return call(Tin, cytnx_complex128(static_cast<cytnx_complex64>(a)),
                    cytnx_complex128(static_cast<cytnx_complex64>(b)));
      },
      py::arg("Tin"), py::arg("a"),
      py::arg_v("b", py::make_scalar(std::complex<float>(0)), "numpy.complex64(0.0)"));
    m.def(
      name,
      [call](const TensorT &Tin, const py::numpy_scalar<std::int64_t> &a,
             const py::numpy_scalar<std::int64_t> &b) {
        return call(Tin, static_cast<cytnx_int64>(a), static_cast<cytnx_int64>(b));
      },
      py::arg("Tin"), py::arg("a"),
      py::arg_v("b", py::make_scalar(static_cast<std::int64_t>(0)), "numpy.int64(0)"));
    m.def(
      name,
      [call](const TensorT &Tin, const py::numpy_scalar<std::uint64_t> &a,
             const py::numpy_scalar<std::uint64_t> &b) {
        return call(Tin, static_cast<cytnx_uint64>(a), static_cast<cytnx_uint64>(b));
      },
      py::arg("Tin"), py::arg("a"),
      py::arg_v("b", py::make_scalar(static_cast<std::uint64_t>(0)), "numpy.uint64(0)"));
    m.def(
      name,
      [call](const TensorT &Tin, const py::numpy_scalar<std::int32_t> &a,
             const py::numpy_scalar<std::int32_t> &b) {
        return call(Tin, static_cast<cytnx_int32>(a), static_cast<cytnx_int32>(b));
      },
      py::arg("Tin"), py::arg("a"),
      py::arg_v("b", py::make_scalar(static_cast<std::int32_t>(0)), "numpy.int32(0)"));
    m.def(
      name,
      [call](const TensorT &Tin, const py::numpy_scalar<std::uint32_t> &a,
             const py::numpy_scalar<std::uint32_t> &b) {
        return call(Tin, static_cast<cytnx_uint32>(a), static_cast<cytnx_uint32>(b));
      },
      py::arg("Tin"), py::arg("a"),
      py::arg_v("b", py::make_scalar(static_cast<std::uint32_t>(0)), "numpy.uint32(0)"));
    m.def(
      name,
      [call](const TensorT &Tin, const py::numpy_scalar<std::int16_t> &a,
             const py::numpy_scalar<std::int16_t> &b) {
        return call(Tin, static_cast<cytnx_int16>(a), static_cast<cytnx_int16>(b));
      },
      py::arg("Tin"), py::arg("a"),
      py::arg_v("b", py::make_scalar(static_cast<std::int16_t>(0)), "numpy.int16(0)"));
    m.def(
      name,
      [call](const TensorT &Tin, const py::numpy_scalar<std::uint16_t> &a,
             const py::numpy_scalar<std::uint16_t> &b) {
        return call(Tin, static_cast<cytnx_uint16>(a), static_cast<cytnx_uint16>(b));
      },
      py::arg("Tin"), py::arg("a"),
      py::arg_v("b", py::make_scalar(static_cast<std::uint16_t>(0)), "numpy.uint16(0)"));
    m.def(
      name,
      [call](const TensorT &Tin, const py::int_ &a, const py::int_ &b) {
        return dispatch_pyint(a, b, [&](auto x, auto y) { return call(Tin, x, y); });
      },
      py::arg("Tin"), py::arg("a"), py::arg("b") = py::int_(0));
    m.def(
      name,
      [call](const TensorT &Tin, const cytnx_double &a, const cytnx_double &b) {
        return call(Tin, a, b);
      },
      py::arg("Tin"), py::arg("a"), py::arg("b") = double(0));
    m.def(
      name,
      [call](const TensorT &Tin, const cytnx_complex128 &a, const cytnx_complex128 &b) {
        return call(Tin, a, b);
      },
      py::arg("Tin"), py::arg("a"), py::arg_v("b", complex128(0), "..."));
  }
}  // namespace

void linalg_binding(py::module &m) {
  // [Submodule linalg]

  pybind11::module m_linalg = m.def_submodule("linalg", "linear algebra related.");
  m_linalg.def(
    "Rand_isometry",
    [](const Tensor &Tin, const cytnx_uint64 &keepdim, const cytnx_uint64 &power_iteration,
       int64_t &seed) {
      if (seed == -1) {
        // If user doesn't specify seed argument
        seed = cytnx::random::__static_random_device();
      }
      return cytnx::linalg::Rand_isometry(Tin, keepdim, power_iteration, seed);
    },
    py::arg("Tin"), py::arg("keepdim"), py::arg("power_iteration") = 2, py::arg("seed") = -1);

  // ---------------------------------------------------------------------------
  // GIL guard discipline (applies to every py::call_guard<py::gil_scoped_release>
  // in the pybind/ layer; network_py.cpp and unitensor_py.cpp refer here):
  //
  //  (a) A guarded lambda body must NEVER touch py::* / CPython API. The guard
  //      releases the GIL for the duration of the call only: arguments are
  //      converted to C++ types BEFORE the guard is constructed, and the return
  //      value is converted back to Python AFTER the guard is destroyed
  //      (pybind11/attr.h: "T scope_guard; return foo(args...)"). Keep the body
  //      pure C++.
  //  (b) call_guard<A, B> constructs guards left to right. Appending
  //      py::scoped_ostream_redirect (whose default constructor imports
  //      sys.stdout and therefore needs the GIL) AFTER gil_scoped_release will
  //      crash. If a binding ever needs both, put the redirect FIRST, or scope
  //      a py::gil_scoped_release manually inside the lambda body around the
  //      compute call. (Redirected writes themselves are safe without the GIL:
  //      pythonbuf::_sync() in pybind11/iostream.h reacquires it.)
  //  (c) Releasing around LinOp-taking solvers (Lanczos/Arnoldi/...) is safe
  //      even though they call back into Python matvec() overrides: the
  //      PYBIND11_OVERRIDE machinery opens with gil_scoped_acquire
  //      (pybind11/pybind11.h:3532, PYBIND11_OVERRIDE_IMPL) before invoking the
  //      Python method.
  //
  // Scope: all O(n^3)-class linalg entry points carry the guard, EXCEPT
  // ExpH/ExpM which are being rewritten in PR #915 (guards added there would
  // conflict). The elementwise operator sweep (Add/Mul/... in tensor_py.cpp)
  // is deliberately deferred.
  // ---------------------------------------------------------------------------
  m_linalg.def(
    "Svd",
    [](const cytnx::Tensor &Tin, const bool &is_UvT) { return cytnx::linalg::Svd(Tin, is_UvT); },
    py::arg("Tin"), py::arg("is_UvT") = true, py::call_guard<py::gil_scoped_release>());
  m_linalg.def(
    "Svd",
    [](const cytnx::UniTensor &Tin, const bool &is_UvT) { return cytnx::linalg::Svd(Tin, is_UvT); },
    py::arg("Tin"), py::arg("is_UvT") = true, py::call_guard<py::gil_scoped_release>());

  m_linalg.def(
    "Gesvd",
    [](const cytnx::Tensor &Tin, const bool &is_U, const bool &is_vT) {
      return cytnx::linalg::Gesvd(Tin, is_U, is_vT);
    },
    py::arg("Tin"), py::arg("is_U") = true, py::arg("is_vT") = true,
    py::call_guard<py::gil_scoped_release>());
  m_linalg.def(
    "Gesvd",
    [](const cytnx::UniTensor &Tin, const bool &is_U, const bool &is_vT) {
      return cytnx::linalg::Gesvd(Tin, is_U, is_vT);
    },
    py::arg("Tin"), py::arg("is_U") = true, py::arg("is_vT") = true,
    py::call_guard<py::gil_scoped_release>());

  m_linalg.def(
    "Rsvd",  // for Tensor
    [](const Tensor &Tin, cytnx_uint64 keepdim, double err, bool is_U, bool is_vT,
       unsigned int return_err, cytnx_uint64 mindim, cytnx_uint64 oversampling_summand,
       double oversampling_factor, cytnx_uint64 power_iteration, int64_t seed) {
      if (seed == -1) {
        // If user doesn't specify seed argument
        seed = cytnx::random::__static_random_device();
      }
      return cytnx::linalg::Rsvd(Tin, keepdim, err, is_U, is_vT, return_err, mindim,
                                 oversampling_summand, oversampling_factor, power_iteration, seed);
    },
    py::arg("Tin"), py::arg("keepdim"), py::arg("err") = double(0), py::arg("is_U") = true,
    py::arg("is_vT") = true, py::arg("return_err") = (unsigned int)(0), py::arg("mindim") = 1,
    py::arg("oversampling_summand") = 10, py::arg("oversampling_factor") = 1.,
    py::arg("power_iteration") = 0, py::arg("seed") = -1, py::call_guard<py::gil_scoped_release>());
  m_linalg.def(
    "Rsvd",  // for UniTensor, without min_blockdim
    [](const cytnx::UniTensor &Tin, cytnx_uint64 keepdim, double err, bool is_U, bool is_vT,
       unsigned int return_err, cytnx_uint64 mindim, cytnx_uint64 oversampling_summand,
       double oversampling_factor, cytnx_uint64 power_iteration, int64_t seed) {
      if (seed == -1) {
        // If user doesn't specify seed argument
        seed = cytnx::random::__static_random_device();
      }
      return cytnx::linalg::Rsvd(Tin, keepdim, err, is_U, is_vT, return_err, mindim,
                                 oversampling_summand, oversampling_factor, power_iteration, seed);
    },
    py::arg("Tin"), py::arg("keepdim"), py::arg("err") = double(0), py::arg("is_U") = true,
    py::arg("is_vT") = true, py::arg("return_err") = (unsigned int)(0), py::arg("mindim") = 1,
    py::arg("oversampling_summand") = 10, py::arg("oversampling_factor") = 1.,
    py::arg("power_iteration") = 0, py::arg("seed") = -1, py::call_guard<py::gil_scoped_release>());
  m_linalg.def(
    "Rsvd",  // for UniTensor, with min_blockdim
    [](const cytnx::UniTensor &Tin, cytnx_uint64 keepdim,
       const std::vector<cytnx_uint64> min_blockdim, double err, bool is_U, bool is_vT,
       unsigned int return_err, cytnx_uint64 mindim, cytnx_uint64 oversampling_summand,
       double oversampling_factor, cytnx_uint64 power_iteration, int64_t seed) {
      if (seed == -1) {
        // If user doesn't specify seed argument
        seed = cytnx::random::__static_random_device();
      }
      return cytnx::linalg::Rsvd(Tin, keepdim, min_blockdim, err, is_U, is_vT, return_err, mindim,
                                 oversampling_summand, oversampling_factor, power_iteration, seed);
    },
    py::arg("Tin"), py::arg("keepdim"), py::arg("min_blockdim"), py::arg("err") = double(0),
    py::arg("is_U") = true, py::arg("is_vT") = true, py::arg("return_err") = (unsigned int)(0),
    py::arg("mindim") = 1, py::arg("oversampling_summand") = 10,
    py::arg("oversampling_factor") = 1., py::arg("power_iteration") = 0, py::arg("seed") = -1,
    py::call_guard<py::gil_scoped_release>());

  m_linalg.def(
    "Gesvd_truncate",
    [](const Tensor &Tin, const cytnx_uint64 &keepdim, const cytnx_double &err, const bool &is_U,
       const bool &is_vT, const unsigned int &return_err, const cytnx_uint64 &mindim) {
      return cytnx::linalg::Gesvd_truncate(Tin, keepdim, err, is_U, is_vT, return_err, mindim);
    },
    py::arg("Tin"), py::arg("keepdim"), py::arg("err") = double(0), py::arg("is_U") = true,
    py::arg("is_vT") = true, py::arg("return_err") = (unsigned int)(0), py::arg("mindim") = 1,
    py::call_guard<py::gil_scoped_release>());
  m_linalg.def(
    "Gesvd_truncate",
    [](const UniTensor &Tin, const cytnx_uint64 &keepdim, const cytnx_double &err, const bool &is_U,
       const bool &is_vT, const unsigned int &return_err, const cytnx_uint64 &mindim) {
      return cytnx::linalg::Gesvd_truncate(Tin, keepdim, err, is_U, is_vT, return_err, mindim);
    },
    py::arg("Tin"), py::arg("keepdim"), py::arg("err") = 0, py::arg("is_U") = true,
    py::arg("is_vT") = true, py::arg("return_err") = (unsigned int)(0), py::arg("mindim") = 1,
    py::call_guard<py::gil_scoped_release>());
  m_linalg.def(
    "Gesvd_truncate",
    [](const UniTensor &Tin, const cytnx_uint64 &keepdim, std::vector<cytnx_uint64> min_blockdim,
       const double &err, const bool &is_U, const bool &is_vT, const unsigned int &return_err,
       const cytnx_uint64 &mindim) {
      return cytnx::linalg::Gesvd_truncate(Tin, keepdim, min_blockdim, err, is_U, is_vT, return_err,
                                           mindim);
    },
    py::arg("Tin"), py::arg("keepdim"), py::arg("min_blockdim"), py::arg("err") = 0,
    py::arg("is_U") = true, py::arg("is_vT") = true, py::arg("return_err") = (unsigned int)(0),
    py::arg("mindim") = 1, py::call_guard<py::gil_scoped_release>());

  m_linalg.def(
    "Svd_truncate",
    [](const Tensor &Tin, const cytnx_uint64 &keepdim, const double &err, const bool &is_UvT,
       const unsigned int &return_err, const cytnx_uint64 &mindim) {
      return cytnx::linalg::Svd_truncate(Tin, keepdim, err, is_UvT, return_err, mindim);
    },
    py::arg("Tin"), py::arg("keepdim"), py::arg("err") = double(0), py::arg("is_UvT") = true,
    py::arg("return_err") = (unsigned int)(0), py::arg("mindim") = 1,
    py::call_guard<py::gil_scoped_release>());
  m_linalg.def(
    "Svd_truncate",
    [](const UniTensor &Tin, const cytnx_uint64 &keepdim, const double &err, const bool &is_UvT,
       const unsigned int &return_err, const cytnx_uint64 &mindim) {
      return cytnx::linalg::Svd_truncate(Tin, keepdim, err, is_UvT, return_err, mindim);
    },
    py::arg("Tin"), py::arg("keepdim"), py::arg("err") = 0, py::arg("is_UvT") = true,
    py::arg("return_err") = (unsigned int)(0), py::arg("mindim") = 1,
    py::call_guard<py::gil_scoped_release>());
  m_linalg.def(
    "Svd_truncate",
    [](const UniTensor &Tin, const cytnx_uint64 &keepdim, std::vector<cytnx_uint64> min_blockdim,
       const double &err, const bool &is_UvT, const unsigned int &return_err,
       const cytnx_uint64 &mindim) {
      return cytnx::linalg::Svd_truncate(Tin, keepdim, min_blockdim, err, is_UvT, return_err,
                                         mindim);
    },
    py::arg("Tin"), py::arg("keepdim"), py::arg("min_blockdim"), py::arg("err") = 0,
    py::arg("is_UvT") = true, py::arg("return_err") = (unsigned int)(0), py::arg("mindim") = 1,
    py::call_guard<py::gil_scoped_release>());

  // m_linalg.def("Eigh", &cytnx::linalg::Eigh, py::arg("Tin"), py::arg("is_V") = true,
  //              py::arg("row_v") = false);
  //  m_linalg.def("Eig", &cytnx::linalg::Eig, py::arg("Tin"), py::arg("is_V") = true,
  //                py::arg("row_v") = false);
  m_linalg.def(
    "Eigh",
    [](const Tensor &Tin, const bool &is_V, const bool &row_v) {
      return cytnx::linalg::Eigh(Tin, is_V, row_v);
    },
    py::arg("Tin"), py::arg("is_V") = true, py::arg("row_v") = false,
    py::call_guard<py::gil_scoped_release>());
  m_linalg.def(
    "Eigh",
    [](const UniTensor &Tin, const bool &is_V, const bool &row_v) {
      return cytnx::linalg::Eigh(Tin, is_V, row_v);
    },
    py::arg("Tin"), py::arg("is_V") = true, py::arg("row_v") = false,
    py::call_guard<py::gil_scoped_release>());

  m_linalg.def(
    "Eig",
    [](const Tensor &Tin, const bool &is_V, const bool &row_v) {
      return cytnx::linalg::Eig(Tin, is_V, row_v);
    },
    py::arg("Tin"), py::arg("is_V") = true, py::arg("row_v") = false,
    py::call_guard<py::gil_scoped_release>());
  m_linalg.def(
    "Eig",
    [](const UniTensor &Tin, const bool &is_V, const bool &row_v) {
      return cytnx::linalg::Eig(Tin, is_V, row_v);
    },
    py::arg("Tin"), py::arg("is_V") = true, py::arg("row_v") = false,
    py::call_guard<py::gil_scoped_release>());

  m_linalg.def("Exp", &cytnx::linalg::Exp, py::arg("Tin"));
  m_linalg.def("Exp_", &cytnx::linalg::Exp_, py::arg("Tio"));
  // Expf/Expf_ are deprecated: Exp/Exp_ are now dtype-preserving (a Float input already yields a
  // Float result), so the float-precision variants are redundant. The -Wdeprecated-declarations
  // warning from calling the [[deprecated]] cytnx::linalg::Expf/Expf_ is suppressed around these
  // standalone binding statements (legal at statement scope, unlike inside a .def() chain).
  #if defined(__GNUC__) || defined(__clang__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  #endif
  m_linalg.def(
    "Expf_",
    [](cytnx::Tensor &Tio) {
      // PyErr_WarnEx returns -1 when the warning is turned into an exception (e.g. under
      // -W error::DeprecationWarning); propagate it instead of continuing.
      if (PyErr_WarnEx(PyExc_DeprecationWarning, "Expf_() is deprecated, use Exp_() instead.", 1) <
          0)
        throw py::error_already_set();
      cytnx::linalg::Expf_(Tio);
    },
    py::arg("Tio"));
  m_linalg.def(
    "Expf",
    [](cytnx::Tensor &Tio) {
      if (PyErr_WarnEx(PyExc_DeprecationWarning, "Expf() is deprecated, use Exp() instead.", 1) < 0)
        throw py::error_already_set();
      return cytnx::linalg::Expf(Tio);
    },
    py::arg("Tio"));
  #if defined(__GNUC__) || defined(__clang__)
    #pragma GCC diagnostic pop
  #endif

  // ExpH / ExpM, [Note] no bool type! One overload per scalar dtype; see
  // bind_exp_scalar for the dtype set and overload ordering.
  auto exph = [](const auto &Tin, auto a, auto b) { return cytnx::linalg::ExpH(Tin, a, b); };
  auto expm = [](const auto &Tin, auto a, auto b) { return cytnx::linalg::ExpM(Tin, a, b); };

  bind_exp_scalar<UniTensor>(m_linalg, "ExpH", exph);
  m_linalg.def(
    "ExpH", [](const UniTensor &Tin) { return cytnx::linalg::ExpH(Tin); }, py::arg("Tin"));
  bind_exp_scalar<Tensor>(m_linalg, "ExpH", exph);
  m_linalg.def(
    "ExpH", [](const Tensor &Tin) { return cytnx::linalg::ExpH(Tin); }, py::arg("Tin"));

  bind_exp_scalar<UniTensor>(m_linalg, "ExpM", expm);
  m_linalg.def(
    "ExpM", [](const UniTensor &Tin) { return cytnx::linalg::ExpM(Tin); }, py::arg("Tin"));
  bind_exp_scalar<Tensor>(m_linalg, "ExpM", expm);
  m_linalg.def(
    "ExpM", [](const Tensor &Tin) { return cytnx::linalg::ExpM(Tin); }, py::arg("Tin"));

  m_linalg.def(
    "Qr",
    [](const cytnx::UniTensor &Tin, const bool &is_tau) { return cytnx::linalg::Qr(Tin, is_tau); },
    py::arg("Tio"), py::arg("is_tau") = false, py::call_guard<py::gil_scoped_release>());
  m_linalg.def(
    "Qr",
    [](const cytnx::Tensor &Tin, const bool &is_tau) { return cytnx::linalg::Qr(Tin, is_tau); },
    py::arg("Tio"), py::arg("is_tau") = false, py::call_guard<py::gil_scoped_release>());
  m_linalg.def(
    "Qdr",
    [](const cytnx::UniTensor &Tin, const bool &is_tau) { return cytnx::linalg::Qdr(Tin, is_tau); },
    py::arg("Tio"), py::arg("is_tau") = false, py::call_guard<py::gil_scoped_release>());
  m_linalg.def(
    "Qdr",
    [](const cytnx::Tensor &Tin, const bool &is_tau) { return cytnx::linalg::Qdr(Tin, is_tau); },
    py::arg("Tio"), py::arg("is_tau") = false, py::call_guard<py::gil_scoped_release>());

  // m_linalg.def("InvM", &cytnx::linalg::InvM, py::arg("Tin"));

  // m_linalg.def("InvM_", &cytnx::linalg::InvM_, py::arg("Tio"));
  m_linalg.def(
    "InvM_", [](cytnx::UniTensor &Tio) { cytnx::linalg::InvM_(Tio); }, py::arg("Tio"),
    py::call_guard<py::gil_scoped_release>());
  m_linalg.def(
    "InvM_", [](cytnx::Tensor &Tio) { cytnx::linalg::InvM_(Tio); }, py::arg("Tio"),
    py::call_guard<py::gil_scoped_release>());

  m_linalg.def(
    "InvM", [](cytnx::UniTensor &Tin) { return cytnx::linalg::InvM(Tin); }, py::arg("Tin"),
    py::call_guard<py::gil_scoped_release>());
  m_linalg.def(
    "InvM", [](cytnx::Tensor &Tin) { return cytnx::linalg::InvM(Tin); }, py::arg("Tin"),
    py::call_guard<py::gil_scoped_release>());

  m_linalg.def(
    "Inv", [](const UniTensor &Tin, double clip) { return cytnx::linalg::Inv(Tin, clip); },
    py::arg("Tin"), py::arg("clip") = -1);
  m_linalg.def(
    "Inv", [](const Tensor &Tin, double clip) { return cytnx::linalg::Inv(Tin, clip); },
    py::arg("Tin"), py::arg("clip") = -1);

  m_linalg.def(
    "Inv_", [](UniTensor &Tin, double clip) { cytnx::linalg::Inv_(Tin, clip); }, py::arg("Tin"),
    py::arg("clip") = -1);
  m_linalg.def(
    "Inv_", [](Tensor &Tin, double clip) { cytnx::linalg::Inv_(Tin, clip); }, py::arg("Tin"),
    py::arg("clip") = -1);

  m_linalg.def(
    "Conj", [](const cytnx::Tensor &Tin) { return cytnx::linalg::Conj(Tin); }, py::arg("Tin"));

  m_linalg.def(
    "Conj_", [](cytnx::Tensor &Tin) { cytnx::linalg::Conj_(Tin); }, py::arg("Tin"));

  m_linalg.def(
    "Conj", [](const cytnx::UniTensor &Tin) { return cytnx::linalg::Conj(Tin); }, py::arg("Tin"));

  m_linalg.def(
    "Conj_", [](cytnx::UniTensor &Tin) { cytnx::linalg::Conj_(Tin); }, py::arg("Tin"));

  m_linalg.def("Matmul", &cytnx::linalg::Matmul, py::arg("T1"), py::arg("T2"),
               py::call_guard<py::gil_scoped_release>());
  m_linalg.def("Matmul_dg", &cytnx::linalg::Matmul_dg, py::arg("T1"), py::arg("T2"),
               py::call_guard<py::gil_scoped_release>());
  m_linalg.def("Diag", &cytnx::linalg::Diag, py::arg("Tin"));
  m_linalg.def("Det", &cytnx::linalg::Det, py::arg("Tin"),
               py::call_guard<py::gil_scoped_release>());
  m_linalg.def("Tensordot", &cytnx::linalg::Tensordot, py::arg("T1"), py::arg("T2"),
               py::arg("indices_1"), py::arg("indices_2"), py::arg("cacheL") = false,
               py::arg("cacheR") = false, py::call_guard<py::gil_scoped_release>());
  m_linalg.def("Tensordot_dg", &cytnx::linalg::Tensordot_dg, py::arg("T1"), py::arg("T2"),
               py::arg("indices_1"), py::arg("indices_2"), py::arg("diag_L"),
               py::call_guard<py::gil_scoped_release>());
  m_linalg.def("Outer", &cytnx::linalg::Outer, py::arg("T1"), py::arg("T2"));
  m_linalg.def("Kron", &cytnx::linalg::Kron, py::arg("T1"), py::arg("T2"),
               py::arg("Tl_pad_left") = false, py::arg("Tr_pad_left") = false,
               py::call_guard<py::gil_scoped_release>());
  m_linalg.def("Vectordot", &cytnx::linalg::Vectordot, py::arg("T1"), py::arg("T2"),
               py::arg("is_conj") = false);
  m_linalg.def("Tridiag", &cytnx::linalg::Tridiag, py::arg("A"), py::arg("B"),
               py::arg("is_V") = true, py::arg("is_row") = false, py::arg("throw_excp") = false);

  // m_linalg.def("Norm", &cytnx::linalg::Norm, py::arg("T1") = cytnx::Tensor());
  // m_linalg.def("Norm", &cytnx::linalg::Norm, py::arg("T1") = cytnx::UniTensor());
  m_linalg.def(
    "Norm", [](cytnx::UniTensor &T1) { return cytnx::linalg::Norm(T1); }, py::arg("T1"));
  m_linalg.def(
    "Norm", [](cytnx::Tensor &T1) { return cytnx::linalg::Norm(T1); }, py::arg("T1"));

  m_linalg.def("Dot", &cytnx::linalg::Dot, py::arg("T1"), py::arg("T2"),
               py::call_guard<py::gil_scoped_release>());

  m_linalg.def(
    "Ger",
    [](const Tensor &x, const Tensor &y, const Scalar &a) { return cytnx::linalg::Ger(x, y, a); },
    py::arg("x"), py::arg("y"), py::arg("a") = Scalar());

  m_linalg.def("Gemm_", &cytnx::linalg::Gemm_, py::arg("a"), py::arg("x"), py::arg("y"),
               py::arg("b"), py::arg("c"), py::call_guard<py::gil_scoped_release>());

  m_linalg.def(
    "Gemm",
    [](const Scalar &a, const Tensor &x, const Tensor &y) { return cytnx::linalg::Gemm(a, x, y); },
    py::arg("a"), py::arg("x"), py::arg("y"), py::call_guard<py::gil_scoped_release>());

  m_linalg.def(
    "Trace",
    [](const cytnx::UniTensor &Tin, const cytnx_int64 &a, const cytnx_int64 &b) {
      return cytnx::linalg::Trace(Tin, a, b);
    },
    py::arg("Tn"), py::arg("axisA") = 0, py::arg("axisB") = 1);

  m_linalg.def(
    "Trace",
    [](const cytnx::UniTensor &Tin, const std::string &a, const std::string &b) {
      return cytnx::linalg::Trace(Tin, a, b);
    },
    py::arg("Tn"), py::arg("axisA"), py::arg("axisB"));

  m_linalg.def(
    "Trace",
    [](const cytnx::Tensor &Tin, const cytnx_int64 &a, const cytnx_int64 &b) {
      return cytnx::linalg::Trace(Tin, a, b);
    },
    py::arg("Tn"), py::arg("axisA") = 0, py::arg("axisB") = 1);

  m_linalg.def(
    "Pow", [](const UniTensor &Tin, const double &p) { return cytnx::linalg::Pow(Tin, p); },
    py::arg("Tin"), py::arg("p"));
  m_linalg.def(
    "Pow", [](const Tensor &Tin, const double &p) { return cytnx::linalg::Pow(Tin, p); },
    py::arg("Tin"), py::arg("p"));

  m_linalg.def(
    "Pow_", [](UniTensor &Tin, const double &p) { cytnx::linalg::Pow_(Tin, p); }, py::arg("Tin"),
    py::arg("p"));
  m_linalg.def(
    "Pow_", [](Tensor &Tin, const double &p) { cytnx::linalg::Pow_(Tin, p); }, py::arg("Tin"),
    py::arg("p"));

  m_linalg.def("Abs", &cytnx::linalg::Abs, py::arg("Tn"));
  m_linalg.def("Abs_", &cytnx::linalg::Abs_, py::arg("Tn"));
  m_linalg.def("Max", &cytnx::linalg::Max, py::arg("Tn"));
  m_linalg.def("Min", &cytnx::linalg::Min, py::arg("Tn"));
  m_linalg.def("Sum", &cytnx::linalg::Sum, py::arg("Tn"));

  m_linalg.def(
    "Directsum",
    [](const Tensor &T1, const Tensor &T2, const std::vector<cytnx_uint64> &shared_axes) {
      return linalg::Directsum(T1, T2, shared_axes);
    },
    py::arg("T1"), py::arg("T2"), py::arg("shared_axes"), py::call_guard<py::gil_scoped_release>());

  m_linalg.def(
    "Hosvd",
    [](const cytnx::Tensor &Tin, const std::vector<cytnx_uint64> &mode, const bool &is_core,
       const bool &is_Ls, const std::vector<cytnx_int64> &truncate_dim) {
      return cytnx::linalg::Hosvd(Tin, mode, is_core, is_Ls, truncate_dim);
    },
    py::arg("Tn"), py::arg("mode"), py::arg("is_core") = true, py::arg("is_Ls") = false,
    py::arg("truncate_dim") = std::vector<cytnx_int64>(), py::call_guard<py::gil_scoped_release>());
  m_linalg.def(
    "Hosvd",
    [](const cytnx::UniTensor &Tin, const std::vector<cytnx_uint64> &mode, const bool &is_core,
       const bool &is_Ls, const std::vector<cytnx_int64> &truncate_dim) {
      return cytnx::linalg::Hosvd(Tin, mode, is_core, is_Ls, truncate_dim);
    },
    py::arg("Tn"), py::arg("mode"), py::arg("is_core") = true, py::arg("is_Ls") = false,
    py::arg("truncate_dim") = std::vector<cytnx_int64>(), py::call_guard<py::gil_scoped_release>());

  m_linalg.def(
    "Arnoldi",
    [](LinOp *Hop, const Tensor &Tin, const std::string which, const cytnx_uint64 &Maxiter,
       const double &CvgCrit, const cytnx_uint64 &k, const bool &is_V, const cytnx_int32 &ncv,
       const bool &verbose) {
      return cytnx::linalg::Arnoldi(Hop, Tin, which, Maxiter, CvgCrit, k, is_V, ncv, verbose);
    },
    py::arg("Hop"), py::arg("Tin"), py::arg("which") = "LM", py::arg("Maxiter") = 10000,
    py::arg("CvgCrit") = 0, py::arg("k") = 1, py::arg("is_V") = true, py::arg("ncv") = 0,
    py::arg("verbose") = false, py::call_guard<py::gil_scoped_release>());
  m_linalg.def(
    "Arnoldi",
    [](LinOp *Hop, const UniTensor &Tin, const std::string which, const cytnx_uint64 &Maxiter,
       const double &CvgCrit, const cytnx_uint64 &k, const bool &is_V, const cytnx_int32 &ncv,
       const bool &verbose) {
      return cytnx::linalg::Arnoldi(Hop, Tin, which, Maxiter, CvgCrit, k, is_V, ncv, verbose);
    },
    py::arg("Hop"), py::arg("Tin"), py::arg("which") = "LM", py::arg("Maxiter") = 10000,
    py::arg("CvgCrit") = 0, py::arg("k") = 1, py::arg("is_V") = true, py::arg("ncv") = 0,
    py::arg("verbose") = false, py::call_guard<py::gil_scoped_release>());

  m_linalg.def(
    "Lanczos",
    [](LinOp *Hop, const Tensor &Tin, const std::string method, const double &CvgCrit,
       const unsigned int &Maxiter, const cytnx_uint64 &k, const bool &is_V, const bool &is_row,
       const cytnx_uint32 &max_krydim, const bool &verbose) {
      return cytnx::linalg::Lanczos(Hop, Tin, method, CvgCrit, Maxiter, k, is_V, is_row, max_krydim,
                                    verbose);
    },
    py::arg("Hop"), py::arg("Tin"), py::arg("method"), py::arg("CvgCrit") = 1.0e-14,
    py::arg("Maxiter") = 10000, py::arg("k") = 1, py::arg("is_V") = true, py::arg("is_row") = false,
    py::arg("max_krydim") = 0, py::arg("verbose") = false,
    py::call_guard<py::gil_scoped_release>());
  m_linalg.def(
    "Lanczos",
    [](LinOp *Hop, const UniTensor &Tin, const std::string method, const double &CvgCrit,
       const unsigned int &Maxiter, const cytnx_uint64 &k, const bool &is_V, const bool &is_row,
       const cytnx_uint32 &max_krydim, const bool &verbose) {
      return cytnx::linalg::Lanczos(Hop, Tin, method, CvgCrit, Maxiter, k, is_V, is_row, max_krydim,
                                    verbose);
    },
    py::arg("Hop"), py::arg("Tin"), py::arg("method"), py::arg("CvgCrit") = 1.0e-14,
    py::arg("Maxiter") = 10000, py::arg("k") = 1, py::arg("is_V") = true, py::arg("is_row") = false,
    py::arg("max_krydim") = 0, py::arg("verbose") = false,
    py::call_guard<py::gil_scoped_release>());

  m_linalg.def(
    "Lanczos",
    [](LinOp *Hop, const Tensor &Tin, const std::string which, const cytnx_uint64 &Maxiter,
       const double &CvgCrit, const cytnx_uint64 &k, const bool &is_V, const cytnx_int32 &ncv,
       const bool &verbose) {
      return cytnx::linalg::Lanczos(Hop, Tin, which, Maxiter, CvgCrit, k, is_V, ncv, verbose);
    },
    py::arg("Hop"), py::arg("Tin"), py::arg("which") = "SA", py::arg("Maxiter") = 10000,
    py::arg("CvgCrit") = 0, py::arg("k") = 1, py::arg("is_V") = true, py::arg("ncv") = 0,
    py::arg("verbose") = false, py::call_guard<py::gil_scoped_release>());
  m_linalg.def(
    "Lanczos",
    [](LinOp *Hop, const UniTensor &Tin, const std::string which, const cytnx_uint64 &Maxiter,
       const double &CvgCrit, const cytnx_uint64 &k, const bool &is_V, const cytnx_int32 &ncv,
       const bool &verbose) {
      return cytnx::linalg::Lanczos(Hop, Tin, which, Maxiter, CvgCrit, k, is_V, ncv, verbose);
    },
    py::arg("Hop"), py::arg("Tin"), py::arg("which") = "SA", py::arg("Maxiter") = 10000,
    py::arg("CvgCrit") = 0, py::arg("k") = 1, py::arg("is_V") = true, py::arg("ncv") = 0,
    py::arg("verbose") = false, py::call_guard<py::gil_scoped_release>());

  m_linalg.def(
    "Lanczos_Exp",
    [](LinOp *Hop, const UniTensor &v, const Scalar &tau, const double &CvgCrit,
       const unsigned int &Maxiter, const bool &verbose) {
      return cytnx::linalg::Lanczos_Exp(Hop, v, tau, CvgCrit, Maxiter, verbose);
    },
    py::arg("Hop"), py::arg("v"), py::arg("tau"), py::arg("CvgCrit") = 1.0e-14,
    py::arg("Maxiter") = 10000, py::arg("verbose") = false,
    py::call_guard<py::gil_scoped_release>());

  m_linalg.def(
    "Lstsq",
    [](const Tensor &A, const Tensor &b, const float &rcond) {
      return cytnx::linalg::Lstsq(A, b, rcond);
    },
    py::arg("A"), py::arg("b"), py::arg("rcond") = float(-1),
    py::call_guard<py::gil_scoped_release>());

  /*
  m_linalg.def("c_Lanczos_ER",
               [](LinOp *Hop, const cytnx_uint64 &k, const bool &is_V, const cytnx_uint64 &maxiter,
                  const double &CvgCrit, const bool &is_row, const Tensor &Tin,
                  const cytnx_uint32 &max_krydim, const bool &verbose) {
                 return cytnx::linalg::Lanczos_ER(Hop, k, is_V, maxiter, CvgCrit, is_row, Tin,
                                                  max_krydim, verbose);
               });

  m_linalg.def("c_Lanczos_Gnd",
               [](LinOp *Hop, const double &CvgCrit, const bool &is_V, const Tensor &Tin,
                  const bool &verbose, const cytnx_uint64 &maxiter) {
                 return cytnx::linalg::Lanczos_Gnd(Hop, CvgCrit, is_V, Tin, verbose, maxiter);
               });

  m_linalg.def("c_Lanczos_Gnd_Ut",
               [](LinOp *Hop, const UniTensor &Tin, const double &CvgCrit, const bool &is_V,
                  const bool &verbose, const cytnx_uint64 &maxiter) {
                 return cytnx::linalg::Lanczos_Gnd_Ut(Hop, Tin, CvgCrit, is_V, verbose, maxiter);
               });
  */
}

#endif
