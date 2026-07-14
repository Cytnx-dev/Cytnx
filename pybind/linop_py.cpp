#include <vector>
#include <map>
#include <random>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/buffer_info.h>
#include <pybind11/functional.h>

#include "cytnx.hpp"
#include "pyint_dispatch.hpp"
// #include "../include/cytnx_error.hpp"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace cytnx;
using pybind_cytnx::dispatch_pyint;

#ifdef BACKEND_TORCH
#else

class PyLinOp : public LinOp {
 public:
  /* inherit constructor */
  using LinOp::LinOp;

  Tensor matvec(const Tensor &Tin) override {
    PYBIND11_OVERLOAD(Tensor, /* Return type */
                      LinOp, /* Parent class */
                      matvec, /* Name of function in C++ (must match Python name) */
                      Tin /* Argument(s) */
    );
  }
  UniTensor matvec(const UniTensor &Tin) override {
    PYBIND11_OVERLOAD(UniTensor, /* Return type */
                      LinOp, /* Parent class */
                      matvec, /* Name of function in C++ (must match Python name) */
                      Tin /* Argument(s) */
    );
  }
};

void linop_binding(py::module &m) {
  py::class_<LinOp, PyLinOp>(m, "LinOp")
    .def(py::init<const std::string &, const cytnx_uint64 &, const int &, const int &>(),
         py::arg("type"), py::arg("nx"), py::arg("dtype") = (int)Type.Double,
         py::arg("device") = (int)Device.cpu)
    // keep-set; registration ORDER matters -- see "KEEP-SET ORDERING" in pybind/pyint_dispatch.hpp.
    // complex64/float/{u,}int{16,32} are dropped: they are already covered
    // by the numpy_scalar overloads immediately below (a plain Python value
    // never reaches them; only a numpy scalar of that exact width would,
    // and that width now has its own numpy_scalar overload). uint64 is
    // dropped in favor of a single py::int_ overload with dispatch_pyint,
    // which reproduces the previous int64-then-uint64 registration order's
    // resolution (int64 when the value fits, uint64 otherwise) without a
    // stub-visible duplicate. The numpy_scalar overloads are registered
    // FIRST, before py::int_/double/complex128: a numpy scalar satisfies
    // __index__/__float__/__complex__, so a plain integral/double/complex128
    // overload registered earlier would greedily consume it in pybind11's
    // conversion pass (see the "__index__ no-convert trap" and "__float__
    // -fallback trap" in pyint_dispatch.hpp) -- this was the case here
    // before this change, e.g. LinOp.set_elem(i, j, np.float32(1.5), True)
    // matched the raw complex128 overload via the float-fallback trap
    // instead of the numpy_scalar<float> overload below.
    .def(
      "set_elem",
      [](LinOp &self, const cytnx::cytnx_uint64 i, const cytnx::cytnx_uint64 j,
         const py::numpy_scalar<std::complex<double>> elem, const bool check_exists) {
        self.set_elem(i, j, static_cast<cytnx::cytnx_complex128>(elem), check_exists);
      },
      py::arg("i"), py::arg("j"), py::arg("elem"), py::arg("check_exists") = true)
    .def(
      "set_elem",
      [](LinOp &self, const cytnx::cytnx_uint64 i, const cytnx::cytnx_uint64 j,
         const py::numpy_scalar<std::complex<float>> elem, const bool check_exists) {
        self.set_elem(i, j, static_cast<cytnx::cytnx_complex64>(elem), check_exists);
      },
      py::arg("i"), py::arg("j"), py::arg("elem"), py::arg("check_exists") = true)
    .def(
      "set_elem",
      [](LinOp &self, const cytnx::cytnx_uint64 i, const cytnx::cytnx_uint64 j,
         const py::numpy_scalar<double> elem, const bool check_exists) {
        self.set_elem(i, j, static_cast<cytnx::cytnx_double>(elem), check_exists);
      },
      py::arg("i"), py::arg("j"), py::arg("elem"), py::arg("check_exists") = true)
    .def(
      "set_elem",
      [](LinOp &self, const cytnx::cytnx_uint64 i, const cytnx::cytnx_uint64 j,
         const py::numpy_scalar<float> elem, const bool check_exists) {
        self.set_elem(i, j, static_cast<cytnx::cytnx_float>(elem), check_exists);
      },
      py::arg("i"), py::arg("j"), py::arg("elem"), py::arg("check_exists") = true)
    .def(
      "set_elem",
      [](LinOp &self, const cytnx::cytnx_uint64 i, const cytnx::cytnx_uint64 j,
         const py::numpy_scalar<int64_t> elem, const bool check_exists) {
        self.set_elem(i, j, static_cast<cytnx::cytnx_int64>(elem), check_exists);
      },
      py::arg("i"), py::arg("j"), py::arg("elem"), py::arg("check_exists") = true)
    .def(
      "set_elem",
      [](LinOp &self, const cytnx::cytnx_uint64 i, const cytnx::cytnx_uint64 j,
         const py::numpy_scalar<uint64_t> elem, const bool check_exists) {
        self.set_elem(i, j, static_cast<cytnx::cytnx_uint64>(elem), check_exists);
      },
      py::arg("i"), py::arg("j"), py::arg("elem"), py::arg("check_exists") = true)
    .def(
      "set_elem",
      [](LinOp &self, const cytnx::cytnx_uint64 i, const cytnx::cytnx_uint64 j,
         const py::numpy_scalar<int32_t> elem, const bool check_exists) {
        self.set_elem(i, j, static_cast<cytnx::cytnx_int32>(elem), check_exists);
      },
      py::arg("i"), py::arg("j"), py::arg("elem"), py::arg("check_exists") = true)
    .def(
      "set_elem",
      [](LinOp &self, const cytnx::cytnx_uint64 i, const cytnx::cytnx_uint64 j,
         const py::numpy_scalar<uint32_t> elem, const bool check_exists) {
        self.set_elem(i, j, static_cast<cytnx::cytnx_uint32>(elem), check_exists);
      },
      py::arg("i"), py::arg("j"), py::arg("elem"), py::arg("check_exists") = true)
    .def(
      "set_elem",
      [](LinOp &self, const cytnx::cytnx_uint64 i, const cytnx::cytnx_uint64 j,
         const py::numpy_scalar<int16_t> elem, const bool check_exists) {
        self.set_elem(i, j, static_cast<cytnx::cytnx_int16>(elem), check_exists);
      },
      py::arg("i"), py::arg("j"), py::arg("elem"), py::arg("check_exists") = true)
    .def(
      "set_elem",
      [](LinOp &self, const cytnx::cytnx_uint64 i, const cytnx::cytnx_uint64 j,
         const py::numpy_scalar<uint16_t> elem, const bool check_exists) {
        self.set_elem(i, j, static_cast<cytnx::cytnx_uint16>(elem), check_exists);
      },
      py::arg("i"), py::arg("j"), py::arg("elem"), py::arg("check_exists") = true)
    .def(
      "set_elem",
      [](LinOp &self, const cytnx::cytnx_uint64 i, const cytnx::cytnx_uint64 j,
         const py::numpy_scalar<bool> elem, const bool check_exists) {
        self.set_elem(i, j, static_cast<cytnx::cytnx_bool>(elem), check_exists);
      },
      py::arg("i"), py::arg("j"), py::arg("elem"), py::arg("check_exists") = true)
    .def(
      "set_elem",
      [](LinOp &self, const cytnx_uint64 i, const cytnx_uint64 j, const py::int_ &elem,
         const bool check_exists) {
        dispatch_pyint(elem, [&](auto v) { self.set_elem(i, j, v, check_exists); });
      },
      py::arg("i"), py::arg("j"), py::arg("elem"), py::arg("check_exists") = true)
    .def("set_elem", &LinOp::set_elem<cytnx_double>, py::arg("i"), py::arg("j"), py::arg("elem"),
         py::arg("check_exists") = true)
    .def("set_elem", &LinOp::set_elem<cytnx_complex128>, py::arg("i"), py::arg("j"),
         py::arg("elem"), py::arg("check_exists") = true)

    //.def("__call__",[](cytnx::LinOp &self, const cytnx_uint64 &i, const cytnx_uint64 &j){
    //        return Tensor(self(i,j));
    //})
    .def(
      "matvec", [](LinOp &self, const Tensor &Tin) -> Tensor { return self.matvec(Tin); },
      py::arg("Tin"))
    .def(
      "matvec", [](LinOp &self, const UniTensor &Tin) -> UniTensor { return self.matvec(Tin); },
      py::arg("Tin"))
    .def("set_device", &LinOp::set_device)
    .def("set_dtype", &LinOp::set_dtype)
    .def("device", &LinOp::device)
    .def("dtype", &LinOp::dtype)
    .def("nx", &LinOp::nx)
    .def(
      "__repr__",
      [](cytnx::LinOp &self) -> std::string {
        self._print();
        return std::string("");
      },
      py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())

    ;  // end of object
}

#endif
