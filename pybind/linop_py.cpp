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
// #include "../include/cytnx_error.hpp"
#include "complex.h"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace cytnx;

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
    .def(py::init<const cytnx_uint64 &, const int &, const int &>(), py::arg("nx"),
         py::arg("dtype") = (int)Type.Double, py::arg("device") = (int)Device.cpu)
    // Python-only backwards-compatibility shim accepting the legacy leading `type` argument. This
    // stays in the bindings indefinitely. It constructs via the plain LinOp(nx, ...) ctor (rather
    // than the deprecated C++ string ctor) so the binding compiles warning-free; `type` must be
    // "mv" (anything else, including the removed "mv_elem", is a hard error).
    .def(py::init([](const std::string &type, const cytnx_uint64 &nx, const int &dtype,
                     const int &device) {
           cytnx_error_msg(
             type != "mv",
             "[ERROR][LinOp] the only supported type is \"mv\"; the \"mv_elem\" path "
             "has been removed. Construct with LinOp(nx, ...) and override matvec().%s",
             "\n");
           return PyLinOp(nx, dtype, device);
         }),
         py::arg("type"), py::arg("nx"), py::arg("dtype") = (int)Type.Double,
         py::arg("device") = (int)Device.cpu)
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

    ;  // end of object
}

#endif
