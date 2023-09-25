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
    .def(py::init<const std::string &, const cytnx_uint64 &, const int &, const int &>(),
         py::arg("type"), py::arg("nx"), py::arg("dtype") = (int)Type.Double,
         py::arg("device") = (int)Device.cpu)
    .def("set_elem", &LinOp::set_elem<cytnx_complex128>, py::arg("i"), py::arg("j"),
         py::arg("elem"), py::arg("check_exists") = true)
    .def("set_elem", &LinOp::set_elem<cytnx_complex64>, py::arg("i"), py::arg("j"), py::arg("elem"),
         py::arg("check_exists") = true)
    .def("set_elem", &LinOp::set_elem<cytnx_double>, py::arg("i"), py::arg("j"), py::arg("elem"),
         py::arg("check_exists") = true)
    .def("set_elem", &LinOp::set_elem<cytnx_float>, py::arg("i"), py::arg("j"), py::arg("elem"),
         py::arg("check_exists") = true)
    .def("set_elem", &LinOp::set_elem<cytnx_int64>, py::arg("i"), py::arg("j"), py::arg("elem"),
         py::arg("check_exists") = true)
    .def("set_elem", &LinOp::set_elem<cytnx_uint64>, py::arg("i"), py::arg("j"), py::arg("elem"),
         py::arg("check_exists") = true)
    .def("set_elem", &LinOp::set_elem<cytnx_int32>, py::arg("i"), py::arg("j"), py::arg("elem"),
         py::arg("check_exists") = true)
    .def("set_elem", &LinOp::set_elem<cytnx_uint32>, py::arg("i"), py::arg("j"), py::arg("elem"),
         py::arg("check_exists") = true)
    .def("set_elem", &LinOp::set_elem<cytnx_int16>, py::arg("i"), py::arg("j"), py::arg("elem"),
         py::arg("check_exists") = true)
    .def("set_elem", &LinOp::set_elem<cytnx_uint16>, py::arg("i"), py::arg("j"), py::arg("elem"),
         py::arg("check_exists") = true)
    .def("set_elem", &LinOp::set_elem<cytnx_bool>, py::arg("i"), py::arg("j"), py::arg("elem"),
         py::arg("check_exists") = true)
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
