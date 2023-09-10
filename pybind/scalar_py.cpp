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

void scalar_binding(py::module &m) {
  py::class_<cytnx::Scalar>(m, "Scalar")
    .def(py::init<>())
    .def(py::init<const cytnx_complex128 &>(), py::arg("a"))
    .def(py::init<const cytnx_complex64 &>(), py::arg("a"))
    .def(py::init<const cytnx_double &>(), py::arg("a"))
    .def(py::init<const cytnx_float &>(), py::arg("a"))
    .def(py::init<const cytnx_uint64 &>(), py::arg("a"))
    .def(py::init<const cytnx_int64 &>(), py::arg("a"))
    .def(py::init<const cytnx_uint32 &>(), py::arg("a"))
    .def(py::init<const cytnx_int32 &>(), py::arg("a"))
    .def(py::init<const cytnx_uint16 &>(), py::arg("a"))
    .def(py::init<const cytnx_int16 &>(), py::arg("a"))
    .def(py::init<const cytnx_bool &>(), py::arg("a"))

    ;  // end of object line

  py::implicitly_convertible<cytnx_double, cytnx::Scalar>();
  py::implicitly_convertible<cytnx_float, cytnx::Scalar>();
  py::implicitly_convertible<cytnx_complex128, cytnx::Scalar>();
  py::implicitly_convertible<cytnx_complex64, cytnx::Scalar>();
  py::implicitly_convertible<cytnx_uint64, cytnx::Scalar>();
  py::implicitly_convertible<cytnx_int64, cytnx::Scalar>();
  py::implicitly_convertible<cytnx_uint32, cytnx::Scalar>();
  py::implicitly_convertible<cytnx_int32, cytnx::Scalar>();
  py::implicitly_convertible<cytnx_uint16, cytnx::Scalar>();
  py::implicitly_convertible<cytnx_int16, cytnx::Scalar>();
  py::implicitly_convertible<cytnx_bool, cytnx::Scalar>();
}

#endif
