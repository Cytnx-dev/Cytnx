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

void algo_binding(py::module &m) {
  // [Submodule algo]
  pybind11::module m_algo = m.def_submodule("algo", "algorithm related.");
  m_algo.def("Sort", &cytnx::algo::Sort, py::arg("Tn"));
  m_algo.def("Concatenate", &cytnx::algo::Concatenate, py::arg("T1"), py::arg("T2"));
  m_algo.def("Vstack", &cytnx::algo::Vstack, py::arg("Tlist"));
  m_algo.def("Hstack", &cytnx::algo::Hstack, py::arg("Tlist"));
  m_algo.def("Vsplit", &cytnx::algo::Vsplit, py::arg("Tin"), py::arg("dims"));
  m_algo.def("Hsplit", &cytnx::algo::Hsplit, py::arg("Tin"), py::arg("dims"));
}
#endif  // BACKEND_TORCH
