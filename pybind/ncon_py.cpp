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

void ncon_binding(py::module &m) {
  m.def(
    "ncon",
    [](const std::vector<UniTensor> &tensor_list_in,
       const std::vector<std::vector<cytnx_int64>> &connect_list_in, const bool check_network,
       const bool optimize, std::vector<cytnx_int64> cont_order,
       const std::vector<std::string> &out_labels) -> UniTensor {
      return ncon(tensor_list_in, connect_list_in, check_network, optimize, cont_order, out_labels);
    },
    py::arg("tensor_list_in"), py::arg("connect_list_in"), py::arg("check_network") = false,
    py::arg("optimize") = false, py::arg("cont_order") = std::vector<cytnx_int64>(),
    py::arg("out_labels") = std::vector<std::string>());
}

#endif
