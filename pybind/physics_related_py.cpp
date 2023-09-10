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

void physics_related_binding(py::module &m) {
  // [Submodule physics]
  pybind11::module m_physics = m.def_submodule("physics", "physics related.");
  m_physics.def(
    "spin",
    [](const cytnx_double &S, const std::string &Comp, const int &device) -> Tensor {
      return cytnx::physics::spin(S, Comp, device);
    },
    py::arg("S"), py::arg("Comp"), py::arg("device") = (int)cytnx::Device.cpu);
  m_physics.def(
    "pauli",
    [](const std::string &Comp, const int &device) -> Tensor {
      return cytnx::physics::pauli(Comp, device);
    },
    py::arg("Comp"), py::arg("device") = (int)cytnx::Device.cpu);

  // [Submodule qgates]
  pybind11::module m_qgates = m.def_submodule("qgates", "quantum gates.");
  m_qgates.def(
    "pauli_z", [](const int &device) -> UniTensor { return cytnx::qgates::pauli_z(device); },
    py::arg("device") = (int)cytnx::Device.cpu);
  m_qgates.def(
    "pauli_y", [](const int &device) -> UniTensor { return cytnx::qgates::pauli_y(device); },
    py::arg("device") = (int)cytnx::Device.cpu);
  m_qgates.def(
    "pauli_x", [](const int &device) -> UniTensor { return cytnx::qgates::pauli_x(device); },
    py::arg("device") = (int)cytnx::Device.cpu);
  m_qgates.def(
    "hadamard", [](const int &device) -> UniTensor { return cytnx::qgates::hadamard(device); },
    py::arg("device") = (int)cytnx::Device.cpu);

  m_qgates.def(
    "swap", [](const int &device) -> UniTensor { return cytnx::qgates::swap(device); },
    py::arg("device") = (int)cytnx::Device.cpu);

  m_qgates.def(
    "sqrt_swap", [](const int &device) -> UniTensor { return cytnx::qgates::sqrt_swap(device); },
    py::arg("device") = (int)cytnx::Device.cpu);

  m_qgates.def(
    "phase_shift",
    [](const cytnx_double &phase, const int &device) -> UniTensor {
      return cytnx::qgates::phase_shift(phase, device);
    },
    py::arg("phase"), py::arg("device") = (int)cytnx::Device.cpu);

  m_qgates.def(
    "toffoli", [](const int &device) -> UniTensor { return cytnx::qgates::toffoli(device); },
    py::arg("device") = (int)cytnx::Device.cpu);

  m_qgates.def(
    "cntl_gate_2q",
    [](const UniTensor &gate_1q) -> UniTensor { return cytnx::qgates::cntl_gate_2q(gate_1q); },
    py::arg("gate_1q"));
}
#endif
