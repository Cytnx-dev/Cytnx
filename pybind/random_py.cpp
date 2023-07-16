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

void random_binding(py::module &m) {
  // [Submodule random]
  pybind11::module m_random = m.def_submodule("random", "random related.");

  m_random.def(
    "Make_normal",
    [](cytnx::Tensor &Tin, const double &mean, const double &std, const long long &seed) {
      cytnx::random::Make_normal(Tin, mean, std, seed);
    },
    py::arg("Tin"), py::arg("mean"), py::arg("std"), py::arg("seed") = std::random_device()());

  m_random.def(
    "Make_normal",
    [](cytnx::Storage &Sin, const double &mean, const double &std, const long long &seed) {
      cytnx::random::Make_normal(Sin, mean, std, seed);
    },
    py::arg("Sin"), py::arg("mean"), py::arg("std"), py::arg("seed") = std::random_device()());

  m_random.def(
    "Make_normal",
    [](cytnx::UniTensor &Tin, const double &mean, const double &std, const long long &seed) {
      cytnx::random::Make_normal(Tin, mean, std, seed);
    },
    py::arg("Tin"), py::arg("mean"), py::arg("std"), py::arg("seed") = std::random_device()());

  m_random.def(
    "Make_uniform",
    [](cytnx::Tensor &Tin, const double &low, const double &high, const long long &seed) {
      cytnx::random::Make_uniform(Tin, low, high, seed);
    },
    py::arg("Tin"), py::arg("low") = double(0), py::arg("high") = double(1.0),
    py::arg("seed") = std::random_device()());

  m_random.def(
    "Make_uniform",
    [](cytnx::Storage &Sin, const double &low, const double &high, const long long &seed) {
      cytnx::random::Make_uniform(Sin, low, high, seed);
    },
    py::arg("Sin"), py::arg("low") = double(0), py::arg("high") = double(1.0),
    py::arg("seed") = std::random_device()());

  m_random.def(
    "Make_uniform",
    [](cytnx::UniTensor &Tin, const double &low, const double &high, const long long &seed) {
      cytnx::random::Make_uniform(Tin, low, high, seed);
    },
    py::arg("Tin"), py::arg("low") = double(0), py::arg("high") = double(1.0),
    py::arg("seed") = std::random_device()());

  m_random.def(
    "normal",
    [](const cytnx_uint64 &Nelem, const double &mean, const double &std, const int &device,
       const unsigned int &seed, const unsigned int &dtype) {
      return cytnx::random::normal(Nelem, mean, std, device, seed, dtype);
    },
    py::arg("Nelem"), py::arg("mean"), py::arg("std"), py::arg("device") = -1,
    py::arg("seed") = std::random_device()(), py::arg("dtype") = (unsigned int)(Type.Double));
  m_random.def(
    "normal",
    [](const std::vector<cytnx_uint64> &Nelem, const double &mean, const double &std,
       const int &device, const unsigned int &seed, const unsigned int &dtype) {
      return cytnx::random::normal(Nelem, mean, std, device, seed, dtype);
    },
    py::arg("Nelem"), py::arg("mean"), py::arg("std"), py::arg("device") = -1,
    py::arg("seed") = std::random_device()(), py::arg("dtype") = (unsigned int)(Type.Double));
  m_random.def(
    "uniform",
    [](const cytnx_uint64 &Nelem, const double &low, const double &high, const int &device,
       const unsigned int &seed, const unsigned int &dtype) {
      return cytnx::random::uniform(Nelem, low, high, device, seed, dtype);
    },
    py::arg("Nelem"), py::arg("low"), py::arg("high"), py::arg("device") = -1,
    py::arg("seed") = std::random_device()(), py::arg("dtype") = (unsigned int)(Type.Double));
  m_random.def(
    "uniform",
    [](const std::vector<cytnx_uint64> &Nelem, const double &low, const double &high,
       const int &device, const unsigned int &seed, const unsigned int &dtype) {
      return cytnx::random::uniform(Nelem, low, high, device, seed, dtype);
    },
    py::arg("Nelem"), py::arg("low"), py::arg("high"), py::arg("device") = -1,
    py::arg("seed") = std::random_device()(), py::arg("dtype") = (unsigned int)(Type.Double));
}
