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

void random_binding(py::module &m) {
  // [Submodule random]
  pybind11::module m_random = m.def_submodule("random", "random related.");

  m_random.def(
    "normal_",
    [](cytnx::Tensor &Tin, const double &mean, const double &std, int64_t &seed) {
      if (seed == -1) {
        // If user doesn't specify seed argument
        seed = cytnx::random::__static_random_device();
      }
      cytnx::random::normal_(Tin, mean, std, seed);
    },
    py::arg("Tin"), py::arg("mean"), py::arg("std"), py::arg("seed") = -1);

  m_random.def(
    "normal_",
    [](cytnx::Storage &Sin, const double &mean, const double &std, int64_t &seed) {
      if (seed == -1) {
        // If user doesn't specify seed argument
        seed = cytnx::random::__static_random_device();
      }
      cytnx::random::normal_(Sin, mean, std, seed);
    },
    py::arg("Sin"), py::arg("mean"), py::arg("std"), py::arg("seed") = -1);

  m_random.def(
    "normal_",
    [](cytnx::UniTensor &Tin, const double &mean, const double &std, int64_t &seed) {
      if (seed == -1) {
        // If user doesn't specify seed argument
        seed = cytnx::random::__static_random_device();
      }
      cytnx::random::normal_(Tin, mean, std, seed);
    },
    py::arg("Tin"), py::arg("mean"), py::arg("std"), py::arg("seed") = -1);

  m_random.def(
    "uniform_",
    [](cytnx::Tensor &Tin, const double &low, const double &high, int64_t &seed) {
      if (seed == -1) {
        // If user doesn't specify seed argument
        seed = cytnx::random::__static_random_device();
      }
      cytnx::random::uniform_(Tin, low, high, seed);
    },
    py::arg("Tin"), py::arg("low") = double(0), py::arg("high") = double(1.0),
    py::arg("seed") = -1);

  m_random.def(
    "uniform_",
    [](cytnx::Storage &Sin, const double &low, const double &high, int64_t &seed) {
      if (seed == -1) {
        // If user doesn't specify seed argument
        seed = cytnx::random::__static_random_device();
      }
      cytnx::random::uniform_(Sin, low, high, seed);
    },
    py::arg("Sin"), py::arg("low") = double(0), py::arg("high") = double(1.0),
    py::arg("seed") = -1);

  m_random.def(
    "uniform_",
    [](cytnx::UniTensor &Tin, const double &low, const double &high, int64_t &seed) {
      if (seed == -1) {
        // If user doesn't specify seed argument
        seed = cytnx::random::__static_random_device();
      }
      cytnx::random::uniform_(Tin, low, high, seed);
    },
    py::arg("Tin"), py::arg("low") = double(0), py::arg("high") = double(1.0),
    py::arg("seed") = -1);

  m_random.def(
    "normal",
    [](const cytnx_uint64 &Nelem, const double &mean, const double &std, const int &device,
       int64_t &seed, const unsigned int &dtype) {
      if (seed == -1) {
        // If user doesn't specify seed argument
        seed = cytnx::random::__static_random_device();
      }
      return cytnx::random::normal(Nelem, mean, std, device, seed, dtype);
    },
    py::arg("Nelem"), py::arg("mean"), py::arg("std"), py::arg("device") = -1, py::arg("seed") = -1,
    py::arg("dtype") = (unsigned int)(Type.Double));
  m_random.def(
    "normal",
    [](const std::vector<cytnx_uint64> &Nelem, const double &mean, const double &std,
       const int &device, int64_t &seed, const unsigned int &dtype) {
      if (seed == -1) {
        // If user doesn't specify seed argument
        seed = cytnx::random::__static_random_device();
      }
      return cytnx::random::normal(Nelem, mean, std, device, seed, dtype);
    },
    py::arg("Nelem"), py::arg("mean"), py::arg("std"), py::arg("device") = -1, py::arg("seed") = -1,
    py::arg("dtype") = (unsigned int)(Type.Double));
  m_random.def(
    "uniform",
    [](const cytnx_uint64 &Nelem, const double &low, const double &high, const int &device,
       int64_t &seed, const unsigned int &dtype) {
      if (seed == -1) {
        // If user doesn't specify seed argument
        seed = cytnx::random::__static_random_device();
      }
      return cytnx::random::uniform(Nelem, low, high, device, seed, dtype);
    },
    py::arg("Nelem"), py::arg("low"), py::arg("high"), py::arg("device") = -1, py::arg("seed") = -1,
    py::arg("dtype") = (unsigned int)(Type.Double));
  m_random.def(
    "uniform",
    [](const std::vector<cytnx_uint64> &Nelem, const double &low, const double &high,
       const int &device, int64_t &seed, const unsigned int &dtype) {
      if (seed == -1) {
        // If user doesn't specify seed argument
        seed = cytnx::random::__static_random_device();
      }
      return cytnx::random::uniform(Nelem, low, high, device, seed, dtype);
    },
    py::arg("Nelem"), py::arg("low"), py::arg("high"), py::arg("device") = -1, py::arg("seed") = -1,
    py::arg("dtype") = (unsigned int)(Type.Double));
}
#endif
