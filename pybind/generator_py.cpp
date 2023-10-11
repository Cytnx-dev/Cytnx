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

void generator_binding(py::module &m) {
  m.def(
    "zeros",
    [](const cytnx_uint64 &Nelem, const unsigned int &dtype, const int &device) -> Tensor {
      return cytnx::zeros(Nelem, dtype, device);
    },
    py::arg("size"), py::arg("dtype") = (unsigned int)(cytnx::Type.Double),
    py::arg("device") = (int)(cytnx::Device.cpu));

  m.def(
    "zeros",
    [](py::object Nelem, const unsigned int &dtype, const int &device) -> Tensor {
      std::vector<cytnx_uint64> tmp = Nelem.cast<std::vector<cytnx_uint64>>();
      return cytnx::zeros(tmp, dtype, device);
    },
    py::arg("size"), py::arg("dtype") = (unsigned int)(cytnx::Type.Double),
    py::arg("device") = (int)(cytnx::Device.cpu));

  m.def(
    "ones",
    [](const cytnx_uint64 &Nelem, const unsigned int &dtype, const int &device) -> Tensor {
      return cytnx::ones(Nelem, dtype, device);
    },
    py::arg("size"), py::arg("dtype") = (unsigned int)(cytnx::Type.Double),
    py::arg("device") = (int)(cytnx::Device.cpu));

  m.def(
    "ones",
    [](py::object Nelem, const unsigned int &dtype, const int &device) -> Tensor {
      std::vector<cytnx_uint64> tmp = Nelem.cast<std::vector<cytnx_uint64>>();
      return cytnx::ones(tmp, dtype, device);
    },
    py::arg("size"), py::arg("dtype") = (unsigned int)(cytnx::Type.Double),
    py::arg("device") = (int)(cytnx::Device.cpu));
  m.def("identity", &cytnx::identity, py::arg("Dim"),
        py::arg("dtype") = (unsigned int)(cytnx::Type.Double),
        py::arg("device") = (int)(cytnx::Device.cpu));
  m.def("eye", &cytnx::identity, py::arg("Dim"),
        py::arg("dtype") = (unsigned int)(cytnx::Type.Double),
        py::arg("device") = (int)(cytnx::Device.cpu));

  m.def(
    "arange", [](const cytnx_uint64 &Nelem) -> Tensor { return cytnx::arange(Nelem); },
    py::arg("size"));

  m.def(
    "arange",
    [](const cytnx_double &start, const cytnx_double &end, const cytnx_double &step,
       const unsigned int &dtype,
       const int &device) -> Tensor { return cytnx::arange(start, end, step, dtype, device); },
    py::arg("start"), py::arg("end"), py::arg("step") = double(1),
    py::arg("dtype") = (unsigned int)(cytnx::Type.Double),
    py::arg("device") = (int)(cytnx::Device.cpu));

  m.def(
    "linspace",
    [](const cytnx_double &start, const cytnx_double &end, const cytnx_uint64 &Nelem,
       const bool &endpoint, const unsigned int &dtype, const int &device) -> Tensor {
      return cytnx::linspace(start, end, Nelem, endpoint, dtype, device);
    },
    py::arg("start"), py::arg("end"), py::arg("Nelem"), py::arg("endpoint") = true,
    py::arg("dtype") = (unsigned int)(cytnx::Type.Double),
    py::arg("device") = (int)(cytnx::Device.cpu));

  m.def("_from_numpy", [](py::buffer b) -> Tensor {
    py::buffer_info info = b.request();

    // std::cout << PyBuffer_IsContiguous(info,'C') << std::endl;

    // check type:
    int dtype;
    std::vector<cytnx_uint64> shape(info.shape.begin(), info.shape.end());
    ssize_t Totbytes;

    if (info.format == py::format_descriptor<cytnx_complex128>::format()) {
      dtype = Type.ComplexDouble;
      Totbytes = info.strides[0] * info.shape[0];
    } else if (info.format == py::format_descriptor<cytnx_complex64>::format()) {
      dtype = Type.ComplexFloat;
      Totbytes = info.strides[0] * info.shape[0];
    } else if (info.format == py::format_descriptor<cytnx_double>::format()) {
      dtype = Type.Double;
      Totbytes = info.strides[0] * info.shape[0];
    } else if (info.format == py::format_descriptor<cytnx_float>::format()) {
      dtype = Type.Float;
      Totbytes = info.strides[0] * info.shape[0];
    } else if (info.format == py::format_descriptor<uint64_t>::format() || info.format == "L") {
      dtype = Type.Uint64;
      Totbytes = info.strides[0] * info.shape[0];
    } else if (info.format == py::format_descriptor<int64_t>::format() || info.format == "l") {
      dtype = Type.Int64;
      Totbytes = info.strides[0] * info.shape[0];
    } else if (info.format == py::format_descriptor<uint32_t>::format()) {
      dtype = Type.Uint32;
      Totbytes = info.strides[0] * info.shape[0];
    } else if (info.format == py::format_descriptor<int32_t>::format()) {
      dtype = Type.Int32;
      Totbytes = info.strides[0] * info.shape[0];
    } else if (info.format == py::format_descriptor<cytnx_bool>::format()) {
      dtype = Type.Bool;
      Totbytes = info.strides[0] * info.shape[0];
    } else {
      cytnx_error_msg(true, "[ERROR] invalid type from numpy.ndarray to Tensor%s", "\n");
    }
    /*
    for( int i=0;i<info.strides.size();i++)
        std::cout << info.strides[i] << " ";
    std::cout << std::endl;
    for( int i=0;i<info.shape.size();i++)
        std::cout << info.shape[i] << " ";
    std::cout << std::endl;

    std::cout << Type.getname(dtype) << std::endl;
    std::cout << Totbytes << std::endl;
    */
    // Totbytes *= cytnx::Type.typeSize(dtype);

    Tensor m;
    m.Init(shape, dtype);
    memcpy(m.storage()._impl->Mem, info.ptr, Totbytes);
    return m;
  });
}

#endif
