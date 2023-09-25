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

// ref: https://developer.lsst.io/v/DM-9089/coding/python_wrappers_for_cpp_with_pybind11.html
// ref: https://pybind11.readthedocs.io/en/stable/advanced/classes.html
// ref: https://block.arch.ethz.ch/blog/2016/07/adding-methods-to-python-classes/
// ref: https://medium.com/@mgarod/dynamically-add-a-method-to-a-class-in-python-c49204b85bd6
void bond_binding(py::module &m);
void symmetry_binding(py::module &m);

#ifdef BACKEND_TORCH
#else

void generator_binding(py::module &m);
void storage_binding(py::module &m);
void tensor_binding(py::module &m);

void network_binding(py::module &m);

class PyLinOp;
void linop_binding(py::module &m);

class cHclass;
void unitensor_binding(py::module &m);

void linalg_binding(py::module &m);
void algo_binding(py::module &m);
void physics_related_binding(py::module &m);
void random_binding(py::module &m);
void tnalgo_binding(py::module &m);
void scalar_binding(py::module &m);

void ncon_binding(py::module &m);
#endif

PYBIND11_MODULE(cytnx, m) {
  m.attr("__version__") = "0.7";
  m.attr("__blasINTsize__") = cytnx::__blasINTsize__;
  m.attr("User_debug") = cytnx::User_debug;

  symmetry_binding(m);
  bond_binding(m);
  py::add_ostream_redirect(m, "ostream_redirect");

  py::enum_<cytnx::__type::__pybind_type>(m, "Type")
    .value("Void", cytnx::__type::__pybind_type::Void)
    .value("ComplexDouble", cytnx::__type::__pybind_type::ComplexDouble)
    .value("ComplexFloat", cytnx::__type::__pybind_type::ComplexFloat)
    .value("Double", cytnx::__type::__pybind_type::Double)
    .value("Float", cytnx::__type::__pybind_type::Float)
    .value("Uint64", cytnx::__type::__pybind_type::Uint64)
    .value("Int64", cytnx::__type::__pybind_type::Int64)
    .value("Uint32", cytnx::__type::__pybind_type::Uint32)
    .value("Int32", cytnx::__type::__pybind_type::Int32)
    .value("Uint16", cytnx::__type::__pybind_type::Uint16)
    .value("Int16", cytnx::__type::__pybind_type::Int16)
    .value("Bool", cytnx::__type::__pybind_type::Bool)
    .export_values();

  auto mdev = m.def_submodule("Device");
  mdev.attr("cpu") = (cytnx_int64)cytnx::Device.cpu;
  mdev.attr("cuda") = (cytnx_int64)cytnx::Device.cuda;
  mdev.attr("Ngpus") = cytnx::Device.Ngpus;
  mdev.attr("Ncpus") = cytnx::Device.Ncpus;
  mdev.def("Print_Property", []() { cytnx::Device.Print_Property(); });
  mdev.def("getname",
           [](const int &device_id) -> std::string { return cytnx::Device.getname(device_id); });
  // mdev.def("cudaDeviceSynchronize",[](){cytnx::Device.cudaDeviceSynchronize();});
#ifdef BACKEND_TORCH
  m.attr("__cytnx_backend__") = std::string("torch");
#else
  m.attr("__cytnx_backend__") = std::string("cytnx");

  m.def("set_mkl_ilp64", &cytnx::set_mkl_ilp64);
  m.def("get_mkl_code", &cytnx::get_mkl_code);

  // global vars
  // m.attr("cytnxdevice") = cytnx::cytnxdevice;
  // m.attr("Type")   = py::cast(cytnx::Type);
  // m.attr("redirect_output") = py::capsule(new py::scoped_ostream_redirect(...),
  //[](void *sor) { delete static_cast<py::scoped_ostream_redirect *>(sor); });

  // py::enum_<cytnx::__device::__pybind_device>(m,"Device",py::arithmetic())
  //     .value("cpu", cytnx::__device::__pybind_device::cpu)
  //	.value("cuda", cytnx::__device::__pybind_device::cuda)
  //	.export_values();

  // m.attr("Device") = py::module::import("enum").attr("IntEnum")
  //     ("Device", py::dict("cpu"_a=(cytnx_int64)cytnx::Device.cpu,
  //     "cuda"_a=(cytnx_int64)cytnx::Device.cuda));

  // m.def(
  //   "ncon",
  //   [](const std::vector<UniTensor> &tensor_list_in,
  //       const std::vector<std::vector<cytnx_int64>> &connect_list_in,
  //       const bool check_network, const bool optimize,
  //       std::vector<cytnx_int64> cont_order ,
  //       const std::vector<std::string> &out_labels) -> UniTensor {
  //     return ncon(tensor_list_in, connect_list_in, check_network, optimize, cont_order,
  //     out_labels);
  //   },
  //   py::arg("tensor_list_in")= std::vector<UniTensor>(),
  //   py::arg("connect_list_in")= std::vector<std::vector<cytnx_int64>>(),
  //   py::arg("check_network")= false,
  //   py::arg("optimize") = false,
  //   py::arg("cont_order") = std::vector<cytnx_int64>(),
  //   py::arg("out_labels") = std::vector<std::string>());

  generator_binding(m);
  scalar_binding(m);
  storage_binding(m);
  tensor_binding(m);
  network_binding(m);
  linop_binding(m);
  unitensor_binding(m);
  linalg_binding(m);
  algo_binding(m);
  physics_related_binding(m);
  random_binding(m);
  tnalgo_binding(m);
  ncon_binding(m);
#endif
}
