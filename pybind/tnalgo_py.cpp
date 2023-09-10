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

void tnalgo_binding(py::module &m) {
  //====================
  // [Submodule tn_algo]
  pybind11::module m_tnalgo = m.def_submodule("tn_algo", "tensor network algorithm related");

  py::class_<tn_algo::MPS>(m_tnalgo, "MPS")
    .def(py::init<>())
    .def(py::init<const cytnx_uint64 &, const cytnx_uint64 &, const cytnx_uint64 &,
                  const cytnx_int64 &, const cytnx_int64 &>(),
         py::arg("N"), py::arg("phys_dim"), py::arg("virt_dim"),
         py::arg("dtype") = cytnx_int64(Type.Double), py::arg("mps_type") = 0)
    .def(py::init<const cytnx_uint64 &, const std::vector<cytnx_uint64> &, const cytnx_uint64 &,
                  const cytnx_int64 &, const cytnx_int64 &>(),
         py::arg("N"), py::arg("vphys_dim"), py::arg("virt_dim"),
         py::arg("dtype") = cytnx_int64(Type.Double), py::arg("mps_type") = 0)

    .def("Init_Msector", &tn_algo::MPS::Init_Msector, py::arg("N"), py::arg("vphys_dim"),
         py::arg("virt_dim"), py::arg("select"), py::arg("dtype") = cytnx_int64(Type.Double),
         py::arg("mps_type") = 0)
    .def("size", &tn_algo::MPS::size)
    .def("mps_type", &tn_algo::MPS::mps_type)
    .def("mps_type_str", &tn_algo::MPS::mps_type_str)
    .def("clone", &tn_algo::MPS::clone)
    .def("data", &tn_algo::MPS::data)
    .def("phys_dim", &tn_algo::MPS::phys_dim, py::arg("idx"))
    .def("virt_dim", &tn_algo::MPS::virt_dim)
    .def("S_loc", &tn_algo::MPS::S_loc)
    .def("c_Into_Lortho", &tn_algo::MPS::Into_Lortho)
    .def("c_S_mvleft", &tn_algo::MPS::S_mvleft)
    .def("c_S_mvright", &tn_algo::MPS::S_mvright)
    .def(
      "Save", [](cytnx::Storage &self, const std::string &fname) { self.Save(fname); },
      py::arg("fname"))
    .def_static(
      "Load", [](const std::string &fname) { return cytnx::tn_algo::MPS::Load(fname); },
      py::arg("fname"))
    .def(
      "__repr__",
      [](cytnx::tn_algo::MPS &self) -> std::string {
        std::cout << self << std::endl;
        return std::string("");
      },
      py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
    .def("norm", [](tn_algo::MPS &self) { return double(self.norm()); })

    ;

  py::class_<tn_algo::MPO>(m_tnalgo, "MPO")
    .def(py::init<>())
    .def("size", &tn_algo::MPO::size)
    .def("append", &tn_algo::MPO::append, py::arg("Tn"))
    .def("assign", &tn_algo::MPO::assign, py::arg("N"), py::arg("Tn"))
    .def("get_all", [](tn_algo::MPO &self) { return self.get_all(); })
    .def("get_op", &tn_algo::MPO::get_op, py::arg("idx"));

  py::class_<tn_algo::DMRG>(m_tnalgo, "DMRG")
    .def(py::init<tn_algo::MPO, tn_algo::MPS, std::vector<tn_algo::MPS>, const double &>(),
         py::arg("mpo"), py::arg("mps"), py::arg("ortho_mps") = std::vector<tn_algo::MPS>(),
         py::arg("weight") = 30)
    .def("initialize", &tn_algo::DMRG::initialize)
    .def(
      "sweep",
      [](tn_algo::DMRG &self, const bool &verbose = false, const cytnx_int64 &maxit = 4000,
         const cytnx_int64 &krydim = 4) {
        auto E = self.sweep(verbose, maxit, krydim);
        return double(E);
      },
      py::arg("verbose") = false, py::arg("maxit") = 4000, py::arg("krydim") = 4)
    .def(
      "sweepv2",
      [](tn_algo::DMRG &self, const bool &verbose = false, const cytnx_int64 &maxit = 4000,
         const cytnx_int64 &krydim = 4) {
        auto E = self.sweepv2(verbose, maxit, krydim);
        return double(E);
      },
      py::arg("verbose") = false, py::arg("maxit") = 4000, py::arg("krydim") = 4);
}
#endif
