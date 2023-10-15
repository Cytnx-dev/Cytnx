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

void network_binding(py::module &m) {
  py::enum_<__ntwk::__nttype>(m, "NtType")
    .value("Regular", __ntwk::__nttype::Regular)
    .value("Fermion", __ntwk::__nttype::Fermion)
    .value("Void", __ntwk::__nttype::Void)
    .export_values();

  py::class_<Network>(m, "Network")
    .def(py::init<>())
    .def(py::init<const std::string &, const int &>(), py::arg("fname"),
         py::arg("network_type") = (int)NtType.Regular)
    .def("_cget_tn_names", [](Network &self) { return self._impl->names; })
    .def("_cget_tn_labels", [](Network &self) { return self._impl->label_arr; })
    .def("_cget_tn_out_labels", [](Network &self) { return self._impl->TOUT_labels; })
    .def("isLoad",
         [](Network &self) -> bool { return self._impl->tensors.size() == 0 ? false : true; })
    .def("isAllset",
         [](Network &self) -> bool {
           bool out = true;
           for (int i = 0; i < self._impl->tensors.size(); i++) {
             if (self._impl->tensors[i].uten_type() == UTenType.Void) out = false;
           }
           return out;
         })
    .def("_cget_filename", [](Network &self) { return self._impl->filename; })
    .def("Fromfile", &Network::Fromfile, py::arg("fname"),
         py::arg("network_type") = (int)NtType.Regular)
    .def("FromString", &Network::FromString, py::arg("contents"),
         py::arg("network_type") = (int)NtType.Regular)
    .def("Savefile", &Network::Savefile, py::arg("fname"))
    .def(
      "PutUniTensor",
      [](Network &self, const std::string &name, const UniTensor &utensor,
         const std::vector<std::string> &label_order) {
        self.PutUniTensor(name, utensor, label_order);
      },
      py::arg("name"), py::arg("utensor"), py::arg("label_order") = std::vector<std::string>())
    .def(
      "PutUniTensor",
      [](Network &self, const cytnx_uint64 &idx, const UniTensor &utensor,
         const std::vector<std::string> &label_order) {
        self.PutUniTensor(idx, utensor, label_order);
      },
      py::arg("idx"), py::arg("utensor"), py::arg("label_order") = std::vector<std::string>())
    .def(
      "PutUniTensors",
      [](Network &self, const std::vector<std::string> &names,
         const std::vector<UniTensor> &utensors) { self.PutUniTensors(names, utensors); },
      py::arg("names"), py::arg("utensors"))

    .def(
      "RmUniTensor", [](Network &self, const std::string &name) { self.RmUniTensor(name); },
      py::arg("name"))
    .def(
      "RmUniTensor", [](Network &self, const cytnx_uint64 &idx) { self.RmUniTensor(idx); },
      py::arg("idx"))
    .def(
      "RmUniTensors",
      [](Network &self, const std::vector<std::string> &names) { self.RmUniTensors(names); },
      py::arg("names"))

    // .def("getOptimalOrder", &Network::getOptimalOrder,
    //      py::arg("network_type") = (int)NtType.Regular)
    // .def("Launch", &Network::Launch, py::arg("optimal") = false, py::arg("contract_order") = "",
    //      py::arg("network_type") = (int)NtType.Regular)

    .def(
      "setOrder",
      [](Network &self, const bool &optimal, const std::string &contract_order) {
        self.setOrder(optimal, contract_order);
      },
      py::arg("optimal") = false, py::arg("contract_order") = "")

    .def("getOrder", [](Network &self) { return self.getOrder(); })

    // .def("getOrder", &Network::getOrder)
    .def("Launch", &Network::Launch, py::arg("network_type") = (int)NtType.Regular)

    .def("construct", &Network::construct, py::arg("alias"), py::arg("labels"),
         py::arg("outlabel") = std::vector<std::string>(), py::arg("outrk"), py::arg("order") = "",
         py::arg("optim") = false, py::arg("network_type") = (int)NtType.Regular)

    .def("clear", &Network::clear)
    .def("clone", &Network::clone)
    .def("__copy__", &Network::clone)
    .def("__deepcopy__", &Network::clone)
    .def(
      "__repr__",
      [](Network &self) -> std::string {
        self.PrintNet();
        return std::string("");
      },
      py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
    .def("PrintNet", &Network::PrintNet)
    .def_static(
      "Contract",
      [](const std::vector<UniTensor> &utensors, const std::string &Tout,
         const std::vector<std::string> &alias = {}, const std::string &contract_order = "") {
        return Network::Contract(utensors, Tout, alias, contract_order);
      },
      py::arg("utensors"), py::arg("Tout"), py::arg("alias") = std::vector<std::string>(),
      py::arg("contract_order") = std::string(""))

    ;  // end of object line
}

#endif
