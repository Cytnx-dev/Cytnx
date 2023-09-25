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
#include "complex.h"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace cytnx;

void symmetry_binding(py::module &m) {
  py::enum_<__sym::__stype>(m, "SymType")
    .value("Z", __sym::__stype::Z)
    .value("U", __sym::__stype::U)
    .export_values();

  py::class_<Qs>(m, "_cQs")
    .def(py::init<const std::vector<cytnx_int64>>(), py::arg("qin"))
    .def(
      "__rshift__", [](Qs &self, const cytnx_uint64 &dim) { return self >> dim; }, py::arg("dim"));

  py::class_<Symmetry>(m, "Symmetry")
    // construction
    .def(py::init<>())
    //.def(py::init<const int &, const int&>())
    .def_static("U1", &Symmetry::U1)
    .def_static("Zn", &Symmetry::Zn)
    .def("clone", &Symmetry::clone)
    .def("stype", &Symmetry::stype)
    .def("stype_str", &Symmetry::stype_str)
    .def("n", &Symmetry::n)
    .def("clone", &Symmetry::clone)
    .def("__copy__", &Symmetry::clone)
    .def("__deepcopy__", &Symmetry::clone)
    .def("__eq__", &Symmetry::operator==)
    .def("check_qnum", &Symmetry::check_qnum, py::arg("qnum"))
    .def("check_qnums", &Symmetry::check_qnums, py::arg("qnums"))
    .def(
      "combine_rule",
      [](Symmetry &self, const cytnx_int64 &inL, const cytnx_int64 &inR, const bool &is_reverse) {
        return self.combine_rule(inL, inR, is_reverse);
      },
      py::arg("qnL"), py::arg("qnR"), py::arg("is_reverse") = false)
    .def(
      "reverse_rule", [](Symmetry &self, const cytnx_int64 &qin) { return self.reverse_rule(qin); },
      py::arg("qin"))

    .def(
      "Save", [](Symmetry &self, const std::string &fname) { self.Save(fname); }, py::arg("fname"))
    .def_static(
      "Load", [](const std::string &fname) { return Symmetry::Load(fname); }, py::arg("fname"))
    .def(
      "__repr__",
      [](Symmetry &self) {
        std::cout << self << std::endl;
        return std::string("");
      },
      py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())

    //.def("combine_rule",&Symmetry::combine_rule,py::arg("qnums_1"),py::arg("qnums_2"))
    //.def("combine_rule_",&Symmetry::combine_rule_,py::arg("qnums_l"),py::arg("qnums_r"))
    //.def("check_qnum", &Symmetry::check_qnum,py::arg("qnum"))
    //.def("check_qnums", &Symmetry::check_qnums, py::arg("qnums"))
    ;
}
