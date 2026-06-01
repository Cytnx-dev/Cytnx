#include <format>
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

namespace {
  cytnx_int64 NormalizeZnInput(const Symmetry &sym, const cytnx_int64 qnum, const char *fn_name,
                               const char *arg_name) {
    if (sym.stype() != SymmetryType::Z) return qnum;
    const cytnx_int64 n = sym.n();
    if (qnum >= 0 && qnum < n) return qnum;
    cytnx_int64 normalized = qnum % n;
    if (normalized < 0) normalized += n;
    const std::string message = std::format(
      "Passing out-of-range Z{} qnum {} to '{}' (argument '{}') is deprecated and will be "
      "rejected in v2.0.0; pass a canonical representative in [0, {}) instead. Normalizing to "
      "{} for now.",
      n, qnum, fn_name, arg_name, n, normalized);
    if (PyErr_WarnEx(PyExc_FutureWarning, message.c_str(), 2) < 0) throw py::error_already_set();
    return normalized;
  }
}  // namespace

void symmetry_binding(py::module &m) {
  py::enum_<SymmetryType>(m, "SymType")
    .value("Z", SymmetryType::Z)
    .value("U", SymmetryType::U)
    .value("fPar", SymmetryType::fPar)
    .value("fNum", SymmetryType::fNum)
    .export_values();

  py::enum_<fermionParity>(m, "fermionParity")
    .value("EVEN", fermionParity::EVEN)
    .value("ODD", fermionParity::ODD)
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
    .def_static("FermionParity", &Symmetry::FermionParity)
    .def_static("FermionNumber", &Symmetry::FermionNumber)
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
      [](const Symmetry &self, const cytnx_int64 &inL, const cytnx_int64 &inR,
         const bool &is_reverse) {
        const cytnx_int64 normL = NormalizeZnInput(self, inL, "combine_rule", "qnL");
        const cytnx_int64 normR = NormalizeZnInput(self, inR, "combine_rule", "qnR");
        return self.combine_rule(normL, normR, is_reverse);
      },
      py::arg("qnL"), py::arg("qnR"), py::arg("is_reverse") = false)
    .def(
      "reverse_rule",
      [](const Symmetry &self, const cytnx_int64 &qin) {
        const cytnx_int64 norm = NormalizeZnInput(self, qin, "reverse_rule", "qin");
        return self.reverse_rule(norm);
      },
      py::arg("qin"))
    .def("get_fermion_parity", &Symmetry::get_fermion_parity, py::arg("qnum"))
    .def("is_fermionic", &Symmetry::is_fermionic)

    .def(
      "Save", [](Symmetry &self, const std::string &fname) { self.Save(fname); }, py::arg("fname"))
    .def_static(
      "Load", [](const std::string &fname) { return Symmetry::Load(fname); }, py::arg("fname"))
    .def(
      "Load_", [](cytnx::Symmetry &self, const std::string &fname) { return self.Load_(fname); },
      py::arg("fname"))

    .def(py::pickle(
      [](const Symmetry &self) {  // __getstate__
        std::ostringstream oss(std::ios::binary);
        self.to_binary(oss);
        return py::bytes(oss.str());
      },
      [](py::bytes state) {  // __setstate__
        std::string data = state;
        std::istringstream iss(data, std::ios::binary);
        Symmetry out;
        out.from_binary(iss);
        return out;
      }))

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
