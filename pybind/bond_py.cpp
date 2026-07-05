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

namespace py = pybind11;
using namespace pybind11::literals;
using namespace cytnx;

void bond_binding(py::module &m) {
  py::enum_<bondType>(m, "bondType")
    .value("BD_BRA", bondType::BD_BRA)
    .value("BD_KET", bondType::BD_KET)
    .value("BD_REG", bondType::BD_REG)
    .value("BD_IN", bondType::BD_IN)
    .value("BD_OUT", bondType::BD_OUT)
    .export_values();

  py::class_<Bond>(m, "Bond")
    // construction
    .def(py::init<>())
    .def(py::init<const cytnx_uint64 &, const bondType &>(), py::arg("dim"),
         py::arg("bond_type") = bondType::BD_REG)
    .def(py::init<const bondType &, const std::vector<std::vector<cytnx_int64>> &,
                  const std::vector<cytnx_uint64> &, const std::vector<Symmetry> &>(),
         py::arg("bond_type"), py::arg("qnums"), py::arg("degs"),
         py::arg("symmetries") = std::vector<Symmetry>())

    .def(py::init<const bondType &,
                  const std::vector<std::pair<std::vector<cytnx_int64>, cytnx_uint64>> &,
                  const std::vector<Symmetry> &>(),
         py::arg("bond_type"), py::arg("qnums"), py::arg("symmetries") = std::vector<Symmetry>())

    .def(py::init<const bondType &, const std::vector<cytnx::Qs> &,
                  const std::vector<cytnx_uint64> &, const std::vector<Symmetry> &>(),
         py::arg("bond_type"), py::arg("qnums"), py::arg("degs"),
         py::arg("symmetries") = std::vector<Symmetry>())

    .def(
      "Init",
      [](Bond &self, const bondType &bd_type, const std::vector<std::vector<cytnx_int64>> &in_qnums,
         const std::vector<cytnx_uint64> &degs,
         const std::vector<Symmetry> &in_syms) { self.Init(bd_type, in_qnums, degs, in_syms); },
      py::arg("bond_type"), py::arg("qnums"), py::arg("degs"),
      py::arg("symmetries") = std::vector<Symmetry>())

    .def(
      "Init",
      [](Bond &self, const cytnx_uint64 &dim, const bondType &bd_type) { self.Init(dim, bd_type); },
      py::arg("dim"), py::arg("bond_type") = bondType::BD_REG)

    .def(
      "Init",
      [](Bond &self, const bondType &bd_type,
         const std::vector<std::pair<std::vector<cytnx_int64>, cytnx_uint64>> &in_qnums_degs,
         const std::vector<Symmetry> &in_syms) { self.Init(bd_type, in_qnums_degs, in_syms); },
      py::arg("bond_type"), py::arg("qnums"), py::arg("symmetries") = std::vector<Symmetry>())

    .def(
      "Init",
      [](Bond &self, const bondType &bd_type, const std::vector<cytnx::Qs> &in_qnums,
         const std::vector<cytnx_uint64> &degs, const std::vector<Symmetry> &in_syms) {
        vec2d<cytnx_int64> qnums(in_qnums.begin(), in_qnums.end());
        self.Init(bd_type, qnums, degs, in_syms);
      },
      py::arg("bond_type"), py::arg("qnums"), py::arg("degs"),
      py::arg("symmetries") = std::vector<Symmetry>())

    .def(
      "__repr__",
      [](Bond &self) {
        std::cout << self << std::endl;
        return std::string("");
      },
      py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
    .def("__eq__", &Bond::operator==)
    .def("type", &Bond::type)
    .def("qnums", [](Bond &self) { return self.qnums(); })
    .def("qnums_clone", [](Bond &self) { return self.qnums_clone(); })
    .def("dim", &Bond::dim)
    .def("Nsym", &Bond::Nsym)
    .def("syms", [](Bond &self) { return self.syms(); })
    .def("syms_clone", [](Bond &self) { return self.syms_clone(); })
    .def("set_type", &Bond::set_type)
    .def("retype", &Bond::retype)
    .def("clear_type", &Bond::clear_type)
    .def("redirect", &Bond::redirect)
    .def("redirect_",
         [](py::object self) {
           self.cast<Bond &>().redirect_();
           return self;
         })
    .def("clone", &Bond::clone)
    .def("__copy__", &Bond::clone)
    .def("__deepcopy__", &Bond::clone)
    .def(
      "combineBond",
      [](Bond &self, const Bond &bd, bool is_grp) { return self.combineBond(bd, is_grp); },
      py::arg("bd"), py::arg("is_grp") = true)
    .def(
      "combineBond",
      [](Bond &self, const std::vector<Bond> &bds, bool is_grp) {
        return self.combineBonds(bds, is_grp);
      },
      py::arg("bds"), py::arg("is_grp") = true)
    .def(
      "combineBond_",
      [](Bond &self, const Bond &bd, bool is_grp) { return self.combineBond_(bd, is_grp); },
      py::arg("bd"), py::arg("is_grp") = true)
    .def(
      "combineBond_",
      [](Bond &self, const std::vector<Bond> &bds, bool is_grp) {
        return self.combineBonds_(bds, is_grp);
      },
      py::arg("bds"), py::arg("is_grp") = true)
    // .def("combineBond", &Bond::combineBond, py::arg("bd"), py::arg("is_grp") = true)
    // .def("combineBond_", &Bond::combineBond_, py::arg("bd"), py::arg("is_grp") = true)
    .def("combineBonds", &Bond::combineBonds, py::arg("bds"), py::arg("is_grp") = true)
    .def("combineBonds_", &Bond::combineBonds_, py::arg("bds"), py::arg("is_grp") = true)
    .def("getDegeneracies", [](Bond &self) { return self.getDegeneracies(); })

    .def(
      "getDegeneracy",
      [](Bond &self, const std::vector<cytnx_int64> &qnum, bool return_indices) -> py::object {
        if (!return_indices) return py::cast(self.getDegeneracy(qnum));
        std::vector<cytnx_uint64> inds;
        auto out = self.getDegeneracy(qnum, inds);
        return py::make_tuple(out, inds);
      },
      py::arg("qnum"), py::arg("return_indices") = false)
    .def(
      "getDegeneracy",
      [](Bond &self, const cytnx::Qs &qnum, bool return_indices) -> py::object {
        std::vector<cytnx_int64> qvec(qnum);
        if (!return_indices) return py::cast(self.getDegeneracy(qvec));
        std::vector<cytnx_uint64> inds;
        auto out = self.getDegeneracy(qvec, inds);
        return py::make_tuple(out, inds);
      },
      py::arg("qnum"), py::arg("return_indices") = false)

    .def("get_fermion_parity", &Bond::get_fermion_parity, py::arg("qnum"))

    .def("group_duplicates_", &Bond::group_duplicates)
    .def("group_duplicates",
         [](Bond &self) {
           std::vector<cytnx_uint64> mprs;
           Bond out = self.group_duplicates(mprs);
           return py::make_tuple(out, mprs);
         })

    .def("has_duplicate_qnums", &Bond::has_duplicate_qnums)
    .def("calc_reverse_qnums", &Bond::calc_reverse_qnums)

    .def(
      "Save", [](Bond &self, const std::string &fname) { self.Save(fname); }, py::arg("fname"))
    .def_static(
      "Load", [](const std::string &fname) { return Bond::Load(fname); }, py::arg("fname"))

    ;  // end of object line
}
