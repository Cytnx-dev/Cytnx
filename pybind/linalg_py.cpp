#include <pybind11/buffer_info.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <map>
#include <random>
#include <vector>

#include "cytnx.hpp"
// #include "../include/cytnx_error.hpp"
#include "complex.h"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace cytnx;

#ifdef BACKEND_TORCH
#else

void linalg_binding(py::module &m) {
  // [Submodule linalg]
  pybind11::module m_linalg = m.def_submodule("linalg", "linear algebra related.");
  m_linalg.def(
    "Svd",
    [](const cytnx::Tensor &Tin, const bool &is_UvT) { return cytnx::linalg::Svd(Tin, is_UvT); },
    py::arg("Tin"), py::arg("is_UvT") = true);
  m_linalg.def(
    "Svd",
    [](const cytnx::UniTensor &Tin, const bool &is_UvT) { return cytnx::linalg::Svd(Tin, is_UvT); },
    py::arg("Tin"), py::arg("is_UvT") = true);

  m_linalg.def(
    "Gesvd",
    [](const cytnx::Tensor &Tin, const bool &is_U, const bool &is_vT) {
      return cytnx::linalg::Gesvd(Tin, is_U, is_vT);
    },
    py::arg("Tin"), py::arg("is_U") = true, py::arg("is_vT") = true);
  m_linalg.def(
    "Gesvd",
    [](const cytnx::UniTensor &Tin, const bool &is_U, const bool &is_vT) {
      return cytnx::linalg::Gesvd(Tin, is_U, is_vT);
    },
    py::arg("Tin"), py::arg("is_U") = true, py::arg("is_vT") = true);

  m_linalg.def(
    "Gesvd_truncate",
    [](const Tensor &Tin, const cytnx_uint64 &keepdim, const cytnx_double &err, const bool &is_U,
       const bool &is_vT, const unsigned int &return_err, const unsigned int &mindim) {
      return cytnx::linalg::Gesvd_truncate(Tin, keepdim, err, is_U, is_vT, return_err, mindim);
    },
    py::arg("Tin"), py::arg("keepdim"), py::arg("err") = double(0), py::arg("is_U") = true,
    py::arg("is_vT") = true, py::arg("return_err") = (unsigned int)(0), py::arg("mindim") = 0);
  m_linalg.def(
    "Gesvd_truncate",
    [](const UniTensor &Tin, const cytnx_uint64 &keepdim, const cytnx_double &err, const bool &is_U,
       const bool &is_vT, const unsigned int &return_err, const unsigned int &mindim) {
      return cytnx::linalg::Gesvd_truncate(Tin, keepdim, err, is_U, is_vT, return_err, mindim);
    },
    py::arg("Tin"), py::arg("keepdim"), py::arg("err") = 0, py::arg("is_U") = true,
    py::arg("is_vT") = true, py::arg("return_err") = (unsigned int)(0), py::arg("mindim") = 0);

  m_linalg.def(
    "Svd_truncate",
    [](const Tensor &Tin, const cytnx_uint64 &keepdim, const cytnx_double &err, const bool &is_UvT,
       const unsigned int &return_err, const unsigned int &mindim) {
      return cytnx::linalg::Svd_truncate(Tin, keepdim, err, is_UvT, return_err, mindim);
    },
    py::arg("Tin"), py::arg("keepdim"), py::arg("err") = double(0), py::arg("is_UvT") = true,
    py::arg("return_err") = (unsigned int)(0), py::arg("mindim") = 0);
  m_linalg.def(
    "Svd_truncate",
    [](const UniTensor &Tin, const cytnx_uint64 &keepdim, const cytnx_double &err,
       const bool &is_UvT, const unsigned int &return_err, const unsigned int &mindim) {
      return cytnx::linalg::Svd_truncate(Tin, keepdim, err, is_UvT, return_err, mindim);
    },
    py::arg("Tin"), py::arg("keepdim"), py::arg("err") = 0, py::arg("is_UvT") = true,
    py::arg("return_err") = (unsigned int)(0), py::arg("mindim") = 0);

  // m_linalg.def("Eigh", &cytnx::linalg::Eigh, py::arg("Tin"), py::arg("is_V") = true,
  //              py::arg("row_v") = false);
  //  m_linalg.def("Eig", &cytnx::linalg::Eig, py::arg("Tin"), py::arg("is_V") = true,
  //                py::arg("row_v") = false);
  m_linalg.def(
    "Eigh",
    [](const Tensor &Tin, const bool &is_V, const bool &row_v) {
      return cytnx::linalg::Eigh(Tin, is_V, row_v);
    },
    py::arg("Tin"), py::arg("is_V") = true, py::arg("row_v") = false);
  m_linalg.def(
    "Eigh",
    [](const UniTensor &Tin, const bool &is_V, const bool &row_v) {
      return cytnx::linalg::Eigh(Tin, is_V, row_v);
    },
    py::arg("Tin"), py::arg("is_V") = true, py::arg("row_v") = false);

  m_linalg.def(
    "Eig",
    [](const Tensor &Tin, const bool &is_V, const bool &row_v) {
      return cytnx::linalg::Eig(Tin, is_V, row_v);
    },
    py::arg("Tin"), py::arg("is_V") = true, py::arg("row_v") = false);
  m_linalg.def(
    "Eig",
    [](const UniTensor &Tin, const bool &is_V, const bool &row_v) {
      return cytnx::linalg::Eig(Tin, is_V, row_v);
    },
    py::arg("Tin"), py::arg("is_V") = true, py::arg("row_v") = false);

  m_linalg.def("Exp", &cytnx::linalg::Exp, py::arg("Tin"));
  m_linalg.def("Exp_", &cytnx::linalg::Exp_, py::arg("Tio"));
  m_linalg.def("Expf_", &cytnx::linalg::Expf_, py::arg("Tio"));
  m_linalg.def("Expf", &cytnx::linalg::Expf, py::arg("Tio"));

  // UT, [Note] no bool type!
  m_linalg.def(
    "ExpH",
    [](const UniTensor &Tin, const cytnx_complex128 &a, const cytnx_complex128 &b) {
      return cytnx::linalg::ExpH(Tin, a, b);
    },
    py::arg("Tin"), py::arg("a"), py::arg("b") = complex128(0));
  m_linalg.def(
    "ExpH",
    [](const UniTensor &Tin, const cytnx_complex64 &a, const cytnx_complex64 &b) {
      return cytnx::linalg::ExpH(Tin, a, b);
    },
    py::arg("Tin"), py::arg("a"), py::arg("b") = complex64(0));
  m_linalg.def(
    "ExpH",
    [](const UniTensor &Tin, const cytnx_double &a, const cytnx_double &b) {
      return cytnx::linalg::ExpH(Tin, a, b);
    },
    py::arg("Tin"), py::arg("a"), py::arg("b") = double(0));
  m_linalg.def(
    "ExpH",
    [](const UniTensor &Tin, const cytnx_float &a, const cytnx_float &b) {
      return cytnx::linalg::ExpH(Tin, a, b);
    },
    py::arg("Tin"), py::arg("a"), py::arg("b") = float(0));
  m_linalg.def(
    "ExpH",
    [](const UniTensor &Tin, const cytnx_uint64 &a, const cytnx_uint64 &b) {
      return cytnx::linalg::ExpH(Tin, a, b);
    },
    py::arg("Tin"), py::arg("a"), py::arg("b") = cytnx_uint64(0));
  m_linalg.def(
    "ExpH",
    [](const UniTensor &Tin, const cytnx_int64 &a, const cytnx_int64 &b) {
      return cytnx::linalg::ExpH(Tin, a, b);
    },
    py::arg("Tin"), py::arg("a"), py::arg("b") = cytnx_int64(0));
  m_linalg.def(
    "ExpH",
    [](const UniTensor &Tin, const cytnx_uint32 &a, const cytnx_uint32 &b) {
      return cytnx::linalg::ExpH(Tin, a, b);
    },
    py::arg("Tin"), py::arg("a"), py::arg("b") = cytnx_uint32(0));
  m_linalg.def(
    "ExpH",
    [](const UniTensor &Tin, const cytnx_int32 &a, const cytnx_int32 &b) {
      return cytnx::linalg::ExpH(Tin, a, b);
    },
    py::arg("Tin"), py::arg("a"), py::arg("b") = cytnx_int32(0));

  m_linalg.def(
    "ExpH",
    [](const UniTensor &Tin, const cytnx_uint16 &a, const cytnx_uint16 &b) {
      return cytnx::linalg::ExpH(Tin, a, b);
    },
    py::arg("Tin"), py::arg("a"), py::arg("b") = cytnx_uint16(0));
  m_linalg.def(
    "ExpH",
    [](const UniTensor &Tin, const cytnx_int16 &a, const cytnx_int16 &b) {
      return cytnx::linalg::ExpH(Tin, a, b);
    },
    py::arg("Tin"), py::arg("a"), py::arg("b") = cytnx_int16(0));

  m_linalg.def(
    "ExpH", [](const UniTensor &Tin) { return cytnx::linalg::ExpH(Tin); }, py::arg("Tin"));

  // Tn
  m_linalg.def(
    "ExpH",
    [](const Tensor &Tin, const cytnx_complex128 &a, const cytnx_complex128 &b) {
      return cytnx::linalg::ExpH(Tin, a, b);
    },
    py::arg("Tin"), py::arg("a"), py::arg("b") = complex128(0));
  m_linalg.def(
    "ExpH",
    [](const Tensor &Tin, const cytnx_complex64 &a, const cytnx_complex64 &b) {
      return cytnx::linalg::ExpH(Tin, a, b);
    },
    py::arg("Tin"), py::arg("a"), py::arg("b") = complex64(0));
  m_linalg.def(
    "ExpH",
    [](const Tensor &Tin, const cytnx_double &a, const cytnx_double &b) {
      return cytnx::linalg::ExpH(Tin, a, b);
    },
    py::arg("Tin"), py::arg("a"), py::arg("b") = double(0));
  m_linalg.def(
    "ExpH",
    [](const Tensor &Tin, const cytnx_float &a, const cytnx_float &b) {
      return cytnx::linalg::ExpH(Tin, a, b);
    },
    py::arg("Tin"), py::arg("a"), py::arg("b") = float(0));
  m_linalg.def(
    "ExpH",
    [](const Tensor &Tin, const cytnx_uint64 &a, const cytnx_uint64 &b) {
      return cytnx::linalg::ExpH(Tin, a, b);
    },
    py::arg("Tin"), py::arg("a"), py::arg("b") = cytnx_uint64(0));
  m_linalg.def(
    "ExpH",
    [](const Tensor &Tin, const cytnx_int64 &a, const cytnx_int64 &b) {
      return cytnx::linalg::ExpH(Tin, a, b);
    },
    py::arg("Tin"), py::arg("a"), py::arg("b") = cytnx_int64(0));
  m_linalg.def(
    "ExpH",
    [](const Tensor &Tin, const cytnx_uint32 &a, const cytnx_uint32 &b) {
      return cytnx::linalg::ExpH(Tin, a, b);
    },
    py::arg("Tin"), py::arg("a"), py::arg("b") = cytnx_uint32(0));
  m_linalg.def(
    "ExpH",
    [](const Tensor &Tin, const cytnx_int32 &a, const cytnx_int32 &b) {
      return cytnx::linalg::ExpH(Tin, a, b);
    },
    py::arg("Tin"), py::arg("a"), py::arg("b") = cytnx_int32(0));

  m_linalg.def(
    "ExpH",
    [](const Tensor &Tin, const cytnx_uint16 &a, const cytnx_uint16 &b) {
      return cytnx::linalg::ExpH(Tin, a, b);
    },
    py::arg("Tin"), py::arg("a"), py::arg("b") = cytnx_uint16(0));
  m_linalg.def(
    "ExpH",
    [](const Tensor &Tin, const cytnx_int16 &a, const cytnx_int16 &b) {
      return cytnx::linalg::ExpH(Tin, a, b);
    },
    py::arg("Tin"), py::arg("a"), py::arg("b") = cytnx_int16(0));

  m_linalg.def(
    "ExpH", [](const Tensor &Tin) { return cytnx::linalg::ExpH(Tin); }, py::arg("Tin"));

  // UT
  m_linalg.def(
    "ExpM",
    [](const UniTensor &Tin, const cytnx_complex128 &a, const cytnx_complex128 &b) {
      return cytnx::linalg::ExpM(Tin, a, b);
    },
    py::arg("Tin"), py::arg("a"), py::arg("b") = 0);
  m_linalg.def(
    "ExpM",
    [](const UniTensor &Tin, const cytnx_complex64 &a, const cytnx_complex64 &b) {
      return cytnx::linalg::ExpM(Tin, a, b);
    },
    py::arg("Tin"), py::arg("a"), py::arg("b") = 0);
  m_linalg.def(
    "ExpM",
    [](const UniTensor &Tin, const cytnx_double &a, const cytnx_double &b) {
      return cytnx::linalg::ExpM(Tin, a, b);
    },
    py::arg("Tin"), py::arg("a"), py::arg("b") = 0);
  m_linalg.def(
    "ExpM",
    [](const UniTensor &Tin, const cytnx_float &a, const cytnx_float &b) {
      return cytnx::linalg::ExpM(Tin, a, b);
    },
    py::arg("Tin"), py::arg("a"), py::arg("b") = 0);
  m_linalg.def(
    "ExpM",
    [](const UniTensor &Tin, const cytnx_uint64 &a, const cytnx_uint64 &b) {
      return cytnx::linalg::ExpM(Tin, a, b);
    },
    py::arg("Tin"), py::arg("a"), py::arg("b") = 0);
  m_linalg.def(
    "ExpM",
    [](const UniTensor &Tin, const cytnx_int64 &a, const cytnx_int64 &b) {
      return cytnx::linalg::ExpM(Tin, a, b);
    },
    py::arg("Tin"), py::arg("a"), py::arg("b") = 0);
  m_linalg.def(
    "ExpM",
    [](const UniTensor &Tin, const cytnx_uint32 &a, const cytnx_uint32 &b) {
      return cytnx::linalg::ExpM(Tin, a, b);
    },
    py::arg("Tin"), py::arg("a"), py::arg("b") = 0);
  m_linalg.def(
    "ExpM",
    [](const UniTensor &Tin, const cytnx_int32 &a, const cytnx_int32 &b) {
      return cytnx::linalg::ExpM(Tin, a, b);
    },
    py::arg("Tin"), py::arg("a"), py::arg("b") = 0);

  m_linalg.def(
    "ExpM",
    [](const UniTensor &Tin, const cytnx_uint16 &a, const cytnx_uint16 &b) {
      return cytnx::linalg::ExpM(Tin, a, b);
    },
    py::arg("Tin"), py::arg("a"), py::arg("b") = 0);
  m_linalg.def(
    "ExpM", [](const UniTensor &Tin) { return cytnx::linalg::ExpM(Tin); }, py::arg("Tin"));

  // Tn
  m_linalg.def(
    "ExpM",
    [](const Tensor &Tin, const cytnx_complex128 &a, const cytnx_complex128 &b) {
      return cytnx::linalg::ExpM(Tin, a, b);
    },
    py::arg("Tin"), py::arg("a"), py::arg("b") = 0);
  m_linalg.def(
    "ExpM",
    [](const Tensor &Tin, const cytnx_complex64 &a, const cytnx_complex64 &b) {
      return cytnx::linalg::ExpM(Tin, a, b);
    },
    py::arg("Tin"), py::arg("a"), py::arg("b") = 0);
  m_linalg.def(
    "ExpM",
    [](const Tensor &Tin, const cytnx_double &a, const cytnx_double &b) {
      return cytnx::linalg::ExpM(Tin, a, b);
    },
    py::arg("Tin"), py::arg("a"), py::arg("b") = 0);
  m_linalg.def(
    "ExpM",
    [](const Tensor &Tin, const cytnx_float &a, const cytnx_float &b) {
      return cytnx::linalg::ExpM(Tin, a, b);
    },
    py::arg("Tin"), py::arg("a"), py::arg("b") = 0);
  m_linalg.def(
    "ExpM",
    [](const Tensor &Tin, const cytnx_uint64 &a, const cytnx_uint64 &b) {
      return cytnx::linalg::ExpM(Tin, a, b);
    },
    py::arg("Tin"), py::arg("a"), py::arg("b") = 0);
  m_linalg.def(
    "ExpM",
    [](const Tensor &Tin, const cytnx_int64 &a, const cytnx_int64 &b) {
      return cytnx::linalg::ExpM(Tin, a, b);
    },
    py::arg("Tin"), py::arg("a"), py::arg("b") = 0);
  m_linalg.def(
    "ExpM",
    [](const Tensor &Tin, const cytnx_uint32 &a, const cytnx_uint32 &b) {
      return cytnx::linalg::ExpM(Tin, a, b);
    },
    py::arg("Tin"), py::arg("a"), py::arg("b") = 0);
  m_linalg.def(
    "ExpM",
    [](const Tensor &Tin, const cytnx_int32 &a, const cytnx_int32 &b) {
      return cytnx::linalg::ExpM(Tin, a, b);
    },
    py::arg("Tin"), py::arg("a"), py::arg("b") = 0);

  m_linalg.def(
    "ExpM",
    [](const Tensor &Tin, const cytnx_uint16 &a, const cytnx_uint16 &b) {
      return cytnx::linalg::ExpM(Tin, a, b);
    },
    py::arg("Tin"), py::arg("a"), py::arg("b") = 0);
  m_linalg.def(
    "ExpM", [](const Tensor &Tin) { return cytnx::linalg::ExpM(Tin); }, py::arg("Tin"));

  m_linalg.def(
    "Qr",
    [](const cytnx::UniTensor &Tin, const bool &is_tau) { return cytnx::linalg::Qr(Tin, is_tau); },
    py::arg("Tio"), py::arg("is_tau") = false);
  m_linalg.def(
    "Qr",
    [](const cytnx::Tensor &Tin, const bool &is_tau) { return cytnx::linalg::Qr(Tin, is_tau); },
    py::arg("Tio"), py::arg("is_tau") = false);
  m_linalg.def(
    "Qdr",
    [](const cytnx::UniTensor &Tin, const bool &is_tau) { return cytnx::linalg::Qdr(Tin, is_tau); },
    py::arg("Tio"), py::arg("is_tau") = false);
  m_linalg.def(
    "Qdr",
    [](const cytnx::Tensor &Tin, const bool &is_tau) { return cytnx::linalg::Qdr(Tin, is_tau); },
    py::arg("Tio"), py::arg("is_tau") = false);

  // m_linalg.def("InvM", &cytnx::linalg::InvM, py::arg("Tin"));

  // m_linalg.def("InvM_", &cytnx::linalg::InvM_, py::arg("Tio"));
  m_linalg.def(
    "InvM_", [](cytnx::UniTensor &Tio) { cytnx::linalg::InvM_(Tio); }, py::arg("Tio"));
  m_linalg.def(
    "InvM_", [](cytnx::Tensor &Tio) { cytnx::linalg::InvM_(Tio); }, py::arg("Tio"));

  m_linalg.def(
    "InvM", [](cytnx::UniTensor &Tin) { return cytnx::linalg::InvM(Tin); }, py::arg("Tin"));
  m_linalg.def(
    "InvM", [](cytnx::Tensor &Tin) { return cytnx::linalg::InvM(Tin); }, py::arg("Tin"));

  m_linalg.def("Inv_", &cytnx::linalg::Inv_, py::arg("Tio"), py::arg("clip"));
  m_linalg.def("Inv", &cytnx::linalg::Inv, py::arg("Tio"), py::arg("clip"));

  m_linalg.def(
    "Conj", [](const cytnx::Tensor &Tin) { return cytnx::linalg::Conj(Tin); }, py::arg("Tin"));

  m_linalg.def(
    "Conj_", [](cytnx::Tensor &Tin) { cytnx::linalg::Conj_(Tin); }, py::arg("Tin"));

  m_linalg.def(
    "Conj", [](const cytnx::UniTensor &Tin) { return cytnx::linalg::Conj(Tin); }, py::arg("Tin"));

  m_linalg.def(
    "Conj_", [](cytnx::UniTensor &Tin) { cytnx::linalg::Conj_(Tin); }, py::arg("Tin"));

  m_linalg.def("Matmul", &cytnx::linalg::Matmul, py::arg("T1"), py::arg("T2"));
  m_linalg.def("Matmul_dg", &cytnx::linalg::Matmul_dg, py::arg("T1"), py::arg("T2"));
  m_linalg.def("Diag", &cytnx::linalg::Diag, py::arg("Tin"));
  m_linalg.def("Det", &cytnx::linalg::Det, py::arg("Tin"));
  m_linalg.def("Tensordot", &cytnx::linalg::Tensordot, py::arg("T1"), py::arg("T2"),
               py::arg("indices_1"), py::arg("indices_2"), py::arg("cacheL") = false,
               py::arg("cacheR") = false);
  m_linalg.def("Tensordot_dg", &cytnx::linalg::Tensordot_dg, py::arg("T1"), py::arg("T2"),
               py::arg("indices_1"), py::arg("indices_2"), py::arg("diag_L"));
  m_linalg.def("Outer", &cytnx::linalg::Outer, py::arg("T1"), py::arg("T2"));
  m_linalg.def("Kron", &cytnx::linalg::Kron, py::arg("T1"), py::arg("T2"),
               py::arg("Tl_pad_left") = false, py::arg("Tr_pad_left") = false);
  m_linalg.def("Vectordot", &cytnx::linalg::Vectordot, py::arg("T1"), py::arg("T2"),
               py::arg("is_conj") = false);
  m_linalg.def("Tridiag", &cytnx::linalg::Tridiag, py::arg("A"), py::arg("B"),
               py::arg("is_V") = true, py::arg("is_row") = false, py::arg("throw_excp") = false);

  // m_linalg.def("Norm", &cytnx::linalg::Norm, py::arg("T1") = cytnx::Tensor());
  // m_linalg.def("Norm", &cytnx::linalg::Norm, py::arg("T1") = cytnx::UniTensor());
  m_linalg.def(
    "Norm", [](cytnx::UniTensor &T1) { return cytnx::linalg::Norm(T1); }, py::arg("T1"));
  m_linalg.def(
    "Norm", [](cytnx::Tensor &T1) { return cytnx::linalg::Norm(T1); }, py::arg("T1"));

  m_linalg.def("Dot", &cytnx::linalg::Dot, py::arg("T1"), py::arg("T2"));
  m_linalg.def(
    "Axpy", [](const Scalar &a, const Tensor &x) { return cytnx::linalg::Axpy(a, x); },
    py::arg("a"), py::arg("x"));
  m_linalg.def(
    "Axpy",
    [](const Scalar &a, const Tensor &x, const Tensor &y) { return cytnx::linalg::Axpy(a, x, y); },
    py::arg("a"), py::arg("x"), py::arg("y"));

  m_linalg.def(
    "Ger",
    [](const Tensor &x, const Tensor &y, const Scalar &a) { return cytnx::linalg::Ger(x, y, a); },
    py::arg("x"), py::arg("y"), py::arg("a") = Scalar());

  m_linalg.def("Axpy_", &cytnx::linalg::Axpy_, py::arg("a"), py::arg("x"), py::arg("y"));

  m_linalg.def("Gemm_", &cytnx::linalg::Gemm_, py::arg("a"), py::arg("x"), py::arg("y"),
               py::arg("b"), py::arg("c"));

  m_linalg.def(
    "Gemm",
    [](const Scalar &a, const Tensor &x, const Tensor &y) { return cytnx::linalg::Gemm(a, x, y); },
    py::arg("a"), py::arg("x"), py::arg("y"));

  m_linalg.def(
    "Trace",
    [](const cytnx::UniTensor &Tin, const cytnx_int64 &a, const cytnx_int64 &b) {
      return cytnx::linalg::Trace(Tin, a, b);
    },
    py::arg("Tn"), py::arg("axisA") = 0, py::arg("axisB") = 1);

  m_linalg.def(
    "Trace",
    [](const cytnx::UniTensor &Tin, const std::string &a, const std::string &b) {
      return cytnx::linalg::Trace(Tin, a, b);
    },
    py::arg("Tn"), py::arg("axisA"), py::arg("axisB"));

  m_linalg.def(
    "Trace",
    [](const cytnx::Tensor &Tin, const cytnx_int64 &a, const cytnx_int64 &b) {
      return cytnx::linalg::Trace(Tin, a, b);
    },
    py::arg("Tn"), py::arg("axisA") = 0, py::arg("axisB") = 1);

  m_linalg.def(
    "Pow", [](const UniTensor &Tin, const double &p) { return cytnx::linalg::Pow(Tin, p); },
    py::arg("Tn"), py::arg("p"));
  m_linalg.def(
    "Pow", [](const Tensor &Tin, const double &p) { return cytnx::linalg::Pow(Tin, p); },
    py::arg("Tn"), py::arg("p"));

  m_linalg.def(
    "Pow_", [](UniTensor &Tin, const double &p) { cytnx::linalg::Pow_(Tin, p); }, py::arg("Tn"),
    py::arg("p"));
  m_linalg.def(
    "Pow_", [](Tensor &Tin, const double &p) { cytnx::linalg::Pow_(Tin, p); }, py::arg("Tn"),
    py::arg("p"));

  m_linalg.def("Abs", &cytnx::linalg::Abs, py::arg("Tn"));
  m_linalg.def("Abs_", &cytnx::linalg::Abs_, py::arg("Tn"));
  m_linalg.def("Max", &cytnx::linalg::Max, py::arg("Tn"));
  m_linalg.def("Min", &cytnx::linalg::Min, py::arg("Tn"));
  m_linalg.def("Sum", &cytnx::linalg::Sum, py::arg("Tn"));

  m_linalg.def(
    "Directsum",
    [](const Tensor &T1, const Tensor &T2, const std::vector<cytnx_uint64> &shared_axes) {
      return linalg::Directsum(T1, T2, shared_axes);
    },
    py::arg("T1"), py::arg("T2"), py::arg("shared_axes"));

  m_linalg.def(
    "Hosvd",
    [](const cytnx::Tensor &Tin, const std::vector<cytnx_uint64> &mode, const bool &is_core,
       const bool &is_Ls, const std::vector<cytnx_int64> &truncate_dim) {
      return cytnx::linalg::Hosvd(Tin, mode, is_core, is_Ls, truncate_dim);
    },
    py::arg("Tn"), py::arg("mode"), py::arg("is_core") = true, py::arg("is_Ls") = false,
    py::arg("truncate_dim") = std::vector<cytnx_int64>());
  m_linalg.def(
    "Hosvd",
    [](const cytnx::UniTensor &Tin, const std::vector<cytnx_uint64> &mode, const bool &is_core,
       const bool &is_Ls, const std::vector<cytnx_int64> &truncate_dim) {
      return cytnx::linalg::Hosvd(Tin, mode, is_core, is_Ls, truncate_dim);
    },
    py::arg("Tn"), py::arg("mode"), py::arg("is_core") = true, py::arg("is_Ls") = false,
    py::arg("truncate_dim") = std::vector<cytnx_int64>());

  m_linalg.def(
    "Arnoldi",
    [](LinOp *Hop, const Tensor &Tin, const std::string which, const cytnx_uint64 &Maxiter,
       const double &CvgCrit, const cytnx_uint64 &k, const bool &is_V, const bool &verbose) {
      return cytnx::linalg::Arnoldi(Hop, Tin, which, Maxiter, CvgCrit, k, is_V, verbose);
    },
    py::arg("Hop"), py::arg("Tin"), py::arg("which") = "LM", py::arg("Maxiter") = 10000,
    py::arg("CvgCrit") = 1.0e-9, py::arg("k") = 1, py::arg("is_V") = true,
    py::arg("verbose") = false);
  m_linalg.def(
    "Arnoldi",
    [](LinOp *Hop, const UniTensor &Tin, const std::string which, const cytnx_uint64 &Maxiter,
       const double &CvgCrit, const cytnx_uint64 &k, const bool &is_V, const bool &verbose) {
      return cytnx::linalg::Arnoldi(Hop, Tin, which, Maxiter, CvgCrit, k, is_V, verbose);
    },
    py::arg("Hop"), py::arg("Tin"), py::arg("which") = "LM", py::arg("Maxiter") = 10000,
    py::arg("CvgCrit") = 1.0e-9, py::arg("k") = 1, py::arg("is_V") = true,
    py::arg("verbose") = false);

  m_linalg.def(
    "Lanczos",
    [](LinOp *Hop, const Tensor &Tin, const std::string method, const double &CvgCrit,
       const unsigned int &Maxiter, const cytnx_uint64 &k, const bool &is_V, const bool &is_row,
       const cytnx_uint32 &max_krydim, const bool &verbose) {
      return cytnx::linalg::Lanczos(Hop, Tin, method, CvgCrit, Maxiter, k, is_V, is_row, max_krydim,
                                    verbose);
    },
    py::arg("Hop"), py::arg("Tin"), py::arg("method"), py::arg("CvgCrit") = 1.0e-14,
    py::arg("Maxiter") = 10000, py::arg("k") = 1, py::arg("is_V") = true, py::arg("is_row") = false,
    py::arg("max_krydim") = 0, py::arg("verbose") = false);
  m_linalg.def(
    "Lanczos",
    [](LinOp *Hop, const UniTensor &Tin, const std::string method, const double &CvgCrit,
       const unsigned int &Maxiter, const cytnx_uint64 &k, const bool &is_V, const bool &is_row,
       const cytnx_uint32 &max_krydim, const bool &verbose) {
      return cytnx::linalg::Lanczos(Hop, Tin, method, CvgCrit, Maxiter, k, is_V, is_row, max_krydim,
                                    verbose);
    },
    py::arg("Hop"), py::arg("Tin"), py::arg("method"), py::arg("CvgCrit") = 1.0e-14,
    py::arg("Maxiter") = 10000, py::arg("k") = 1, py::arg("is_V") = true, py::arg("is_row") = false,
    py::arg("max_krydim") = 0, py::arg("verbose") = false);

  m_linalg.def(
    "Lanczos_Exp",
    [](LinOp *Hop, const UniTensor &v, const Scalar &tau, const double &CvgCrit,
       const unsigned int &Maxiter, const bool &verbose) {
      return cytnx::linalg::Lanczos_Exp(Hop, v, tau, CvgCrit, Maxiter, verbose);
    },
    py::arg("Hop"), py::arg("v"), py::arg("tau"), py::arg("CvgCrit") = 1.0e-14,
    py::arg("Maxiter") = 10000, py::arg("verbose") = false);

  m_linalg.def(
    "Lstsq",
    [](const Tensor &A, const Tensor &b, const float &rcond) {
      return cytnx::linalg::Lstsq(A, b, rcond);
    },
    py::arg("A"), py::arg("b"), py::arg("rcond") = float(-1));

  /*
  m_linalg.def("c_Lanczos_ER",
               [](LinOp *Hop, const cytnx_uint64 &k, const bool &is_V, const cytnx_uint64 &maxiter,
                  const double &CvgCrit, const bool &is_row, const Tensor &Tin,
                  const cytnx_uint32 &max_krydim, const bool &verbose) {
                 return cytnx::linalg::Lanczos_ER(Hop, k, is_V, maxiter, CvgCrit, is_row, Tin,
                                                  max_krydim, verbose);
               });

  m_linalg.def("c_Lanczos_Gnd",
               [](LinOp *Hop, const double &CvgCrit, const bool &is_V, const Tensor &Tin,
                  const bool &verbose, const cytnx_uint64 &maxiter) {
                 return cytnx::linalg::Lanczos_Gnd(Hop, CvgCrit, is_V, Tin, verbose, maxiter);
               });

  m_linalg.def("c_Lanczos_Gnd_Ut",
               [](LinOp *Hop, const UniTensor &Tin, const double &CvgCrit, const bool &is_V,
                  const bool &verbose, const cytnx_uint64 &maxiter) {
                 return cytnx::linalg::Lanczos_Gnd_Ut(Hop, Tin, CvgCrit, is_V, verbose, maxiter);
               });
  */
}

#endif
