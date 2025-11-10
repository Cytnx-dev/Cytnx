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

void scalar_binding(py::module &m) {
  py::class_<cytnx::Scalar>(m, "Scalar")
    .def(py::init<>())
    .def(py::init<const cytnx::cytnx_complex128 &>(), py::arg("a"))
    .def(py::init<const cytnx::cytnx_complex64 &>(), py::arg("a"))
    .def(py::init<const cytnx::cytnx_double &>(), py::arg("a"))
    .def(py::init<const cytnx::cytnx_float &>(), py::arg("a"))
    .def(py::init<const cytnx::cytnx_uint64 &>(), py::arg("a"))
    .def(py::init<const cytnx::cytnx_int64 &>(), py::arg("a"))
    .def(py::init<const cytnx::cytnx_uint32 &>(), py::arg("a"))
    .def(py::init<const cytnx::cytnx_int32 &>(), py::arg("a"))
    .def(py::init<const cytnx::cytnx_uint16 &>(), py::arg("a"))
    .def(py::init<const cytnx::cytnx_int16 &>(), py::arg("a"))
    .def(py::init<const cytnx::cytnx_bool &>(), py::arg("a"))
    .def(
      "__init__",
      [](Scalar &self, const py::numpy_scalar<std::complex<double>> value) {
        new (&self) Scalar(static_cast<cytnx::cytnx_complex128>(value));
      },
      py::arg("a"))
    .def(
      "__init__",
      [](Scalar &self, const py::numpy_scalar<std::complex<float>> value) {
        new (&self) Scalar(static_cast<cytnx::cytnx_complex64>(value));
      },
      py::arg("a"))
    .def(
      "__init__",
      [](Scalar &self, const py::numpy_scalar<double> value) {
        new (&self) Scalar(static_cast<cytnx::cytnx_double>(value));
      },
      py::arg("a"))
    .def(
      "__init__",
      [](Scalar &self, const py::numpy_scalar<float> value) {
        new (&self) Scalar(static_cast<cytnx::cytnx_float>(value));
      },
      py::arg("a"))
    .def(
      "__init__",
      [](Scalar &self, const py::numpy_scalar<int64_t> value) {
        new (&self) Scalar(static_cast<cytnx::cytnx_int64>(value));
      },
      py::arg("a"))
    .def(
      "__init__",
      [](Scalar &self, const py::numpy_scalar<uint64_t> value) {
        new (&self) Scalar(static_cast<cytnx::cytnx_uint64>(value));
      },
      py::arg("a"))
    .def(
      "__init__",
      [](Scalar &self, const py::numpy_scalar<int32_t> value) {
        new (&self) Scalar(static_cast<cytnx::cytnx_int32>(value));
      },
      py::arg("a"))
    .def(
      "__init__",
      [](Scalar &self, const py::numpy_scalar<uint32_t> value) {
        new (&self) Scalar(static_cast<cytnx::cytnx_uint32>(value));
      },
      py::arg("a"))
    .def(
      "__init__",
      [](Scalar &self, const py::numpy_scalar<int16_t> value) {
        new (&self) Scalar(static_cast<cytnx::cytnx_int16>(value));
      },
      py::arg("a"))
    .def(
      "__init__",
      [](Scalar &self, const py::numpy_scalar<uint16_t> value) {
        new (&self) Scalar(static_cast<cytnx::cytnx_uint16>(value));
      },
      py::arg("a"))
    .def(
      "__init__",
      [](Scalar &self, const py::numpy_scalar<bool> value) {
        new (&self) Scalar(static_cast<cytnx::cytnx_bool>(value));
      },
      py::arg("a"))

    // ---- Static methods ----
    .def_static("maxval", &Scalar::maxval)
    .def_static("minval", &Scalar::minval)

    // ---- Methods ----
    .def("astype", &Scalar::astype)
    .def("conj", &Scalar::conj)
    .def("real", &Scalar::real)
    .def("imag", &Scalar::imag)
    .def("abs", &Scalar::abs)
    .def("sqrt", &Scalar::sqrt)
    .def("iabs", &Scalar::iabs)
    .def("isqrt", &Scalar::isqrt)
    .def("dtype", &Scalar::dtype)
    .def("print", &Scalar::print)

    // ---- Arithmetic in-place ----
    .def("__iadd__",
         [](Scalar &self, const Scalar &rhs) -> Scalar & {
           self += rhs;
           return self;
         })
    .def("__isub__",
         [](Scalar &self, const Scalar &rhs) -> Scalar & {
           self -= rhs;
           return self;
         })
    .def("__imul__",
         [](Scalar &self, const Scalar &rhs) -> Scalar & {
           self *= rhs;
           return self;
         })
    .def("__itruediv__",
         [](Scalar &self, const Scalar &rhs) -> Scalar & {
           self /= rhs;
           return self;
         })

    // ---- Arithmetic binary ----
    .def(py::self + py::self)
    .def(py::self - py::self)
    .def(py::self * py::self)
    .def(py::self / py::self)

    // ---- Comparison ----
    .def(py::self == py::self)
    .def(py::self != py::self)
    .def(py::self < py::self)
    .def(py::self <= py::self)
    .def(py::self > py::self)
    .def(py::self >= py::self)

    // ---- Conversion operators ----
    .def("__float__", [](const Scalar &s) { return static_cast<cytnx_double>(s); })
    .def("__int__", [](const Scalar &s) { return static_cast<cytnx_int64>(s); })
    .def("__complex__",
         [](const Scalar &s) {
           cytnx_double re = static_cast<cytnx_double>(s.real());
           cytnx_double im = static_cast<cytnx_double>(s.imag());
           return std::complex<cytnx_double>(re, im);
         })

    // ---- String representation ----
    .def("__repr__",
         [](const Scalar &s) {
           std::ostringstream ss;
           ss << s;
           return ss.str();
         })

  // ============================================================================
  // numpy scalar arithmetic overloads
  // ============================================================================

  // Macro for all numpy scalar types (float32, int32, etc.)
  #define FOR_EACH_NUMPY_TYPE(OPNAME, OP)                                                  \
    .def("__" #OPNAME "__",                                                                \
         [](cytnx::Scalar &self, const py::numpy_scalar<std::complex<double>> &rhs) {      \
           return self OP static_cast<cytnx::cytnx_complex128>(rhs);                                    \
         })                                                                                \
      .def("__" #OPNAME "__",                                                              \
           [](cytnx::Scalar &self, const py::numpy_scalar<std::complex<float>> &rhs) {     \
             return self OP static_cast<cytnx::cytnx_complex64>(rhs);                                   \
           })                                                                              \
      .def("__" #OPNAME "__",                                                              \
           [](cytnx::Scalar &self, const py::numpy_scalar<double> &rhs) {                  \
             return self OP static_cast<cytnx::cytnx_double>(rhs);                                      \
           })                                                                              \
      .def("__" #OPNAME "__",                                                              \
           [](cytnx::Scalar &self, const py::numpy_scalar<float> &rhs) {                   \
             return self OP static_cast<cytnx::cytnx_float>(rhs);                                       \
           })                                                                              \
      .def("__" #OPNAME "__",                                                              \
           [](cytnx::Scalar &self, const py::numpy_scalar<int64_t> &rhs) {                 \
             return self OP static_cast<cytnx::cytnx_int64>(rhs);                                       \
           })                                                                              \
      .def("__" #OPNAME "__",                                                              \
           [](cytnx::Scalar &self, const py::numpy_scalar<uint64_t> &rhs) {                \
             return self OP static_cast<cytnx::cytnx_uint64>(rhs);                                      \
           })                                                                              \
      .def("__" #OPNAME "__",                                                              \
           [](cytnx::Scalar &self, const py::numpy_scalar<int32_t> &rhs) {                 \
             return self OP static_cast<cytnx::cytnx_int32>(rhs);                                       \
           })                                                                              \
      .def("__" #OPNAME "__",                                                              \
           [](cytnx::Scalar &self, const py::numpy_scalar<uint32_t> &rhs) {                \
             return self OP static_cast<cytnx::cytnx_uint32>(rhs);                                      \
           })                                                                              \
      .def("__" #OPNAME "__",                                                              \
           [](cytnx::Scalar &self, const py::numpy_scalar<int16_t> &rhs) {                 \
             return self OP static_cast<cytnx::cytnx_int16>(rhs);                                       \
           })                                                                              \
      .def("__" #OPNAME "__",                                                              \
           [](cytnx::Scalar &self, const py::numpy_scalar<uint16_t> &rhs) {                \
             return self OP static_cast<cytnx::cytnx_uint16>(rhs);                                      \
           })                                                                              \
      .def("__" #OPNAME "__", [](cytnx::Scalar &self, const py::numpy_scalar<bool> &rhs) { \
        return self OP static_cast<cytnx::cytnx_bool>(rhs);                                                          \
      })

  // does not give the right type because of type casting to python objects before handling to
  // pybind
  #define FOR_EACH_NUMPY_RTYPE(OPNAME, OP)                                                  \
    .def("__r" #OPNAME "__",                                                                \
         [](cytnx::Scalar &self, const py::numpy_scalar<std::complex<double>> &lhs) {       \
           return (cytnx::Scalar(static_cast<cytnx::cytnx_complex128>(lhs)))OP self;                     \
         })                                                                                 \
      .def("__r" #OPNAME "__",                                                              \
           [](cytnx::Scalar &self, const py::numpy_scalar<std::complex<float>> &lhs) {      \
             return (cytnx::Scalar(static_cast<cytnx::cytnx_complex64>(lhs)))OP self;                    \
           })                                                                               \
      .def("__r" #OPNAME "__",                                                              \
           [](cytnx::Scalar &self, const py::numpy_scalar<double> &lhs) {                   \
             return (cytnx::Scalar(static_cast<cytnx::cytnx_double>(lhs)))OP self;                       \
           })                                                                               \
      .def("__r" #OPNAME "__",                                                              \
           [](cytnx::Scalar &self, const py::numpy_scalar<float> &lhs) {                    \
             return (cytnx::Scalar(static_cast<cytnx::cytnx_float>(lhs)))OP self;                        \
           })                                                                               \
      .def("__r" #OPNAME "__",                                                              \
           [](cytnx::Scalar &self, const py::numpy_scalar<int64_t> &lhs) {                  \
             return (cytnx::Scalar(static_cast<cytnx::cytnx_int64>(lhs)))OP self;                        \
           })                                                                               \
      .def("__r" #OPNAME "__",                                                              \
           [](cytnx::Scalar &self, const py::numpy_scalar<uint64_t> &lhs) {                 \
             return (cytnx::Scalar(static_cast<cytnx::cytnx_uint64>(lhs)))OP self;                       \
           })                                                                               \
      .def("__r" #OPNAME "__",                                                              \
           [](cytnx::Scalar &self, const py::numpy_scalar<int32_t> &lhs) {                  \
             return (cytnx::Scalar(static_cast<cytnx::cytnx_int32>(lhs)))OP self;                        \
           })                                                                               \
      .def("__r" #OPNAME "__",                                                              \
           [](cytnx::Scalar &self, const py::numpy_scalar<uint32_t> &lhs) {                 \
             return (cytnx::Scalar(static_cast<cytnx::cytnx_uint32>(lhs)))OP self;                       \
           })                                                                               \
      .def("__r" #OPNAME "__",                                                              \
           [](cytnx::Scalar &self, const py::numpy_scalar<int16_t> &lhs) {                  \
             return (cytnx::Scalar(static_cast<cytnx::cytnx_int16>(lhs)))OP self;                        \
           })                                                                               \
      .def("__r" #OPNAME "__",                                                              \
           [](cytnx::Scalar &self, const py::numpy_scalar<uint16_t> &lhs) {                 \
             return (cytnx::Scalar(static_cast<cytnx::cytnx_uint16>(lhs)))OP self;                       \
           })                                                                               \
      .def("__r" #OPNAME "__", [](cytnx::Scalar &self, const py::numpy_scalar<bool> &lhs) { \
        return (cytnx::Scalar(static_cast<cytnx::cytnx_bool>(lhs)))OP self;                                           \
      })

  #define FOR_EACH_NUMPY_ITYPE(OPNAME, OP)                                                     \
    .def("__i" #OPNAME "__",                                                                   \
         [](cytnx::Scalar &self,                                                               \
            const py::numpy_scalar<std::complex<double>> &rhs) -> cytnx::Scalar & {            \
           self OP static_cast<cytnx::cytnx_complex128>(rhs);                                               \
           return self;                                                                        \
         })                                                                                    \
      .def("__i" #OPNAME "__",                                                                 \
           [](cytnx::Scalar &self,                                                             \
              const py::numpy_scalar<std::complex<float>> &rhs) -> cytnx::Scalar & {           \
             self OP static_cast<cytnx::cytnx_complex64>(rhs);                                              \
             return self;                                                                      \
           })                                                                                  \
      .def("__i" #OPNAME "__",                                                                 \
           [](cytnx::Scalar &self, const py::numpy_scalar<double> &rhs) -> cytnx::Scalar & {   \
             self OP static_cast<cytnx::cytnx_double>(rhs);                                                 \
             return self;                                                                      \
           })                                                                                  \
      .def("__i" #OPNAME "__",                                                                 \
           [](cytnx::Scalar &self, const py::numpy_scalar<float> &rhs) -> cytnx::Scalar & {    \
             self OP static_cast<cytnx::cytnx_float>(rhs);                                                  \
             return self;                                                                      \
           })                                                                                  \
      .def("__i" #OPNAME "__",                                                                 \
           [](cytnx::Scalar &self, const py::numpy_scalar<int64_t> &rhs) -> cytnx::Scalar & {  \
             self OP static_cast<cytnx::cytnx_int64>(rhs);                                                  \
             return self;                                                                      \
           })                                                                                  \
      .def("__i" #OPNAME "__",                                                                 \
           [](cytnx::Scalar &self, const py::numpy_scalar<uint64_t> &rhs) -> cytnx::Scalar & { \
             self OP static_cast<cytnx::cytnx_uint64>(rhs);                                                 \
             return self;                                                                      \
           })                                                                                  \
      .def("__i" #OPNAME "__",                                                                 \
           [](cytnx::Scalar &self, const py::numpy_scalar<int32_t> &rhs) -> cytnx::Scalar & {  \
             self OP static_cast<cytnx::cytnx_int32>(rhs);                                                  \
             return self;                                                                      \
           })                                                                                  \
      .def("__i" #OPNAME "__",                                                                 \
           [](cytnx::Scalar &self, const py::numpy_scalar<uint32_t> &rhs) -> cytnx::Scalar & { \
             self OP static_cast<cytnx::cytnx_uint32>(rhs);                                                 \
             return self;                                                                      \
           })                                                                                  \
      .def("__i" #OPNAME "__",                                                                 \
           [](cytnx::Scalar &self, const py::numpy_scalar<int16_t> &rhs) -> cytnx::Scalar & {  \
             self OP static_cast<cytnx::cytnx_int16>(rhs);                                                  \
             return self;                                                                      \
           })                                                                                  \
      .def("__i" #OPNAME "__",                                                                 \
           [](cytnx::Scalar &self, const py::numpy_scalar<uint16_t> &rhs) -> cytnx::Scalar & { \
             self OP static_cast<cytnx::cytnx_uint16>(rhs);                                                 \
             return self;                                                                      \
           })                                                                                  \
      .def("__i" #OPNAME "__",                                                                 \
           [](cytnx::Scalar &self, const py::numpy_scalar<bool> &rhs) -> cytnx::Scalar & {     \
             self OP static_cast<cytnx::cytnx_bool>(rhs);                                                                \
             return self;                                                                      \
           })

    // Apply macros for all operators
    FOR_EACH_NUMPY_TYPE(add, +) FOR_EACH_NUMPY_TYPE(sub, -) FOR_EACH_NUMPY_TYPE(mul, *)
      FOR_EACH_NUMPY_TYPE(truediv, /)

        FOR_EACH_NUMPY_RTYPE(add, +) FOR_EACH_NUMPY_RTYPE(sub, -) FOR_EACH_NUMPY_RTYPE(mul, *)
          FOR_EACH_NUMPY_RTYPE(truediv, /)

            FOR_EACH_NUMPY_ITYPE(add, +=) FOR_EACH_NUMPY_ITYPE(sub, -=)
              FOR_EACH_NUMPY_ITYPE(mul, *=) FOR_EACH_NUMPY_ITYPE(truediv, /=)

    ;  // end of object line

  py::implicitly_convertible<cytnx::cytnx_double, cytnx::Scalar>();
  py::implicitly_convertible<cytnx::cytnx_float, cytnx::Scalar>();
  py::implicitly_convertible<cytnx::cytnx_complex128, cytnx::Scalar>();
  py::implicitly_convertible<cytnx::cytnx_complex64, cytnx::Scalar>();
  py::implicitly_convertible<cytnx::cytnx_uint64, cytnx::Scalar>();
  py::implicitly_convertible<cytnx::cytnx_int64, cytnx::Scalar>();
  py::implicitly_convertible<cytnx::cytnx_uint32, cytnx::Scalar>();
  py::implicitly_convertible<cytnx::cytnx_int32, cytnx::Scalar>();
  py::implicitly_convertible<cytnx::cytnx_uint16, cytnx::Scalar>();
  py::implicitly_convertible<cytnx::cytnx_int16, cytnx::Scalar>();
  py::implicitly_convertible<cytnx::cytnx_bool, cytnx::Scalar>();
}

#endif
