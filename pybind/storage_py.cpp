#include <map>
#include <random>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/buffer_info.h>
#include <pybind11/functional.h>

#include "cytnx.hpp"
#include "pyint_dispatch.hpp"
// #include "../include/cytnx_error.hpp"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace cytnx;
using pybind_cytnx::dispatch_pyint;

#ifdef BACKEND_TORCH
#else

void storage_binding(py::module &m) {
  py::class_<cytnx::Storage>(m, "Storage")
    .def("numpy",
         [](Storage &self) -> py::array {
           // device on GPU? move to cpu:ref it;
           Storage tmpIN;
           if (self.device() >= 0) {
             tmpIN = self.to(Device.cpu);
           } else {
             tmpIN = self.clone();
           }

           // calculate stride:
           std::size_t type_size = Type.typeSize(tmpIN.dtype());
           std::vector<ssize_t> stride(1, type_size);
           std::vector<ssize_t> shape(1, tmpIN.size());

           py::buffer_info npbuf;
           std::string chr_dtype;
           if (tmpIN.dtype() == Type.ComplexDouble) {
             chr_dtype = py::format_descriptor<cytnx_complex128>::format();
           } else if (tmpIN.dtype() == Type.ComplexFloat) {
             chr_dtype = py::format_descriptor<cytnx_complex64>::format();
           } else if (tmpIN.dtype() == Type.Double) {
             chr_dtype = py::format_descriptor<cytnx_double>::format();
           } else if (tmpIN.dtype() == Type.Float) {
             chr_dtype = py::format_descriptor<cytnx_float>::format();
           } else if (tmpIN.dtype() == Type.Uint64) {
             chr_dtype = py::format_descriptor<cytnx_uint64>::format();
           } else if (tmpIN.dtype() == Type.Int64) {
             chr_dtype = py::format_descriptor<cytnx_int64>::format();
           } else if (tmpIN.dtype() == Type.Uint32) {
             chr_dtype = py::format_descriptor<cytnx_uint32>::format();
           } else if (tmpIN.dtype() == Type.Int32) {
             chr_dtype = py::format_descriptor<cytnx_int32>::format();
           } else if (tmpIN.dtype() == Type.Bool) {
             chr_dtype = py::format_descriptor<cytnx_bool>::format();
           } else {
             cytnx_error_msg(true, "[ERROR] Void Type Tensor cannot convert to numpy ndarray%s",
                             "\n");
           }

           // Call `.release()` to avoid the memory passed to numpy being freed.
           npbuf = py::buffer_info(tmpIN.release(), type_size,
                                   chr_dtype,  // pss format
                                   /* rank= */ 1, shape, stride);
           return py::array(npbuf);
         })

    // construction
    .def(py::init<>())
    .def(py::init<const cytnx::Storage &>())
    .def(py::init<boost::intrusive_ptr<cytnx::Storage_base>>())
    .def(py::init<const unsigned long long &, const unsigned int &, int, const bool &>(),
         py::arg("size"), py::arg("dtype") = (cytnx_uint64)Type.Double, py::arg("device") = -1,
         py::arg("init_zero") = true)
    .def("Init", &cytnx::Storage::Init, py::arg("size"),
         py::arg("dtype") = (cytnx_uint64)Type.Double, py::arg("device") = -1,
         py::arg("init_zero") = true)

    .def("dtype", &cytnx::Storage::dtype)
    .def("dtype_str", &cytnx::Storage::dtype_str)
    .def("device", &cytnx::Storage::device)
    .def("device_str", &cytnx::Storage::device_str)

    //[note] this is an interesting binding, since we want if new_type==self.dtype() to return self,
    //       the pybind cannot handle this. The direct binding will make a "new" instance in terms
    //       of python's consideration. The solution is to move the definition into python side.
    //       (see cytnx/Storage_conti.py)
    //.def("astype", &cytnx::Storage::astype,py::arg("new_type"))
    .def(
      "astype_different_type",
      [](cytnx::Storage &self, const cytnx_uint64 &new_type) {
        cytnx_error_msg(self.dtype() == new_type,
                        "[ERROR][pybind][astype_diffferent_type] same type for astype() should be "
                        "handle in python side.%s",
                        "\n");
        return self.astype(new_type);
      },
      py::arg("new_type"))

    .def("__getitem__",
         [](cytnx::Storage &self, const unsigned long long &idx) {
           cytnx_error_msg(idx > self.size(), "idx exceed the size of storage.%s", "\n");
           py::object out;
           if (self.dtype() == cytnx::Type.Double)
             out = py::cast(self.at<cytnx::cytnx_double>(idx));
           else if (self.dtype() == cytnx::Type.Float)
             out = py::cast(self.at<cytnx::cytnx_float>(idx));
           else if (self.dtype() == cytnx::Type.ComplexDouble)
             out = py::cast(self.at<cytnx::cytnx_complex128>(idx));
           else if (self.dtype() == cytnx::Type.ComplexFloat)
             out = py::cast(self.at<cytnx::cytnx_complex64>(idx));
           else if (self.dtype() == cytnx::Type.Uint64)
             out = py::cast(self.at<cytnx::cytnx_uint64>(idx));
           else if (self.dtype() == cytnx::Type.Int64)
             out = py::cast(self.at<cytnx::cytnx_int64>(idx));
           else if (self.dtype() == cytnx::Type.Uint32)
             out = py::cast(self.at<cytnx::cytnx_uint32>(idx));
           else if (self.dtype() == cytnx::Type.Int32)
             out = py::cast(self.at<cytnx::cytnx_int32>(idx));
           else if (self.dtype() == cytnx::Type.Uint16)
             out = py::cast(self.at<cytnx::cytnx_uint16>(idx));
           else if (self.dtype() == cytnx::Type.Int16)
             out = py::cast(self.at<cytnx::cytnx_int16>(idx));
           else if (self.dtype() == cytnx::Type.Bool)
             out = py::cast(self.at<cytnx::cytnx_bool>(idx));
           else
             cytnx_error_msg(true, "%s", "[ERROR] try to get element from a void Storage.");

           return out;
         })
    .def("__setitem__",
         [](cytnx::Storage &self, const unsigned long long &idx, py::object in) {
           cytnx_error_msg(idx > self.size(), "idx exceed the size of storage.%s", "\n");
           py::object out;
           if (self.dtype() == cytnx::Type.Double)
             self.at<cytnx::cytnx_double>(idx) = in.cast<cytnx::cytnx_double>();
           else if (self.dtype() == cytnx::Type.Float)
             self.at<cytnx::cytnx_float>(idx) = in.cast<cytnx::cytnx_float>();
           else if (self.dtype() == cytnx::Type.ComplexDouble)
             self.at<cytnx::cytnx_complex128>(idx) = in.cast<cytnx::cytnx_complex128>();
           else if (self.dtype() == cytnx::Type.ComplexFloat)
             self.at<cytnx::cytnx_complex64>(idx) = in.cast<cytnx::cytnx_complex64>();
           else if (self.dtype() == cytnx::Type.Uint64)
             self.at<cytnx::cytnx_uint64>(idx) = in.cast<cytnx::cytnx_uint64>();
           else if (self.dtype() == cytnx::Type.Int64)
             self.at<cytnx::cytnx_int64>(idx) = in.cast<cytnx::cytnx_int64>();
           else if (self.dtype() == cytnx::Type.Uint32)
             self.at<cytnx::cytnx_uint32>(idx) = in.cast<cytnx::cytnx_uint32>();
           else if (self.dtype() == cytnx::Type.Int32)
             self.at<cytnx::cytnx_int32>(idx) = in.cast<cytnx::cytnx_int32>();
           else if (self.dtype() == cytnx::Type.Uint16)
             self.at<cytnx::cytnx_uint16>(idx) = in.cast<cytnx::cytnx_uint16>();
           else if (self.dtype() == cytnx::Type.Int16)
             self.at<cytnx::cytnx_int16>(idx) = in.cast<cytnx::cytnx_int16>();
           else if (self.dtype() == cytnx::Type.Bool)
             self.at<cytnx::cytnx_bool>(idx) = in.cast<cytnx::cytnx_bool>();
           else
             cytnx_error_msg(true, "%s", "[ERROR] try to get element from a void Storage.");
         })

    .def(
      "__repr__",
      [](cytnx::Storage &self) -> std::string {
        std::cout << self << std::endl;
        return std::string("");
      },
      py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
    .def("__len__", [](cytnx::Storage &self) -> cytnx::cytnx_uint64 { return self.size(); })

    .def("to_", &cytnx::Storage::to_, py::arg("device"))

    // handle same device from cytnx/Storage_conti.py
    .def(
      "to_different_device",
      [](cytnx::Storage &self, const cytnx_int64 &device) {
        cytnx_error_msg(self.device() == device,
                        "[ERROR][pybind][to_diffferent_device] same device for to() should be "
                        "handle in python side.%s",
                        "\n");
        return self.to(device);
      },
      py::arg("device"))

    .def("resize", &cytnx::Storage::resize)
    .def("capacity", &cytnx::Storage::capacity)
    .def("clone", &cytnx::Storage::clone)
    .def("__copy__", &cytnx::Storage::clone)
    .def("__deepcopy__", &cytnx::Storage::clone)
    .def("size", &cytnx::Storage::size)
    .def("print_info", &cytnx::Storage::print_info,
         py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
    .def("set_zeros", &cytnx::Storage::set_zeros)
    .def("__eq__",
         [](cytnx::Storage &self, const cytnx::Storage &rhs) -> bool { return self == rhs; })

    // keep-set; registration ORDER matters -- see "KEEP-SET ORDERING" in pybind/pyint_dispatch.hpp.
    .def(
      "fill",
      [](Storage &self, const py::numpy_scalar<float> &val) {
        self.fill(static_cast<cytnx_float>(val));
      },
      py::arg("val"))
    .def(
      "fill",
      [](Storage &self, const py::numpy_scalar<std::complex<float>> &val) {
        self.fill(static_cast<cytnx_complex64>(val));
      },
      py::arg("val"))
    .def(
      "fill",
      [](Storage &self, const py::numpy_scalar<int64_t> &val) {
        self.fill(static_cast<cytnx_int64>(val));
      },
      py::arg("val"))
    .def(
      "fill",
      [](Storage &self, const py::numpy_scalar<uint64_t> &val) {
        self.fill(static_cast<cytnx_uint64>(val));
      },
      py::arg("val"))
    .def(
      "fill",
      [](Storage &self, const py::numpy_scalar<int32_t> &val) {
        self.fill(static_cast<cytnx_int32>(val));
      },
      py::arg("val"))
    .def(
      "fill",
      [](Storage &self, const py::numpy_scalar<uint32_t> &val) {
        self.fill(static_cast<cytnx_uint32>(val));
      },
      py::arg("val"))
    .def(
      "fill",
      [](Storage &self, const py::numpy_scalar<int16_t> &val) {
        self.fill(static_cast<cytnx_int16>(val));
      },
      py::arg("val"))
    .def(
      "fill",
      [](Storage &self, const py::numpy_scalar<uint16_t> &val) {
        self.fill(static_cast<cytnx_uint16>(val));
      },
      py::arg("val"))
    .def(
      "fill",
      [](Storage &self, const py::numpy_scalar<bool> &val) {
        self.fill(static_cast<cytnx_bool>(val));
      },
      py::arg("val"))
    .def(
      "fill",
      [](Storage &self, const py::int_ &val) {
        dispatch_pyint(val, [&](auto v) { self.fill(v); });
      },
      py::arg("val"))
    .def(
      "fill", [](Storage &self, const cytnx_double &val) { self.fill(val); }, py::arg("val"))
    .def(
      "fill", [](Storage &self, const cytnx_complex128 &val) { self.fill(val); }, py::arg("val"))

    // keep-set; registration ORDER matters -- see "KEEP-SET ORDERING" in pybind/pyint_dispatch.hpp.
    .def(
      "append",
      [](Storage &self, const py::numpy_scalar<float> &val) {
        self.append(static_cast<cytnx_float>(val));
      },
      py::arg("val"))
    .def(
      "append",
      [](Storage &self, const py::numpy_scalar<std::complex<float>> &val) {
        self.append(static_cast<cytnx_complex64>(val));
      },
      py::arg("val"))
    .def(
      "append",
      [](Storage &self, const py::numpy_scalar<int64_t> &val) {
        self.append(static_cast<cytnx_int64>(val));
      },
      py::arg("val"))
    .def(
      "append",
      [](Storage &self, const py::numpy_scalar<uint64_t> &val) {
        self.append(static_cast<cytnx_uint64>(val));
      },
      py::arg("val"))
    .def(
      "append",
      [](Storage &self, const py::numpy_scalar<int32_t> &val) {
        self.append(static_cast<cytnx_int32>(val));
      },
      py::arg("val"))
    .def(
      "append",
      [](Storage &self, const py::numpy_scalar<uint32_t> &val) {
        self.append(static_cast<cytnx_uint32>(val));
      },
      py::arg("val"))
    .def(
      "append",
      [](Storage &self, const py::numpy_scalar<int16_t> &val) {
        self.append(static_cast<cytnx_int16>(val));
      },
      py::arg("val"))
    .def(
      "append",
      [](Storage &self, const py::numpy_scalar<uint16_t> &val) {
        self.append(static_cast<cytnx_uint16>(val));
      },
      py::arg("val"))
    .def(
      "append",
      [](Storage &self, const py::numpy_scalar<bool> &val) {
        self.append(static_cast<cytnx_bool>(val));
      },
      py::arg("val"))
    .def(
      "append",
      [](Storage &self, const py::int_ &val) {
        dispatch_pyint(val, [&](auto v) { self.append(v); });
      },
      py::arg("val"))
    .def(
      "append", [](Storage &self, const cytnx_double &val) { self.append(val); }, py::arg("val"))
    .def(
      "append", [](Storage &self, const cytnx_complex128 &val) { self.append(val); },
      py::arg("val"))

    // keep-set; registration ORDER matters -- see "KEEP-SET ORDERING" in
    // pybind/pyint_dispatch.hpp. complex64/float/{u,}int{32,16} are covered:
    // a plain Python list's elements match the complex128/double/uint64/
    // int64 casters in pybind11's first (no-convert) overload-resolution
    // pass regardless of registration order, so the narrower-width
    // overloads are unreachable duplicates (confirmed empirically:
    // from_pylist([1.5]) -> Double, from_pylist([1+2j]) -> ComplexDouble).
    // bool is registered first: Python bool implements __index__ (via int),
    // so typing considers Sequence[bool] a subtype of the SupportsFloat/
    // SupportsComplex unions double/complex128 render as, and this
    // ordering is also required for correctness -- from_pylist([True,
    // False]) must produce a Bool-dtype Storage, not Uint64. double is
    // registered before complex128 for the same covariance reason
    // (Sequence[SupportsFloat | SupportsIndex] is a subtype of
    // Sequence[SupportsComplex | SupportsFloat | SupportsIndex]). The
    // plain-int case is a single py::sequence overload below: int64 wins
    // unless any element needs uint64's range, matching dispatch_pyint's
    // int64-preferred convention used by every other keep-set in this
    // codebase.
    //
    // Empty-list default: an empty Python list has no elements for any
    // vector<T> caster to check, so pybind11's no-convert pass accepts it
    // for every from_pylist overload equally and the FIRST-registered one
    // wins regardless of T -- from_pylist([]) is at the mercy of
    // registration order alone, and now defaults to ComplexFloat rather
    // than the ComplexDouble it defaulted to before this keep-set (numpy
    // integer/bool scalars must be checked ahead of the plain Bool/py::int_
    // overloads for correctness, and complex128 must stay registered LAST
    // -- see the note below -- so there is no ordering that gives every
    // non-empty list its correct dtype AND keeps complex128 first for the
    // empty case; moving it first was tried and reverted, since mypy has
    // no notion of pybind11's runtime no-convert/convert passes and treats
    // an earlier "complex" stub signature as unconditionally shadowing
    // every later, narrower one, regardless of what pybind11 actually does
    // at runtime -- reintroducing the exact overload-cannot-match class of
    // bug this PR exists to fix). from_pylist([]) with an explicit
    // ComplexDouble intent should use from_pylist([], device) plus a
    // separate astype(), or Storage(0, Type.ComplexDouble) directly.
    //
    // numpy_scalar block: numpy integer/bool scalars are not subclasses of
    // Python int/bool (unlike np.float64/np.complex128, which are Python
    // float/complex subclasses and so already match the double/complex128
    // overloads below), so without a dedicated overload here a list of them
    // matches nothing in pybind11's no-convert pass and falls through to
    // the convert pass, where the plain Bool overload's truthiness-based
    // conversion silently accepts any object as True/False -- e.g.
    // from_pylist([np.int32(2)]) produced a Bool storage instead of
    // raising or dispatching to Int32/Int64. Registering these ahead of
    // the plain Bool overload (matching every other keep-set's numpy_scalar-
    // first ordering) lets them win in the no-convert pass instead.
    .def_static(
      "from_pylist",
      [](const std::vector<py::numpy_scalar<std::complex<float>>> &pylist, const int &device) {
        std::vector<cytnx_complex64> vals;
        vals.reserve(pylist.size());
        for (const auto &v : pylist) vals.push_back(static_cast<cytnx_complex64>(v));
        return cytnx::Storage::from_vector<cytnx_complex64>(vals, device);
      },
      py::arg("pylist"), py::arg("device") = (int)cytnx::Device.cpu)
    .def_static(
      "from_pylist",
      [](const std::vector<py::numpy_scalar<float>> &pylist, const int &device) {
        std::vector<cytnx_float> vals;
        vals.reserve(pylist.size());
        for (const auto &v : pylist) vals.push_back(static_cast<cytnx_float>(v));
        return cytnx::Storage::from_vector<cytnx_float>(vals, device);
      },
      py::arg("pylist"), py::arg("device") = (int)cytnx::Device.cpu)
    .def_static(
      "from_pylist",
      [](const std::vector<py::numpy_scalar<int64_t>> &pylist, const int &device) {
        std::vector<cytnx_int64> vals;
        vals.reserve(pylist.size());
        for (const auto &v : pylist) vals.push_back(static_cast<cytnx_int64>(v));
        return cytnx::Storage::from_vector<cytnx_int64>(vals, device);
      },
      py::arg("pylist"), py::arg("device") = (int)cytnx::Device.cpu)
    .def_static(
      "from_pylist",
      [](const std::vector<py::numpy_scalar<uint64_t>> &pylist, const int &device) {
        std::vector<cytnx_uint64> vals;
        vals.reserve(pylist.size());
        for (const auto &v : pylist) vals.push_back(static_cast<cytnx_uint64>(v));
        return cytnx::Storage::from_vector<cytnx_uint64>(vals, device);
      },
      py::arg("pylist"), py::arg("device") = (int)cytnx::Device.cpu)
    .def_static(
      "from_pylist",
      [](const std::vector<py::numpy_scalar<int32_t>> &pylist, const int &device) {
        std::vector<cytnx_int32> vals;
        vals.reserve(pylist.size());
        for (const auto &v : pylist) vals.push_back(static_cast<cytnx_int32>(v));
        return cytnx::Storage::from_vector<cytnx_int32>(vals, device);
      },
      py::arg("pylist"), py::arg("device") = (int)cytnx::Device.cpu)
    .def_static(
      "from_pylist",
      [](const std::vector<py::numpy_scalar<uint32_t>> &pylist, const int &device) {
        std::vector<cytnx_uint32> vals;
        vals.reserve(pylist.size());
        for (const auto &v : pylist) vals.push_back(static_cast<cytnx_uint32>(v));
        return cytnx::Storage::from_vector<cytnx_uint32>(vals, device);
      },
      py::arg("pylist"), py::arg("device") = (int)cytnx::Device.cpu)
    .def_static(
      "from_pylist",
      [](const std::vector<py::numpy_scalar<int16_t>> &pylist, const int &device) {
        std::vector<cytnx_int16> vals;
        vals.reserve(pylist.size());
        for (const auto &v : pylist) vals.push_back(static_cast<cytnx_int16>(v));
        return cytnx::Storage::from_vector<cytnx_int16>(vals, device);
      },
      py::arg("pylist"), py::arg("device") = (int)cytnx::Device.cpu)
    .def_static(
      "from_pylist",
      [](const std::vector<py::numpy_scalar<uint16_t>> &pylist, const int &device) {
        std::vector<cytnx_uint16> vals;
        vals.reserve(pylist.size());
        for (const auto &v : pylist) vals.push_back(static_cast<cytnx_uint16>(v));
        return cytnx::Storage::from_vector<cytnx_uint16>(vals, device);
      },
      py::arg("pylist"), py::arg("device") = (int)cytnx::Device.cpu)
    // cytnx has no Int8/Uint8 dtype, so these widen to the narrowest integer
    // dtype cytnx does have (Int16/Uint16) rather than falling through to
    // the double overload above, matching Scalar's constructor fix (#1053).
    .def_static(
      "from_pylist",
      [](const std::vector<py::numpy_scalar<int8_t>> &pylist, const int &device) {
        std::vector<cytnx_int16> vals;
        vals.reserve(pylist.size());
        for (const auto &v : pylist) vals.push_back(static_cast<cytnx_int16>(static_cast<int8_t>(v)));
        return cytnx::Storage::from_vector<cytnx_int16>(vals, device);
      },
      py::arg("pylist"), py::arg("device") = (int)cytnx::Device.cpu)
    .def_static(
      "from_pylist",
      [](const std::vector<py::numpy_scalar<uint8_t>> &pylist, const int &device) {
        std::vector<cytnx_uint16> vals;
        vals.reserve(pylist.size());
        for (const auto &v : pylist)
          vals.push_back(static_cast<cytnx_uint16>(static_cast<uint8_t>(v)));
        return cytnx::Storage::from_vector<cytnx_uint16>(vals, device);
      },
      py::arg("pylist"), py::arg("device") = (int)cytnx::Device.cpu)
    .def_static(
      "from_pylist",
      [](const std::vector<py::numpy_scalar<bool>> &pylist, const int &device) {
        std::vector<cytnx_bool> vals;
        vals.reserve(pylist.size());
        for (const auto &v : pylist) vals.push_back(static_cast<cytnx_bool>(v));
        return cytnx::Storage::from_vector<cytnx_bool>(vals, device);
      },
      py::arg("pylist"), py::arg("device") = (int)cytnx::Device.cpu)
    // Plain Bool overload: requires every element to already be an exact
    // Python bool (py::bool_'s caster does a strict isinstance check with
    // no truthiness-based convert-pass fallback, unlike the arithmetic
    // cytnx_bool/C++-bool caster from_vector<cytnx_bool> would get if bound
    // directly) -- see the numpy_scalar block above for why the lenient
    // fallback is unsafe.
    .def_static(
      "from_pylist",
      [](const std::vector<py::bool_> &pylist, const int &device) {
        std::vector<cytnx_bool> vals;
        vals.reserve(pylist.size());
        for (const py::bool_ &v : pylist) vals.push_back(v.cast<cytnx_bool>());
        return cytnx::Storage::from_vector<cytnx_bool>(vals, device);
      },
      py::arg("pylist"), py::arg("device") = (int)cytnx::Device.cpu)
    .def_static(
      "from_pylist",
      [](const std::vector<py::int_> &pylist, const int &device) {
        bool needs_uint64 = false;
        for (const py::int_ &v : pylist) {
          int overflow = 0;
          PyLong_AsLongLongAndOverflow(v.ptr(), &overflow);
          if (overflow < 0) {
            cytnx_error_msg(
              true, "[ERROR] integer scalar out of the supported int64/uint64 range.%s", "\n");
          } else if (overflow > 0) {
            needs_uint64 = true;
          }
        }
        if (needs_uint64) {
          std::vector<cytnx_uint64> vals;
          vals.reserve(pylist.size());
          for (const py::int_ &v : pylist) vals.push_back(v.cast<cytnx_uint64>());
          return cytnx::Storage::from_vector<cytnx_uint64>(vals, device);
        }
        std::vector<cytnx_int64> vals;
        vals.reserve(pylist.size());
        for (const py::int_ &v : pylist) vals.push_back(v.cast<cytnx_int64>());
        return cytnx::Storage::from_vector<cytnx_int64>(vals, device);
      },
      py::arg("pylist"), py::arg("device") = (int)cytnx::Device.cpu)
    .def_static("from_pylist", &cytnx::Storage::from_vector<cytnx_double>, py::arg("pylist"),
                py::arg("device") = (int)cytnx::Device.cpu)
    .def_static("from_pylist", &cytnx::Storage::from_vector<cytnx_complex128>, py::arg("pylist"),
                py::arg("device") = (int)cytnx::Device.cpu)

    .def("pylist",
         [](cytnx::Storage &self) -> py::object {
           switch (self.dtype()) {
             case (unsigned int)Type.ComplexDouble:
               return py::cast(self.vector<cytnx_complex128>());
             case (unsigned int)Type.ComplexFloat:
               return py::cast(self.vector<cytnx_complex64>());
             case (unsigned int)Type.Double:
               return py::cast(self.vector<cytnx_double>());
             case (unsigned int)Type.Float:
               return py::cast(self.vector<cytnx_float>());
             case (unsigned int)Type.Uint64:
               return py::cast(self.vector<cytnx_uint64>());
             case (unsigned int)Type.Int64:
               return py::cast(self.vector<cytnx_int64>());
             case (unsigned int)Type.Uint32:
               return py::cast(self.vector<cytnx_uint32>());
             case (unsigned int)Type.Int32:
               return py::cast(self.vector<cytnx_int32>());
             case (unsigned int)Type.Uint16:
               return py::cast(self.vector<cytnx_uint16>());
             case (unsigned int)Type.Int16:
               return py::cast(self.vector<cytnx_int16>());
             case (unsigned int)Type.Bool:
               return py::cast(self.vector<cytnx_bool>());
             default:
               cytnx_error_msg(true, "%s", "[ERROR] Storage.pylist: invalid Storage dtype!");
               return py::none();
           }
         })

    .def(
      "Save", [](cytnx::Storage &self, const std::string &fname) { self.Save(fname); },
      py::arg("fname"))
    .def(
      "Tofile", [](cytnx::Storage &self, const std::string &fname) { self.Tofile(fname); },
      py::arg("fname"))
    .def_static(
      "Load", [](const std::string &fname) { return cytnx::Storage::Load(fname); },
      py::arg("fname"))
    .def_static(
      "Fromfile",
      [](const std::string &fname, const unsigned int &dtype, const cytnx_int64 &count) {
        return cytnx::Storage::Fromfile(fname, dtype, count);
      },
      py::arg("fname"), py::arg("dtype"), py::arg("count") = (cytnx_int64)(-1))
    .def("real", &cytnx::Storage::real)
    .def("imag", &cytnx::Storage::imag)

    ;  // end of object line
}
#endif
