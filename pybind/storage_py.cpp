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
           std::vector<ssize_t> stride(1, Type.typeSize(tmpIN.dtype()));
           std::vector<ssize_t> shape(1, tmpIN.size());
           // ssize_t accu = tmpIN.size();

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

           npbuf = py::buffer_info(tmpIN._impl->Mem,  // ptr
                                   Type.typeSize(tmpIN.dtype()),  // size of elem
                                   chr_dtype,  // pss format
                                   1,  // rank
                                   shape,  // shape
                                   stride  // stride
           );
           py::array out(npbuf);
           // delegate numpy array with it's ptr, and swap a auxiliary ptr for intrusive_ptr to
           // free.
           void *pswap = malloc(sizeof(bool));
           tmpIN._impl->Mem = pswap;
           return out;
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
    .def("__len__", [](cytnx::Storage &self) { return self.size(); })
    .def("print_info", &cytnx::Storage::print_info,
         py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
    .def("set_zeros", &cytnx::Storage::set_zeros)
    .def("__eq__",
         [](cytnx::Storage &self, const cytnx::Storage &rhs) -> bool { return self == rhs; })

    .def("fill", &cytnx::Storage::fill<cytnx::cytnx_complex128>, py::arg("val"))
    .def("fill", &cytnx::Storage::fill<cytnx::cytnx_complex64>, py::arg("val"))
    .def("fill", &cytnx::Storage::fill<cytnx::cytnx_double>, py::arg("val"))
    .def("fill", &cytnx::Storage::fill<cytnx::cytnx_float>, py::arg("val"))
    .def("fill", &cytnx::Storage::fill<cytnx::cytnx_int64>, py::arg("val"))
    .def("fill", &cytnx::Storage::fill<cytnx::cytnx_uint64>, py::arg("val"))
    .def("fill", &cytnx::Storage::fill<cytnx::cytnx_int32>, py::arg("val"))
    .def("fill", &cytnx::Storage::fill<cytnx::cytnx_uint32>, py::arg("val"))
    .def("fill", &cytnx::Storage::fill<cytnx::cytnx_int16>, py::arg("val"))
    .def("fill", &cytnx::Storage::fill<cytnx::cytnx_uint16>, py::arg("val"))
    .def("fill", &cytnx::Storage::fill<cytnx::cytnx_bool>, py::arg("val"))

    .def("append", &cytnx::Storage::append<cytnx::cytnx_complex128>, py::arg("val"))
    .def("append", &cytnx::Storage::append<cytnx::cytnx_complex64>, py::arg("val"))
    .def("append", &cytnx::Storage::append<cytnx::cytnx_double>, py::arg("val"))
    .def("append", &cytnx::Storage::append<cytnx::cytnx_float>, py::arg("val"))
    .def("append", &cytnx::Storage::append<cytnx::cytnx_int64>, py::arg("val"))
    .def("append", &cytnx::Storage::append<cytnx::cytnx_uint64>, py::arg("val"))
    .def("append", &cytnx::Storage::append<cytnx::cytnx_int32>, py::arg("val"))
    .def("append", &cytnx::Storage::append<cytnx::cytnx_uint32>, py::arg("val"))
    .def("append", &cytnx::Storage::append<cytnx::cytnx_int16>, py::arg("val"))
    .def("append", &cytnx::Storage::append<cytnx::cytnx_uint16>, py::arg("val"))
    .def("append", &cytnx::Storage::append<cytnx::cytnx_bool>, py::arg("val"))

    .def_static("from_pylist", &cytnx::Storage::from_vector<cytnx_complex128>, py::arg("pylist"),
                py::arg("device") = (int)cytnx::Device.cpu)
    .def_static("from_pylist", &cytnx::Storage::from_vector<cytnx_complex64>, py::arg("pylist"),
                py::arg("device") = (int)cytnx::Device.cpu)
    .def_static("from_pylist", &cytnx::Storage::from_vector<cytnx_double>, py::arg("pylist"),
                py::arg("device") = (int)cytnx::Device.cpu)
    .def_static("from_pylist", &cytnx::Storage::from_vector<cytnx_float>, py::arg("pylist"),
                py::arg("device") = (int)cytnx::Device.cpu)
    .def_static("from_pylist", &cytnx::Storage::from_vector<cytnx_uint64>, py::arg("pylist"),
                py::arg("device") = (int)cytnx::Device.cpu)
    .def_static("from_pylist", &cytnx::Storage::from_vector<cytnx_int64>, py::arg("pylist"),
                py::arg("device") = (int)cytnx::Device.cpu)
    .def_static("from_pylist", &cytnx::Storage::from_vector<cytnx_uint32>, py::arg("pylist"),
                py::arg("device") = (int)cytnx::Device.cpu)
    .def_static("from_pylist", &cytnx::Storage::from_vector<cytnx_int32>, py::arg("pylist"),
                py::arg("device") = (int)cytnx::Device.cpu)
    .def_static("from_pylist", &cytnx::Storage::from_vector<cytnx_uint16>, py::arg("pylist"),
                py::arg("device") = (int)cytnx::Device.cpu)
    .def_static("from_pylist", &cytnx::Storage::from_vector<cytnx_int16>, py::arg("pylist"),
                py::arg("device") = (int)cytnx::Device.cpu)
    .def_static("from_pylist", &cytnx::Storage::from_vector<cytnx_bool>, py::arg("pylist"),
                py::arg("device") = (int)cytnx::Device.cpu)

    .def("c_pylist_complex128", &cytnx::Storage::vector<cytnx_complex128>)
    .def("c_pylist_complex64", &cytnx::Storage::vector<cytnx_complex64>)
    .def("c_pylist_double", &cytnx::Storage::vector<cytnx_double>)
    .def("c_pylist_float", &cytnx::Storage::vector<cytnx_float>)
    .def("c_pylist_uint64", &cytnx::Storage::vector<cytnx_uint64>)
    .def("c_pylist_int64", &cytnx::Storage::vector<cytnx_int64>)
    .def("c_pylist_uint32", &cytnx::Storage::vector<cytnx_uint32>)
    .def("c_pylist_int32", &cytnx::Storage::vector<cytnx_int32>)
    .def("c_pylist_uint16", &cytnx::Storage::vector<cytnx_uint16>)
    .def("c_pylist_int16", &cytnx::Storage::vector<cytnx_int16>)
    .def("c_pylist_bool", &cytnx::Storage::vector<cytnx_bool>)

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
