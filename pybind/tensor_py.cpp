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

template <class T>
void f_Tensor_setitem_scal(cytnx::Tensor &self, py::object locators, const T &rc) {
  cytnx_error_msg(self.shape().size() == 0, "[ERROR] try to setelem to a empty Tensor%s", "\n");

  ssize_t start, stop, step, slicelength;
  std::vector<cytnx::Accessor> accessors;
  if (py::isinstance<py::tuple>(locators)) {
    py::tuple Args = locators.cast<py::tuple>();
    cytnx_uint64 cnt = 0;
    // mixing of slice and ints
    for (cytnx_uint32 axis = 0; axis < Args.size(); axis++) {
      cnt++;
      // check type:
      if (py::isinstance<py::slice>(Args[axis])) {
        py::slice sls = Args[axis].cast<py::slice>();
        if (!sls.compute((ssize_t)self.shape()[axis], &start, &stop, &step, &slicelength))
          throw py::error_already_set();
        // std::cout << start << " " << stop << " " << step << slicelength << std::endl;
        // if(slicelength == self.shape()[axis]) accessors.push_back(cytnx::Accessor::all());
        accessors.push_back(
          cytnx::Accessor::range(cytnx_int64(start), cytnx_int64(stop), cytnx_int64(step)));
      } else {
        accessors.push_back(cytnx::Accessor(Args[axis].cast<cytnx_int64>()));
      }
    }
    while (cnt < self.shape().size()) {
      cnt++;
      accessors.push_back(Accessor::all());
    }
  } else {
    // only int
    for (cytnx_uint32 i = 0; i < self.shape().size(); i++) {
      if (i == 0)
        accessors.push_back(cytnx::Accessor(locators.cast<cytnx_int64>()));
      else
        accessors.push_back(cytnx::Accessor::all());
    }
  }

  self.set(accessors, rc);
}

void tensor_binding(py::module &m) {
  py::class_<cytnx::Tensor>(m, "Tensor")
    .def(
      "numpy",
      [](Tensor &self, const bool &share_mem) -> py::array {
        // device on GPU? move to cpu:ref it;
        Tensor tmpIN;
        if (self.device() >= 0) {
          if (share_mem) {
            cytnx_error_msg(true,
                            "[ERROR] the Tensor is on GPU. to have share_mem=True, move the Tensor "
                            "back to CPU by .to(Device.cpu).%s",
                            "\n");
          } else {
            tmpIN = self.to(Device.cpu);
          }
        } else {
          tmpIN = self;
        }
        if (tmpIN.is_contiguous()) {
          if (share_mem)
            tmpIN = self;
          else
            tmpIN = self.clone();
        } else {
          if (share_mem) {
            cytnx_error_msg(true,
                            "[ERROR] calling numpy(share_mem=true) require the current Tensor is "
                            "contiguous. \n Call contiguous() first before convert to numpy.%s",
                            "\n");
          } else
            tmpIN = self.contiguous();
        }

        // calculate stride:
        std::vector<ssize_t> stride(tmpIN.shape().size());
        std::vector<ssize_t> shape(tmpIN.shape().begin(), tmpIN.shape().end());
        ssize_t accu = 1;
        for (int i = shape.size() - 1; i >= 0; i--) {
          stride[i] = accu * Type.typeSize(tmpIN.dtype());
          accu *= shape[i];
        }
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
          cytnx_error_msg(true, "[ERROR] Void Type Tensor cannot convert to numpy ndarray%s", "\n");
        }

        npbuf = py::buffer_info(tmpIN.storage()._impl->Mem,  // ptr
                                Type.typeSize(tmpIN.dtype()),  // size of elem
                                chr_dtype,  // pss format
                                tmpIN.rank(),  // rank
                                shape,  // shape
                                stride  // stride
        );
        py::array out(npbuf);
        // delegate numpy array with it's ptr, and swap a auxiliary ptr for intrusive_ptr to
        // free.
        if (share_mem == false) {
          void *pswap = malloc(sizeof(bool));
          tmpIN.storage()._impl->Mem = pswap;
        }
        return out;
      },
      py::arg("share_mem") = false)
    // construction
    .def(py::init<>())
    .def(py::init<const cytnx::Tensor &>())
    .def(
      py::init<const std::vector<cytnx::cytnx_uint64> &, const unsigned int &, int, const bool &>(),
      py::arg("shape"), py::arg("dtype") = (cytnx_uint64)cytnx::Type.Double,
      py::arg("device") = (int)cytnx::Device.cpu, py::arg("init_zero") = true)
    .def("Init", &cytnx::Tensor::Init, py::arg("shape"),
         py::arg("dtype") = (cytnx_uint64)cytnx::Type.Double,
         py::arg("device") = (int)cytnx::Device.cpu, py::arg("init_zero") = true)
    .def("dtype", &cytnx::Tensor::dtype)
    .def("dtype_str", &cytnx::Tensor::dtype_str)
    .def("device", &cytnx::Tensor::device)
    .def("device_str", &cytnx::Tensor::device_str)
    .def("shape", &cytnx::Tensor::shape)
    .def("rank", &cytnx::Tensor::rank)
    .def("clone", &cytnx::Tensor::clone)
    .def("__copy__", &cytnx::Tensor::clone)
    .def("__deepcopy__", &cytnx::Tensor::clone)
    //.def("to", &cytnx::Tensor::to, py::arg("device"))
    // handle same device from cytnx/Tensor_conti.py
    .def(
      "to_different_device",
      [](cytnx::Tensor &self, const cytnx_int64 &device) {
        cytnx_error_msg(self.device() == device,
                        "[ERROR][pybind][to_diffferent_device] same device for to() should be "
                        "handle in python side.%s",
                        "\n");
        return self.to(device);
      },
      py::arg("device"))

    .def("to_", &cytnx::Tensor::to_, py::arg("device"))
    .def("is_contiguous", &cytnx::Tensor::is_contiguous)
    .def("permute_",
         [](cytnx::Tensor &self, py::args args) {
           std::vector<cytnx::cytnx_uint64> c_args = args.cast<std::vector<cytnx::cytnx_uint64>>();
           // std::cout << c_args.size() << std::endl;
           self.permute_(c_args);
         })
    .def("permute",
         [](cytnx::Tensor &self, py::args args) -> cytnx::Tensor {
           std::vector<cytnx::cytnx_uint64> c_args = args.cast<std::vector<cytnx::cytnx_uint64>>();
           // std::cout << c_args.size() << std::endl;
           return self.permute(c_args);
         })
    .def("same_data", &cytnx::Tensor::same_data)
    .def("flatten", &cytnx::Tensor::flatten)
    .def("flatten_", &cytnx::Tensor::flatten_)
    .def("make_contiguous", &cytnx::Tensor::contiguous)  // this will be rename by python side conti
    .def("contiguous_", &cytnx::Tensor::contiguous_)
    .def("reshape_",
         [](cytnx::Tensor &self, py::args args) {
           std::vector<cytnx::cytnx_int64> c_args = args.cast<std::vector<cytnx::cytnx_int64>>();
           self.reshape_(c_args);
         })
    .def("reshape",
         [](cytnx::Tensor &self, py::args args) -> cytnx::Tensor {
           std::vector<cytnx::cytnx_int64> c_args = args.cast<std::vector<cytnx::cytnx_int64>>();
           return self.reshape(c_args);
         })
    //.def("astype", &cytnx::Tensor::astype,py::arg("new_type"))
    .def(
      "astype_different_dtype",
      [](cytnx::Tensor &self, const cytnx_uint64 &dtype) {
        cytnx_error_msg(self.dtype() == dtype,
                        "[ERROR][pybind][astype_diffferent_device] same dtype for astype() should "
                        "be handle in python side.%s",
                        "\n");
        return self.astype(dtype);
      },
      py::arg("new_type"))

    .def("item",
         [](cytnx::Tensor &self) {
           py::object out;
           if (self.dtype() == cytnx::Type.Double)
             out = py::cast(self.item<cytnx::cytnx_double>());
           else if (self.dtype() == cytnx::Type.Float)
             out = py::cast(self.item<cytnx::cytnx_float>());
           else if (self.dtype() == cytnx::Type.ComplexDouble)
             out = py::cast(self.item<cytnx::cytnx_complex128>());
           else if (self.dtype() == cytnx::Type.ComplexFloat)
             out = py::cast(self.item<cytnx::cytnx_complex64>());
           else if (self.dtype() == cytnx::Type.Uint64)
             out = py::cast(self.item<cytnx::cytnx_uint64>());
           else if (self.dtype() == cytnx::Type.Int64)
             out = py::cast(self.item<cytnx::cytnx_int64>());
           else if (self.dtype() == cytnx::Type.Uint32)
             out = py::cast(self.item<cytnx::cytnx_uint32>());
           else if (self.dtype() == cytnx::Type.Int32)
             out = py::cast(self.item<cytnx::cytnx_int32>());
           else if (self.dtype() == cytnx::Type.Uint16)
             out = py::cast(self.item<cytnx::cytnx_uint16>());
           else if (self.dtype() == cytnx::Type.Int16)
             out = py::cast(self.item<cytnx::cytnx_int16>());
           else if (self.dtype() == cytnx::Type.Bool)
             out = py::cast(self.item<cytnx::cytnx_bool>());
           else
             cytnx_error_msg(true, "%s", "[ERROR] try to get element from a void Storage.");
           return out;
         })
    .def("storage", &cytnx::Tensor::storage)
    .def("real", &cytnx::Tensor::real)
    .def("imag", &cytnx::Tensor::imag)
    .def(
      "__repr__",
      [](cytnx::Tensor &self) -> std::string {
        std::cout << self << std::endl;
        return std::string("");
      },
      py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
    .def("fill", &cytnx::Tensor::fill<cytnx::cytnx_complex128>, py::arg("val"))
    .def("fill", &cytnx::Tensor::fill<cytnx::cytnx_complex64>, py::arg("val"))
    .def("fill", &cytnx::Tensor::fill<cytnx::cytnx_double>, py::arg("val"))
    .def("fill", &cytnx::Tensor::fill<cytnx::cytnx_float>, py::arg("val"))
    .def("fill", &cytnx::Tensor::fill<cytnx::cytnx_int64>, py::arg("val"))
    .def("fill", &cytnx::Tensor::fill<cytnx::cytnx_uint64>, py::arg("val"))
    .def("fill", &cytnx::Tensor::fill<cytnx::cytnx_int32>, py::arg("val"))
    .def("fill", &cytnx::Tensor::fill<cytnx::cytnx_uint32>, py::arg("val"))
    .def("fill", &cytnx::Tensor::fill<cytnx::cytnx_int16>, py::arg("val"))
    .def("fill", &cytnx::Tensor::fill<cytnx::cytnx_uint16>, py::arg("val"))
    .def("fill", &cytnx::Tensor::fill<cytnx::cytnx_bool>, py::arg("val"))

    .def("append", &cytnx::Tensor::append<cytnx::cytnx_complex128>, py::arg("val"))
    .def("append", &cytnx::Tensor::append<cytnx::cytnx_complex64>, py::arg("val"))
    .def("append", &cytnx::Tensor::append<cytnx::cytnx_double>, py::arg("val"))
    .def("append", &cytnx::Tensor::append<cytnx::cytnx_float>, py::arg("val"))
    .def("append", &cytnx::Tensor::append<cytnx::cytnx_int64>, py::arg("val"))
    .def("append", &cytnx::Tensor::append<cytnx::cytnx_uint64>, py::arg("val"))
    .def("append", &cytnx::Tensor::append<cytnx::cytnx_int32>, py::arg("val"))
    .def("append", &cytnx::Tensor::append<cytnx::cytnx_uint32>, py::arg("val"))
    .def("append", &cytnx::Tensor::append<cytnx::cytnx_int16>, py::arg("val"))
    .def("append", &cytnx::Tensor::append<cytnx::cytnx_uint16>, py::arg("val"))
    .def("append", &cytnx::Tensor::append<cytnx::cytnx_bool>, py::arg("val"))
    .def(
      "append", [](cytnx::Tensor &self, const cytnx::Tensor &rhs) { self.append(rhs); },
      py::arg("val"))
    .def(
      "append", [](cytnx::Tensor &self, const cytnx::Storage &rhs) { self.append(rhs); },
      py::arg("val"))

    .def(
      "Save", [](cytnx::Tensor &self, const std::string &fname) { self.Save(fname); },
      py::arg("fname"))
    .def_static(
      "Load", [](const std::string &fname) { return cytnx::Tensor::Load(fname); }, py::arg("fname"))

    .def(
      "Tofile", [](cytnx::Tensor &self, const std::string &fname) { self.Tofile(fname); },
      py::arg("fname"))
    .def_static(
      "Fromfile",
      [](const std::string &fname, const unsigned int &dtype, const cytnx_int64 &count) {
        return cytnx::Tensor::Load(fname);
      },
      py::arg("fname"), py::arg("dtype"), py::arg("count") = (cytnx_int64)(-1))

    .def_static(
      "from_storage",
      [](const Storage &sin, const bool &is_clone) {
        Tensor out;
        if (is_clone)
          out = cytnx::Tensor::from_storage(sin.clone());
        else
          out = cytnx::Tensor::from_storage(sin);
        return out;
      },
      py::arg("sin"), py::arg("is_clone") = false)

    .def("__len__",
         [](const cytnx::Tensor &self) {
           if (self.dtype() == Type.Void) {
             cytnx_error_msg(true, "[ERROR] uninitialize Tensor does not have len!%s", "\n");

           } else {
             return self.shape()[0];
           }
         })
    .def("__getitem__",
         [](const cytnx::Tensor &self, py::object locators) {
           cytnx_error_msg(self.shape().size() == 0, "[ERROR] try to getitem from a empty Tensor%s",
                           "\n");

           ssize_t start, stop, step, slicelength;
           std::vector<cytnx::Accessor> accessors;
           if (py::isinstance<py::tuple>(locators)) {
             py::tuple Args = locators.cast<py::tuple>();
             cytnx_uint64 cnt = 0;
             // mixing of slice and ints
             for (cytnx_uint32 axis = 0; axis < Args.size(); axis++) {
               cnt++;
               // check type:
               if (py::isinstance<py::slice>(Args[axis])) {
                 py::slice sls = Args[axis].cast<py::slice>();
                 if (!sls.compute((ssize_t)self.shape()[axis], &start, &stop, &step, &slicelength))
                   throw py::error_already_set();
                 // std::cout << start << " " << stop << " " << step << slicelength << std::endl;
                 // if(slicelength == self.shape()[axis])
                 // accessors.push_back(cytnx::Accessor::all());
                 accessors.push_back(cytnx::Accessor::range(cytnx_int64(start), cytnx_int64(stop),
                                                            cytnx_int64(step)));
               } else {
                 accessors.push_back(cytnx::Accessor(Args[axis].cast<cytnx_int64>()));
               }
             }
             while (cnt < self.shape().size()) {
               cnt++;
               accessors.push_back(Accessor::all());
             }
           } else if (py::isinstance<py::slice>(locators)) {
             py::slice sls = locators.cast<py::slice>();
             if (!sls.compute((ssize_t)self.shape()[0], &start, &stop, &step, &slicelength))
               throw py::error_already_set();
             // if(slicelength == self.shape()[0]) accessors.push_back(cytnx::Accessor::all());
             std::cout << start << " " << stop << " " << step << std::endl;
             accessors.push_back(cytnx::Accessor::range(start, stop, step));
             for (cytnx_uint32 axis = 1; axis < self.shape().size(); axis++) {
               accessors.push_back(Accessor::all());
             }

           } else {
             // std::cout << "int locators" << std::endl;
             // std::cout << locators.cast<cytnx_int64>() << std::endl;
             //  only int
             for (cytnx_uint32 i = 0; i < self.shape().size(); i++) {
               if (i == 0)
                 accessors.push_back(cytnx::Accessor(locators.cast<cytnx_int64>()));
               else
                 accessors.push_back(cytnx::Accessor::all());
             }
           }

           return self.get(accessors);
         })

    .def("__setitem__",
         [](cytnx::Tensor &self, py::object locators, const cytnx::Tensor &rhs) {
           cytnx_error_msg(self.shape().size() == 0, "[ERROR] try to setelem to a empty Tensor%s",
                           "\n");

           ssize_t start, stop, step, slicelength;
           std::vector<cytnx::Accessor> accessors;
           if (py::isinstance<py::tuple>(locators)) {
             py::tuple Args = locators.cast<py::tuple>();
             cytnx_uint64 cnt = 0;
             // mixing of slice and ints
             for (cytnx_uint32 axis = 0; axis < Args.size(); axis++) {
               cnt++;
               // check type:
               if (py::isinstance<py::slice>(Args[axis])) {
                 py::slice sls = Args[axis].cast<py::slice>();
                 if (!sls.compute((ssize_t)self.shape()[axis], &start, &stop, &step, &slicelength))
                   throw py::error_already_set();
                 // std::cout << start << " " << stop << " " << step << slicelength << std::endl;
                 // if(slicelength == self.shape()[axis])
                 // accessors.push_back(cytnx::Accessor::all());
                 accessors.push_back(cytnx::Accessor::range(cytnx_int64(start), cytnx_int64(stop),
                                                            cytnx_int64(step)));
               } else {
                 accessors.push_back(cytnx::Accessor(Args[axis].cast<cytnx_int64>()));
               }
             }
             while (cnt < self.shape().size()) {
               cnt++;
               accessors.push_back(Accessor::all());
             }
           } else if (py::isinstance<py::slice>(locators)) {
             py::slice sls = locators.cast<py::slice>();
             if (!sls.compute((ssize_t)self.shape()[0], &start, &stop, &step, &slicelength))
               throw py::error_already_set();
             // if(slicelength == self.shape()[0]) accessors.push_back(cytnx::Accessor::all());
             accessors.push_back(cytnx::Accessor::range(start, stop, step));
             for (cytnx_uint32 axis = 1; axis < self.shape().size(); axis++) {
               accessors.push_back(Accessor::all());
             }

           } else {
             // only int
             for (cytnx_uint32 i = 0; i < self.shape().size(); i++) {
               if (i == 0)
                 accessors.push_back(cytnx::Accessor(locators.cast<cytnx_int64>()));
               else
                 accessors.push_back(cytnx::Accessor::all());
             }
           }

           self.set(accessors, rhs);
         })
    .def("__setitem__", &f_Tensor_setitem_scal<cytnx_complex128>)
    .def("__setitem__", &f_Tensor_setitem_scal<cytnx_complex64>)
    .def("__setitem__", &f_Tensor_setitem_scal<cytnx_double>)
    .def("__setitem__", &f_Tensor_setitem_scal<cytnx_float>)
    .def("__setitem__", &f_Tensor_setitem_scal<cytnx_int64>)
    .def("__setitem__", &f_Tensor_setitem_scal<cytnx_uint64>)
    .def("__setitem__", &f_Tensor_setitem_scal<cytnx_int32>)
    .def("__setitem__", &f_Tensor_setitem_scal<cytnx_uint32>)
    .def("__setitem__", &f_Tensor_setitem_scal<cytnx_int16>)
    .def("__setitem__", &f_Tensor_setitem_scal<cytnx_uint16>)
    .def("__setitem__", &f_Tensor_setitem_scal<cytnx_bool>)
    // arithmetic >>
    .def("__neg__",
         [](cytnx::Tensor &self) {
           if (self.dtype() == Type.Double) {
             return cytnx::linalg::Mul(cytnx_double(-1), self);
           } else if (self.dtype() == Type.ComplexDouble) {
             return cytnx::linalg::Mul(cytnx_complex128(-1, 0), self);
           } else if (self.dtype() == Type.Float) {
             return cytnx::linalg::Mul(cytnx_float(-1), self);
           } else if (self.dtype() == Type.ComplexFloat) {
             return cytnx::linalg::Mul(cytnx_complex64(-1, 0), self);
           } else {
             return cytnx::linalg::Mul(-1, self);
           }
         })
    .def("__pos__", [](cytnx::Tensor &self) { return self; })
    .def("__add__", [](cytnx::Tensor &self, const cytnx::Tensor &rhs) { return self.Add(rhs); })
    .def("__add__",
         [](cytnx::Tensor &self, const cytnx::cytnx_complex128 &rhs) { return self.Add(rhs); })
    .def("__add__",
         [](cytnx::Tensor &self, const cytnx::cytnx_complex64 &rhs) { return self.Add(rhs); })
    .def("__add__",
         [](cytnx::Tensor &self, const cytnx::cytnx_double &rhs) { return self.Add(rhs); })
    .def("__add__",
         [](cytnx::Tensor &self, const cytnx::cytnx_float &rhs) { return self.Add(rhs); })
    .def("__add__",
         [](cytnx::Tensor &self, const cytnx::cytnx_int64 &rhs) { return self.Add(rhs); })
    .def("__add__",
         [](cytnx::Tensor &self, const cytnx::cytnx_uint64 &rhs) { return self.Add(rhs); })
    .def("__add__",
         [](cytnx::Tensor &self, const cytnx::cytnx_int32 &rhs) { return self.Add(rhs); })
    .def("__add__",
         [](cytnx::Tensor &self, const cytnx::cytnx_uint32 &rhs) { return self.Add(rhs); })
    .def("__add__",
         [](cytnx::Tensor &self, const cytnx::cytnx_int16 &rhs) { return self.Add(rhs); })
    .def("__add__",
         [](cytnx::Tensor &self, const cytnx::cytnx_uint16 &rhs) { return self.Add(rhs); })
    .def("__add__", [](cytnx::Tensor &self, const cytnx::cytnx_bool &rhs) { return self.Add(rhs); })

    .def("__radd__",
         [](cytnx::Tensor &self, const cytnx::cytnx_complex128 &lhs) {
           return cytnx::linalg::Add(lhs, self);
         })
    .def("__radd__",
         [](cytnx::Tensor &self, const cytnx::cytnx_complex64 &lhs) {
           return cytnx::linalg::Add(lhs, self);
         })
    .def("__radd__", [](cytnx::Tensor &self,
                        const cytnx::cytnx_double &lhs) { return cytnx::linalg::Add(lhs, self); })
    .def("__radd__", [](cytnx::Tensor &self,
                        const cytnx::cytnx_float &lhs) { return cytnx::linalg::Add(lhs, self); })
    .def("__radd__", [](cytnx::Tensor &self,
                        const cytnx::cytnx_int64 &lhs) { return cytnx::linalg::Add(lhs, self); })
    .def("__radd__", [](cytnx::Tensor &self,
                        const cytnx::cytnx_uint64 &lhs) { return cytnx::linalg::Add(lhs, self); })
    .def("__radd__", [](cytnx::Tensor &self,
                        const cytnx::cytnx_int32 &lhs) { return cytnx::linalg::Add(lhs, self); })
    .def("__radd__", [](cytnx::Tensor &self,
                        const cytnx::cytnx_uint32 &lhs) { return cytnx::linalg::Add(lhs, self); })
    .def("__radd__", [](cytnx::Tensor &self,
                        const cytnx::cytnx_int16 &lhs) { return cytnx::linalg::Add(lhs, self); })
    .def("__radd__", [](cytnx::Tensor &self,
                        const cytnx::cytnx_uint16 &lhs) { return cytnx::linalg::Add(lhs, self); })
    .def("__radd__", [](cytnx::Tensor &self,
                        const cytnx::cytnx_bool &lhs) { return cytnx::linalg::Add(lhs, self); })
    .def("c__iadd__",
         [](cytnx::Tensor &self, const cytnx::Tensor &rhs) {
           return self.Add_(rhs);
         })  // these will return self!
    .def("c__iadd__",
         [](cytnx::Tensor &self, const cytnx::cytnx_complex128 &rhs) { return self.Add_(rhs); })
    .def("c__iadd__",
         [](cytnx::Tensor &self, const cytnx::cytnx_complex64 &rhs) { return self.Add_(rhs); })
    .def("c__iadd__",
         [](cytnx::Tensor &self, const cytnx::cytnx_double &rhs) { return self.Add_(rhs); })
    .def("c__iadd__",
         [](cytnx::Tensor &self, const cytnx::cytnx_float &rhs) { return self.Add_(rhs); })
    .def("c__iadd__",
         [](cytnx::Tensor &self, const cytnx::cytnx_int64 &rhs) { return self.Add_(rhs); })
    .def("c__iadd__",
         [](cytnx::Tensor &self, const cytnx::cytnx_uint64 &rhs) { return self.Add_(rhs); })
    .def("c__iadd__",
         [](cytnx::Tensor &self, const cytnx::cytnx_int32 &rhs) { return self.Add_(rhs); })
    .def("c__iadd__",
         [](cytnx::Tensor &self, const cytnx::cytnx_uint32 &rhs) { return self.Add_(rhs); })
    .def("c__iadd__",
         [](cytnx::Tensor &self, const cytnx::cytnx_int16 &rhs) { return self.Add_(rhs); })
    .def("c__iadd__",
         [](cytnx::Tensor &self, const cytnx::cytnx_uint16 &rhs) { return self.Add_(rhs); })
    .def("c__iadd__",
         [](cytnx::Tensor &self, const cytnx::cytnx_bool &rhs) { return self.Add_(rhs); })

    .def("__sub__", [](cytnx::Tensor &self, const cytnx::Tensor &rhs) { return self.Sub(rhs); })
    .def("__sub__",
         [](cytnx::Tensor &self, const cytnx::cytnx_complex128 &rhs) { return self.Sub(rhs); })
    .def("__sub__",
         [](cytnx::Tensor &self, const cytnx::cytnx_complex64 &rhs) { return self.Sub(rhs); })
    .def("__sub__",
         [](cytnx::Tensor &self, const cytnx::cytnx_double &rhs) { return self.Sub(rhs); })
    .def("__sub__",
         [](cytnx::Tensor &self, const cytnx::cytnx_float &rhs) { return self.Sub(rhs); })
    .def("__sub__",
         [](cytnx::Tensor &self, const cytnx::cytnx_int64 &rhs) { return self.Sub(rhs); })
    .def("__sub__",
         [](cytnx::Tensor &self, const cytnx::cytnx_uint64 &rhs) { return self.Sub(rhs); })
    .def("__sub__",
         [](cytnx::Tensor &self, const cytnx::cytnx_int32 &rhs) { return self.Sub(rhs); })
    .def("__sub__",
         [](cytnx::Tensor &self, const cytnx::cytnx_uint32 &rhs) { return self.Sub(rhs); })
    .def("__sub__",
         [](cytnx::Tensor &self, const cytnx::cytnx_int16 &rhs) { return self.Sub(rhs); })
    .def("__sub__",
         [](cytnx::Tensor &self, const cytnx::cytnx_uint16 &rhs) { return self.Sub(rhs); })
    .def("__sub__", [](cytnx::Tensor &self, const cytnx::cytnx_bool &rhs) { return self.Sub(rhs); })

    .def("__rsub__",
         [](cytnx::Tensor &self, const cytnx::cytnx_complex128 &lhs) {
           return cytnx::linalg::Sub(lhs, self);
         })
    .def("__rsub__",
         [](cytnx::Tensor &self, const cytnx::cytnx_complex64 &lhs) {
           return cytnx::linalg::Sub(lhs, self);
         })
    .def("__rsub__", [](cytnx::Tensor &self,
                        const cytnx::cytnx_double &lhs) { return cytnx::linalg::Sub(lhs, self); })
    .def("__rsub__", [](cytnx::Tensor &self,
                        const cytnx::cytnx_float &lhs) { return cytnx::linalg::Sub(lhs, self); })
    .def("__rsub__", [](cytnx::Tensor &self,
                        const cytnx::cytnx_int64 &lhs) { return cytnx::linalg::Sub(lhs, self); })
    .def("__rsub__", [](cytnx::Tensor &self,
                        const cytnx::cytnx_uint64 &lhs) { return cytnx::linalg::Sub(lhs, self); })
    .def("__rsub__", [](cytnx::Tensor &self,
                        const cytnx::cytnx_int32 &lhs) { return cytnx::linalg::Sub(lhs, self); })
    .def("__rsub__", [](cytnx::Tensor &self,
                        const cytnx::cytnx_uint32 &lhs) { return cytnx::linalg::Sub(lhs, self); })
    .def("__rsub__", [](cytnx::Tensor &self,
                        const cytnx::cytnx_int16 &lhs) { return cytnx::linalg::Sub(lhs, self); })
    .def("__rsub__", [](cytnx::Tensor &self,
                        const cytnx::cytnx_uint16 &lhs) { return cytnx::linalg::Sub(lhs, self); })
    .def("__rsub__", [](cytnx::Tensor &self,
                        const cytnx::cytnx_bool &lhs) { return cytnx::linalg::Sub(lhs, self); })
    .def("c__isub__",
         [](cytnx::Tensor &self, const cytnx::Tensor &rhs) {
           return self.Sub_(rhs);
         })  // these will return self!
    .def("c__isub__",
         [](cytnx::Tensor &self, const cytnx::cytnx_complex128 &rhs) { return self.Sub_(rhs); })
    .def("c__isub__",
         [](cytnx::Tensor &self, const cytnx::cytnx_complex64 &rhs) { return self.Sub_(rhs); })
    .def("c__isub__",
         [](cytnx::Tensor &self, const cytnx::cytnx_double &rhs) { return self.Sub_(rhs); })
    .def("c__isub__",
         [](cytnx::Tensor &self, const cytnx::cytnx_float &rhs) { return self.Sub_(rhs); })
    .def("c__isub__",
         [](cytnx::Tensor &self, const cytnx::cytnx_int64 &rhs) { return self.Sub_(rhs); })
    .def("c__isub__",
         [](cytnx::Tensor &self, const cytnx::cytnx_uint64 &rhs) { return self.Sub_(rhs); })
    .def("c__isub__",
         [](cytnx::Tensor &self, const cytnx::cytnx_int32 &rhs) { return self.Sub_(rhs); })
    .def("c__isub__",
         [](cytnx::Tensor &self, const cytnx::cytnx_uint32 &rhs) { return self.Sub_(rhs); })
    .def("c__isub__",
         [](cytnx::Tensor &self, const cytnx::cytnx_int16 &rhs) { return self.Sub_(rhs); })
    .def("c__isub__",
         [](cytnx::Tensor &self, const cytnx::cytnx_uint16 &rhs) { return self.Sub_(rhs); })
    .def("c__isub__",
         [](cytnx::Tensor &self, const cytnx::cytnx_bool &rhs) { return self.Sub_(rhs); })

    .def("__mul__", [](cytnx::Tensor &self, const cytnx::Tensor &rhs) { return self.Mul(rhs); })
    .def("__mul__",
         [](cytnx::Tensor &self, const cytnx::cytnx_complex128 &rhs) { return self.Mul(rhs); })
    .def("__mul__",
         [](cytnx::Tensor &self, const cytnx::cytnx_complex64 &rhs) { return self.Mul(rhs); })
    .def("__mul__",
         [](cytnx::Tensor &self, const cytnx::cytnx_double &rhs) { return self.Mul(rhs); })
    .def("__mul__",
         [](cytnx::Tensor &self, const cytnx::cytnx_float &rhs) { return self.Mul(rhs); })
    .def("__mul__",
         [](cytnx::Tensor &self, const cytnx::cytnx_int64 &rhs) { return self.Mul(rhs); })
    .def("__mul__",
         [](cytnx::Tensor &self, const cytnx::cytnx_uint64 &rhs) { return self.Mul(rhs); })
    .def("__mul__",
         [](cytnx::Tensor &self, const cytnx::cytnx_int32 &rhs) { return self.Mul(rhs); })
    .def("__mul__",
         [](cytnx::Tensor &self, const cytnx::cytnx_uint32 &rhs) { return self.Mul(rhs); })
    .def("__mul__",
         [](cytnx::Tensor &self, const cytnx::cytnx_int16 &rhs) { return self.Mul(rhs); })
    .def("__mul__",
         [](cytnx::Tensor &self, const cytnx::cytnx_uint16 &rhs) { return self.Mul(rhs); })
    .def("__mul__", [](cytnx::Tensor &self, const cytnx::cytnx_bool &rhs) { return self.Mul(rhs); })

    .def("__rmul__",
         [](cytnx::Tensor &self, const cytnx::cytnx_complex128 &lhs) {
           return cytnx::linalg::Mul(lhs, self);
         })
    .def("__rmul__",
         [](cytnx::Tensor &self, const cytnx::cytnx_complex64 &lhs) {
           return cytnx::linalg::Mul(lhs, self);
         })
    .def("__rmul__", [](cytnx::Tensor &self,
                        const cytnx::cytnx_double &lhs) { return cytnx::linalg::Mul(lhs, self); })
    .def("__rmul__", [](cytnx::Tensor &self,
                        const cytnx::cytnx_float &lhs) { return cytnx::linalg::Mul(lhs, self); })
    .def("__rmul__", [](cytnx::Tensor &self,
                        const cytnx::cytnx_int64 &lhs) { return cytnx::linalg::Mul(lhs, self); })
    .def("__rmul__", [](cytnx::Tensor &self,
                        const cytnx::cytnx_uint64 &lhs) { return cytnx::linalg::Mul(lhs, self); })
    .def("__rmul__", [](cytnx::Tensor &self,
                        const cytnx::cytnx_int32 &lhs) { return cytnx::linalg::Mul(lhs, self); })
    .def("__rmul__", [](cytnx::Tensor &self,
                        const cytnx::cytnx_uint32 &lhs) { return cytnx::linalg::Mul(lhs, self); })
    .def("__rmul__", [](cytnx::Tensor &self,
                        const cytnx::cytnx_int16 &lhs) { return cytnx::linalg::Mul(lhs, self); })
    .def("__rmul__", [](cytnx::Tensor &self,
                        const cytnx::cytnx_uint16 &lhs) { return cytnx::linalg::Mul(lhs, self); })
    .def("__rmul__", [](cytnx::Tensor &self,
                        const cytnx::cytnx_bool &lhs) { return cytnx::linalg::Mul(lhs, self); })

    .def("__mod__", [](cytnx::Tensor &self, const cytnx::Tensor &rhs) { return self.Mod(rhs); })
    .def("__mod__",
         [](cytnx::Tensor &self, const cytnx::cytnx_complex128 &rhs) { return self.Mod(rhs); })
    .def("__mod__",
         [](cytnx::Tensor &self, const cytnx::cytnx_complex64 &rhs) { return self.Mod(rhs); })
    .def("__mod__",
         [](cytnx::Tensor &self, const cytnx::cytnx_double &rhs) { return self.Mod(rhs); })
    .def("__mod__",
         [](cytnx::Tensor &self, const cytnx::cytnx_float &rhs) { return self.Mod(rhs); })
    .def("__mod__",
         [](cytnx::Tensor &self, const cytnx::cytnx_int64 &rhs) { return self.Mod(rhs); })
    .def("__mod__",
         [](cytnx::Tensor &self, const cytnx::cytnx_uint64 &rhs) { return self.Mod(rhs); })
    .def("__mod__",
         [](cytnx::Tensor &self, const cytnx::cytnx_int32 &rhs) { return self.Mod(rhs); })
    .def("__mod__",
         [](cytnx::Tensor &self, const cytnx::cytnx_uint32 &rhs) { return self.Mod(rhs); })
    .def("__mod__",
         [](cytnx::Tensor &self, const cytnx::cytnx_int16 &rhs) { return self.Mod(rhs); })
    .def("__mod__",
         [](cytnx::Tensor &self, const cytnx::cytnx_uint16 &rhs) { return self.Mod(rhs); })
    .def("__mod__", [](cytnx::Tensor &self, const cytnx::cytnx_bool &rhs) { return self.Mod(rhs); })

    .def("__rmod__",
         [](cytnx::Tensor &self, const cytnx::cytnx_complex128 &lhs) {
           return cytnx::linalg::Mod(lhs, self);
         })
    .def("__rmod__",
         [](cytnx::Tensor &self, const cytnx::cytnx_complex64 &lhs) {
           return cytnx::linalg::Mod(lhs, self);
         })
    .def("__rmod__", [](cytnx::Tensor &self,
                        const cytnx::cytnx_double &lhs) { return cytnx::linalg::Mod(lhs, self); })
    .def("__rmod__", [](cytnx::Tensor &self,
                        const cytnx::cytnx_float &lhs) { return cytnx::linalg::Mod(lhs, self); })
    .def("__rmod__", [](cytnx::Tensor &self,
                        const cytnx::cytnx_int64 &lhs) { return cytnx::linalg::Mod(lhs, self); })
    .def("__rmod__", [](cytnx::Tensor &self,
                        const cytnx::cytnx_uint64 &lhs) { return cytnx::linalg::Mod(lhs, self); })
    .def("__rmod__", [](cytnx::Tensor &self,
                        const cytnx::cytnx_int32 &lhs) { return cytnx::linalg::Mod(lhs, self); })
    .def("__rmod__", [](cytnx::Tensor &self,
                        const cytnx::cytnx_uint32 &lhs) { return cytnx::linalg::Mod(lhs, self); })
    .def("__rmod__", [](cytnx::Tensor &self,
                        const cytnx::cytnx_int16 &lhs) { return cytnx::linalg::Mod(lhs, self); })
    .def("__rmod__", [](cytnx::Tensor &self,
                        const cytnx::cytnx_uint16 &lhs) { return cytnx::linalg::Mod(lhs, self); })
    .def("__rmod__", [](cytnx::Tensor &self,
                        const cytnx::cytnx_bool &lhs) { return cytnx::linalg::Mod(lhs, self); })
    .def("c__imul__",
         [](cytnx::Tensor &self, const cytnx::Tensor &rhs) {
           return self.Mul_(rhs);
         })  // these will return self!
    .def("c__imul__",
         [](cytnx::Tensor &self, const cytnx::cytnx_complex128 &rhs) { return self.Mul_(rhs); })
    .def("c__imul__",
         [](cytnx::Tensor &self, const cytnx::cytnx_complex64 &rhs) { return self.Mul_(rhs); })
    .def("c__imul__",
         [](cytnx::Tensor &self, const cytnx::cytnx_double &rhs) { return self.Mul_(rhs); })
    .def("c__imul__",
         [](cytnx::Tensor &self, const cytnx::cytnx_float &rhs) { return self.Mul_(rhs); })
    .def("c__imul__",
         [](cytnx::Tensor &self, const cytnx::cytnx_int64 &rhs) { return self.Mul_(rhs); })
    .def("c__imul__",
         [](cytnx::Tensor &self, const cytnx::cytnx_uint64 &rhs) { return self.Mul_(rhs); })
    .def("c__imul__",
         [](cytnx::Tensor &self, const cytnx::cytnx_int32 &rhs) { return self.Mul_(rhs); })
    .def("c__imul__",
         [](cytnx::Tensor &self, const cytnx::cytnx_uint32 &rhs) { return self.Mul_(rhs); })
    .def("c__imul__",
         [](cytnx::Tensor &self, const cytnx::cytnx_int16 &rhs) { return self.Mul_(rhs); })
    .def("c__imul__",
         [](cytnx::Tensor &self, const cytnx::cytnx_uint16 &rhs) { return self.Mul_(rhs); })
    .def("c__imul__",
         [](cytnx::Tensor &self, const cytnx::cytnx_bool &rhs) { return self.Mul_(rhs); })

    .def("__truediv__", [](cytnx::Tensor &self, const cytnx::Tensor &rhs) { return self.Div(rhs); })
    .def("__truediv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_complex128 &rhs) { return self.Div(rhs); })
    .def("__truediv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_complex64 &rhs) { return self.Div(rhs); })
    .def("__truediv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_double &rhs) { return self.Div(rhs); })
    .def("__truediv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_float &rhs) { return self.Div(rhs); })
    .def("__truediv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_int64 &rhs) { return self.Div(rhs); })
    .def("__truediv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_uint64 &rhs) { return self.Div(rhs); })
    .def("__truediv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_int32 &rhs) { return self.Div(rhs); })
    .def("__truediv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_uint32 &rhs) { return self.Div(rhs); })
    .def("__truediv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_int16 &rhs) { return self.Div(rhs); })
    .def("__truediv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_uint16 &rhs) { return self.Div(rhs); })
    .def("__truediv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_bool &rhs) { return self.Div(rhs); })

    .def("__rtruediv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_complex128 &lhs) {
           return cytnx::linalg::Div(lhs, self);
         })
    .def("__rtruediv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_complex64 &lhs) {
           return cytnx::linalg::Div(lhs, self);
         })
    .def("__rtruediv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_double &lhs) {
           return cytnx::linalg::Div(lhs, self);
         })
    .def("__rtruediv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_float &lhs) {
           return cytnx::linalg::Div(lhs, self);
         })
    .def("__rtruediv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_int64 &lhs) {
           return cytnx::linalg::Div(lhs, self);
         })
    .def("__rtruediv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_uint64 &lhs) {
           return cytnx::linalg::Div(lhs, self);
         })
    .def("__rtruediv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_int32 &lhs) {
           return cytnx::linalg::Div(lhs, self);
         })
    .def("__rtruediv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_uint32 &lhs) {
           return cytnx::linalg::Div(lhs, self);
         })
    .def("__rtruediv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_int16 &lhs) {
           return cytnx::linalg::Div(lhs, self);
         })
    .def("__rtruediv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_uint16 &lhs) {
           return cytnx::linalg::Div(lhs, self);
         })
    .def("__rtruediv__", [](cytnx::Tensor &self,
                            const cytnx::cytnx_bool &lhs) { return cytnx::linalg::Div(lhs, self); })
    .def("c__itruediv__",
         [](cytnx::Tensor &self, const cytnx::Tensor &rhs) {
           return self.Div_(rhs);
         })  // these will return self!
    .def("c__itruediv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_complex128 &rhs) { return self.Div_(rhs); })
    .def("c__itruediv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_complex64 &rhs) { return self.Div_(rhs); })
    .def("c__itruediv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_double &rhs) { return self.Div_(rhs); })
    .def("c__itruediv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_float &rhs) { return self.Div_(rhs); })
    .def("c__itruediv__",
         [](cytnx::Tensor &self,
            const cytnx::cytnx_int64 &rhs) {  // std::cout << "vchkp_i64" << std::endl;
           return self.Div_(rhs);
         })
    .def("c__itruediv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_uint64 &rhs) { return self.Div_(rhs); })
    .def("c__itruediv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_int32 &rhs) { return self.Div_(rhs); })
    .def("c__itruediv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_uint32 &rhs) { return self.Div_(rhs); })
    .def("c__itruediv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_int16 &rhs) { return self.Div_(rhs); })
    .def("c__itruediv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_uint16 &rhs) { return self.Div_(rhs); })
    .def("c__itruediv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_bool &rhs) { return self.Div_(rhs); })

    .def("__floordiv__",
         [](cytnx::Tensor &self, const cytnx::Tensor &rhs) { return self.Div(rhs); })
    .def("__floordiv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_complex128 &rhs) { return self.Div(rhs); })
    .def("__floordiv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_complex64 &rhs) { return self.Div(rhs); })
    .def("__floordiv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_double &rhs) { return self.Div(rhs); })
    .def("__floordiv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_float &rhs) { return self.Div(rhs); })
    .def("__floordiv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_int64 &rhs) { return self.Div(rhs); })
    .def("__floordiv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_uint64 &rhs) { return self.Div(rhs); })
    .def("__floordiv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_int32 &rhs) { return self.Div(rhs); })
    .def("__floordiv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_uint32 &rhs) { return self.Div(rhs); })
    .def("__floordiv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_int16 &rhs) { return self.Div(rhs); })
    .def("__floordiv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_uint16 &rhs) { return self.Div(rhs); })
    .def("__floordiv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_bool &rhs) { return self.Div(rhs); })

    .def("__rfloordiv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_complex128 &lhs) {
           return cytnx::linalg::Div(lhs, self);
         })
    .def("__rfloordiv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_complex64 &lhs) {
           return cytnx::linalg::Div(lhs, self);
         })
    .def("__rfloordiv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_double &lhs) {
           return cytnx::linalg::Div(lhs, self);
         })
    .def("__rfloordiv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_float &lhs) {
           return cytnx::linalg::Div(lhs, self);
         })
    .def("__rfloordiv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_int64 &lhs) {
           return cytnx::linalg::Div(lhs, self);
         })
    .def("__rfloordiv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_uint64 &lhs) {
           return cytnx::linalg::Div(lhs, self);
         })
    .def("__rfloordiv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_int32 &lhs) {
           return cytnx::linalg::Div(lhs, self);
         })
    .def("__rfloordiv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_uint32 &lhs) {
           return cytnx::linalg::Div(lhs, self);
         })
    .def("__rfloordiv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_int16 &lhs) {
           return cytnx::linalg::Div(lhs, self);
         })
    .def("__rfloordiv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_uint16 &lhs) {
           return cytnx::linalg::Div(lhs, self);
         })
    .def("__rfloordiv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_bool &lhs) {
           return cytnx::linalg::Div(lhs, self);
         })

    .def("c__ifloordiv__",
         [](cytnx::Tensor &self, const cytnx::Tensor &rhs) {
           return self.Div_(rhs);
         })  // these will return self!
    .def("c__ifloordiv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_complex128 &rhs) { return self.Div_(rhs); })
    .def("c__ifloordiv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_complex64 &rhs) { return self.Div_(rhs); })
    .def("c__ifloordiv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_double &rhs) { return self.Div_(rhs); })
    .def("c__ifloordiv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_float &rhs) { return self.Div_(rhs); })
    .def("c__ifloordiv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_int64 &rhs) { return self.Div_(rhs); })
    .def("c__ifloordiv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_uint64 &rhs) { return self.Div_(rhs); })
    .def("c__ifloordiv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_int32 &rhs) { return self.Div_(rhs); })
    .def("c__ifloordiv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_uint32 &rhs) { return self.Div_(rhs); })
    .def("c__ifloordiv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_int16 &rhs) { return self.Div_(rhs); })
    .def("c__ifloordiv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_uint16 &rhs) { return self.Div_(rhs); })
    .def("c__ifloordiv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_bool &rhs) { return self.Div_(rhs); })

    .def("__eq__", [](cytnx::Tensor &self, const cytnx::Tensor &rhs) { return self == rhs; })
    .def("__eq__",
         [](cytnx::Tensor &self, const cytnx::cytnx_complex128 &rhs) { return self == rhs; })
    .def("__eq__",
         [](cytnx::Tensor &self, const cytnx::cytnx_complex64 &rhs) { return self == rhs; })
    .def("__eq__", [](cytnx::Tensor &self, const cytnx::cytnx_double &rhs) { return self == rhs; })
    .def("__eq__", [](cytnx::Tensor &self, const cytnx::cytnx_float &rhs) { return self == rhs; })
    .def("__eq__", [](cytnx::Tensor &self, const cytnx::cytnx_int64 &rhs) { return self == rhs; })
    .def("__eq__", [](cytnx::Tensor &self, const cytnx::cytnx_uint64 &rhs) { return self == rhs; })
    .def("__eq__", [](cytnx::Tensor &self, const cytnx::cytnx_int32 &rhs) { return self == rhs; })
    .def("__eq__", [](cytnx::Tensor &self, const cytnx::cytnx_uint32 &rhs) { return self == rhs; })
    .def("__eq__", [](cytnx::Tensor &self, const cytnx::cytnx_int16 &rhs) { return self == rhs; })
    .def("__eq__", [](cytnx::Tensor &self, const cytnx::cytnx_uint16 &rhs) { return self == rhs; })
    .def("__eq__", [](cytnx::Tensor &self, const cytnx::cytnx_bool &rhs) { return self == rhs; })

    .def("__pow__", [](cytnx::Tensor &self,
                       const cytnx::cytnx_double &p) { return cytnx::linalg::Pow(self, p); })
    .def("c__ipow__",
         [](cytnx::Tensor &self, const cytnx::cytnx_double &p) { cytnx::linalg::Pow_(self, p); })
    .def("__matmul__", [](cytnx::Tensor &self,
                          const cytnx::Tensor &rhs) { return cytnx::linalg::Dot(self, rhs); })
    .def("c__imatmul__",
         [](cytnx::Tensor &self, const cytnx::Tensor &rhs) {
           self = cytnx::linalg::Dot(self, rhs);
           return self;
         })
    // linalg >>
    .def("Svd", &cytnx::Tensor::Svd, py::arg("is_UvT") = true)
    .def("Eigh", &cytnx::Tensor::Eigh, py::arg("is_V") = true, py::arg("row_v") = false)
    .def("cInvM_", &cytnx::Tensor::InvM_)
    .def("InvM", &cytnx::Tensor::InvM)
    .def("cInv_", &cytnx::Tensor::Inv_, py::arg("clip"))
    .def("Inv", &cytnx::Tensor::Inv, py::arg("clip"))
    .def("cConj_", &cytnx::Tensor::Conj_)
    .def("Conj", &cytnx::Tensor::Conj)
    .def("cExp_", &cytnx::Tensor::Exp_)
    .def("Exp", &cytnx::Tensor::Exp)
    .def("Pow", &cytnx::Tensor::Pow)
    .def("cPow_", &cytnx::Tensor::Pow_)
    .def("Abs", &cytnx::Tensor::Abs)
    .def("cAbs_", &cytnx::Tensor::Abs_)
    .def("Max", &cytnx::Tensor::Max)
    .def("Min", &cytnx::Tensor::Min)
    .def("Norm", &cytnx::Tensor::Norm)
    .def("Trace", &cytnx::Tensor::Trace)

    ;  // end of object line
}
#endif
