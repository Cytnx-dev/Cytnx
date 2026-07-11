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
#include "pyint_dispatch.hpp"

namespace py = pybind11;
using namespace pybind11::literals;
using pybind_cytnx::dispatch_pyint;

#ifdef BACKEND_TORCH
#else

namespace {
  bool is_empty_tuple(py::handle object) {
    return py::isinstance<py::tuple>(object) &&
           py::reinterpret_borrow<py::tuple>(object).size() == 0;
  }

  void check_tuple_rank(py::tuple args, cytnx::cytnx_uint64 rank, const char *type_name) {
    const auto index_count = static_cast<cytnx::cytnx_uint64>(args.size());
    cytnx_error_msg(index_count > rank,
                    "[ERROR] too many indices for %s: got %llu indices for rank-%llu object.%s",
                    type_name, static_cast<unsigned long long>(index_count),
                    static_cast<unsigned long long>(rank), "\n");
  }
}  // namespace

template <class T>
void f_Tensor_setitem_scal(cytnx::Tensor &self, py::object locators, const T &rc) {
  cytnx_error_msg(self.dtype() == cytnx::Type.Void,
                  "[ERROR] try to setelem to an uninitialized Tensor%s", "\n");
  if (self.rank() == 0) {
    cytnx_error_msg(!is_empty_tuple(locators),
                    "[ERROR] rank-0 Tensor can only be indexed with ().%s", "\n");
    self.set(std::vector<cytnx::Accessor>{}, rc);
    return;
  }

  ssize_t start, stop, step, slicelength;
  std::vector<cytnx::Accessor> accessors;
  if (py::isinstance<py::tuple>(locators)) {
    py::tuple Args = locators.cast<py::tuple>();
    check_tuple_rank(Args, self.rank(), "Tensor");
    cytnx::cytnx_uint64 cnt = 0;
    // mixing of slice and ints
    for (cytnx::cytnx_uint32 axis = 0; axis < Args.size(); axis++) {
      cnt++;
      // check type:
      if (py::isinstance<py::slice>(Args[axis])) {
        py::slice sls = Args[axis].cast<py::slice>();
        if (!sls.compute((ssize_t)self.shape()[axis], &start, &stop, &step, &slicelength))
          throw py::error_already_set();
        accessors.push_back(cytnx::Accessor::range(
          cytnx::cytnx_int64(start), cytnx::cytnx_int64(stop), cytnx::cytnx_int64(step)));
      } else {
        accessors.push_back(cytnx::Accessor(Args[axis].cast<cytnx::cytnx_int64>()));
      }
    }
    while (cnt < self.shape().size()) {
      cnt++;
      accessors.push_back(cytnx::Accessor::all());
    }
  } else {
    // only int
    for (cytnx::cytnx_uint32 i = 0; i < self.shape().size(); i++) {
      if (i == 0)
        accessors.push_back(cytnx::Accessor(locators.cast<cytnx::cytnx_int64>()));
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
      [](cytnx::Tensor &self, const bool &share_mem) -> py::array {
        // device on GPU? move to cpu:ref it;
        cytnx::Tensor tmpIN;
        if (self.device() >= 0) {
          if (share_mem) {
            cytnx_error_msg(true,
                            "[ERROR] the Tensor is on GPU. to have share_mem=True, move the Tensor "
                            "back to CPU by .to(Device.cpu).%s",
                            "\n");
          } else {
            tmpIN = self.to(cytnx::Device.cpu);
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
        for (auto i = shape.size(); i-- > 0;) {
          stride[i] = accu * cytnx::Type.typeSize(tmpIN.dtype());
          accu *= shape[i];
        }
        py::buffer_info npbuf;
        std::string chr_dtype;
        if (tmpIN.dtype() == cytnx::Type.ComplexDouble) {
          chr_dtype = py::format_descriptor<cytnx::cytnx_complex128>::format();
        } else if (tmpIN.dtype() == cytnx::Type.ComplexFloat) {
          chr_dtype = py::format_descriptor<cytnx::cytnx_complex64>::format();
        } else if (tmpIN.dtype() == cytnx::Type.Double) {
          chr_dtype = py::format_descriptor<cytnx::cytnx_double>::format();
        } else if (tmpIN.dtype() == cytnx::Type.Float) {
          chr_dtype = py::format_descriptor<cytnx::cytnx_float>::format();
        } else if (tmpIN.dtype() == cytnx::Type.Uint64) {
          chr_dtype = py::format_descriptor<cytnx::cytnx_uint64>::format();
        } else if (tmpIN.dtype() == cytnx::Type.Int64) {
          chr_dtype = py::format_descriptor<cytnx::cytnx_int64>::format();
        } else if (tmpIN.dtype() == cytnx::Type.Uint32) {
          chr_dtype = py::format_descriptor<cytnx::cytnx_uint32>::format();
        } else if (tmpIN.dtype() == cytnx::Type.Int32) {
          chr_dtype = py::format_descriptor<cytnx::cytnx_int32>::format();
        } else if (tmpIN.dtype() == cytnx::Type.Uint16) {
          chr_dtype = py::format_descriptor<cytnx::cytnx_uint16>::format();
        } else if (tmpIN.dtype() == cytnx::Type.Int16) {
          chr_dtype = py::format_descriptor<cytnx::cytnx_int16>::format();
        } else if (tmpIN.dtype() == cytnx::Type.Bool) {
          chr_dtype = py::format_descriptor<cytnx::cytnx_bool>::format();
        } else {
          cytnx_error_msg(true, "[ERROR] Void Type Tensor cannot convert to numpy ndarray%s", "\n");
        }

        npbuf = py::buffer_info(tmpIN.storage()._impl->data(),  // ptr
                                cytnx::Type.typeSize(tmpIN.dtype()),  // size of elem
                                chr_dtype,  // pss format
                                tmpIN.rank(),  // rank
                                shape,  // shape
                                stride  // stride
        );

        if (!share_mem) {
          // Avoid the memory passed to numpy being freed.
          tmpIN.storage().release();
        }

        return py::array(npbuf);
      },
      py::arg("share_mem") = false)
    // construction
    .def(py::init<>())
    .def(py::init<const cytnx::Tensor &>())
    .def(
      py::init<const std::vector<cytnx::cytnx_uint64> &, unsigned int, int, bool>(),
      py::arg("shape"), py::arg("dtype") = (cytnx::cytnx_uint64)cytnx::Type.Double,
      py::arg("device") = (int)cytnx::Device.cpu, py::arg("init_zero") = true)
    .def("Init",
         [](cytnx::Tensor &self, const std::vector<cytnx::cytnx_uint64> &shape,
            unsigned int dtype, int device, bool init_zero) {
           self.Init(shape, dtype, device, init_zero);
         },
         py::arg("shape"),
         py::arg("dtype") = (cytnx::cytnx_uint64)cytnx::Type.Double,
         py::arg("device") = (int)cytnx::Device.cpu, py::arg("init_zero") = true)
    .def("dtype", &cytnx::Tensor::dtype)
    .def("dtype_str", &cytnx::Tensor::dtype_str)
    .def("device", &cytnx::Tensor::device)
    .def("device_str", &cytnx::Tensor::device_str)
    .def("shape", &cytnx::Tensor::shape)
    .def("rank", &cytnx::Tensor::rank)
    .def("size", &cytnx::Tensor::size)
    .def("is_void", &cytnx::Tensor::is_void)
    .def("is_scalar", &cytnx::Tensor::is_scalar)
    .def("is_empty", &cytnx::Tensor::is_empty)
    .def("clone", &cytnx::Tensor::clone)
    .def("__copy__", &cytnx::Tensor::clone)
    .def("__deepcopy__", &cytnx::Tensor::clone)
    //.def("to", &cytnx::Tensor::to, py::arg("device"))
    // handle same device from cytnx/Tensor_conti.py
    .def(
      "to_different_device",
      [](cytnx::Tensor &self, const cytnx::cytnx_int64 &device) {
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
           return &self.permute_(c_args);
         })
    .def("permute",
         [](cytnx::Tensor &self, py::args args) -> cytnx::Tensor {
           std::vector<cytnx::cytnx_uint64> c_args = args.cast<std::vector<cytnx::cytnx_uint64>>();
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
           return &self.reshape_(c_args);
         })
    .def("reshape",
         [](cytnx::Tensor &self, py::args args) -> cytnx::Tensor {
           std::vector<cytnx::cytnx_int64> c_args = args.cast<std::vector<cytnx::cytnx_int64>>();
           return self.reshape(c_args);
         })
    //.def("astype", &cytnx::Tensor::astype,py::arg("new_type"))
    .def(
      "astype_different_dtype",
      [](cytnx::Tensor &self, const cytnx::cytnx_uint64 &dtype) {
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
      [](const std::string &fname, const unsigned int &dtype, const cytnx::cytnx_int64 &count) {
        return cytnx::Tensor::Fromfile(fname, dtype, count);
      },
      py::arg("fname"), py::arg("dtype"), py::arg("count") = cytnx::cytnx_int64(-1))

    .def_static(
      "from_storage",
      [](const cytnx::Storage &sin, const bool &is_clone) {
        cytnx::Tensor out;
        if (is_clone)
          out = cytnx::Tensor::from_storage(sin.clone());
        else
          out = cytnx::Tensor::from_storage(sin);
        return out;
      },
      py::arg("sin"), py::arg("is_clone") = false)

    .def("__len__",
         [](const cytnx::Tensor &self) {
           if (self.dtype() == cytnx::Type.Void) {
             cytnx_error_msg(true, "[ERROR] uninitialize Tensor does not have len!%s", "\n");
           }
           cytnx_error_msg(self.rank() == 0, "[ERROR] rank-0 Tensor does not have len!%s", "\n");
           return self.shape()[0];
         })
    .def("__getitem__",
         [](const cytnx::Tensor &self, py::object locators) {
           cytnx_error_msg(self.dtype() == cytnx::Type.Void,
                           "[ERROR] try to getitem from an uninitialized Tensor%s", "\n");
           if (self.rank() == 0) {
             cytnx_error_msg(!is_empty_tuple(locators),
                             "[ERROR] rank-0 Tensor can only be indexed with ().%s", "\n");
             return self.get({});
           }

           ssize_t start, stop, step, slicelength;
           std::vector<cytnx::Accessor> accessors;
           if (py::isinstance<py::tuple>(locators)) {
             py::tuple Args = locators.cast<py::tuple>();
             check_tuple_rank(Args, self.rank(), "Tensor");
             cytnx::cytnx_uint64 cnt = 0;
             // mixing of slice and ints
             for (cytnx::cytnx_uint32 axis = 0; axis < Args.size(); axis++) {
               cnt++;
               // check type:
               if (py::isinstance<py::slice>(Args[axis])) {
                 py::slice sls = Args[axis].cast<py::slice>();
                 if (!sls.compute((ssize_t)self.shape()[axis], &start, &stop, &step, &slicelength))
                   throw py::error_already_set();
                 accessors.push_back(cytnx::Accessor::range(
                   cytnx::cytnx_int64(start), cytnx::cytnx_int64(stop), cytnx::cytnx_int64(step)));
               } else {
                 accessors.push_back(cytnx::Accessor(Args[axis].cast<cytnx::cytnx_int64>()));
               }
             }
             while (cnt < self.shape().size()) {
               cnt++;
               accessors.push_back(cytnx::Accessor::all());
             }
           } else if (py::isinstance<py::slice>(locators)) {
             py::slice sls = locators.cast<py::slice>();
             if (!sls.compute((ssize_t)self.shape()[0], &start, &stop, &step, &slicelength))
               throw py::error_already_set();
             // if(slicelength == self.shape()[0]) accessors.push_back(cytnx::Accessor::all());
             accessors.push_back(cytnx::Accessor::range(start, stop, step));
             for (cytnx::cytnx_uint32 axis = 1; axis < self.shape().size(); axis++) {
               accessors.push_back(cytnx::Accessor::all());
             }

           } else {
             //  only int
             for (cytnx::cytnx_uint32 i = 0; i < self.shape().size(); i++) {
               if (i == 0)
                 accessors.push_back(cytnx::Accessor(locators.cast<cytnx::cytnx_int64>()));
               else
                 accessors.push_back(cytnx::Accessor::all());
             }
           }

           return self.get(accessors);
         })

    .def("__setitem__",
         [](cytnx::Tensor &self, py::object locators, const cytnx::Tensor &rhs) {
           cytnx_error_msg(self.dtype() == cytnx::Type.Void,
                           "[ERROR] try to setelem to an uninitialized Tensor%s", "\n");
           if (self.rank() == 0) {
             cytnx_error_msg(!is_empty_tuple(locators),
                             "[ERROR] rank-0 Tensor can only be indexed with ().%s", "\n");
             self.set(std::vector<cytnx::Accessor>{}, rhs);
             return;
           }

           ssize_t start, stop, step, slicelength;
           std::vector<cytnx::Accessor> accessors;
           if (py::isinstance<py::tuple>(locators)) {
             py::tuple Args = locators.cast<py::tuple>();
             check_tuple_rank(Args, self.rank(), "Tensor");
             cytnx::cytnx_uint64 cnt = 0;
             // mixing of slice and ints
             for (cytnx::cytnx_uint32 axis = 0; axis < Args.size(); axis++) {
               cnt++;
               // check type:
               if (py::isinstance<py::slice>(Args[axis])) {
                 py::slice sls = Args[axis].cast<py::slice>();
                 if (!sls.compute((ssize_t)self.shape()[axis], &start, &stop, &step, &slicelength))
                   throw py::error_already_set();
                 accessors.push_back(cytnx::Accessor::range(
                   cytnx::cytnx_int64(start), cytnx::cytnx_int64(stop), cytnx::cytnx_int64(step)));
               } else {
                 accessors.push_back(cytnx::Accessor(Args[axis].cast<cytnx::cytnx_int64>()));
               }
             }
             while (cnt < self.shape().size()) {
               cnt++;
               accessors.push_back(cytnx::Accessor::all());
             }
           } else if (py::isinstance<py::slice>(locators)) {
             py::slice sls = locators.cast<py::slice>();
             if (!sls.compute((ssize_t)self.shape()[0], &start, &stop, &step, &slicelength))
               throw py::error_already_set();
             // if(slicelength == self.shape()[0]) accessors.push_back(cytnx::Accessor::all());
             accessors.push_back(cytnx::Accessor::range(start, stop, step));
             for (cytnx::cytnx_uint32 axis = 1; axis < self.shape().size(); axis++) {
               accessors.push_back(cytnx::Accessor::all());
             }

           } else {
             // only int
             for (cytnx::cytnx_uint32 i = 0; i < self.shape().size(); i++) {
               if (i == 0)
                 accessors.push_back(cytnx::Accessor(locators.cast<cytnx::cytnx_int64>()));
               else
                 accessors.push_back(cytnx::Accessor::all());
             }
           }

           self.set(accessors, rhs);
         })
    // __setitem__ keep-set; registration ORDER matters -- see "KEEP-SET
    // ORDERING" in pybind/pyint_dispatch.hpp. Group-specific note: before
    // this rewrite, __setitem__ had zero numpy_scalar overloads AND
    // cytnx_complex128 registered first, so `t[0] = np.float32(x)` on a
    // real-dtype Tensor was captured by the complex128 overload (the
    // __float__-fallback trap) and raised "cannot assign complex element to
    // real container" instead of just mis-preserving dtype.
    .def("__setitem__",
         [](cytnx::Tensor &self, py::object locators, const py::numpy_scalar<float> &rc) {
           f_Tensor_setitem_scal(self, locators, static_cast<cytnx::cytnx_float>(rc));
         })
    .def("__setitem__",
         [](cytnx::Tensor &self, py::object locators,
            const py::numpy_scalar<std::complex<float>> &rc) {
           f_Tensor_setitem_scal(self, locators, static_cast<cytnx::cytnx_complex64>(rc));
         })
    .def("__setitem__",
         [](cytnx::Tensor &self, py::object locators, const py::numpy_scalar<int64_t> &rc) {
           f_Tensor_setitem_scal(self, locators, static_cast<cytnx::cytnx_int64>(rc));
         })
    .def("__setitem__",
         [](cytnx::Tensor &self, py::object locators, const py::numpy_scalar<uint64_t> &rc) {
           f_Tensor_setitem_scal(self, locators, static_cast<cytnx::cytnx_uint64>(rc));
         })
    .def("__setitem__",
         [](cytnx::Tensor &self, py::object locators, const py::numpy_scalar<int32_t> &rc) {
           f_Tensor_setitem_scal(self, locators, static_cast<cytnx::cytnx_int32>(rc));
         })
    .def("__setitem__",
         [](cytnx::Tensor &self, py::object locators, const py::numpy_scalar<uint32_t> &rc) {
           f_Tensor_setitem_scal(self, locators, static_cast<cytnx::cytnx_uint32>(rc));
         })
    .def("__setitem__",
         [](cytnx::Tensor &self, py::object locators, const py::numpy_scalar<int16_t> &rc) {
           f_Tensor_setitem_scal(self, locators, static_cast<cytnx::cytnx_int16>(rc));
         })
    .def("__setitem__",
         [](cytnx::Tensor &self, py::object locators, const py::numpy_scalar<uint16_t> &rc) {
           f_Tensor_setitem_scal(self, locators, static_cast<cytnx::cytnx_uint16>(rc));
         })
    .def("__setitem__",
         [](cytnx::Tensor &self, py::object locators, const py::numpy_scalar<bool> &rc) {
           f_Tensor_setitem_scal(self, locators, static_cast<cytnx::cytnx_bool>(rc));
         })
    .def("__setitem__",
         [](cytnx::Tensor &self, py::object locators, const py::int_ &rc) {
           dispatch_pyint(
             rc, [&](auto v) { f_Tensor_setitem_scal(self, locators, v); });
         })
    .def("__setitem__", &f_Tensor_setitem_scal<cytnx::cytnx_double>)
    .def("__setitem__", &f_Tensor_setitem_scal<cytnx::cytnx_complex128>)
    .def("__setitem__", &f_Tensor_setitem_scal<cytnx::Scalar>)
    // arithmetic >>
    .def("__neg__",
         [](cytnx::Tensor &self) {
           if (self.dtype() == cytnx::Type.Double) {
             return cytnx::linalg::Mul(cytnx::cytnx_double(-1), self);
           } else if (self.dtype() == cytnx::Type.ComplexDouble) {
             return cytnx::linalg::Mul(cytnx::cytnx_complex128(-1, 0), self);
           } else if (self.dtype() == cytnx::Type.Float) {
             return cytnx::linalg::Mul(cytnx::cytnx_float(-1), self);
           } else if (self.dtype() == cytnx::Type.ComplexFloat) {
             return cytnx::linalg::Mul(cytnx::cytnx_complex64(-1, 0), self);
           } else {
             return cytnx::linalg::Mul(-1, self);
           }
         })
    .def("__pos__", [](cytnx::Tensor &self) { return self; })
    // keep-set; registration ORDER matters -- see "KEEP-SET ORDERING" in pybind/pyint_dispatch.hpp.
    .def("__add__", [](cytnx::Tensor &self, const cytnx::Tensor &rhs) { return self.Add(rhs); })
    .def("__add__",
         [](cytnx::Tensor &self, const py::numpy_scalar<float> &rhs) {
           return self.Add(static_cast<cytnx::cytnx_float>(rhs));
         })
    .def("__add__",
         [](cytnx::Tensor &self, const py::numpy_scalar<std::complex<float>> &rhs) {
           return self.Add(static_cast<cytnx::cytnx_complex64>(rhs));
         })
    .def("__add__",
         [](cytnx::Tensor &self, const py::numpy_scalar<int64_t> &rhs) {
           return self.Add(static_cast<cytnx::cytnx_int64>(rhs));
         })
    .def("__add__",
         [](cytnx::Tensor &self, const py::numpy_scalar<uint64_t> &rhs) {
           return self.Add(static_cast<cytnx::cytnx_uint64>(rhs));
         })
    .def("__add__",
         [](cytnx::Tensor &self, const py::numpy_scalar<int32_t> &rhs) {
           return self.Add(static_cast<cytnx::cytnx_int32>(rhs));
         })
    .def("__add__",
         [](cytnx::Tensor &self, const py::numpy_scalar<uint32_t> &rhs) {
           return self.Add(static_cast<cytnx::cytnx_uint32>(rhs));
         })
    .def("__add__",
         [](cytnx::Tensor &self, const py::numpy_scalar<int16_t> &rhs) {
           return self.Add(static_cast<cytnx::cytnx_int16>(rhs));
         })
    .def("__add__",
         [](cytnx::Tensor &self, const py::numpy_scalar<uint16_t> &rhs) {
           return self.Add(static_cast<cytnx::cytnx_uint16>(rhs));
         })
    .def("__add__",
         [](cytnx::Tensor &self, const py::numpy_scalar<bool> &rhs) {
           return self.Add(static_cast<cytnx::cytnx_bool>(rhs));
         })
    .def("__add__",
         [](cytnx::Tensor &self, const py::int_ &rhs) {
           return dispatch_pyint(rhs, [&](auto v) { return self.Add(v); });
         })
    .def("__add__",
         [](cytnx::Tensor &self, const cytnx::cytnx_double &rhs) { return self.Add(rhs); })
    .def("__add__",
         [](cytnx::Tensor &self, const cytnx::cytnx_complex128 &rhs) { return self.Add(rhs); })
    .def("__add__", [](cytnx::Tensor &self, const cytnx::Scalar &rhs) { return self.Add(rhs); })

    // keep-set; registration ORDER matters -- see "KEEP-SET ORDERING" in pybind/pyint_dispatch.hpp.
    // NOTE (pre-existing, out of scope here): a numpy scalar on the LEFT
    // (e.g. np.float32(1.0) + t) does not reach this __r*__ binding at all --
    // Tensor defines __iter__, so numpy's ufunc machinery treats it as an
    // array-like and tries to iterate it instead, raising
    // "TypeError: 'TensorIterator' object is not iterable" (issue #692).
    .def("__radd__",
         [](cytnx::Tensor &self, const py::numpy_scalar<float> &lhs) {
           return cytnx::linalg::Add(static_cast<cytnx::cytnx_float>(lhs), self);
         })
    .def("__radd__",
         [](cytnx::Tensor &self, const py::numpy_scalar<std::complex<float>> &lhs) {
           return cytnx::linalg::Add(static_cast<cytnx::cytnx_complex64>(lhs), self);
         })
    .def("__radd__",
         [](cytnx::Tensor &self, const py::numpy_scalar<int64_t> &lhs) {
           return cytnx::linalg::Add(static_cast<cytnx::cytnx_int64>(lhs), self);
         })
    .def("__radd__",
         [](cytnx::Tensor &self, const py::numpy_scalar<uint64_t> &lhs) {
           return cytnx::linalg::Add(static_cast<cytnx::cytnx_uint64>(lhs), self);
         })
    .def("__radd__",
         [](cytnx::Tensor &self, const py::numpy_scalar<int32_t> &lhs) {
           return cytnx::linalg::Add(static_cast<cytnx::cytnx_int32>(lhs), self);
         })
    .def("__radd__",
         [](cytnx::Tensor &self, const py::numpy_scalar<uint32_t> &lhs) {
           return cytnx::linalg::Add(static_cast<cytnx::cytnx_uint32>(lhs), self);
         })
    .def("__radd__",
         [](cytnx::Tensor &self, const py::numpy_scalar<int16_t> &lhs) {
           return cytnx::linalg::Add(static_cast<cytnx::cytnx_int16>(lhs), self);
         })
    .def("__radd__",
         [](cytnx::Tensor &self, const py::numpy_scalar<uint16_t> &lhs) {
           return cytnx::linalg::Add(static_cast<cytnx::cytnx_uint16>(lhs), self);
         })
    .def("__radd__",
         [](cytnx::Tensor &self, const py::numpy_scalar<bool> &lhs) {
           return cytnx::linalg::Add(static_cast<cytnx::cytnx_bool>(lhs), self);
         })
    .def("__radd__",
         [](cytnx::Tensor &self, const py::int_ &lhs) {
           return dispatch_pyint(lhs, [&](auto v) { return cytnx::linalg::Add(v, self); });
         })
    .def("__radd__",
         [](cytnx::Tensor &self, const cytnx::cytnx_double &lhs) { return cytnx::linalg::Add(lhs, self); })
    .def("__radd__",
         [](cytnx::Tensor &self, const cytnx::cytnx_complex128 &lhs) { return cytnx::linalg::Add(lhs, self); })
    .def("__radd__", [](cytnx::Tensor &self, const cytnx::Scalar &lhs) { return cytnx::linalg::Add(lhs, self); })

    // keep-set; registration ORDER matters -- see "KEEP-SET ORDERING" in pybind/pyint_dispatch.hpp.
    .def("__iadd__",
         [](py::object self, const cytnx::Tensor &rhs) {
           self.cast<cytnx::Tensor &>().Add_(rhs);
           return self;
         })
    .def("__iadd__",
         [](py::object self, const py::numpy_scalar<float> &rhs) {
           self.cast<cytnx::Tensor &>().Add_(static_cast<cytnx::cytnx_float>(rhs));
           return self;
         })
    .def("__iadd__",
         [](py::object self, const py::numpy_scalar<std::complex<float>> &rhs) {
           self.cast<cytnx::Tensor &>().Add_(static_cast<cytnx::cytnx_complex64>(rhs));
           return self;
         })
    .def("__iadd__",
         [](py::object self, const py::numpy_scalar<int64_t> &rhs) {
           self.cast<cytnx::Tensor &>().Add_(static_cast<cytnx::cytnx_int64>(rhs));
           return self;
         })
    .def("__iadd__",
         [](py::object self, const py::numpy_scalar<uint64_t> &rhs) {
           self.cast<cytnx::Tensor &>().Add_(static_cast<cytnx::cytnx_uint64>(rhs));
           return self;
         })
    .def("__iadd__",
         [](py::object self, const py::numpy_scalar<int32_t> &rhs) {
           self.cast<cytnx::Tensor &>().Add_(static_cast<cytnx::cytnx_int32>(rhs));
           return self;
         })
    .def("__iadd__",
         [](py::object self, const py::numpy_scalar<uint32_t> &rhs) {
           self.cast<cytnx::Tensor &>().Add_(static_cast<cytnx::cytnx_uint32>(rhs));
           return self;
         })
    .def("__iadd__",
         [](py::object self, const py::numpy_scalar<int16_t> &rhs) {
           self.cast<cytnx::Tensor &>().Add_(static_cast<cytnx::cytnx_int16>(rhs));
           return self;
         })
    .def("__iadd__",
         [](py::object self, const py::numpy_scalar<uint16_t> &rhs) {
           self.cast<cytnx::Tensor &>().Add_(static_cast<cytnx::cytnx_uint16>(rhs));
           return self;
         })
    .def("__iadd__",
         [](py::object self, const py::numpy_scalar<bool> &rhs) {
           self.cast<cytnx::Tensor &>().Add_(static_cast<cytnx::cytnx_bool>(rhs));
           return self;
         })
    .def("__iadd__",
         [](py::object self, const py::int_ &rhs) {
           dispatch_pyint(rhs, [&](auto v) { self.cast<cytnx::Tensor &>().Add_(v); });
           return self;
         })
    .def("__iadd__",
         [](py::object self, const cytnx::cytnx_double &rhs) {
           self.cast<cytnx::Tensor &>().Add_(rhs);
           return self;
         })
    .def("__iadd__",
         [](py::object self, const cytnx::cytnx_complex128 &rhs) {
           self.cast<cytnx::Tensor &>().Add_(rhs);
           return self;
         })
    .def("__iadd__",
         [](py::object self, const cytnx::Scalar &rhs) {
           self.cast<cytnx::Tensor &>().Add_(rhs);
           return self;
         })

    // keep-set; registration ORDER matters -- see "KEEP-SET ORDERING" in pybind/pyint_dispatch.hpp.
    .def("__sub__", [](cytnx::Tensor &self, const cytnx::Tensor &rhs) { return self.Sub(rhs); })
    .def("__sub__",
         [](cytnx::Tensor &self, const py::numpy_scalar<float> &rhs) {
           return self.Sub(static_cast<cytnx::cytnx_float>(rhs));
         })
    .def("__sub__",
         [](cytnx::Tensor &self, const py::numpy_scalar<std::complex<float>> &rhs) {
           return self.Sub(static_cast<cytnx::cytnx_complex64>(rhs));
         })
    .def("__sub__",
         [](cytnx::Tensor &self, const py::numpy_scalar<int64_t> &rhs) {
           return self.Sub(static_cast<cytnx::cytnx_int64>(rhs));
         })
    .def("__sub__",
         [](cytnx::Tensor &self, const py::numpy_scalar<uint64_t> &rhs) {
           return self.Sub(static_cast<cytnx::cytnx_uint64>(rhs));
         })
    .def("__sub__",
         [](cytnx::Tensor &self, const py::numpy_scalar<int32_t> &rhs) {
           return self.Sub(static_cast<cytnx::cytnx_int32>(rhs));
         })
    .def("__sub__",
         [](cytnx::Tensor &self, const py::numpy_scalar<uint32_t> &rhs) {
           return self.Sub(static_cast<cytnx::cytnx_uint32>(rhs));
         })
    .def("__sub__",
         [](cytnx::Tensor &self, const py::numpy_scalar<int16_t> &rhs) {
           return self.Sub(static_cast<cytnx::cytnx_int16>(rhs));
         })
    .def("__sub__",
         [](cytnx::Tensor &self, const py::numpy_scalar<uint16_t> &rhs) {
           return self.Sub(static_cast<cytnx::cytnx_uint16>(rhs));
         })
    .def("__sub__",
         [](cytnx::Tensor &self, const py::numpy_scalar<bool> &rhs) {
           return self.Sub(static_cast<cytnx::cytnx_bool>(rhs));
         })
    .def("__sub__",
         [](cytnx::Tensor &self, const py::int_ &rhs) {
           return dispatch_pyint(rhs, [&](auto v) { return self.Sub(v); });
         })
    .def("__sub__",
         [](cytnx::Tensor &self, const cytnx::cytnx_double &rhs) { return self.Sub(rhs); })
    .def("__sub__",
         [](cytnx::Tensor &self, const cytnx::cytnx_complex128 &rhs) { return self.Sub(rhs); })
    .def("__sub__", [](cytnx::Tensor &self, const cytnx::Scalar &rhs) { return self.Sub(rhs); })

    // keep-set; registration ORDER matters -- see "KEEP-SET ORDERING" in pybind/pyint_dispatch.hpp.
    // NOTE (pre-existing, out of scope here): a numpy scalar on the LEFT
    // (e.g. np.float32(1.0) + t) does not reach this __r*__ binding at all --
    // Tensor defines __iter__, so numpy's ufunc machinery treats it as an
    // array-like and tries to iterate it instead, raising
    // "TypeError: 'TensorIterator' object is not iterable" (issue #692).
    .def("__rsub__",
         [](cytnx::Tensor &self, const py::numpy_scalar<float> &lhs) {
           return cytnx::linalg::Sub(static_cast<cytnx::cytnx_float>(lhs), self);
         })
    .def("__rsub__",
         [](cytnx::Tensor &self, const py::numpy_scalar<std::complex<float>> &lhs) {
           return cytnx::linalg::Sub(static_cast<cytnx::cytnx_complex64>(lhs), self);
         })
    .def("__rsub__",
         [](cytnx::Tensor &self, const py::numpy_scalar<int64_t> &lhs) {
           return cytnx::linalg::Sub(static_cast<cytnx::cytnx_int64>(lhs), self);
         })
    .def("__rsub__",
         [](cytnx::Tensor &self, const py::numpy_scalar<uint64_t> &lhs) {
           return cytnx::linalg::Sub(static_cast<cytnx::cytnx_uint64>(lhs), self);
         })
    .def("__rsub__",
         [](cytnx::Tensor &self, const py::numpy_scalar<int32_t> &lhs) {
           return cytnx::linalg::Sub(static_cast<cytnx::cytnx_int32>(lhs), self);
         })
    .def("__rsub__",
         [](cytnx::Tensor &self, const py::numpy_scalar<uint32_t> &lhs) {
           return cytnx::linalg::Sub(static_cast<cytnx::cytnx_uint32>(lhs), self);
         })
    .def("__rsub__",
         [](cytnx::Tensor &self, const py::numpy_scalar<int16_t> &lhs) {
           return cytnx::linalg::Sub(static_cast<cytnx::cytnx_int16>(lhs), self);
         })
    .def("__rsub__",
         [](cytnx::Tensor &self, const py::numpy_scalar<uint16_t> &lhs) {
           return cytnx::linalg::Sub(static_cast<cytnx::cytnx_uint16>(lhs), self);
         })
    .def("__rsub__",
         [](cytnx::Tensor &self, const py::numpy_scalar<bool> &lhs) {
           return cytnx::linalg::Sub(static_cast<cytnx::cytnx_bool>(lhs), self);
         })
    .def("__rsub__",
         [](cytnx::Tensor &self, const py::int_ &lhs) {
           return dispatch_pyint(lhs, [&](auto v) { return cytnx::linalg::Sub(v, self); });
         })
    .def("__rsub__",
         [](cytnx::Tensor &self, const cytnx::cytnx_double &lhs) { return cytnx::linalg::Sub(lhs, self); })
    .def("__rsub__",
         [](cytnx::Tensor &self, const cytnx::cytnx_complex128 &lhs) { return cytnx::linalg::Sub(lhs, self); })
    .def("__rsub__", [](cytnx::Tensor &self, const cytnx::Scalar &lhs) { return cytnx::linalg::Sub(lhs, self); })

    // keep-set; registration ORDER matters -- see "KEEP-SET ORDERING" in pybind/pyint_dispatch.hpp.
    .def("__isub__",
         [](py::object self, const cytnx::Tensor &rhs) {
           self.cast<cytnx::Tensor &>().Sub_(rhs);
           return self;
         })
    .def("__isub__",
         [](py::object self, const py::numpy_scalar<float> &rhs) {
           self.cast<cytnx::Tensor &>().Sub_(static_cast<cytnx::cytnx_float>(rhs));
           return self;
         })
    .def("__isub__",
         [](py::object self, const py::numpy_scalar<std::complex<float>> &rhs) {
           self.cast<cytnx::Tensor &>().Sub_(static_cast<cytnx::cytnx_complex64>(rhs));
           return self;
         })
    .def("__isub__",
         [](py::object self, const py::numpy_scalar<int64_t> &rhs) {
           self.cast<cytnx::Tensor &>().Sub_(static_cast<cytnx::cytnx_int64>(rhs));
           return self;
         })
    .def("__isub__",
         [](py::object self, const py::numpy_scalar<uint64_t> &rhs) {
           self.cast<cytnx::Tensor &>().Sub_(static_cast<cytnx::cytnx_uint64>(rhs));
           return self;
         })
    .def("__isub__",
         [](py::object self, const py::numpy_scalar<int32_t> &rhs) {
           self.cast<cytnx::Tensor &>().Sub_(static_cast<cytnx::cytnx_int32>(rhs));
           return self;
         })
    .def("__isub__",
         [](py::object self, const py::numpy_scalar<uint32_t> &rhs) {
           self.cast<cytnx::Tensor &>().Sub_(static_cast<cytnx::cytnx_uint32>(rhs));
           return self;
         })
    .def("__isub__",
         [](py::object self, const py::numpy_scalar<int16_t> &rhs) {
           self.cast<cytnx::Tensor &>().Sub_(static_cast<cytnx::cytnx_int16>(rhs));
           return self;
         })
    .def("__isub__",
         [](py::object self, const py::numpy_scalar<uint16_t> &rhs) {
           self.cast<cytnx::Tensor &>().Sub_(static_cast<cytnx::cytnx_uint16>(rhs));
           return self;
         })
    .def("__isub__",
         [](py::object self, const py::numpy_scalar<bool> &rhs) {
           self.cast<cytnx::Tensor &>().Sub_(static_cast<cytnx::cytnx_bool>(rhs));
           return self;
         })
    .def("__isub__",
         [](py::object self, const py::int_ &rhs) {
           dispatch_pyint(rhs, [&](auto v) { self.cast<cytnx::Tensor &>().Sub_(v); });
           return self;
         })
    .def("__isub__",
         [](py::object self, const cytnx::cytnx_double &rhs) {
           self.cast<cytnx::Tensor &>().Sub_(rhs);
           return self;
         })
    .def("__isub__",
         [](py::object self, const cytnx::cytnx_complex128 &rhs) {
           self.cast<cytnx::Tensor &>().Sub_(rhs);
           return self;
         })
    .def("__isub__",
         [](py::object self, const cytnx::Scalar &rhs) {
           self.cast<cytnx::Tensor &>().Sub_(rhs);
           return self;
         })

    // keep-set; registration ORDER matters -- see "KEEP-SET ORDERING" in pybind/pyint_dispatch.hpp.
    .def("__mul__", [](cytnx::Tensor &self, const cytnx::Tensor &rhs) { return self.Mul(rhs); })
    .def("__mul__",
         [](cytnx::Tensor &self, const py::numpy_scalar<float> &rhs) {
           return self.Mul(static_cast<cytnx::cytnx_float>(rhs));
         })
    .def("__mul__",
         [](cytnx::Tensor &self, const py::numpy_scalar<std::complex<float>> &rhs) {
           return self.Mul(static_cast<cytnx::cytnx_complex64>(rhs));
         })
    .def("__mul__",
         [](cytnx::Tensor &self, const py::numpy_scalar<int64_t> &rhs) {
           return self.Mul(static_cast<cytnx::cytnx_int64>(rhs));
         })
    .def("__mul__",
         [](cytnx::Tensor &self, const py::numpy_scalar<uint64_t> &rhs) {
           return self.Mul(static_cast<cytnx::cytnx_uint64>(rhs));
         })
    .def("__mul__",
         [](cytnx::Tensor &self, const py::numpy_scalar<int32_t> &rhs) {
           return self.Mul(static_cast<cytnx::cytnx_int32>(rhs));
         })
    .def("__mul__",
         [](cytnx::Tensor &self, const py::numpy_scalar<uint32_t> &rhs) {
           return self.Mul(static_cast<cytnx::cytnx_uint32>(rhs));
         })
    .def("__mul__",
         [](cytnx::Tensor &self, const py::numpy_scalar<int16_t> &rhs) {
           return self.Mul(static_cast<cytnx::cytnx_int16>(rhs));
         })
    .def("__mul__",
         [](cytnx::Tensor &self, const py::numpy_scalar<uint16_t> &rhs) {
           return self.Mul(static_cast<cytnx::cytnx_uint16>(rhs));
         })
    .def("__mul__",
         [](cytnx::Tensor &self, const py::numpy_scalar<bool> &rhs) {
           return self.Mul(static_cast<cytnx::cytnx_bool>(rhs));
         })
    .def("__mul__",
         [](cytnx::Tensor &self, const py::int_ &rhs) {
           return dispatch_pyint(rhs, [&](auto v) { return self.Mul(v); });
         })
    .def("__mul__",
         [](cytnx::Tensor &self, const cytnx::cytnx_double &rhs) { return self.Mul(rhs); })
    .def("__mul__",
         [](cytnx::Tensor &self, const cytnx::cytnx_complex128 &rhs) { return self.Mul(rhs); })
    .def("__mul__", [](cytnx::Tensor &self, const cytnx::Scalar &rhs) { return self.Mul(rhs); })

    // keep-set; registration ORDER matters -- see "KEEP-SET ORDERING" in pybind/pyint_dispatch.hpp.
    // NOTE (pre-existing, out of scope here): a numpy scalar on the LEFT
    // (e.g. np.float32(1.0) + t) does not reach this __r*__ binding at all --
    // Tensor defines __iter__, so numpy's ufunc machinery treats it as an
    // array-like and tries to iterate it instead, raising
    // "TypeError: 'TensorIterator' object is not iterable" (issue #692).
    .def("__rmul__",
         [](cytnx::Tensor &self, const py::numpy_scalar<float> &lhs) {
           return cytnx::linalg::Mul(static_cast<cytnx::cytnx_float>(lhs), self);
         })
    .def("__rmul__",
         [](cytnx::Tensor &self, const py::numpy_scalar<std::complex<float>> &lhs) {
           return cytnx::linalg::Mul(static_cast<cytnx::cytnx_complex64>(lhs), self);
         })
    .def("__rmul__",
         [](cytnx::Tensor &self, const py::numpy_scalar<int64_t> &lhs) {
           return cytnx::linalg::Mul(static_cast<cytnx::cytnx_int64>(lhs), self);
         })
    .def("__rmul__",
         [](cytnx::Tensor &self, const py::numpy_scalar<uint64_t> &lhs) {
           return cytnx::linalg::Mul(static_cast<cytnx::cytnx_uint64>(lhs), self);
         })
    .def("__rmul__",
         [](cytnx::Tensor &self, const py::numpy_scalar<int32_t> &lhs) {
           return cytnx::linalg::Mul(static_cast<cytnx::cytnx_int32>(lhs), self);
         })
    .def("__rmul__",
         [](cytnx::Tensor &self, const py::numpy_scalar<uint32_t> &lhs) {
           return cytnx::linalg::Mul(static_cast<cytnx::cytnx_uint32>(lhs), self);
         })
    .def("__rmul__",
         [](cytnx::Tensor &self, const py::numpy_scalar<int16_t> &lhs) {
           return cytnx::linalg::Mul(static_cast<cytnx::cytnx_int16>(lhs), self);
         })
    .def("__rmul__",
         [](cytnx::Tensor &self, const py::numpy_scalar<uint16_t> &lhs) {
           return cytnx::linalg::Mul(static_cast<cytnx::cytnx_uint16>(lhs), self);
         })
    .def("__rmul__",
         [](cytnx::Tensor &self, const py::numpy_scalar<bool> &lhs) {
           return cytnx::linalg::Mul(static_cast<cytnx::cytnx_bool>(lhs), self);
         })
    .def("__rmul__",
         [](cytnx::Tensor &self, const py::int_ &lhs) {
           return dispatch_pyint(lhs, [&](auto v) { return cytnx::linalg::Mul(v, self); });
         })
    .def("__rmul__",
         [](cytnx::Tensor &self, const cytnx::cytnx_double &lhs) { return cytnx::linalg::Mul(lhs, self); })
    .def("__rmul__",
         [](cytnx::Tensor &self, const cytnx::cytnx_complex128 &lhs) { return cytnx::linalg::Mul(lhs, self); })
    .def("__rmul__", [](cytnx::Tensor &self, const cytnx::Scalar &lhs) { return cytnx::linalg::Mul(lhs, self); })

    // keep-set; registration ORDER matters -- see "KEEP-SET ORDERING" in pybind/pyint_dispatch.hpp.
    .def("__imul__",
         [](py::object self, const cytnx::Tensor &rhs) {
           self.cast<cytnx::Tensor &>().Mul_(rhs);
           return self;
         })
    .def("__imul__",
         [](py::object self, const py::numpy_scalar<float> &rhs) {
           self.cast<cytnx::Tensor &>().Mul_(static_cast<cytnx::cytnx_float>(rhs));
           return self;
         })
    .def("__imul__",
         [](py::object self, const py::numpy_scalar<std::complex<float>> &rhs) {
           self.cast<cytnx::Tensor &>().Mul_(static_cast<cytnx::cytnx_complex64>(rhs));
           return self;
         })
    .def("__imul__",
         [](py::object self, const py::numpy_scalar<int64_t> &rhs) {
           self.cast<cytnx::Tensor &>().Mul_(static_cast<cytnx::cytnx_int64>(rhs));
           return self;
         })
    .def("__imul__",
         [](py::object self, const py::numpy_scalar<uint64_t> &rhs) {
           self.cast<cytnx::Tensor &>().Mul_(static_cast<cytnx::cytnx_uint64>(rhs));
           return self;
         })
    .def("__imul__",
         [](py::object self, const py::numpy_scalar<int32_t> &rhs) {
           self.cast<cytnx::Tensor &>().Mul_(static_cast<cytnx::cytnx_int32>(rhs));
           return self;
         })
    .def("__imul__",
         [](py::object self, const py::numpy_scalar<uint32_t> &rhs) {
           self.cast<cytnx::Tensor &>().Mul_(static_cast<cytnx::cytnx_uint32>(rhs));
           return self;
         })
    .def("__imul__",
         [](py::object self, const py::numpy_scalar<int16_t> &rhs) {
           self.cast<cytnx::Tensor &>().Mul_(static_cast<cytnx::cytnx_int16>(rhs));
           return self;
         })
    .def("__imul__",
         [](py::object self, const py::numpy_scalar<uint16_t> &rhs) {
           self.cast<cytnx::Tensor &>().Mul_(static_cast<cytnx::cytnx_uint16>(rhs));
           return self;
         })
    .def("__imul__",
         [](py::object self, const py::numpy_scalar<bool> &rhs) {
           self.cast<cytnx::Tensor &>().Mul_(static_cast<cytnx::cytnx_bool>(rhs));
           return self;
         })
    .def("__imul__",
         [](py::object self, const py::int_ &rhs) {
           dispatch_pyint(rhs, [&](auto v) { self.cast<cytnx::Tensor &>().Mul_(v); });
           return self;
         })
    .def("__imul__",
         [](py::object self, const cytnx::cytnx_double &rhs) {
           self.cast<cytnx::Tensor &>().Mul_(rhs);
           return self;
         })
    .def("__imul__",
         [](py::object self, const cytnx::cytnx_complex128 &rhs) {
           self.cast<cytnx::Tensor &>().Mul_(rhs);
           return self;
         })
    .def("__imul__",
         [](py::object self, const cytnx::Scalar &rhs) {
           self.cast<cytnx::Tensor &>().Mul_(rhs);
           return self;
         })

    // keep-set; registration ORDER matters -- see "KEEP-SET ORDERING" in pybind/pyint_dispatch.hpp.
    .def("__truediv__", [](cytnx::Tensor &self, const cytnx::Tensor &rhs) { return self.Div(rhs); })
    .def("__truediv__",
         [](cytnx::Tensor &self, const py::numpy_scalar<float> &rhs) {
           return self.Div(static_cast<cytnx::cytnx_float>(rhs));
         })
    .def("__truediv__",
         [](cytnx::Tensor &self, const py::numpy_scalar<std::complex<float>> &rhs) {
           return self.Div(static_cast<cytnx::cytnx_complex64>(rhs));
         })
    .def("__truediv__",
         [](cytnx::Tensor &self, const py::numpy_scalar<int64_t> &rhs) {
           return self.Div(static_cast<cytnx::cytnx_int64>(rhs));
         })
    .def("__truediv__",
         [](cytnx::Tensor &self, const py::numpy_scalar<uint64_t> &rhs) {
           return self.Div(static_cast<cytnx::cytnx_uint64>(rhs));
         })
    .def("__truediv__",
         [](cytnx::Tensor &self, const py::numpy_scalar<int32_t> &rhs) {
           return self.Div(static_cast<cytnx::cytnx_int32>(rhs));
         })
    .def("__truediv__",
         [](cytnx::Tensor &self, const py::numpy_scalar<uint32_t> &rhs) {
           return self.Div(static_cast<cytnx::cytnx_uint32>(rhs));
         })
    .def("__truediv__",
         [](cytnx::Tensor &self, const py::numpy_scalar<int16_t> &rhs) {
           return self.Div(static_cast<cytnx::cytnx_int16>(rhs));
         })
    .def("__truediv__",
         [](cytnx::Tensor &self, const py::numpy_scalar<uint16_t> &rhs) {
           return self.Div(static_cast<cytnx::cytnx_uint16>(rhs));
         })
    .def("__truediv__",
         [](cytnx::Tensor &self, const py::numpy_scalar<bool> &rhs) {
           return self.Div(static_cast<cytnx::cytnx_bool>(rhs));
         })
    .def("__truediv__",
         [](cytnx::Tensor &self, const py::int_ &rhs) {
           return dispatch_pyint(rhs, [&](auto v) { return self.Div(v); });
         })
    .def("__truediv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_double &rhs) { return self.Div(rhs); })
    .def("__truediv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_complex128 &rhs) { return self.Div(rhs); })
    .def("__truediv__", [](cytnx::Tensor &self, const cytnx::Scalar &rhs) { return self.Div(rhs); })

    // keep-set; registration ORDER matters -- see "KEEP-SET ORDERING" in pybind/pyint_dispatch.hpp.
    // NOTE (pre-existing, out of scope here): a numpy scalar on the LEFT
    // (e.g. np.float32(1.0) + t) does not reach this __r*__ binding at all --
    // Tensor defines __iter__, so numpy's ufunc machinery treats it as an
    // array-like and tries to iterate it instead, raising
    // "TypeError: 'TensorIterator' object is not iterable" (issue #692).
    .def("__rtruediv__",
         [](cytnx::Tensor &self, const py::numpy_scalar<float> &lhs) {
           return cytnx::linalg::Div(static_cast<cytnx::cytnx_float>(lhs), self);
         })
    .def("__rtruediv__",
         [](cytnx::Tensor &self, const py::numpy_scalar<std::complex<float>> &lhs) {
           return cytnx::linalg::Div(static_cast<cytnx::cytnx_complex64>(lhs), self);
         })
    .def("__rtruediv__",
         [](cytnx::Tensor &self, const py::numpy_scalar<int64_t> &lhs) {
           return cytnx::linalg::Div(static_cast<cytnx::cytnx_int64>(lhs), self);
         })
    .def("__rtruediv__",
         [](cytnx::Tensor &self, const py::numpy_scalar<uint64_t> &lhs) {
           return cytnx::linalg::Div(static_cast<cytnx::cytnx_uint64>(lhs), self);
         })
    .def("__rtruediv__",
         [](cytnx::Tensor &self, const py::numpy_scalar<int32_t> &lhs) {
           return cytnx::linalg::Div(static_cast<cytnx::cytnx_int32>(lhs), self);
         })
    .def("__rtruediv__",
         [](cytnx::Tensor &self, const py::numpy_scalar<uint32_t> &lhs) {
           return cytnx::linalg::Div(static_cast<cytnx::cytnx_uint32>(lhs), self);
         })
    .def("__rtruediv__",
         [](cytnx::Tensor &self, const py::numpy_scalar<int16_t> &lhs) {
           return cytnx::linalg::Div(static_cast<cytnx::cytnx_int16>(lhs), self);
         })
    .def("__rtruediv__",
         [](cytnx::Tensor &self, const py::numpy_scalar<uint16_t> &lhs) {
           return cytnx::linalg::Div(static_cast<cytnx::cytnx_uint16>(lhs), self);
         })
    .def("__rtruediv__",
         [](cytnx::Tensor &self, const py::numpy_scalar<bool> &lhs) {
           return cytnx::linalg::Div(static_cast<cytnx::cytnx_bool>(lhs), self);
         })
    .def("__rtruediv__",
         [](cytnx::Tensor &self, const py::int_ &lhs) {
           return dispatch_pyint(lhs, [&](auto v) { return cytnx::linalg::Div(v, self); });
         })
    .def("__rtruediv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_double &lhs) { return cytnx::linalg::Div(lhs, self); })
    .def("__rtruediv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_complex128 &lhs) { return cytnx::linalg::Div(lhs, self); })
    .def("__rtruediv__", [](cytnx::Tensor &self, const cytnx::Scalar &lhs) { return cytnx::linalg::Div(lhs, self); })

    // keep-set; registration ORDER matters -- see "KEEP-SET ORDERING" in pybind/pyint_dispatch.hpp.
    .def("__itruediv__",
         [](py::object self, const cytnx::Tensor &rhs) {
           self.cast<cytnx::Tensor &>().Div_(rhs);
           return self;
         })
    .def("__itruediv__",
         [](py::object self, const py::numpy_scalar<float> &rhs) {
           self.cast<cytnx::Tensor &>().Div_(static_cast<cytnx::cytnx_float>(rhs));
           return self;
         })
    .def("__itruediv__",
         [](py::object self, const py::numpy_scalar<std::complex<float>> &rhs) {
           self.cast<cytnx::Tensor &>().Div_(static_cast<cytnx::cytnx_complex64>(rhs));
           return self;
         })
    .def("__itruediv__",
         [](py::object self, const py::numpy_scalar<int64_t> &rhs) {
           self.cast<cytnx::Tensor &>().Div_(static_cast<cytnx::cytnx_int64>(rhs));
           return self;
         })
    .def("__itruediv__",
         [](py::object self, const py::numpy_scalar<uint64_t> &rhs) {
           self.cast<cytnx::Tensor &>().Div_(static_cast<cytnx::cytnx_uint64>(rhs));
           return self;
         })
    .def("__itruediv__",
         [](py::object self, const py::numpy_scalar<int32_t> &rhs) {
           self.cast<cytnx::Tensor &>().Div_(static_cast<cytnx::cytnx_int32>(rhs));
           return self;
         })
    .def("__itruediv__",
         [](py::object self, const py::numpy_scalar<uint32_t> &rhs) {
           self.cast<cytnx::Tensor &>().Div_(static_cast<cytnx::cytnx_uint32>(rhs));
           return self;
         })
    .def("__itruediv__",
         [](py::object self, const py::numpy_scalar<int16_t> &rhs) {
           self.cast<cytnx::Tensor &>().Div_(static_cast<cytnx::cytnx_int16>(rhs));
           return self;
         })
    .def("__itruediv__",
         [](py::object self, const py::numpy_scalar<uint16_t> &rhs) {
           self.cast<cytnx::Tensor &>().Div_(static_cast<cytnx::cytnx_uint16>(rhs));
           return self;
         })
    .def("__itruediv__",
         [](py::object self, const py::numpy_scalar<bool> &rhs) {
           self.cast<cytnx::Tensor &>().Div_(static_cast<cytnx::cytnx_bool>(rhs));
           return self;
         })
    .def("__itruediv__",
         [](py::object self, const py::int_ &rhs) {
           dispatch_pyint(rhs, [&](auto v) { self.cast<cytnx::Tensor &>().Div_(v); });
           return self;
         })
    .def("__itruediv__",
         [](py::object self, const cytnx::cytnx_double &rhs) {
           self.cast<cytnx::Tensor &>().Div_(rhs);
           return self;
         })
    .def("__itruediv__",
         [](py::object self, const cytnx::cytnx_complex128 &rhs) {
           self.cast<cytnx::Tensor &>().Div_(rhs);
           return self;
         })
    .def("__itruediv__",
         [](py::object self, const cytnx::Scalar &rhs) {
           self.cast<cytnx::Tensor &>().Div_(rhs);
           return self;
         })

    // keep-set; registration ORDER matters -- see "KEEP-SET ORDERING" in pybind/pyint_dispatch.hpp.
    .def("__floordiv__", [](cytnx::Tensor &self, const cytnx::Tensor &rhs) { return self.Div(rhs); })
    .def("__floordiv__",
         [](cytnx::Tensor &self, const py::numpy_scalar<float> &rhs) {
           return self.Div(static_cast<cytnx::cytnx_float>(rhs));
         })
    .def("__floordiv__",
         [](cytnx::Tensor &self, const py::numpy_scalar<std::complex<float>> &rhs) {
           return self.Div(static_cast<cytnx::cytnx_complex64>(rhs));
         })
    .def("__floordiv__",
         [](cytnx::Tensor &self, const py::numpy_scalar<int64_t> &rhs) {
           return self.Div(static_cast<cytnx::cytnx_int64>(rhs));
         })
    .def("__floordiv__",
         [](cytnx::Tensor &self, const py::numpy_scalar<uint64_t> &rhs) {
           return self.Div(static_cast<cytnx::cytnx_uint64>(rhs));
         })
    .def("__floordiv__",
         [](cytnx::Tensor &self, const py::numpy_scalar<int32_t> &rhs) {
           return self.Div(static_cast<cytnx::cytnx_int32>(rhs));
         })
    .def("__floordiv__",
         [](cytnx::Tensor &self, const py::numpy_scalar<uint32_t> &rhs) {
           return self.Div(static_cast<cytnx::cytnx_uint32>(rhs));
         })
    .def("__floordiv__",
         [](cytnx::Tensor &self, const py::numpy_scalar<int16_t> &rhs) {
           return self.Div(static_cast<cytnx::cytnx_int16>(rhs));
         })
    .def("__floordiv__",
         [](cytnx::Tensor &self, const py::numpy_scalar<uint16_t> &rhs) {
           return self.Div(static_cast<cytnx::cytnx_uint16>(rhs));
         })
    .def("__floordiv__",
         [](cytnx::Tensor &self, const py::numpy_scalar<bool> &rhs) {
           return self.Div(static_cast<cytnx::cytnx_bool>(rhs));
         })
    .def("__floordiv__",
         [](cytnx::Tensor &self, const py::int_ &rhs) {
           return dispatch_pyint(rhs, [&](auto v) { return self.Div(v); });
         })
    .def("__floordiv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_double &rhs) { return self.Div(rhs); })
    .def("__floordiv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_complex128 &rhs) { return self.Div(rhs); })
    .def("__floordiv__", [](cytnx::Tensor &self, const cytnx::Scalar &rhs) { return self.Div(rhs); })

    // keep-set; registration ORDER matters -- see "KEEP-SET ORDERING" in pybind/pyint_dispatch.hpp.
    // NOTE (pre-existing, out of scope here): a numpy scalar on the LEFT
    // (e.g. np.float32(1.0) + t) does not reach this __r*__ binding at all --
    // Tensor defines __iter__, so numpy's ufunc machinery treats it as an
    // array-like and tries to iterate it instead, raising
    // "TypeError: 'TensorIterator' object is not iterable" (issue #692).
    .def("__rfloordiv__",
         [](cytnx::Tensor &self, const py::numpy_scalar<float> &lhs) {
           return cytnx::linalg::Div(static_cast<cytnx::cytnx_float>(lhs), self);
         })
    .def("__rfloordiv__",
         [](cytnx::Tensor &self, const py::numpy_scalar<std::complex<float>> &lhs) {
           return cytnx::linalg::Div(static_cast<cytnx::cytnx_complex64>(lhs), self);
         })
    .def("__rfloordiv__",
         [](cytnx::Tensor &self, const py::numpy_scalar<int64_t> &lhs) {
           return cytnx::linalg::Div(static_cast<cytnx::cytnx_int64>(lhs), self);
         })
    .def("__rfloordiv__",
         [](cytnx::Tensor &self, const py::numpy_scalar<uint64_t> &lhs) {
           return cytnx::linalg::Div(static_cast<cytnx::cytnx_uint64>(lhs), self);
         })
    .def("__rfloordiv__",
         [](cytnx::Tensor &self, const py::numpy_scalar<int32_t> &lhs) {
           return cytnx::linalg::Div(static_cast<cytnx::cytnx_int32>(lhs), self);
         })
    .def("__rfloordiv__",
         [](cytnx::Tensor &self, const py::numpy_scalar<uint32_t> &lhs) {
           return cytnx::linalg::Div(static_cast<cytnx::cytnx_uint32>(lhs), self);
         })
    .def("__rfloordiv__",
         [](cytnx::Tensor &self, const py::numpy_scalar<int16_t> &lhs) {
           return cytnx::linalg::Div(static_cast<cytnx::cytnx_int16>(lhs), self);
         })
    .def("__rfloordiv__",
         [](cytnx::Tensor &self, const py::numpy_scalar<uint16_t> &lhs) {
           return cytnx::linalg::Div(static_cast<cytnx::cytnx_uint16>(lhs), self);
         })
    .def("__rfloordiv__",
         [](cytnx::Tensor &self, const py::numpy_scalar<bool> &lhs) {
           return cytnx::linalg::Div(static_cast<cytnx::cytnx_bool>(lhs), self);
         })
    .def("__rfloordiv__",
         [](cytnx::Tensor &self, const py::int_ &lhs) {
           return dispatch_pyint(lhs, [&](auto v) { return cytnx::linalg::Div(v, self); });
         })
    .def("__rfloordiv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_double &lhs) { return cytnx::linalg::Div(lhs, self); })
    .def("__rfloordiv__",
         [](cytnx::Tensor &self, const cytnx::cytnx_complex128 &lhs) { return cytnx::linalg::Div(lhs, self); })
    .def("__rfloordiv__", [](cytnx::Tensor &self, const cytnx::Scalar &lhs) { return cytnx::linalg::Div(lhs, self); })

    // keep-set; registration ORDER matters -- see "KEEP-SET ORDERING" in pybind/pyint_dispatch.hpp.
    .def("__ifloordiv__",
         [](py::object self, const cytnx::Tensor &rhs) {
           self.cast<cytnx::Tensor &>().Div_(rhs);
           return self;
         })
    .def("__ifloordiv__",
         [](py::object self, const py::numpy_scalar<float> &rhs) {
           self.cast<cytnx::Tensor &>().Div_(static_cast<cytnx::cytnx_float>(rhs));
           return self;
         })
    .def("__ifloordiv__",
         [](py::object self, const py::numpy_scalar<std::complex<float>> &rhs) {
           self.cast<cytnx::Tensor &>().Div_(static_cast<cytnx::cytnx_complex64>(rhs));
           return self;
         })
    .def("__ifloordiv__",
         [](py::object self, const py::numpy_scalar<int64_t> &rhs) {
           self.cast<cytnx::Tensor &>().Div_(static_cast<cytnx::cytnx_int64>(rhs));
           return self;
         })
    .def("__ifloordiv__",
         [](py::object self, const py::numpy_scalar<uint64_t> &rhs) {
           self.cast<cytnx::Tensor &>().Div_(static_cast<cytnx::cytnx_uint64>(rhs));
           return self;
         })
    .def("__ifloordiv__",
         [](py::object self, const py::numpy_scalar<int32_t> &rhs) {
           self.cast<cytnx::Tensor &>().Div_(static_cast<cytnx::cytnx_int32>(rhs));
           return self;
         })
    .def("__ifloordiv__",
         [](py::object self, const py::numpy_scalar<uint32_t> &rhs) {
           self.cast<cytnx::Tensor &>().Div_(static_cast<cytnx::cytnx_uint32>(rhs));
           return self;
         })
    .def("__ifloordiv__",
         [](py::object self, const py::numpy_scalar<int16_t> &rhs) {
           self.cast<cytnx::Tensor &>().Div_(static_cast<cytnx::cytnx_int16>(rhs));
           return self;
         })
    .def("__ifloordiv__",
         [](py::object self, const py::numpy_scalar<uint16_t> &rhs) {
           self.cast<cytnx::Tensor &>().Div_(static_cast<cytnx::cytnx_uint16>(rhs));
           return self;
         })
    .def("__ifloordiv__",
         [](py::object self, const py::numpy_scalar<bool> &rhs) {
           self.cast<cytnx::Tensor &>().Div_(static_cast<cytnx::cytnx_bool>(rhs));
           return self;
         })
    .def("__ifloordiv__",
         [](py::object self, const py::int_ &rhs) {
           dispatch_pyint(rhs, [&](auto v) { self.cast<cytnx::Tensor &>().Div_(v); });
           return self;
         })
    .def("__ifloordiv__",
         [](py::object self, const cytnx::cytnx_double &rhs) {
           self.cast<cytnx::Tensor &>().Div_(rhs);
           return self;
         })
    .def("__ifloordiv__",
         [](py::object self, const cytnx::cytnx_complex128 &rhs) {
           self.cast<cytnx::Tensor &>().Div_(rhs);
           return self;
         })
    .def("__ifloordiv__",
         [](py::object self, const cytnx::Scalar &rhs) {
           self.cast<cytnx::Tensor &>().Div_(rhs);
           return self;
         })

    // keep-set; registration ORDER matters -- see "KEEP-SET ORDERING" in pybind/pyint_dispatch.hpp.
    .def("__mod__", [](cytnx::Tensor &self, const cytnx::Tensor &rhs) { return self.Mod(rhs); })
    .def("__mod__",
         [](cytnx::Tensor &self, const py::numpy_scalar<float> &rhs) {
           return self.Mod(static_cast<cytnx::cytnx_float>(rhs));
         })
    .def("__mod__",
         [](cytnx::Tensor &self, const py::numpy_scalar<std::complex<float>> &rhs) {
           return self.Mod(static_cast<cytnx::cytnx_complex64>(rhs));
         })
    .def("__mod__",
         [](cytnx::Tensor &self, const py::numpy_scalar<int64_t> &rhs) {
           return self.Mod(static_cast<cytnx::cytnx_int64>(rhs));
         })
    .def("__mod__",
         [](cytnx::Tensor &self, const py::numpy_scalar<uint64_t> &rhs) {
           return self.Mod(static_cast<cytnx::cytnx_uint64>(rhs));
         })
    .def("__mod__",
         [](cytnx::Tensor &self, const py::numpy_scalar<int32_t> &rhs) {
           return self.Mod(static_cast<cytnx::cytnx_int32>(rhs));
         })
    .def("__mod__",
         [](cytnx::Tensor &self, const py::numpy_scalar<uint32_t> &rhs) {
           return self.Mod(static_cast<cytnx::cytnx_uint32>(rhs));
         })
    .def("__mod__",
         [](cytnx::Tensor &self, const py::numpy_scalar<int16_t> &rhs) {
           return self.Mod(static_cast<cytnx::cytnx_int16>(rhs));
         })
    .def("__mod__",
         [](cytnx::Tensor &self, const py::numpy_scalar<uint16_t> &rhs) {
           return self.Mod(static_cast<cytnx::cytnx_uint16>(rhs));
         })
    .def("__mod__",
         [](cytnx::Tensor &self, const py::numpy_scalar<bool> &rhs) {
           return self.Mod(static_cast<cytnx::cytnx_bool>(rhs));
         })
    .def("__mod__",
         [](cytnx::Tensor &self, const py::int_ &rhs) {
           return dispatch_pyint(rhs, [&](auto v) { return self.Mod(v); });
         })
    .def("__mod__",
         [](cytnx::Tensor &self, const cytnx::cytnx_double &rhs) { return self.Mod(rhs); })
    .def("__mod__",
         [](cytnx::Tensor &self, const cytnx::cytnx_complex128 &rhs) { return self.Mod(rhs); })
    .def("__mod__", [](cytnx::Tensor &self, const cytnx::Scalar &rhs) { return self.Mod(rhs); })

    // keep-set; registration ORDER matters -- see "KEEP-SET ORDERING" in pybind/pyint_dispatch.hpp.
    // NOTE (pre-existing, out of scope here): a numpy scalar on the LEFT
    // (e.g. np.float32(1.0) + t) does not reach this __r*__ binding at all --
    // Tensor defines __iter__, so numpy's ufunc machinery treats it as an
    // array-like and tries to iterate it instead, raising
    // "TypeError: 'TensorIterator' object is not iterable" (issue #692).
    .def("__rmod__",
         [](cytnx::Tensor &self, const py::numpy_scalar<float> &lhs) {
           return cytnx::linalg::Mod(static_cast<cytnx::cytnx_float>(lhs), self);
         })
    .def("__rmod__",
         [](cytnx::Tensor &self, const py::numpy_scalar<std::complex<float>> &lhs) {
           return cytnx::linalg::Mod(static_cast<cytnx::cytnx_complex64>(lhs), self);
         })
    .def("__rmod__",
         [](cytnx::Tensor &self, const py::numpy_scalar<int64_t> &lhs) {
           return cytnx::linalg::Mod(static_cast<cytnx::cytnx_int64>(lhs), self);
         })
    .def("__rmod__",
         [](cytnx::Tensor &self, const py::numpy_scalar<uint64_t> &lhs) {
           return cytnx::linalg::Mod(static_cast<cytnx::cytnx_uint64>(lhs), self);
         })
    .def("__rmod__",
         [](cytnx::Tensor &self, const py::numpy_scalar<int32_t> &lhs) {
           return cytnx::linalg::Mod(static_cast<cytnx::cytnx_int32>(lhs), self);
         })
    .def("__rmod__",
         [](cytnx::Tensor &self, const py::numpy_scalar<uint32_t> &lhs) {
           return cytnx::linalg::Mod(static_cast<cytnx::cytnx_uint32>(lhs), self);
         })
    .def("__rmod__",
         [](cytnx::Tensor &self, const py::numpy_scalar<int16_t> &lhs) {
           return cytnx::linalg::Mod(static_cast<cytnx::cytnx_int16>(lhs), self);
         })
    .def("__rmod__",
         [](cytnx::Tensor &self, const py::numpy_scalar<uint16_t> &lhs) {
           return cytnx::linalg::Mod(static_cast<cytnx::cytnx_uint16>(lhs), self);
         })
    .def("__rmod__",
         [](cytnx::Tensor &self, const py::numpy_scalar<bool> &lhs) {
           return cytnx::linalg::Mod(static_cast<cytnx::cytnx_bool>(lhs), self);
         })
    .def("__rmod__",
         [](cytnx::Tensor &self, const py::int_ &lhs) {
           return dispatch_pyint(lhs, [&](auto v) { return cytnx::linalg::Mod(v, self); });
         })
    .def("__rmod__",
         [](cytnx::Tensor &self, const cytnx::cytnx_double &lhs) { return cytnx::linalg::Mod(lhs, self); })
    .def("__rmod__",
         [](cytnx::Tensor &self, const cytnx::cytnx_complex128 &lhs) { return cytnx::linalg::Mod(lhs, self); })
    .def("__rmod__", [](cytnx::Tensor &self, const cytnx::Scalar &lhs) { return cytnx::linalg::Mod(lhs, self); })

    // keep-set; registration ORDER matters -- see "KEEP-SET ORDERING" in pybind/pyint_dispatch.hpp.
    .def("__eq__", [](cytnx::Tensor &self, const cytnx::Tensor &rhs) { return self == rhs; })
    .def("__eq__",
         [](cytnx::Tensor &self, const py::numpy_scalar<float> &rhs) {
           return self == static_cast<cytnx::cytnx_float>(rhs);
         })
    .def("__eq__",
         [](cytnx::Tensor &self, const py::numpy_scalar<std::complex<float>> &rhs) {
           return self == static_cast<cytnx::cytnx_complex64>(rhs);
         })
    .def("__eq__",
         [](cytnx::Tensor &self, const py::numpy_scalar<int64_t> &rhs) {
           return self == static_cast<cytnx::cytnx_int64>(rhs);
         })
    .def("__eq__",
         [](cytnx::Tensor &self, const py::numpy_scalar<uint64_t> &rhs) {
           return self == static_cast<cytnx::cytnx_uint64>(rhs);
         })
    .def("__eq__",
         [](cytnx::Tensor &self, const py::numpy_scalar<int32_t> &rhs) {
           return self == static_cast<cytnx::cytnx_int32>(rhs);
         })
    .def("__eq__",
         [](cytnx::Tensor &self, const py::numpy_scalar<uint32_t> &rhs) {
           return self == static_cast<cytnx::cytnx_uint32>(rhs);
         })
    .def("__eq__",
         [](cytnx::Tensor &self, const py::numpy_scalar<int16_t> &rhs) {
           return self == static_cast<cytnx::cytnx_int16>(rhs);
         })
    .def("__eq__",
         [](cytnx::Tensor &self, const py::numpy_scalar<uint16_t> &rhs) {
           return self == static_cast<cytnx::cytnx_uint16>(rhs);
         })
    .def("__eq__",
         [](cytnx::Tensor &self, const py::numpy_scalar<bool> &rhs) {
           return self == static_cast<cytnx::cytnx_bool>(rhs);
         })
    .def("__eq__",
         [](cytnx::Tensor &self, const py::int_ &rhs) {
           return dispatch_pyint(rhs, [&](auto v) { return self == v; });
         })
    .def("__eq__", [](cytnx::Tensor &self, const cytnx::cytnx_double &rhs) { return self == rhs; })
    .def("__eq__",
         [](cytnx::Tensor &self, const cytnx::cytnx_complex128 &rhs) { return self == rhs; })
    .def("__eq__", [](cytnx::Tensor &self, const cytnx::Scalar &rhs) { return self == rhs; })

    // __ne__ (#928/#916/#692 background): cytnx has no elementwise
    // operator!= or Neq/logical-not kernel (checked: no `Neq`, no
    // `logical_not`, no `operator!` anywhere in include/linalg.hpp or
    // src/linalg/). Leaving __ne__ unbound is NOT safe: Python's default
    // dunder-less `!=` falls back to `not (self == rhs)`, and __eq__ already
    // returns an elementwise Bool Tensor here, so `not Tensor` collapses
    // through __bool__/__len__ to a single bare `False` for any non-empty
    // operand -- a silently wrong scalar instead of an elementwise
    // comparison (exactly the bug #928/#916 describe). A cheap, correct
    // elementwise implementation composes two EXISTING kernels instead of
    // requiring a new C++ kernel: `self.Cpr(rhs)` (the same call __eq__
    // uses) gives the elementwise equality as a Bool Tensor; arithmetic
    // `1 - eq` on a Bool tensor promotes to Int64 (0/1) through the normal
    // type-promotion path, and `.astype(Type.Bool)` casts it back to a
    // proper elementwise Bool Tensor with the negated values. Verified
    // empirically (both 1-D and 2-D, contiguous) to match numpy elementwise
    // != semantics. Keep-set mirrors __eq__ (including the trailing Scalar
    // overload); registration ORDER matters -- see "KEEP-SET ORDERING" in
    // pybind/pyint_dispatch.hpp.
    .def("__ne__",
         [](cytnx::Tensor &self, const cytnx::Tensor &rhs) {
           return (1 - self.Cpr(rhs)).astype(cytnx::Type.Bool);
         })
    .def("__ne__",
         [](cytnx::Tensor &self, const py::numpy_scalar<float> &rhs) {
           return (1 - self.Cpr(static_cast<cytnx::cytnx_float>(rhs))).astype(cytnx::Type.Bool);
         })
    .def("__ne__",
         [](cytnx::Tensor &self, const py::numpy_scalar<std::complex<float>> &rhs) {
           return (1 - self.Cpr(static_cast<cytnx::cytnx_complex64>(rhs)))
             .astype(cytnx::Type.Bool);
         })
    .def("__ne__",
         [](cytnx::Tensor &self, const py::numpy_scalar<int64_t> &rhs) {
           return (1 - self.Cpr(static_cast<cytnx::cytnx_int64>(rhs))).astype(cytnx::Type.Bool);
         })
    .def("__ne__",
         [](cytnx::Tensor &self, const py::numpy_scalar<uint64_t> &rhs) {
           return (1 - self.Cpr(static_cast<cytnx::cytnx_uint64>(rhs))).astype(cytnx::Type.Bool);
         })
    .def("__ne__",
         [](cytnx::Tensor &self, const py::numpy_scalar<int32_t> &rhs) {
           return (1 - self.Cpr(static_cast<cytnx::cytnx_int32>(rhs))).astype(cytnx::Type.Bool);
         })
    .def("__ne__",
         [](cytnx::Tensor &self, const py::numpy_scalar<uint32_t> &rhs) {
           return (1 - self.Cpr(static_cast<cytnx::cytnx_uint32>(rhs))).astype(cytnx::Type.Bool);
         })
    .def("__ne__",
         [](cytnx::Tensor &self, const py::numpy_scalar<int16_t> &rhs) {
           return (1 - self.Cpr(static_cast<cytnx::cytnx_int16>(rhs))).astype(cytnx::Type.Bool);
         })
    .def("__ne__",
         [](cytnx::Tensor &self, const py::numpy_scalar<uint16_t> &rhs) {
           return (1 - self.Cpr(static_cast<cytnx::cytnx_uint16>(rhs))).astype(cytnx::Type.Bool);
         })
    .def("__ne__",
         [](cytnx::Tensor &self, const py::numpy_scalar<bool> &rhs) {
           return (1 - self.Cpr(static_cast<cytnx::cytnx_bool>(rhs))).astype(cytnx::Type.Bool);
         })
    .def("__ne__",
         [](cytnx::Tensor &self, const py::int_ &rhs) {
           return dispatch_pyint(rhs, [&](auto v) {
             return (1 - self.Cpr(v)).astype(cytnx::Type.Bool);
           });
         })
    .def("__ne__",
         [](cytnx::Tensor &self, const cytnx::cytnx_double &rhs) {
           return (1 - self.Cpr(rhs)).astype(cytnx::Type.Bool);
         })
    .def("__ne__",
         [](cytnx::Tensor &self, const cytnx::cytnx_complex128 &rhs) {
           return (1 - self.Cpr(rhs)).astype(cytnx::Type.Bool);
         })
    .def("__ne__",
         [](cytnx::Tensor &self, const cytnx::Scalar &rhs) {
           return (1 - (self == rhs)).astype(cytnx::Type.Bool);
         })

    // __bool__ (numpy semantics, #928/#916/#692 background): previously
    // unbound, so Python's truthiness fell through to __len__ (`if tensor:`
    // just checked shape()[0] != 0), which never raises for a multi-element
    // tensor and gives no meaningful truth value at all for an uninitialized
    // (Void-dtype, shape []) Tensor beyond a low-level cytnx_error_msg
    // RuntimeError. numpy raises ValueError for an array with more than one
    // element ("truth value ... is ambiguous") and also for a size-0 array;
    // cytnx never allows a genuinely empty (0-length) dimension, so the
    // size-0 case that matters here is the uninitialized/Void Tensor.
    // NOTE: this is a user-visible behavior change -- `if tensor:` used to
    // be equivalent to `if len(tensor):` (shape()[0] truthiness) and now
    // raises ValueError for any multi-element tensor instead.
    .def("__bool__",
         [](cytnx::Tensor &self) {
           if (self.dtype() == cytnx::Type.Void) {
             throw py::value_error("the truth value of an uninitialized Tensor is ambiguous.");
           }
           cytnx::cytnx_uint64 numel = 1;
           for (auto &d : self.shape()) numel *= d;
           if (numel != 1) {
             throw py::value_error(
               "the truth value of a Tensor with more than one element is ambiguous.");
           }
           if (self.dtype() == cytnx::Type.Double)
             return bool(self.item<cytnx::cytnx_double>());
           else if (self.dtype() == cytnx::Type.Float)
             return bool(self.item<cytnx::cytnx_float>());
           else if (self.dtype() == cytnx::Type.ComplexDouble)
             return self.item<cytnx::cytnx_complex128>() != cytnx::cytnx_complex128(0, 0);
           else if (self.dtype() == cytnx::Type.ComplexFloat)
             return self.item<cytnx::cytnx_complex64>() != cytnx::cytnx_complex64(0, 0);
           else if (self.dtype() == cytnx::Type.Uint64)
             return bool(self.item<cytnx::cytnx_uint64>());
           else if (self.dtype() == cytnx::Type.Int64)
             return bool(self.item<cytnx::cytnx_int64>());
           else if (self.dtype() == cytnx::Type.Uint32)
             return bool(self.item<cytnx::cytnx_uint32>());
           else if (self.dtype() == cytnx::Type.Int32)
             return bool(self.item<cytnx::cytnx_int32>());
           else if (self.dtype() == cytnx::Type.Uint16)
             return bool(self.item<cytnx::cytnx_uint16>());
           else if (self.dtype() == cytnx::Type.Int16)
             return bool(self.item<cytnx::cytnx_int16>());
           else  // Bool
             return bool(self.item<cytnx::cytnx_bool>());
         })

    // __pow__/__ipow__: linalg::Pow/Tensor::Pow_ only take a plain double
    // exponent, so the keep-set here is narrower than the arithmetic ops --
    // there is no elementwise Tensor**Tensor kernel, and the output dtype
    // follows the BASE tensor, not the exponent, so no numpy integer-dtype-
    // preservation concern applies to the exponent itself. Still bind
    // py::int_ and numpy_scalar<float> explicitly (rather than relying on
    // pybind11's implicit double conversion) so #916-style stubs can show a
    // precise signature once the stub pipeline (#915) lands, instead of only
    // working by accident through the convert-pass fallback.
    .def("__pow__",
         [](cytnx::Tensor &self, const py::numpy_scalar<float> &p) {
           return cytnx::linalg::Pow(self, static_cast<cytnx::cytnx_double>(
                                             static_cast<cytnx::cytnx_float>(p)));
         })
    .def("__pow__",
         [](cytnx::Tensor &self, const py::int_ &p) {
           return dispatch_pyint(
             p, [&](auto v) { return cytnx::linalg::Pow(self, static_cast<cytnx::cytnx_double>(v)); });
         })
    .def("__pow__", [](cytnx::Tensor &self,
                       const cytnx::cytnx_double &p) { return cytnx::linalg::Pow(self, p); })
    .def("__ipow__",
         [](py::object self, const py::numpy_scalar<float> &p) {
           self.cast<cytnx::Tensor &>().Pow_(
             static_cast<cytnx::cytnx_double>(static_cast<cytnx::cytnx_float>(p)));
           return self;
         })
    .def("__ipow__",
         [](py::object self, const py::int_ &p) {
           dispatch_pyint(p, [&](auto v) {
             self.cast<cytnx::Tensor &>().Pow_(static_cast<cytnx::cytnx_double>(v));
           });
           return self;
         })
    .def("__ipow__",
         [](py::object self, const cytnx::cytnx_double &p) {
           self.cast<cytnx::Tensor &>().Pow_(p);
           return self;
         })
    .def("__matmul__", [](cytnx::Tensor &self,
                          const cytnx::Tensor &rhs) { return cytnx::linalg::Dot(self, rhs); })
    .def("__imatmul__",
         [](py::object self, const cytnx::Tensor &rhs) {
           self.cast<cytnx::Tensor &>() = cytnx::linalg::Dot(self.cast<cytnx::Tensor &>(), rhs);
           return self;
         })
    // linalg >>
    .def("Svd", &cytnx::Tensor::Svd, py::arg("is_UvT") = true)
    .def("Eigh", &cytnx::Tensor::Eigh, py::arg("is_V") = true, py::arg("row_v") = false)
    .def("InvM_",
         [](py::object self) {
           self.cast<cytnx::Tensor &>().InvM_();
           return self;
         })
    .def("InvM", &cytnx::Tensor::InvM)
    .def("Inv_",
         [](py::object self, const double &clip) {
           self.cast<cytnx::Tensor &>().Inv_(clip);
           return self;
         },
         py::arg("clip") = -1)
    .def("Inv", &cytnx::Tensor::Inv, py::arg("clip") = -1)
    .def("Conj_",
         [](py::object self) {
           self.cast<cytnx::Tensor &>().Conj_();
           return self;
         })
    .def("Conj", &cytnx::Tensor::Conj)
    .def("Exp_",
         [](py::object self) {
           self.cast<cytnx::Tensor &>().Exp_();
           return self;
         })
    .def("Exp", &cytnx::Tensor::Exp)
    .def("Pow", &cytnx::Tensor::Pow)
    .def("Pow_",
         [](py::object self, const cytnx::cytnx_double &p) {
           self.cast<cytnx::Tensor &>().Pow_(p);
           return self;
         })
    .def("Abs", &cytnx::Tensor::Abs)
    .def("Abs_",
         [](py::object self) {
           self.cast<cytnx::Tensor &>().Abs_();
           return self;
         })
    .def("Max", &cytnx::Tensor::Max)
    .def("Min", &cytnx::Tensor::Min)
    .def("Norm", &cytnx::Tensor::Norm)
    .def("Trace", &cytnx::Tensor::Trace)

    ;  // end of object line
}
#endif
