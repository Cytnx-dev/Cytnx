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

class cHclass {
 public:
  Scalar::Sproxy proxy;

  cHclass(const Scalar::Sproxy &inproxy) { this->proxy = inproxy; }
  cHclass(const cHclass &rhs) { this->proxy = rhs.proxy.copy(); }
  cHclass &operator=(cHclass &rhs) {
    this->proxy = rhs.proxy.copy();
    return *this;
  }

  bool exists() const { return this->proxy.exists(); }
  int dtype() const { return this->proxy._insimpl->dtype; }

  cytnx_double get_elem_d() const { return cytnx_double(Scalar(this->proxy)); }
  cytnx_float get_elem_f() const { return cytnx_float(Scalar(this->proxy)); }
  cytnx_complex128 get_elem_cd() const { return complex128(Scalar(this->proxy)); }
  cytnx_complex64 get_elem_cf() const { return complex64(Scalar(this->proxy)); }
  cytnx_uint64 get_elem_u64() const { return cytnx_uint64(Scalar(this->proxy)); }
  cytnx_int64 get_elem_i64() const { return cytnx_int64(Scalar(this->proxy)); }
  cytnx_uint32 get_elem_u32() const { return cytnx_uint32(Scalar(this->proxy)); }
  cytnx_int32 get_elem_i32() const { return cytnx_int32(Scalar(this->proxy)); }
  cytnx_uint16 get_elem_u16() const { return cytnx_uint16(Scalar(this->proxy)); }
  cytnx_int16 get_elem_i16() const { return cytnx_int16(Scalar(this->proxy)); }
  cytnx_bool get_elem_b() const { return cytnx_bool(Scalar(this->proxy)); }

  template <class T>
  void set_elem(const T &elem) {
    // std::cout << typeid(T).name() << std::endl;
    this->proxy = elem;
  }
};

template <class T>
void f_UniTensor_setelem_scal(UniTensor &self, const std::vector<cytnx_uint64> &locator,
                              const T &rc) {
  self.set_elem(locator, rc);
}

template <class T>
void f_UniTensor_setelem_scal_int(UniTensor &self, const cytnx_uint64 &locator, const T &rc) {
  const std::vector<cytnx_uint64> tmp = {locator};
  self.set_elem(tmp, rc);
}

void unitensor_binding(py::module &m) {
  py::class_<cHclass>(m, "Helpclass")
    .def("exists", &cHclass::exists)
    .def("dtype", &cHclass::dtype)
    .def("get_elem_d", &cHclass::get_elem_d)
    .def("get_elem_f", &cHclass::get_elem_f)
    .def("get_elem_cd", &cHclass::get_elem_cd)
    .def("get_elem_cf", &cHclass::get_elem_cf)
    .def("get_elem_i64", &cHclass::get_elem_i64)
    .def("get_elem_u64", &cHclass::get_elem_u64)
    .def("get_elem_i32", &cHclass::get_elem_i32)
    .def("get_elem_u32", &cHclass::get_elem_u32)
    .def("get_elem_i16", &cHclass::get_elem_i16)
    .def("get_elem_u16", &cHclass::get_elem_u16)
    .def("get_elem_b", &cHclass::get_elem_b)

    .def("set_elem", &cHclass::set_elem<double>)
    .def("set_elem", &cHclass::set_elem<float>)
    .def("set_elem", &cHclass::set_elem<cytnx_complex128>)
    .def("set_elem", &cHclass::set_elem<cytnx_complex64>)
    .def("set_elem", &cHclass::set_elem<cytnx_uint64>)
    .def("set_elem", &cHclass::set_elem<cytnx_int64>)
    .def("set_elem", &cHclass::set_elem<cytnx_uint32>)
    .def("set_elem", &cHclass::set_elem<cytnx_int32>)
    .def("set_elem", &cHclass::set_elem<cytnx_uint16>)
    .def("set_elem", &cHclass::set_elem<cytnx_int16>)
    .def("set_elem", &cHclass::set_elem<cytnx_bool>);

  // entry.UniTensor
  py::class_<UniTensor>(m, "UniTensor")
    .def(py::init<>())
    .def(py::init<const cytnx::Tensor &, const bool &, const cytnx_int64 &, const std::vector<std::string> &, const std::string &>(), py::arg("Tin"),
         py::arg("is_diag") = false, py::arg("rowrank") = (cytnx_int64)(-1), py::arg("labels") = std::vector<std::string>(), py::arg("name")="")


    .def(py::init<const std::vector<Bond> &, const std::vector<std::string> &, const cytnx_int64 &,
                  const unsigned int &, const int &, const bool &, const std::string &>(),
         py::arg("bonds"), py::arg("labels") = std::vector<std::string>(),
         py::arg("rowrank") = (cytnx_int64)(-1),
         py::arg("dtype") = (unsigned int)(cytnx::Type.Double),
         py::arg("device") = (int)cytnx::Device.cpu, py::arg("is_diag") = false, py::arg("name")="")


    .def("Init",[](UniTensor &self, const Tensor &in_tensor, const bool &is_diag, const cytnx_int64 &rowrank, const std::vector<std::string> &labels, const std::string &name){
                    self.Init(in_tensor,is_diag,rowrank,labels,name);
                },py::arg("Tin"),py::arg("is_diag")=false,py::arg("rowrank")=(cytnx_int64)(-1), py::arg("labels") = std::vector<std::string>(), py::arg("name")="")


    .def("Init",[](UniTensor &self, const std::vector<Bond> &bonds, const std::vector<std::string> &in_labels,
                   const cytnx_int64 &rowrank, const unsigned int &dtype,
                   const int &device, const bool &is_diag, const std::string &name){
                    self.Init(bonds,in_labels,rowrank,dtype,device,is_diag,name);
                },
         py::arg("bonds"), py::arg("labels") = std::vector<std::string>(),
         py::arg("rowrank") = (cytnx_int64)(-1),
         py::arg("dtype") = (unsigned int)(cytnx::Type.Double),
         py::arg("device") = (int)cytnx::Device.cpu, py::arg("is_diag") = false, py::arg("name")="")
    .def("c_set_name", &UniTensor::set_name)


    .def("c_set_label", [](UniTensor &self, const cytnx_int64 &idx, const std::string &new_label){
                            return self.set_label(idx,new_label);
                        },py::arg("idx"), py::arg("new_label"))

    .def("c_set_label", [](UniTensor &self, const std::string &old_label, const std::string &new_label){
                            return self.set_label(old_label,new_label);
                        },py::arg("old_label"), py::arg("new_label"))


    .def("c_set_labels",[](UniTensor &self, const std::vector<std::string> &new_labels){
                            return self.set_labels(new_labels);
                        },py::arg("new_labels"))


    .def("c_set_rowrank_", &UniTensor::set_rowrank_, py::arg("new_rowrank"))

    .def("set_rowrank", &UniTensor::set_rowrank, py::arg("new_rowrank"))
    .def("relabel",[](UniTensor &self, const std::vector<std::string> &new_labels){
                        return self.relabel(new_labels);
                    }, py::arg("new_labels"))
    .def("relabels",[](UniTensor &self, const std::vector<std::string> &new_labels){
                        return self.relabels(new_labels);
                    }, py::arg("new_labels"))

     .def("c_relabel_",[](UniTensor &self, const std::vector<std::string> &new_labels){
                        self.relabel_(new_labels);
                    }, py::arg("new_labels"))
     .def("c_relabels_",[](UniTensor &self, const std::vector<std::string> &new_labels){
                        self.relabels_(new_labels);
                    }, py::arg("new_labels"))


    .def("relabel", [](UniTensor &self, const cytnx_int64 &idx, const std::string &new_label){
                            return self.relabel(idx,new_label);
                        },py::arg("idx"), py::arg("new_label"))

     .def("c_relabel_", [](UniTensor &self, const cytnx_int64 &idx, const std::string &new_label){
                            self.relabel_(idx,new_label);
                        },py::arg("idx"), py::arg("new_label"))

    .def("relabel", [](UniTensor &self, const std::string &old_label, const std::string &new_label){
                            return self.relabel(old_label,new_label);
                        },py::arg("old_label"), py::arg("new_label"))
     .def("c_relabel_", [](UniTensor &self, const std::string &old_label, const std::string &new_label){
                            self.relabel_(old_label,new_label);
                        },py::arg("old_label"), py::arg("new_label"))


    .def("relabel",[](UniTensor &self, const std::vector<std::string> &old_labels, const std::vector<std::string> &new_labels){
                        return self.relabel(old_labels,new_labels);
                    } ,py::arg("old_labels"), py::arg("new_labels"))

    .def("c_relabel_",[](UniTensor &self, const std::vector<std::string> &old_labels, const std::vector<std::string> &new_labels){
                        self.relabel_(old_labels,new_labels);
                    } ,py::arg("old_labels"), py::arg("new_labels"))

    .def("relabels",[](UniTensor &self, const std::vector<std::string> &old_labels, const std::vector<std::string> &new_labels){
                        return self.relabels(old_labels,new_labels);
                    } ,py::arg("old_labels"), py::arg("new_labels"))

    .def("c_relabels_",[](UniTensor &self, const std::vector<std::string> &old_labels, const std::vector<std::string> &new_labels){
                        self.relabels_(old_labels,new_labels);
                    } ,py::arg("old_labels"), py::arg("new_labels"))



    .def("rowrank", &UniTensor::rowrank)
    .def("Nblocks", &UniTensor::Nblocks)
    .def("rank", &UniTensor::rank)
    .def("uten_type", &UniTensor::uten_type)
    .def("uten_type_str",&UniTensor::uten_type_str)
    .def("syms", &UniTensor::syms)
    .def("dtype", &UniTensor::dtype)
    .def("dtype_str", &UniTensor::dtype_str)
    .def("device", &UniTensor::device)
    .def("device_str", &UniTensor::device_str)
    .def("name", &UniTensor::name)
    .def("is_blockform", &UniTensor::is_blockform)

    .def("get_index",&UniTensor::get_index)

    .def("reshape",
         [](UniTensor &self, py::args args, py::kwargs kwargs) -> UniTensor {
           std::vector<cytnx::cytnx_int64> c_args = args.cast<std::vector<cytnx::cytnx_int64>>();
           cytnx_uint64 rowrank = 0;

           if (kwargs) {
             if (kwargs.contains("rowrank")) rowrank = kwargs["rowrank"].cast<cytnx::cytnx_int64>();
           }

           return self.reshape(c_args, rowrank);
         })
    .def("reshape_",
         [](UniTensor &self, py::args args, py::kwargs kwargs) {
           std::vector<cytnx::cytnx_int64> c_args = args.cast<std::vector<cytnx::cytnx_int64>>();
           cytnx_uint64 rowrank = 0;

           if (kwargs) {
             if (kwargs.contains("rowrank")) rowrank = kwargs["rowrank"].cast<cytnx::cytnx_int64>();
           }

           self.reshape_(c_args, rowrank);
         })
    .def("elem_exists", &UniTensor::elem_exists)
    .def("item",
         [](UniTensor &self) {
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
             cytnx_error_msg(true, "%s", "[ERROR] try to get element from a empty UniTensor.");
           return out;
         })

    .def("c_at", [](UniTensor &self, const std::vector<cytnx_uint64> &locator){
                  Scalar::Sproxy tmp = self.at(locator);
                  //std::cout << "ok" << std::endl;
                  return cHclass(tmp);
               },py::arg("locator"))


    .def("c_at",[](UniTensor &self, const std::vector<std::string> &labels, const std::vector<cytnx_uint64> &locator){
                  Scalar::Sproxy tmp = self.at(labels,locator);
                  //std::cout << "ok" << std::endl;
                  return cHclass(tmp);
               },py::arg("labels"), py::arg("locator"))




    .def("__getitem__",
         [](const UniTensor &self, py::object locators) {
           cytnx_error_msg(self.shape().size() == 0,
                           "[ERROR] try to getitem from a empty UniTensor%s", "\n");
           cytnx_error_msg(
             self.uten_type() != UTenType.Dense,
             "[ERROR] cannot get element using [] from Block/SparseUniTensor. Use at() instead.%s", "\n");

           ssize_t start, stop, step, slicelength;
           std::vector<cytnx::Accessor> accessors;
           if (self.is_diag()){
               if (py::isinstance<py::tuple>(locators)) {
                    cytnx_error_msg(true,
                    "[ERROR] cannot get element using [tuple] on is_diag=True UniTensor since the block is rank-1, consider [int] or [int:int] instead.%s", "\n");
               } else if (py::isinstance<py::slice>(locators)) {
                    py::slice sls = locators.cast<py::slice>();
                    if (!sls.compute((ssize_t)self.shape()[0], &start, &stop, &step, &slicelength))
                         throw py::error_already_set();
                    accessors.push_back(cytnx::Accessor::range(start, stop, step));
               } else {
                    accessors.push_back(cytnx::Accessor(locators.cast<cytnx_int64>()));
               }
           }else{
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
          }
           return self.get(accessors);
         })
    .def("__setitem__",
         [](UniTensor &self, py::object locators, const cytnx::Tensor &rhs) {
           cytnx_error_msg(self.shape().size() == 0,
                           "[ERROR] try to setelem to a empty UniTensor%s", "\n");
           cytnx_error_msg(
             self.uten_type() == UTenType.Sparse,
             "[ERROR] cannot set element using [] from SparseUniTensor. Use at() instead.%s", "\n");

           ssize_t start, stop, step, slicelength;
           std::vector<cytnx::Accessor> accessors;
          if (self.is_diag()){
               if (py::isinstance<py::tuple>(locators)) {
                    cytnx_error_msg(true,
                    "[ERROR] cannot get element using [tuple] on is_diag=True UniTensor since the block is rank-1, consider [int] or [int:int] instead.%s", "\n");
               } else if (py::isinstance<py::slice>(locators)) {
                    py::slice sls = locators.cast<py::slice>();
                    if (!sls.compute((ssize_t)self.shape()[0], &start, &stop, &step, &slicelength))
                         throw py::error_already_set();
                    accessors.push_back(cytnx::Accessor::range(start, stop, step));
               } else {
                    accessors.push_back(cytnx::Accessor(locators.cast<cytnx_int64>()));
               }
          }else{
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
           }
           self.set(accessors, rhs);
         })
    .def("__setitem__",
         [](UniTensor &self, py::object locators, const cytnx::UniTensor &rhs) {
           cytnx_error_msg(self.shape().size() == 0,
                           "[ERROR] try to setelem to a empty UniTensor%s", "\n");
           cytnx_error_msg(
             self.uten_type() != UTenType.Dense,
             "[ERROR] cannot set element using [] from Blcok/SparseUniTensor. Use at() instead.%s", "\n");

           ssize_t start, stop, step, slicelength;
           std::vector<cytnx::Accessor> accessors;
          if (self.is_diag()){
               if (py::isinstance<py::tuple>(locators)) {
                    cytnx_error_msg(true,
                    "[ERROR] cannot get element using [tuple] on is_diag=True UniTensor since the block is rank-1, consider [int] or [int:int] instead.%s", "\n");
               } else if (py::isinstance<py::slice>(locators)) {
                    py::slice sls = locators.cast<py::slice>();
                    if (!sls.compute((ssize_t)self.shape()[0], &start, &stop, &step, &slicelength))
                         throw py::error_already_set();
                    accessors.push_back(cytnx::Accessor::range(start, stop, step));
               } else {
                    accessors.push_back(cytnx::Accessor(locators.cast<cytnx_int64>()));
               }
          }else{
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
           }

           self.set(accessors, rhs.get_block());
         })
    .def("get_elem",
         [](UniTensor &self, const std::vector<cytnx_uint64> &locator) {
           py::object out;
           if (self.dtype() == cytnx::Type.Double)
             out = py::cast(self.get_elem<cytnx::cytnx_double>(locator));
           else if (self.dtype() == cytnx::Type.Float)
             out = py::cast(self.get_elem<cytnx::cytnx_float>(locator));
           else if (self.dtype() == cytnx::Type.ComplexDouble)
             out = py::cast(self.get_elem<cytnx::cytnx_complex128>(locator));
           else if (self.dtype() == cytnx::Type.ComplexFloat)
             out = py::cast(self.get_elem<cytnx::cytnx_complex64>(locator));
           else
             cytnx_error_msg(true, "%s", "[ERROR] try to get element from a void Storage.");
           return out;
         })

    .def("set_elem", &f_UniTensor_setelem_scal<cytnx_complex128>)
    .def("set_elem", &f_UniTensor_setelem_scal<cytnx_complex64>)
    .def("set_elem", &f_UniTensor_setelem_scal<cytnx_double>)
    .def("set_elem", &f_UniTensor_setelem_scal<cytnx_float>)
    .def("set_elem", &f_UniTensor_setelem_scal<cytnx_int64>)
    .def("set_elem", &f_UniTensor_setelem_scal<cytnx_uint64>)
    .def("set_elem", &f_UniTensor_setelem_scal<cytnx_int32>)
    .def("set_elem", &f_UniTensor_setelem_scal<cytnx_uint32>)
    .def("set_elem", &f_UniTensor_setelem_scal<cytnx_int16>)
    .def("set_elem", &f_UniTensor_setelem_scal<cytnx_uint16>)
    .def("set_elem", &f_UniTensor_setelem_scal<cytnx_bool>)

    .def("set_elem", &f_UniTensor_setelem_scal_int<cytnx_complex128>)
    .def("set_elem", &f_UniTensor_setelem_scal_int<cytnx_complex64>)
    .def("set_elem", &f_UniTensor_setelem_scal_int<cytnx_double>)
    .def("set_elem", &f_UniTensor_setelem_scal_int<cytnx_float>)
    .def("set_elem", &f_UniTensor_setelem_scal_int<cytnx_int64>)
    .def("set_elem", &f_UniTensor_setelem_scal_int<cytnx_uint64>)
    .def("set_elem", &f_UniTensor_setelem_scal_int<cytnx_int32>)
    .def("set_elem", &f_UniTensor_setelem_scal_int<cytnx_uint32>)
    .def("set_elem", &f_UniTensor_setelem_scal_int<cytnx_int16>)
    .def("set_elem", &f_UniTensor_setelem_scal_int<cytnx_uint16>)
    .def("set_elem", &f_UniTensor_setelem_scal_int<cytnx_bool>)

    .def("__setitem__", &f_UniTensor_setelem_scal<cytnx_complex128>)
    .def("__setitem__", &f_UniTensor_setelem_scal<cytnx_complex64>)
    .def("__setitem__", &f_UniTensor_setelem_scal<cytnx_double>)
    .def("__setitem__", &f_UniTensor_setelem_scal<cytnx_float>)
    .def("__setitem__", &f_UniTensor_setelem_scal<cytnx_int64>)
    .def("__setitem__", &f_UniTensor_setelem_scal<cytnx_uint64>)
    .def("__setitem__", &f_UniTensor_setelem_scal<cytnx_int32>)
    .def("__setitem__", &f_UniTensor_setelem_scal<cytnx_uint32>)
    .def("__setitem__", &f_UniTensor_setelem_scal<cytnx_int16>)
    .def("__setitem__", &f_UniTensor_setelem_scal<cytnx_uint16>)
    .def("__setitem__", &f_UniTensor_setelem_scal<cytnx_bool>)

    .def("__setitem__", &f_UniTensor_setelem_scal_int<cytnx_complex128>)
    .def("__setitem__", &f_UniTensor_setelem_scal_int<cytnx_complex64>)
    .def("__setitem__", &f_UniTensor_setelem_scal_int<cytnx_double>)
    .def("__setitem__", &f_UniTensor_setelem_scal_int<cytnx_float>)
    .def("__setitem__", &f_UniTensor_setelem_scal_int<cytnx_int64>)
    .def("__setitem__", &f_UniTensor_setelem_scal_int<cytnx_uint64>)
    .def("__setitem__", &f_UniTensor_setelem_scal_int<cytnx_int32>)
    .def("__setitem__", &f_UniTensor_setelem_scal_int<cytnx_uint32>)
    .def("__setitem__", &f_UniTensor_setelem_scal_int<cytnx_int16>)
    .def("__setitem__", &f_UniTensor_setelem_scal_int<cytnx_uint16>)
    .def("__setitem__", &f_UniTensor_setelem_scal_int<cytnx_bool>)

    .def("is_contiguous", &UniTensor::is_contiguous)
    .def("is_diag", &UniTensor::is_diag)
    .def("is_tag", &UniTensor::is_tag)
    .def("is_braket_form", &UniTensor::is_braket_form)
    .def("same_data", &UniTensor::same_data)
    .def("labels", &UniTensor::labels)
    .def("bonds", [](UniTensor &self) { return self.bonds(); })
    .def("bond_", [](UniTensor &self, const cytnx_uint64 &idx){return self.bond_(idx);} ,py::arg("idx"))
    .def("bond_", [](UniTensor &self, const std::string &label){return self.bond_(label);} ,py::arg("label"))
    .def("bond", [](UniTensor &self, const cytnx_uint64 &idx){return self.bond(idx);} ,py::arg("idx"))
    .def("bond", [](UniTensor &self, const std::string &label){return self.bond(label);} ,py::arg("label"))
    .def("shape", &UniTensor::shape)
    .def("to_", &UniTensor::to_)
    .def(
      "to_different_device",
      [](UniTensor &self, const cytnx_int64 &device) {
        cytnx_error_msg(self.device() == device,
                        "[ERROR][pybind][to_diffferent_device] same device for to() should be "
                        "handle in python side.%s",
                        "\n");
        return self.to(device);
      },
      py::arg("device"))
    .def("clone", &UniTensor::clone)
    .def("__copy__", &UniTensor::clone)
    .def("__deepcopy__", &UniTensor::clone)
    .def(
      "Save", [](UniTensor &self, const std::string &fname) { self.Save(fname); }, py::arg("fname"))
    .def_static(
      "Load", [](const std::string &fname) { return UniTensor::Load(fname); }, py::arg("fname"))
    //.def("permute",&UniTensor::permute,py::arg("mapper"),py::arg("rowrank")=(cytnx_int64)-1,py::arg("by_label")=false)
    //.def("permute_",&UniTensor::permute_,py::arg("mapper"),py::arg("rowrank")=(cytnx_int64)-1,py::arg("by_label")=false)
    .def(
      "astype_different_type",
      [](cytnx::UniTensor &self, const cytnx_uint64 &new_type) {
        cytnx_error_msg(self.dtype() == new_type,
                        "[ERROR][pybind][astype_diffferent_type] same type for astype() should be "
                        "handle in python side.%s",
                        "\n");
        return self.astype(new_type);
      },
      py::arg("new_type"))

    // [Deprecated by_label!]
    .def("permute_", [](UniTensor &self, const std::vector<cytnx_int64> &mapper, const cytnx_int64 &rowrank){

                        self.permute_(mapper,rowrank);

                },py::arg("mapper"), py::arg("rowrank")=(cytnx_int64)(-1))

    .def("permute_", [](UniTensor &self, const std::vector<std::string> &mapper, const cytnx_int64 &rowrank){
                        self.permute_(mapper,rowrank);
                },py::arg("mapper"), py::arg("rowrank")=(cytnx_int64)(-1))

    .def("permute", [](UniTensor &self, const std::vector<cytnx_int64> &mapper, const cytnx_int64 &rowrank){

                        return self.permute(mapper,rowrank);

                },py::arg("mapper"), py::arg("rowrank")=(cytnx_int64)(-1))

    .def("permute", [](UniTensor &self, const std::vector<std::string> &mapper, const cytnx_int64 &rowrank){
                        return self.permute(mapper,rowrank);
                },py::arg("mapper"), py::arg("rowrank")=(cytnx_int64)(-1))




    .def("make_contiguous", &UniTensor::contiguous)
    .def("contiguous_", &UniTensor::contiguous_)
    .def("print_diagram", &UniTensor::print_diagram, py::arg("bond_info") = false,
         py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
    .def("print_blocks", &UniTensor::print_blocks, py::arg("full_info") = true,
         py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
    .def("print_block", &UniTensor::print_block, py::arg("idx"), py::arg("full_info") = true,
         py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())

    .def("group_basis_", &UniTensor::group_basis_)
    .def("group_basis", &UniTensor::group_basis)
    .def(
      "get_block",
      [](const UniTensor &self, const cytnx_uint64 &idx) { return self.get_block(idx); },
      py::arg("idx") = (cytnx_uint64)(0))

    .def(
      "get_block",
      [](const UniTensor &self, const std::vector<cytnx_int64> &qnum, const bool &force) {
        return self.get_block(qnum, force);
      },
      py::arg("qnum"), py::arg("force") = false)
    .def(
      "get_block",
      [](const UniTensor &self, const std::vector<cytnx_uint64> &qnum, const bool &force) {
        return self.get_block(qnum, force);
      },
      py::arg("qnum"), py::arg("force") = false)

    .def(
      "get_block",
      [](const UniTensor &self, const std::vector<std::string> &label, const std::vector<cytnx_int64> &qnum, const bool &force) {
        return self.get_block(label, qnum, force);
      },
      py::arg("labels"), py::arg("qnum"), py::arg("force") = false)
    .def(
      "get_block",
      [](const UniTensor &self, const std::vector<std::string> &label, const std::vector<cytnx_uint64> &qnum, const bool &force) {
        return self.get_block(label,qnum, force);
      },
      py::arg("labels"), py::arg("qnum"), py::arg("force") = false)
    .def(
      "get_block_",
      [](UniTensor &self, const std::vector<cytnx_int64> &qnum, const bool &force) {
        return self.get_block_(qnum, force);
      },
      py::arg("qnum"), py::arg("force") = false)
    .def(
      "get_block_",
      [](UniTensor &self, const std::vector<cytnx_uint64> &qnum, const bool &force) {
        return self.get_block_(qnum, force);
      },
      py::arg("qnum"), py::arg("force") = false)

    .def(
      "get_block_",
      [](UniTensor &self, const std::vector<std::string> &labels, const std::vector<cytnx_int64> &qnum, const bool &force) {
        return self.get_block_(labels, qnum, force);
      },
      py::arg("labels"), py::arg("qnum"), py::arg("force") = false)
    .def(
      "get_block_",
      [](UniTensor &self, const std::vector<std::string> &labels, const std::vector<cytnx_uint64> &qnum, const bool &force) {
        return self.get_block_(labels,qnum, force);
      },
      py::arg("labels"), py::arg("qnum"), py::arg("force") = false)


    .def(
      "get_block_", [](UniTensor &self, const cytnx_uint64 &idx) { return self.get_block_(idx); },
      py::arg("idx") = (cytnx_uint64)(0))
    .def("get_blocks", [](const UniTensor &self) { return self.get_blocks(); })
    .def(
      "get_blocks_",
      [](const UniTensor &self, const bool &slient) { return self.get_blocks_(slient); },
      py::arg("slient") = false)
    .def(
      "get_blocks_", [](UniTensor &self, const bool &slient) { return self.get_blocks_(slient); },
      py::arg("slient") = false)
    .def(
      "put_block",
      [](UniTensor &self, const cytnx::Tensor &in, const cytnx_uint64 &idx) {
        self.put_block(in, idx);
      },
      py::arg("in"), py::arg("idx") = (cytnx_uint64)(0))

    .def(
      "put_block",
      [](UniTensor &self, const cytnx::Tensor &in, const std::vector<cytnx_int64> &qnum,
         const bool &force) { self.put_block(in, qnum, force); },
      py::arg("in"), py::arg("qidx"), py::arg("force") = false)
     .def(
      "put_block",
      [](UniTensor &self, cytnx::Tensor &in, const std::vector<std::string> &lbls, const std::vector<cytnx_int64> &qnum,
         const bool &force) { self.put_block(in, lbls, qnum, force); },
      py::arg("in"), py::arg("labels"), py::arg("qidx"), py::arg("force") = false)
    .def(
      "put_block_",
      [](UniTensor &self, cytnx::Tensor &in, const cytnx_uint64 &idx) { self.put_block_(in, idx); },
      py::arg("in"), py::arg("idx") = (cytnx_uint64)(0))

    .def(
      "put_block_",
      [](UniTensor &self, cytnx::Tensor &in, const std::vector<cytnx_int64> &qnum,
         const bool &force) { self.put_block_(in, qnum, force); },
      py::arg("in"), py::arg("qidx"), py::arg("force") = false)
     .def(
      "put_block_",
      [](UniTensor &self, cytnx::Tensor &in, const std::vector<std::string> &lbls, const std::vector<cytnx_int64> &qnum,
         const bool &force) { self.put_block_(in, lbls, qnum, force); },
      py::arg("in"), py::arg("labels"), py::arg("qidx"), py::arg("force") = false)
    .def(
      "__repr__",
      [](UniTensor &self) -> std::string {
        std::cout << self << std::endl;
        return std::string("");
      },
      py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
    .def("to_dense", &UniTensor::to_dense)
    .def("to_dense_", &UniTensor::to_dense_)
    .def("combineBonds",
         [](UniTensor &self, const std::vector<cytnx_int64> &indicators, const bool &force,
            const bool &by_label)
         {
            if(by_label){
                cytnx_warning_msg(true,"[Deprecated notice] by_label option is going to be deprecated. using string will automatically recognized as labels.%s","\n");
                self.combineBonds(indicators,force,by_label);
            }else{
                self.combineBonds(indicators,force);
            }
         },
         py::arg("indicators"), py::arg("force") = false, py::arg("by_label") = false)

    .def("combineBonds",
         [](UniTensor &self, const std::vector<std::string> &indicators, const bool &force)
         {
            self.combineBonds(indicators,force);
         },
         py::arg("indicators"), py::arg("force") = false)



    .def("contract", &UniTensor::contract)

    .def("getTotalQnums", &UniTensor::getTotalQnums, py::arg("physical")=false)

    .def("get_blocks_qnums", &UniTensor::get_blocks_qnums)

    // arithmetic >>
    .def("__neg__",
         [](UniTensor &self) {
           if (self.dtype() == Type.Double) {
             return linalg::Mul(cytnx_double(-1), self);
           } else if (self.dtype() == Type.ComplexDouble) {
             return linalg::Mul(cytnx_complex128(-1, 0), self);
           } else if (self.dtype() == Type.Float) {
             return linalg::Mul(cytnx_float(-1), self);
           } else if (self.dtype() == Type.ComplexFloat) {
             return linalg::Mul(cytnx_complex64(-1, 0), self);
           } else {
             return linalg::Mul(-1, self);
           }
         })
    .def("__pos__", [](UniTensor &self) { return self; })
    .def("__add__", [](UniTensor &self, const UniTensor &rhs) { return linalg::Add(self, rhs); })
    .def("__add__",
         [](UniTensor &self, const cytnx::cytnx_complex128 &rhs) { return linalg::Add(self, rhs); })
    .def("__add__",
         [](UniTensor &self, const cytnx::cytnx_complex64 &rhs) { return linalg::Add(self, rhs); })
    .def("__add__",
         [](UniTensor &self, const cytnx::cytnx_double &rhs) { return linalg::Add(self, rhs); })
    .def("__add__",
         [](UniTensor &self, const cytnx::cytnx_float &rhs) { return linalg::Add(self, rhs); })
    .def("__add__",
         [](UniTensor &self, const cytnx::cytnx_int64 &rhs) { return linalg::Add(self, rhs); })
    .def("__add__",
         [](UniTensor &self, const cytnx::cytnx_uint64 &rhs) { return linalg::Add(self, rhs); })
    .def("__add__",
         [](UniTensor &self, const cytnx::cytnx_int32 &rhs) { return linalg::Add(self, rhs); })
    .def("__add__",
         [](UniTensor &self, const cytnx::cytnx_uint32 &rhs) { return linalg::Add(self, rhs); })
    .def("__add__",
         [](UniTensor &self, const cytnx::cytnx_int16 &rhs) { return linalg::Add(self, rhs); })
    .def("__add__",
         [](UniTensor &self, const cytnx::cytnx_uint16 &rhs) { return linalg::Add(self, rhs); })
    .def("__add__",
         [](UniTensor &self, const cytnx::cytnx_bool &rhs) { return linalg::Add(self, rhs); })

    .def("__radd__",
         [](UniTensor &self, const cytnx::cytnx_complex128 &lhs) { return linalg::Add(lhs, self); })
    .def("__radd__",
         [](UniTensor &self, const cytnx::cytnx_complex64 &lhs) { return linalg::Add(lhs, self); })
    .def("__radd__",
         [](UniTensor &self, const cytnx::cytnx_double &lhs) { return linalg::Add(lhs, self); })
    .def("__radd__",
         [](UniTensor &self, const cytnx::cytnx_float &lhs) { return linalg::Add(lhs, self); })
    .def("__radd__",
         [](UniTensor &self, const cytnx::cytnx_int64 &lhs) { return linalg::Add(lhs, self); })
    .def("__radd__",
         [](UniTensor &self, const cytnx::cytnx_uint64 &lhs) { return linalg::Add(lhs, self); })
    .def("__radd__",
         [](UniTensor &self, const cytnx::cytnx_int32 &lhs) { return linalg::Add(lhs, self); })
    .def("__radd__",
         [](UniTensor &self, const cytnx::cytnx_uint32 &lhs) { return linalg::Add(lhs, self); })
    .def("__radd__",
         [](UniTensor &self, const cytnx::cytnx_int16 &lhs) { return linalg::Add(lhs, self); })
    .def("__radd__",
         [](UniTensor &self, const cytnx::cytnx_uint16 &lhs) { return linalg::Add(lhs, self); })
    .def("__radd__",
         [](UniTensor &self, const cytnx::cytnx_bool &lhs) { return linalg::Add(lhs, self); })

    .def("__iadd__",
         [](UniTensor &self, const UniTensor &rhs) {
           return self.Add_(rhs);
         })  // these will return self!
    .def("__iadd__",
         [](UniTensor &self, const cytnx::cytnx_complex128 &rhs) { return self.Add_(rhs); })
    .def("__iadd__",
         [](UniTensor &self, const cytnx::cytnx_complex64 &rhs) { return self.Add_(rhs); })
    .def("__iadd__", [](UniTensor &self, const cytnx::cytnx_double &rhs) { return self.Add_(rhs); })
    .def("__iadd__", [](UniTensor &self, const cytnx::cytnx_float &rhs) { return self.Add_(rhs); })
    .def("__iadd__", [](UniTensor &self, const cytnx::cytnx_int64 &rhs) { return self.Add_(rhs); })
    .def("__iadd__", [](UniTensor &self, const cytnx::cytnx_uint64 &rhs) { return self.Add_(rhs); })
    .def("__iadd__", [](UniTensor &self, const cytnx::cytnx_int32 &rhs) { return self.Add_(rhs); })
    .def("__iadd__", [](UniTensor &self, const cytnx::cytnx_uint32 &rhs) { return self.Add_(rhs); })
    .def("__iadd__", [](UniTensor &self, const cytnx::cytnx_int16 &rhs) { return self.Add_(rhs); })
    .def("__iadd__", [](UniTensor &self, const cytnx::cytnx_uint16 &rhs) { return self.Add_(rhs); })
    .def("__iadd__", [](UniTensor &self, const cytnx::cytnx_bool &rhs) { return self.Add_(rhs); })

    .def("__sub__", [](UniTensor &self, const UniTensor &rhs) { return linalg::Sub(self, rhs); })
    .def("__sub__",
         [](UniTensor &self, const cytnx::cytnx_complex128 &rhs) { return linalg::Sub(self, rhs); })
    .def("__sub__",
         [](UniTensor &self, const cytnx::cytnx_complex64 &rhs) { return linalg::Sub(self, rhs); })
    .def("__sub__",
         [](UniTensor &self, const cytnx::cytnx_double &rhs) { return linalg::Sub(self, rhs); })
    .def("__sub__",
         [](UniTensor &self, const cytnx::cytnx_float &rhs) { return linalg::Sub(self, rhs); })
    .def("__sub__",
         [](UniTensor &self, const cytnx::cytnx_int64 &rhs) { return linalg::Sub(self, rhs); })
    .def("__sub__",
         [](UniTensor &self, const cytnx::cytnx_uint64 &rhs) { return linalg::Sub(self, rhs); })
    .def("__sub__",
         [](UniTensor &self, const cytnx::cytnx_int32 &rhs) { return linalg::Sub(self, rhs); })
    .def("__sub__",
         [](UniTensor &self, const cytnx::cytnx_uint32 &rhs) { return linalg::Sub(self, rhs); })
    .def("__sub__",
         [](UniTensor &self, const cytnx::cytnx_int16 &rhs) { return linalg::Sub(self, rhs); })
    .def("__sub__",
         [](UniTensor &self, const cytnx::cytnx_uint16 &rhs) { return linalg::Sub(self, rhs); })
    .def("__sub__",
         [](UniTensor &self, const cytnx::cytnx_bool &rhs) { return linalg::Sub(self, rhs); })

    .def("__rsub__",
         [](UniTensor &self, const cytnx::cytnx_complex128 &lhs) { return linalg::Sub(lhs, self); })
    .def("__rsub__",
         [](UniTensor &self, const cytnx::cytnx_complex64 &lhs) { return linalg::Sub(lhs, self); })
    .def("__rsub__",
         [](UniTensor &self, const cytnx::cytnx_double &lhs) { return linalg::Sub(lhs, self); })
    .def("__rsub__",
         [](UniTensor &self, const cytnx::cytnx_float &lhs) { return linalg::Sub(lhs, self); })
    .def("__rsub__",
         [](UniTensor &self, const cytnx::cytnx_int64 &lhs) { return linalg::Sub(lhs, self); })
    .def("__rsub__",
         [](UniTensor &self, const cytnx::cytnx_uint64 &lhs) { return linalg::Sub(lhs, self); })
    .def("__rsub__",
         [](UniTensor &self, const cytnx::cytnx_int32 &lhs) { return linalg::Sub(lhs, self); })
    .def("__rsub__",
         [](UniTensor &self, const cytnx::cytnx_uint32 &lhs) { return linalg::Sub(lhs, self); })
    .def("__rsub__",
         [](UniTensor &self, const cytnx::cytnx_int16 &lhs) { return linalg::Sub(lhs, self); })
    .def("__rsub__",
         [](UniTensor &self, const cytnx::cytnx_uint16 &lhs) { return linalg::Sub(lhs, self); })
    .def("__rsub__",
         [](UniTensor &self, const cytnx::cytnx_bool &lhs) { return linalg::Sub(lhs, self); })

    .def("__isub__",
         [](UniTensor &self, const UniTensor &rhs) {
           return self.Sub_(rhs);
         })  // these will return self!
    .def("__isub__",
         [](UniTensor &self, const cytnx::cytnx_complex128 &rhs) { return self.Sub_(rhs); })
    .def("__isub__",
         [](UniTensor &self, const cytnx::cytnx_complex64 &rhs) { return self.Sub_(rhs); })
    .def("__isub__", [](UniTensor &self, const cytnx::cytnx_double &rhs) { return self.Sub_(rhs); })
    .def("__isub__", [](UniTensor &self, const cytnx::cytnx_float &rhs) { return self.Sub_(rhs); })
    .def("__isub__", [](UniTensor &self, const cytnx::cytnx_int64 &rhs) { return self.Sub_(rhs); })
    .def("__isub__", [](UniTensor &self, const cytnx::cytnx_uint64 &rhs) { return self.Sub_(rhs); })
    .def("__isub__", [](UniTensor &self, const cytnx::cytnx_int32 &rhs) { return self.Sub_(rhs); })
    .def("__isub__", [](UniTensor &self, const cytnx::cytnx_uint32 &rhs) { return self.Sub_(rhs); })
    .def("__isub__", [](UniTensor &self, const cytnx::cytnx_int16 &rhs) { return self.Sub_(rhs); })
    .def("__isub__", [](UniTensor &self, const cytnx::cytnx_uint16 &rhs) { return self.Sub_(rhs); })
    .def("__isub__", [](UniTensor &self, const cytnx::cytnx_bool &rhs) { return self.Sub_(rhs); })

    .def("__mul__", [](UniTensor &self, const UniTensor &rhs) { return linalg::Mul(self, rhs); })
    .def("__mul__",
         [](UniTensor &self, const cytnx::cytnx_complex128 &rhs) { return linalg::Mul(self, rhs); })
    .def("__mul__",
         [](UniTensor &self, const cytnx::cytnx_complex64 &rhs) { return linalg::Mul(self, rhs); })
    .def("__mul__",
         [](UniTensor &self, const cytnx::cytnx_double &rhs) { return linalg::Mul(self, rhs); })
    .def("__mul__",
         [](UniTensor &self, const cytnx::cytnx_float &rhs) { return linalg::Mul(self, rhs); })
    .def("__mul__",
         [](UniTensor &self, const cytnx::cytnx_int64 &rhs) { return linalg::Mul(self, rhs); })
    .def("__mul__",
         [](UniTensor &self, const cytnx::cytnx_uint64 &rhs) { return linalg::Mul(self, rhs); })
    .def("__mul__",
         [](UniTensor &self, const cytnx::cytnx_int32 &rhs) { return linalg::Mul(self, rhs); })
    .def("__mul__",
         [](UniTensor &self, const cytnx::cytnx_uint32 &rhs) { return linalg::Mul(self, rhs); })
    .def("__mul__",
         [](UniTensor &self, const cytnx::cytnx_int16 &rhs) { return linalg::Mul(self, rhs); })
    .def("__mul__",
         [](UniTensor &self, const cytnx::cytnx_uint16 &rhs) { return linalg::Mul(self, rhs); })
    .def("__mul__",
         [](UniTensor &self, const cytnx::cytnx_bool &rhs) { return linalg::Mul(self, rhs); })

    .def("__rmul__",
         [](UniTensor &self, const cytnx::cytnx_complex128 &lhs) { return linalg::Mul(lhs, self); })
    .def("__rmul__",
         [](UniTensor &self, const cytnx::cytnx_complex64 &lhs) { return linalg::Mul(lhs, self); })
    .def("__rmul__",
         [](UniTensor &self, const cytnx::cytnx_double &lhs) { return linalg::Mul(lhs, self); })
    .def("__rmul__",
         [](UniTensor &self, const cytnx::cytnx_float &lhs) { return linalg::Mul(lhs, self); })
    .def("__rmul__",
         [](UniTensor &self, const cytnx::cytnx_int64 &lhs) { return linalg::Mul(lhs, self); })
    .def("__rmul__",
         [](UniTensor &self, const cytnx::cytnx_uint64 &lhs) { return linalg::Mul(lhs, self); })
    .def("__rmul__",
         [](UniTensor &self, const cytnx::cytnx_int32 &lhs) { return linalg::Mul(lhs, self); })
    .def("__rmul__",
         [](UniTensor &self, const cytnx::cytnx_uint32 &lhs) { return linalg::Mul(lhs, self); })
    .def("__rmul__",
         [](UniTensor &self, const cytnx::cytnx_int16 &lhs) { return linalg::Mul(lhs, self); })
    .def("__rmul__",
         [](UniTensor &self, const cytnx::cytnx_uint16 &lhs) { return linalg::Mul(lhs, self); })
    .def("__rmul__",
         [](UniTensor &self, const cytnx::cytnx_bool &lhs) { return linalg::Mul(lhs, self); })

    /*
    .def("__mod__",[](UniTensor &self, const UniTensor &rhs){return self.Mod(rhs);})
    .def("__mod__",[](UniTensor &self, const cytnx::cytnx_complex128&rhs){return self.Mod(rhs);})
    .def("__mod__",[](UniTensor &self, const cytnx::cytnx_complex64 &rhs){return self.Mod(rhs);})
    .def("__mod__",[](UniTensor &self, const cytnx::cytnx_double    &rhs){return self.Mod(rhs);})
    .def("__mod__",[](UniTensor &self, const cytnx::cytnx_float     &rhs){return self.Mod(rhs);})
    .def("__mod__",[](UniTensor &self, const cytnx::cytnx_int64     &rhs){return self.Mod(rhs);})
    .def("__mod__",[](UniTensor &self, const cytnx::cytnx_uint64    &rhs){return self.Mod(rhs);})
    .def("__mod__",[](UniTensor &self, const cytnx::cytnx_int32     &rhs){return self.Mod(rhs);})
    .def("__mod__",[](UniTensor &self, const cytnx::cytnx_uint32    &rhs){return self.Mod(rhs);})
    .def("__mod__",[](UniTensor &self, const cytnx::cytnx_int16     &rhs){return self.Mod(rhs);})
    .def("__mod__",[](UniTensor &self, const cytnx::cytnx_uint16    &rhs){return self.Mod(rhs);})
    .def("__mod__",[](UniTensor &self, const cytnx::cytnx_bool    &rhs){return self.Mod(rhs);})

    .def("__rmod__",[](UniTensor &self, const cytnx::cytnx_complex128&lhs){return
    linalg::Mod(lhs,self);}) .def("__rmod__",[](UniTensor &self, const cytnx::cytnx_complex64
    &lhs){return linalg::Mod(lhs,self);}) .def("__rmod__",[](UniTensor &self, const
    cytnx::cytnx_double    &lhs){return linalg::Mod(lhs,self);}) .def("__rmod__",[](UniTensor &self,
    const cytnx::cytnx_float     &lhs){return linalg::Mod(lhs,self);}) .def("__rmod__",[](UniTensor
    &self, const cytnx::cytnx_int64     &lhs){return linalg::Mod(lhs,self);})
    .def("__rmod__",[](UniTensor &self, const cytnx::cytnx_uint64    &lhs){return
    linalg::Mod(lhs,self);}) .def("__rmod__",[](UniTensor &self, const cytnx::cytnx_int32
    &lhs){return linalg::Mod(lhs,self);}) .def("__rmod__",[](UniTensor &self, const
    cytnx::cytnx_uint32    &lhs){return linalg::Mod(lhs,self);}) .def("__rmod__",[](UniTensor &self,
    const cytnx::cytnx_int16     &lhs){return linalg::Mod(lhs,self);}) .def("__rmod__",[](UniTensor
    &self, const cytnx::cytnx_uint16    &lhs){return linalg::Mod(lhs,self);})
    .def("__rmod__",[](UniTensor &self, const cytnx::cytnx_bool      &lhs){return
    linalg::Mod(lhs,self);})
    */

    .def("__imul__",
         [](UniTensor &self, const UniTensor &rhs) {
           return self.Mul_(rhs);
         })  // these will return self!
    .def("__imul__",
         [](UniTensor &self, const cytnx::cytnx_complex128 &rhs) { return self.Mul_(rhs); })
    .def("__imul__",
         [](UniTensor &self, const cytnx::cytnx_complex64 &rhs) { return self.Mul_(rhs); })
    .def("__imul__", [](UniTensor &self, const cytnx::cytnx_double &rhs) { return self.Mul_(rhs); })
    .def("__imul__", [](UniTensor &self, const cytnx::cytnx_float &rhs) { return self.Mul_(rhs); })
    .def("__imul__", [](UniTensor &self, const cytnx::cytnx_int64 &rhs) { return self.Mul_(rhs); })
    .def("__imul__", [](UniTensor &self, const cytnx::cytnx_uint64 &rhs) { return self.Mul_(rhs); })
    .def("__imul__", [](UniTensor &self, const cytnx::cytnx_int32 &rhs) { return self.Mul_(rhs); })
    .def("__imul__", [](UniTensor &self, const cytnx::cytnx_uint32 &rhs) { return self.Mul_(rhs); })
    .def("__imul__", [](UniTensor &self, const cytnx::cytnx_int16 &rhs) { return self.Mul_(rhs); })
    .def("__imul__", [](UniTensor &self, const cytnx::cytnx_uint16 &rhs) { return self.Mul_(rhs); })
    .def("__imul__", [](UniTensor &self, const cytnx::cytnx_bool &rhs) { return self.Mul_(rhs); })

    .def("__truediv__",
         [](UniTensor &self, const UniTensor &rhs) { return linalg::Div(self, rhs); })
    .def("__truediv__",
         [](UniTensor &self, const cytnx::cytnx_complex128 &rhs) { return linalg::Div(self, rhs); })
    .def("__truediv__",
         [](UniTensor &self, const cytnx::cytnx_complex64 &rhs) { return linalg::Div(self, rhs); })
    .def("__truediv__",
         [](UniTensor &self, const cytnx::cytnx_double &rhs) { return linalg::Div(self, rhs); })
    .def("__truediv__",
         [](UniTensor &self, const cytnx::cytnx_float &rhs) { return linalg::Div(self, rhs); })
    .def("__truediv__",
         [](UniTensor &self, const cytnx::cytnx_int64 &rhs) { return linalg::Div(self, rhs); })
    .def("__truediv__",
         [](UniTensor &self, const cytnx::cytnx_uint64 &rhs) { return linalg::Div(self, rhs); })
    .def("__truediv__",
         [](UniTensor &self, const cytnx::cytnx_int32 &rhs) { return linalg::Div(self, rhs); })
    .def("__truediv__",
         [](UniTensor &self, const cytnx::cytnx_uint32 &rhs) { return linalg::Div(self, rhs); })
    .def("__truediv__",
         [](UniTensor &self, const cytnx::cytnx_int16 &rhs) { return linalg::Div(self, rhs); })
    .def("__truediv__",
         [](UniTensor &self, const cytnx::cytnx_uint16 &rhs) { return linalg::Div(self, rhs); })
    .def("__truediv__",
         [](UniTensor &self, const cytnx::cytnx_bool &rhs) { return linalg::Div(self, rhs); })

    .def("__rtruediv__",
         [](UniTensor &self, const cytnx::cytnx_complex128 &lhs) { return linalg::Div(lhs, self); })
    .def("__rtruediv__",
         [](UniTensor &self, const cytnx::cytnx_complex64 &lhs) { return linalg::Div(lhs, self); })
    .def("__rtruediv__",
         [](UniTensor &self, const cytnx::cytnx_double &lhs) { return linalg::Div(lhs, self); })
    .def("__rtruediv__",
         [](UniTensor &self, const cytnx::cytnx_float &lhs) { return linalg::Div(lhs, self); })
    .def("__rtruediv__",
         [](UniTensor &self, const cytnx::cytnx_int64 &lhs) { return linalg::Div(lhs, self); })
    .def("__rtruediv__",
         [](UniTensor &self, const cytnx::cytnx_uint64 &lhs) { return linalg::Div(lhs, self); })
    .def("__rtruediv__",
         [](UniTensor &self, const cytnx::cytnx_int32 &lhs) { return linalg::Div(lhs, self); })
    .def("__rtruediv__",
         [](UniTensor &self, const cytnx::cytnx_uint32 &lhs) { return linalg::Div(lhs, self); })
    .def("__rtruediv__",
         [](UniTensor &self, const cytnx::cytnx_int16 &lhs) { return linalg::Div(lhs, self); })
    .def("__rtruediv__",
         [](UniTensor &self, const cytnx::cytnx_uint16 &lhs) { return linalg::Div(lhs, self); })
    .def("__rtruediv__",
         [](UniTensor &self, const cytnx::cytnx_bool &lhs) { return linalg::Div(lhs, self); })

    .def("__itruediv__",
         [](UniTensor &self, const UniTensor &rhs) {
           return self.Div_(rhs);
         })  // these will return self!
    .def("__itruediv__",
         [](UniTensor &self, const cytnx::cytnx_complex128 &rhs) { return self.Div_(rhs); })
    .def("__itruediv__",
         [](UniTensor &self, const cytnx::cytnx_complex64 &rhs) { return self.Div_(rhs); })
    .def("__itruediv__",
         [](UniTensor &self, const cytnx::cytnx_double &rhs) { return self.Div_(rhs); })
    .def("__itruediv__",
         [](UniTensor &self, const cytnx::cytnx_float &rhs) { return self.Div_(rhs); })
    .def("__itruediv__",
         [](UniTensor &self, const cytnx::cytnx_int64 &rhs) { return self.Div_(rhs); })
    .def("__itruediv__",
         [](UniTensor &self, const cytnx::cytnx_uint64 &rhs) { return self.Div_(rhs); })
    .def("__itruediv__",
         [](UniTensor &self, const cytnx::cytnx_int32 &rhs) { return self.Div_(rhs); })
    .def("__itruediv__",
         [](UniTensor &self, const cytnx::cytnx_uint32 &rhs) { return self.Div_(rhs); })
    .def("__itruediv__",
         [](UniTensor &self, const cytnx::cytnx_int16 &rhs) { return self.Div_(rhs); })
    .def("__itruediv__",
         [](UniTensor &self, const cytnx::cytnx_uint16 &rhs) { return self.Div_(rhs); })
    .def("__itruediv__",
         [](UniTensor &self, const cytnx::cytnx_bool &rhs) { return self.Div_(rhs); })
    .def("__floordiv__",
         [](UniTensor &self, const UniTensor &rhs) { return linalg::Div(self, rhs); })
    .def("__floordiv__",
         [](UniTensor &self, const cytnx::cytnx_complex128 &rhs) { return linalg::Div(self, rhs); })
    .def("__floordiv__",
         [](UniTensor &self, const cytnx::cytnx_complex64 &rhs) { return linalg::Div(self, rhs); })
    .def("__floordiv__",
         [](UniTensor &self, const cytnx::cytnx_double &rhs) { return linalg::Div(self, rhs); })
    .def("__floordiv__",
         [](UniTensor &self, const cytnx::cytnx_float &rhs) { return linalg::Div(self, rhs); })
    .def("__floordiv__",
         [](UniTensor &self, const cytnx::cytnx_int64 &rhs) { return linalg::Div(self, rhs); })
    .def("__floordiv__",
         [](UniTensor &self, const cytnx::cytnx_uint64 &rhs) { return linalg::Div(self, rhs); })
    .def("__floordiv__",
         [](UniTensor &self, const cytnx::cytnx_int32 &rhs) { return linalg::Div(self, rhs); })
    .def("__floordiv__",
         [](UniTensor &self, const cytnx::cytnx_uint32 &rhs) { return linalg::Div(self, rhs); })
    .def("__floordiv__",
         [](UniTensor &self, const cytnx::cytnx_int16 &rhs) { return linalg::Div(self, rhs); })
    .def("__floordiv__",
         [](UniTensor &self, const cytnx::cytnx_uint16 &rhs) { return linalg::Div(self, rhs); })
    .def("__floordiv__",
         [](UniTensor &self, const cytnx::cytnx_bool &rhs) { return linalg::Div(self, rhs); })

    .def("__rfloordiv__",
         [](UniTensor &self, const cytnx::cytnx_complex128 &lhs) { return linalg::Div(lhs, self); })
    .def("__rfloordiv__",
         [](UniTensor &self, const cytnx::cytnx_complex64 &lhs) { return linalg::Div(lhs, self); })
    .def("__rfloordiv__",
         [](UniTensor &self, const cytnx::cytnx_double &lhs) { return linalg::Div(lhs, self); })
    .def("__rfloordiv__",
         [](UniTensor &self, const cytnx::cytnx_float &lhs) { return linalg::Div(lhs, self); })
    .def("__rfloordiv__",
         [](UniTensor &self, const cytnx::cytnx_int64 &lhs) { return linalg::Div(lhs, self); })
    .def("__rfloordiv__",
         [](UniTensor &self, const cytnx::cytnx_uint64 &lhs) { return linalg::Div(lhs, self); })
    .def("__rfloordiv__",
         [](UniTensor &self, const cytnx::cytnx_int32 &lhs) { return linalg::Div(lhs, self); })
    .def("__rfloordiv__",
         [](UniTensor &self, const cytnx::cytnx_uint32 &lhs) { return linalg::Div(lhs, self); })
    .def("__rfloordiv__",
         [](UniTensor &self, const cytnx::cytnx_int16 &lhs) { return linalg::Div(lhs, self); })
    .def("__rfloordiv__",
         [](UniTensor &self, const cytnx::cytnx_uint16 &lhs) { return linalg::Div(lhs, self); })
    .def("__rfloordiv__",
         [](UniTensor &self, const cytnx::cytnx_bool &lhs) { return linalg::Div(lhs, self); })

    .def("__ifloordiv__",
         [](UniTensor &self, const UniTensor &rhs) {
           return self.Div_(rhs);
         })  // these will return self!
    .def("__ifloordiv__",
         [](UniTensor &self, const cytnx::cytnx_complex128 &rhs) { return self.Div_(rhs); })
    .def("__ifloordiv__",
         [](UniTensor &self, const cytnx::cytnx_complex64 &rhs) { return self.Div_(rhs); })
    .def("__ifloordiv__",
         [](UniTensor &self, const cytnx::cytnx_double &rhs) { return self.Div_(rhs); })
    .def("__ifloordiv__",
         [](UniTensor &self, const cytnx::cytnx_float &rhs) { return self.Div_(rhs); })
    .def("__ifloordiv__",
         [](UniTensor &self, const cytnx::cytnx_int64 &rhs) { return self.Div_(rhs); })
    .def("__ifloordiv__",
         [](UniTensor &self, const cytnx::cytnx_uint64 &rhs) { return self.Div_(rhs); })
    .def("__ifloordiv__",
         [](UniTensor &self, const cytnx::cytnx_int32 &rhs) { return self.Div_(rhs); })
    .def("__ifloordiv__",
         [](UniTensor &self, const cytnx::cytnx_uint32 &rhs) { return self.Div_(rhs); })
    .def("__ifloordiv__",
         [](UniTensor &self, const cytnx::cytnx_int16 &rhs) { return self.Div_(rhs); })
    .def("__ifloordiv__",
         [](UniTensor &self, const cytnx::cytnx_uint16 &rhs) { return self.Div_(rhs); })
    .def("__ifloordiv__",
         [](UniTensor &self, const cytnx::cytnx_bool &rhs) { return self.Div_(rhs); })
    .def("__pow__", [](UniTensor &self, const cytnx::cytnx_double &p) { return self.Pow(p); })
    .def("c__ipow__", [](UniTensor &self, const cytnx::cytnx_double &p) { self.Pow_(p); })
    .def("Pow", &UniTensor::Pow)
    .def("cPow_", &UniTensor::Pow_)
    .def("cConj_", &UniTensor::Conj_)
    .def("Conj", &UniTensor::Conj)

    .def("cTrace_", [](UniTensor &self, const cytnx_int64 &a, const cytnx_int64 &b){

                            return self.Trace_(a,b);

                    },
                    py::arg("a")=0, py::arg("b")=1)

    .def("cTrace_", [](UniTensor &self, const std::string &a, const std::string &b){
                    return self.Trace_(a,b);
                 },
                 py::arg("a"), py::arg("b"))


    .def("Trace", [](UniTensor &self, const cytnx_int64 &a, const cytnx_int64 &b){

                        return self.Trace(a,b);

                 },
                 py::arg("a")=0, py::arg("b")=1)

    .def("Trace", [](UniTensor &self, const std::string &a, const std::string &b){
                    return self.Trace(a,b);
                 },
                 py::arg("a"), py::arg("b"))

    .def("Norm", &UniTensor::Norm)
    .def("cTranspose_", &UniTensor::Transpose_)
    .def("Transpose", &UniTensor::Transpose)
    .def("cnormalize_", &UniTensor::normalize_)
    .def("normalize", &UniTensor::normalize)

    .def("cDagger_", &UniTensor::Dagger_)
    .def("Dagger", &UniTensor::Dagger)
    .def("ctag", &UniTensor::tag)
    .def("truncate",[](UniTensor &self, const cytnx_int64 &bond_idx, const cytnx_uint64 &dim){

                         return self.truncate(bond_idx, dim);

                    },
                    py::arg("bond_idx"), py::arg("dim"))
    .def("truncate",[](UniTensor &self, const std::string &label, const cytnx_uint64 &dim){
                        return self.truncate(label, dim);
                    },
                    py::arg("label"), py::arg("dim"))

    .def("ctruncate_",[](UniTensor &self, const cytnx_int64 &bond_idx, const cytnx_uint64 &dim){

                            return self.truncate_(bond_idx, dim);

                    },
                    py::arg("bond_idx"), py::arg("dim"))

    .def("ctruncate_",[](UniTensor &self, const std::string &label, const cytnx_uint64 &dim){
                        return self.truncate(label, dim);
                    },
                    py::arg("label"), py::arg("dim"))

    //[Generator]
    .def_static("identity", [](const cytnx_uint64 &dim, const std::vector<std::string> &in_labels,
                  const cytnx_bool &is_diag,
                  const unsigned int &dtype,
                  const int &device,
                  const std::string &name)
                {
                  return UniTensor::identity(dim, in_labels, is_diag, dtype, device, name);
                }, py::arg("dim"), py::arg("labels") = std::vector<std::string>(),
                   py::arg("is_diag") = false,
                   py::arg("dtype") = (unsigned int)Type.Double,
                   py::arg("device") = int(Device.cpu),
                   py::arg("name") = std::string(""))
     .def_static("eye", [](const cytnx_uint64 &dim, const std::vector<std::string> &in_labels,
                  const cytnx_bool &is_diag,
                  const unsigned int &dtype,
                  const int &device,
                  const std::string &name)
                {
                  return UniTensor::eye(dim, in_labels, is_diag, dtype, device, name);
                }, py::arg("dim"), py::arg("labels") = std::vector<std::string>(),
                   py::arg("is_diag") = false,
                   py::arg("dtype") = (unsigned int)Type.Double,
                   py::arg("device") = int(Device.cpu),
                   py::arg("name") = std::string(""))
    .def_static("ones", [](const cytnx_uint64 &Nelem, const std::vector<std::string> &in_labels,
                  const unsigned int &dtype,
                  const int &device,
                  const std::string &name)
                {
                  return UniTensor::ones(Nelem, in_labels,dtype,device,name);
                }, py::arg("Nelem"), py::arg("labels") = std::vector<std::string>(), py::arg("dtype") = (unsigned int)Type.Double,
                   py::arg("device") = int(Device.cpu),
                   py::arg("name") = std::string(""))
    .def_static("ones", [](const std::vector<cytnx_uint64> &shape, const std::vector<std::string> &in_labels,
                  const unsigned int &dtype,
                  const int &device,
                  const std::string &name)
                {
                  return UniTensor::ones(shape, in_labels,dtype,device,name);
                }, py::arg("shape"), py::arg("labels") = std::vector<std::string>(), py::arg("dtype") = (unsigned int)Type.Double,
                   py::arg("device") = int(Device.cpu),
                   py::arg("name") = std::string(""))
     .def_static("zeros", [](const cytnx_uint64 &Nelem, const std::vector<std::string> &in_labels,
                  const unsigned int &dtype,
                  const int &device,
                  const std::string &name)
                {
                  return UniTensor::zeros(Nelem, in_labels,dtype,device,name);
                }, py::arg("Nelem"), py::arg("labels") = std::vector<std::string>(), py::arg("dtype") = (unsigned int)Type.Double,
                   py::arg("device") = int(Device.cpu),
                   py::arg("name") = std::string(""))
     .def_static("zeros", [](const std::vector<cytnx_uint64> &shape, const std::vector<std::string> &in_labels,
                  const unsigned int &dtype,
                  const int &device,
                  const std::string &name)
                {
                  return UniTensor::zeros(shape, in_labels,dtype,device,name);
                }, py::arg("shape"), py::arg("labels") = std::vector<std::string>(), py::arg("dtype") = (unsigned int)Type.Double,
                   py::arg("device") = int(Device.cpu),
                   py::arg("name") = std::string(""))
     .def_static("arange", [](const cytnx_uint64 &Nelem, const std::vector<std::string> &in_labels,
                  const std::string &name)
                {
                  return UniTensor::arange(Nelem, in_labels,name);
                }, py::arg("Nelem"), py::arg("labels") = std::vector<std::string>(),
                   py::arg("name") = std::string(""))
     .def_static("arange", [](const cytnx_double &start,const cytnx_double &end
     ,const cytnx_double &step, const std::vector<std::string> &in_labels,const unsigned int &dtype, const int &device,
                  const std::string &name)
                {
                  return UniTensor::arange(start,end,step, in_labels,dtype,device,name);
                }, py::arg("start"),py::arg("end"),py::arg("step")=cytnx_double(1), py::arg("labels") = std::vector<std::string>(), py::arg("dtype") = (unsigned int)Type.Double,
                   py::arg("device") = int(Device.cpu),
                   py::arg("name") = std::string(""))
     .def_static("linspace", [](const cytnx_double &start,const cytnx_double &end
     ,const cytnx_uint64 &Nelem,const bool &endpoint,const std::vector<std::string> &in_labels,const unsigned int &dtype, const int &device,
                  const std::string &name)
                {
                  return UniTensor::linspace(start,end,Nelem, endpoint, in_labels,dtype,device,name);
                }, py::arg("start"),py::arg("end"),py::arg("Nelem"),py::arg("endpoint")=true,  py::arg("labels") = std::vector<std::string>(), py::arg("dtype") = (unsigned int)Type.Double,
                   py::arg("device") = int(Device.cpu),
                   py::arg("name") = std::string(""))
     .def_static("normal", [](const cytnx_uint64 &Nelem, const double &mean, const double &std,
                              const std::vector<std::string> &in_labels,
                              int64_t &seed, const unsigned int &dtype,
							  const int &device, const std::string &name)
                {
                    if(seed==-1){
                         // If user doesn't specify seed argument
                         seed = cytnx::random::__static_random_device();
                    }
                  return UniTensor::normal(Nelem, mean, std, in_labels, seed, dtype, device, name);
                },
				py::arg("Nelem"), py::arg("mean"), py::arg("std"),
				py::arg("in_labels")=std::vector<std::string>(), py::arg("seed")= -1,
				py::arg("dtype") = (unsigned int)Type.Double, py::arg("device") = int(Device.cpu),
                py::arg("name") = std::string(""))
     .def_static("normal", [](const std::vector<cytnx_uint64> &shape, const double &mean, const double &std,
                              const std::vector<std::string> &in_labels,
                              int64_t &seed, const unsigned int &dtype,
							  const int &device, const std::string &name)
                {
                    if(seed==-1){
                         // If user doesn't specify seed argument
                         seed = cytnx::random::__static_random_device();
                    }
                  return UniTensor::normal(shape, mean, std, in_labels, seed, dtype, device, name);
                },
				py::arg("shape"), py::arg("mean"), py::arg("std"),
				py::arg("in_labels")=std::vector<std::string>(), py::arg("seed")= -1,
				py::arg("dtype") = (unsigned int)Type.Double, py::arg("device") = int(Device.cpu),
                py::arg("name") = std::string(""))
     .def_static("uniform", [](const cytnx_uint64 &Nelem, const double &low, const double &high,
                              const std::vector<std::string> &in_labels,
                              int64_t &seed, const unsigned int &dtype,
							  const int &device, const std::string &name)
                {
                    if(seed==-1){
                         // If user doesn't specify seed argument
                         seed = cytnx::random::__static_random_device();
                    }
                  return UniTensor::uniform(Nelem, low, high, in_labels, seed, dtype, device, name);
                },
				py::arg("Nelem"), py::arg("low"), py::arg("high"),
				py::arg("in_labels")=std::vector<std::string>(), py::arg("seed")= -1,
				py::arg("dtype") = (unsigned int)Type.Double, py::arg("device") = int(Device.cpu),
                py::arg("name") = std::string(""))
     .def_static("uniform", [](const std::vector<cytnx_uint64> &shape, const double &low, const double &high,
                              const std::vector<std::string> &in_labels,
                              int64_t &seed, const unsigned int &dtype,
							  const int &device, const std::string &name)
                {
                    if(seed==-1){
                         // If user doesn't specify seed argument
                         seed = cytnx::random::__static_random_device();
                    }
                  return UniTensor::uniform(shape, low, high, in_labels, seed, dtype, device, name);
                },
				py::arg("shape"), py::arg("low"), py::arg("high"),
				py::arg("in_labels")=std::vector<std::string>(), py::arg("seed")= -1,
				py::arg("dtype") = (unsigned int)Type.Double, py::arg("device") = int(Device.cpu),
                py::arg("name") = std::string(""))
     .def("normal_", [](UniTensor &self, const double &mean, const double &std,
					    int64_t &seed)
                {
                    if(seed==-1){
                         // If user doesn't specify seed argument
                         seed = cytnx::random::__static_random_device();
                    }
                  self.normal_(mean, std, seed);
                },
				py::arg("mean"), py::arg("std"), py::arg("seed")= -1)
     .def("uniform_", [](UniTensor &self, const double &low, const double &high,
					     int64_t &seed)
                {
                    if(seed==-1){
                         // If user doesn't specify seed argument
                         seed = cytnx::random::__static_random_device();
                    }
                  self.uniform_(low, high, seed);
                },
				py::arg("low"), py::arg("high"), py::arg("seed")= -1)

     .def("cfrom", [](UniTensor &self, const UniTensor &in, const bool &force){
                         self.convert_from(in,force);
                         },
                    py::arg("Tin"), py::arg("force") = false)
     .def("get_qindices",  [](UniTensor &self, const cytnx_uint64 &bidx){return self.get_qindices(bidx);});
  ;  // end of object line

  //   m.def("Contract", Contract, py::arg("Tl"), py::arg("Tr"), py::arg("cacheL") = false,
  //         py::arg("cacheR") = false);
  m.def(
    "Contract",
    [](const UniTensor &inL, const UniTensor &inR, const bool &cacheL,
       const bool &cacheR) -> UniTensor { return Contract(inL, inR, cacheL, cacheR); },
    py::arg("Tl"), py::arg("Tr"), py::arg("cacheL") = false, py::arg("cacheR") = false);
  m.def(
    "Contract",
    [](const std::vector<UniTensor> &TNs, const std::string &order,
       const bool &optimal) -> UniTensor { return Contract(TNs, order, optimal); },
    py::arg("TNs"), py::arg("order") = "", py::arg("optimal") = true);
  m.def(
    "Contracts",
    [](const std::vector<UniTensor> &TNs, const std::string &order,
       const bool &optimal) -> UniTensor { return Contracts(TNs, order, optimal); },
    py::arg("TNs"), py::arg("order") = "", py::arg("optimal") = true);
}
#endif
