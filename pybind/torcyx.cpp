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
#include "torcyx.hpp"
#include "complex.h"
//#include <torch/python.h>
#include <torch/torch.h>

using namespace torcyx;
namespace py = pybind11;
using namespace pybind11::literals;

//ref: https://developer.lsst.io/v/DM-9089/coding/python_wrappers_for_cpp_with_pybind11.html
//ref: https://pybind11.readthedocs.io/en/stable/advanced/classes.html
//ref: https://block.arch.ethz.ch/blog/2016/07/adding-methods-to-python-classes/
//ref: https://medium.com/@mgarod/dynamically-add-a-method-to-a-class-in-python-c49204b85bd6



//=============================================
/*
void f_UniTensor_setelem_scal_d(UniTensor &self, const std::vector<cytnx_uint64> &locator, const cytnx::cytnx_double&rc){
    if(self.dtype() == cytnx::Type.Double){ 
        self.set_elem<cytnx::cytnx_double>(locator,rc);
    }else if(self.dtype() == cytnx::Type.Float){ 
        self.set_elem<cytnx::cytnx_float>(locator,rc);
    }else if(self.dtype() == cytnx::Type.ComplexDouble) {
        self.set_elem<cytnx::cytnx_complex128>(locator,cytnx_complex128(rc,0));
    }else if(self.dtype() == cytnx::Type.ComplexFloat){
        self.set_elem<cytnx::cytnx_complex64>(locator,cytnx_complex64(rc,0));
    }else if(self.dtype() == cytnx::Type.Uint64){
        self.set_elem<cytnx::cytnx_uint64>(locator,rc);
    }else if(self.dtype() == cytnx::Type.Int64){
        self.set_elem<cytnx::cytnx_int64>(locator,rc);
    }else if(self.dtype() == cytnx::Type.Uint32){
        self.set_elem<cytnx::cytnx_uint32>(locator,rc);
    }else if(self.dtype() == cytnx::Type.Int32) {
        self.set_elem<cytnx::cytnx_int32>(locator,rc);
    }else if(self.dtype() == cytnx::Type.Uint16) {
        self.set_elem<cytnx::cytnx_uint16>(locator,rc);
    }else if(self.dtype() == cytnx::Type.Int16){
        self.set_elem<cytnx::cytnx_int16>(locator,rc);
    }else if(self.dtype() == cytnx::Type.Bool) {
        self.set_elem<cytnx::cytnx_bool>(locator,rc);
    }else cytnx_error_msg(true, "%s","[ERROR] try to get element from a void Storage.");
}

void f_UniTensor_setelem_scal_f(UniTensor &self, const std::vector<cytnx_uint64> &locator, const cytnx::cytnx_float&rc){
    if(self.dtype() == cytnx::Type.Double) 
        self.set_elem<cytnx::cytnx_double>(locator,rc);
    else if(self.dtype() == cytnx::Type.Float) 
        self.set_elem<cytnx::cytnx_float>(locator,rc);
    else if(self.dtype() == cytnx::Type.ComplexDouble) 
        self.set_elem<cytnx::cytnx_complex128>(locator,cytnx_complex128(rc,0));
    else if(self.dtype() == cytnx::Type.ComplexFloat) 
        self.set_elem<cytnx::cytnx_complex64>(locator,cytnx_complex64(rc,0));
    else if(self.dtype() == cytnx::Type.Uint64) 
        self.set_elem<cytnx::cytnx_uint64>(locator,rc);
    else if(self.dtype() == cytnx::Type.Int64) 
        self.set_elem<cytnx::cytnx_int64>(locator,rc);
    else if(self.dtype() == cytnx::Type.Uint32) 
        self.set_elem<cytnx::cytnx_uint32>(locator,rc);
    else if(self.dtype() == cytnx::Type.Int32) 
        self.set_elem<cytnx::cytnx_int32>(locator,rc);
    else if(self.dtype() == cytnx::Type.Uint16) 
        self.set_elem<cytnx::cytnx_uint16>(locator,rc);
    else if(self.dtype() == cytnx::Type.Int16) 
        self.set_elem<cytnx::cytnx_int16>(locator,rc);
    else if(self.dtype() == cytnx::Type.Bool) 
        self.set_elem<cytnx::cytnx_bool>(locator,rc);
    else cytnx_error_msg(true, "%s","[ERROR] try to get element from a void Storage.");
}
void f_UniTensor_setelem_scal_u64(UniTensor &self, const std::vector<cytnx_uint64> &locator, const cytnx::cytnx_uint64&rc){
    if(self.dtype() == cytnx::Type.Double) 
        self.set_elem<cytnx::cytnx_double>(locator,rc);
    else if(self.dtype() == cytnx::Type.Float) 
        self.set_elem<cytnx::cytnx_float>(locator,rc);
    else if(self.dtype() == cytnx::Type.ComplexDouble) 
        self.set_elem<cytnx::cytnx_complex128>(locator,cytnx_complex128(rc,0));
    else if(self.dtype() == cytnx::Type.ComplexFloat) 
        self.set_elem<cytnx::cytnx_complex64>(locator,cytnx_complex64(rc,0));
    else if(self.dtype() == cytnx::Type.Uint64) 
        self.set_elem<cytnx::cytnx_uint64>(locator,rc);
    else if(self.dtype() == cytnx::Type.Int64) 
        self.set_elem<cytnx::cytnx_int64>(locator,rc);
    else if(self.dtype() == cytnx::Type.Uint32) 
        self.set_elem<cytnx::cytnx_uint32>(locator,rc);
    else if(self.dtype() == cytnx::Type.Int32) 
        self.set_elem<cytnx::cytnx_int32>(locator,rc);
    else if(self.dtype() == cytnx::Type.Uint16) 
        self.set_elem<cytnx::cytnx_uint16>(locator,rc);
    else if(self.dtype() == cytnx::Type.Int16) 
        self.set_elem<cytnx::cytnx_int16>(locator,rc);
    else if(self.dtype() == cytnx::Type.Bool) 
        self.set_elem<cytnx::cytnx_bool>(locator,rc);
    else cytnx_error_msg(true, "%s","[ERROR] try to get element from a void Storage.");
}
void f_UniTensor_setelem_scal_i64(UniTensor &self, const std::vector<cytnx_uint64> &locator, const cytnx::cytnx_int64&rc){
    if(self.dtype() == cytnx::Type.Double) 
        self.set_elem<cytnx::cytnx_double>(locator,rc);
    else if(self.dtype() == cytnx::Type.Float) 
        self.set_elem<cytnx::cytnx_float>(locator,rc);
    else if(self.dtype() == cytnx::Type.ComplexDouble) 
        self.set_elem<cytnx::cytnx_complex128>(locator,cytnx_complex128(rc,0));
    else if(self.dtype() == cytnx::Type.ComplexFloat) 
        self.set_elem<cytnx::cytnx_complex64>(locator,cytnx_complex64(rc,0));
    else if(self.dtype() == cytnx::Type.Uint64) 
        self.set_elem<cytnx::cytnx_uint64>(locator,rc);
    else if(self.dtype() == cytnx::Type.Int64) 
        self.set_elem<cytnx::cytnx_int64>(locator,rc);
    else if(self.dtype() == cytnx::Type.Uint32) 
        self.set_elem<cytnx::cytnx_uint32>(locator,rc);
    else if(self.dtype() == cytnx::Type.Int32) 
        self.set_elem<cytnx::cytnx_int32>(locator,rc);
    else if(self.dtype() == cytnx::Type.Uint16) 
        self.set_elem<cytnx::cytnx_uint16>(locator,rc);
    else if(self.dtype() == cytnx::Type.Int16) 
        self.set_elem<cytnx::cytnx_int16>(locator,rc);
    else if(self.dtype() == cytnx::Type.Bool) 
        self.set_elem<cytnx::cytnx_bool>(locator,rc);
    else cytnx_error_msg(true, "%s","[ERROR] try to get element from a void Storage.");
}
void f_UniTensor_setelem_scal_u32(UniTensor &self, const std::vector<cytnx_uint64> &locator, const cytnx::cytnx_uint32&rc){
    if(self.dtype() == cytnx::Type.Double) 
        self.set_elem<cytnx::cytnx_double>(locator,rc);
    else if(self.dtype() == cytnx::Type.Float) 
        self.set_elem<cytnx::cytnx_float>(locator,rc);
    else if(self.dtype() == cytnx::Type.ComplexDouble) 
        self.set_elem<cytnx::cytnx_complex128>(locator,cytnx_complex128(rc,0));
    else if(self.dtype() == cytnx::Type.ComplexFloat) 
        self.set_elem<cytnx::cytnx_complex64>(locator,cytnx_complex64(rc,0));
    else if(self.dtype() == cytnx::Type.Uint64) 
        self.set_elem<cytnx::cytnx_uint64>(locator,rc);
    else if(self.dtype() == cytnx::Type.Int64) 
        self.set_elem<cytnx::cytnx_int64>(locator,rc);
    else if(self.dtype() == cytnx::Type.Uint32) 
        self.set_elem<cytnx::cytnx_uint32>(locator,rc);
    else if(self.dtype() == cytnx::Type.Int32) 
        self.set_elem<cytnx::cytnx_int32>(locator,rc);
    else if(self.dtype() == cytnx::Type.Uint16) 
        self.set_elem<cytnx::cytnx_uint16>(locator,rc);
    else if(self.dtype() == cytnx::Type.Int16) 
        self.set_elem<cytnx::cytnx_int16>(locator,rc);
    else if(self.dtype() == cytnx::Type.Bool) 
        self.set_elem<cytnx::cytnx_bool>(locator,rc);
    else cytnx_error_msg(true, "%s","[ERROR] try to get element from a void Storage.");
}
void f_UniTensor_setelem_scal_i32(UniTensor &self, const std::vector<cytnx_uint64> &locator, const cytnx::cytnx_int32&rc){
    if(self.dtype() == cytnx::Type.Double) 
        self.set_elem<cytnx::cytnx_double>(locator,rc);
    else if(self.dtype() == cytnx::Type.Float) 
        self.set_elem<cytnx::cytnx_float>(locator,rc);
    else if(self.dtype() == cytnx::Type.ComplexDouble) 
        self.set_elem<cytnx::cytnx_complex128>(locator,cytnx_complex128(rc,0));
    else if(self.dtype() == cytnx::Type.ComplexFloat) 
        self.set_elem<cytnx::cytnx_complex64>(locator,cytnx_complex64(rc,0));
    else if(self.dtype() == cytnx::Type.Uint64) 
        self.set_elem<cytnx::cytnx_uint64>(locator,rc);
    else if(self.dtype() == cytnx::Type.Int64) 
        self.set_elem<cytnx::cytnx_int64>(locator,rc);
    else if(self.dtype() == cytnx::Type.Uint32) 
        self.set_elem<cytnx::cytnx_uint32>(locator,rc);
    else if(self.dtype() == cytnx::Type.Int32) 
        self.set_elem<cytnx::cytnx_int32>(locator,rc);
    else if(self.dtype() == cytnx::Type.Uint16) 
        self.set_elem<cytnx::cytnx_uint16>(locator,rc);
    else if(self.dtype() == cytnx::Type.Int16) 
        self.set_elem<cytnx::cytnx_int16>(locator,rc);
    else if(self.dtype() == cytnx::Type.Bool) 
        self.set_elem<cytnx::cytnx_bool>(locator,rc);
    else cytnx_error_msg(true, "%s","[ERROR] try to get element from a void Storage.");
}
void f_UniTensor_setelem_scal_u16(UniTensor &self, const std::vector<cytnx_uint64> &locator, const cytnx::cytnx_uint16&rc){
    if(self.dtype() == cytnx::Type.Double) 
        self.set_elem<cytnx::cytnx_double>(locator,rc);
    else if(self.dtype() == cytnx::Type.Float) 
        self.set_elem<cytnx::cytnx_float>(locator,rc);
    else if(self.dtype() == cytnx::Type.ComplexDouble) 
        self.set_elem<cytnx::cytnx_complex128>(locator,cytnx_complex128(rc,0));
    else if(self.dtype() == cytnx::Type.ComplexFloat) 
        self.set_elem<cytnx::cytnx_complex64>(locator,cytnx_complex64(rc,0));
    else if(self.dtype() == cytnx::Type.Uint64) 
        self.set_elem<cytnx::cytnx_uint64>(locator,rc);
    else if(self.dtype() == cytnx::Type.Int64) 
        self.set_elem<cytnx::cytnx_int64>(locator,rc);
    else if(self.dtype() == cytnx::Type.Uint32) 
        self.set_elem<cytnx::cytnx_uint32>(locator,rc);
    else if(self.dtype() == cytnx::Type.Int32) 
        self.set_elem<cytnx::cytnx_int32>(locator,rc);
    else if(self.dtype() == cytnx::Type.Uint16) 
        self.set_elem<cytnx::cytnx_uint16>(locator,rc);
    else if(self.dtype() == cytnx::Type.Int16) 
        self.set_elem<cytnx::cytnx_int16>(locator,rc);
    else if(self.dtype() == cytnx::Type.Bool) 
        self.set_elem<cytnx::cytnx_bool>(locator,rc);
    else cytnx_error_msg(true, "%s","[ERROR] try to get element from a void Storage.");
}
void f_UniTensor_setelem_scal_i16(UniTensor &self, const std::vector<cytnx_uint64> &locator, const cytnx::cytnx_int16&rc){
    if(self.dtype() == cytnx::Type.Double) 
        self.set_elem<cytnx::cytnx_double>(locator,rc);
    else if(self.dtype() == cytnx::Type.Float) 
        self.set_elem<cytnx::cytnx_float>(locator,rc);
    else if(self.dtype() == cytnx::Type.ComplexDouble) 
        self.set_elem<cytnx::cytnx_complex128>(locator,cytnx_complex128(rc,0));
    else if(self.dtype() == cytnx::Type.ComplexFloat) 
        self.set_elem<cytnx::cytnx_complex64>(locator,cytnx_complex64(rc,0));
    else if(self.dtype() == cytnx::Type.Uint64) 
        self.set_elem<cytnx::cytnx_uint64>(locator,rc);
    else if(self.dtype() == cytnx::Type.Int64) 
        self.set_elem<cytnx::cytnx_int64>(locator,rc);
    else if(self.dtype() == cytnx::Type.Uint32) 
        self.set_elem<cytnx::cytnx_uint32>(locator,rc);
    else if(self.dtype() == cytnx::Type.Int32) 
        self.set_elem<cytnx::cytnx_int32>(locator,rc);
    else if(self.dtype() == cytnx::Type.Uint16) 
        self.set_elem<cytnx::cytnx_uint16>(locator,rc);
    else if(self.dtype() == cytnx::Type.Int16) 
        self.set_elem<cytnx::cytnx_int16>(locator,rc);
    else if(self.dtype() == cytnx::Type.Bool) 
        self.set_elem<cytnx::cytnx_bool>(locator,rc);
    else cytnx_error_msg(true, "%s","[ERROR] try to get element from a void Storage.");
}
void f_UniTensor_setelem_scal_b(UniTensor &self, const std::vector<cytnx_uint64> &locator, const cytnx::cytnx_bool&rc){
    if(self.dtype() == cytnx::Type.Double) 
        self.set_elem<cytnx::cytnx_double>(locator,rc);
    else if(self.dtype() == cytnx::Type.Float) 
        self.set_elem<cytnx::cytnx_float>(locator,rc);
    else if(self.dtype() == cytnx::Type.ComplexDouble) 
        self.set_elem<cytnx::cytnx_complex128>(locator,cytnx_complex128(rc,0));
    else if(self.dtype() == cytnx::Type.ComplexFloat) 
        self.set_elem<cytnx::cytnx_complex64>(locator,cytnx_complex64(rc,0));
    else if(self.dtype() == cytnx::Type.Uint64) 
        self.set_elem<cytnx::cytnx_uint64>(locator,rc);
    else if(self.dtype() == cytnx::Type.Int64) 
        self.set_elem<cytnx::cytnx_int64>(locator,rc);
    else if(self.dtype() == cytnx::Type.Uint32) 
        self.set_elem<cytnx::cytnx_uint32>(locator,rc);
    else if(self.dtype() == cytnx::Type.Int32) 
        self.set_elem<cytnx::cytnx_int32>(locator,rc);
    else if(self.dtype() == cytnx::Type.Uint16) 
        self.set_elem<cytnx::cytnx_uint16>(locator,rc);
    else if(self.dtype() == cytnx::Type.Int16) 
        self.set_elem<cytnx::cytnx_int16>(locator,rc);
    else if(self.dtype() == cytnx::Type.Bool) 
        self.set_elem<cytnx::cytnx_bool>(locator,rc);
    else cytnx_error_msg(true, "%s","[ERROR] try to get element from a void Storage.");
}
void f_UniTensor_setelem_scal_cd(UniTensor &self, const std::vector<cytnx_uint64> &locator, const cytnx::cytnx_complex128 &rc){
    if(self.dtype() == cytnx::Type.Double){
        cytnx_error_msg(true,"%s","[ERROR] cannot assign complex to real container.\n");
    }else if(self.dtype() == cytnx::Type.Float){
        cytnx_error_msg(true,"%s","[ERROR] cannot assign complex to real container.\n");
    }else if(self.dtype() == cytnx::Type.ComplexDouble){
        self.set_elem<cytnx::cytnx_complex128>(locator,rc);
    }else if(self.dtype() == cytnx::Type.ComplexFloat){
        self.set_elem<cytnx::cytnx_complex64>(locator,cytnx::cytnx_complex64(rc.real(),rc.imag()));
    }else if(self.dtype() == cytnx::Type.Uint64){
        cytnx_error_msg(true,"%s","[ERROR] cannot assign complex to real container.\n");
    }else if(self.dtype() == cytnx::Type.Int64){
        cytnx_error_msg(true,"%s","[ERROR] cannot assign complex to real container.\n");
    }else if(self.dtype() == cytnx::Type.Uint32){ 
        cytnx_error_msg(true,"%s","[ERROR] cannot assign complex to real container.\n");
    }else if(self.dtype() == cytnx::Type.Int32){ 
        cytnx_error_msg(true,"%s","[ERROR] cannot assign complex to real container.\n");
    }else if(self.dtype() == cytnx::Type.Uint16){ 
        cytnx_error_msg(true,"%s","[ERROR] cannot assign complex to real container.\n");
    }else if(self.dtype() == cytnx::Type.Int16){ 
        cytnx_error_msg(true,"%s","[ERROR] cannot assign complex to real container.\n");
    }else if(self.dtype() == cytnx::Type.Bool){ 
        cytnx_error_msg(true,"%s","[ERROR] cannot assign complex to real container.\n");
    }else cytnx_error_msg(true, "%s","[ERROR] try to get element from a void Storage.");
}
void f_UniTensor_setelem_scal_cf(UniTensor &self, const std::vector<cytnx_uint64> &locator, const cytnx::cytnx_complex64 &rc){
    if(self.dtype() == cytnx::Type.Double){
        cytnx_error_msg(true,"%s","[ERROR] cannot assign complex to real container.\n");
    }else if(self.dtype() == cytnx::Type.Float){ 
        cytnx_error_msg(true,"%s","[ERROR] cannot assign complex to real container.\n");
    }else if(self.dtype() == cytnx::Type.ComplexDouble){
        self.set_elem<cytnx::cytnx_complex128>(locator,cytnx::cytnx_complex128(rc.real(),rc.imag()));
    }else if(self.dtype() == cytnx::Type.ComplexFloat){ 
        self.set_elem<cytnx::cytnx_complex64>(locator,rc);
    }else if(self.dtype() == cytnx::Type.Uint64) {
        cytnx_error_msg(true,"%s","[ERROR] cannot assign complex to real container.\n");
    }else if(self.dtype() == cytnx::Type.Int64) {
        cytnx_error_msg(true,"%s","[ERROR] cannot assign complex to real container.\n");
    }else if(self.dtype() == cytnx::Type.Uint32) {
        cytnx_error_msg(true,"%s","[ERROR] cannot assign complex to real container.\n");
    }else if(self.dtype() == cytnx::Type.Int32) {
        cytnx_error_msg(true,"%s","[ERROR] cannot assign complex to real container.\n");
    }else if(self.dtype() == cytnx::Type.Uint16) {
        cytnx_error_msg(true,"%s","[ERROR] cannot assign complex to real container.\n");
    }else if(self.dtype() == cytnx::Type.Int16) {
        cytnx_error_msg(true,"%s","[ERROR] cannot assign complex to real container.\n");
    }else if(self.dtype() == cytnx::Type.Bool) {
        cytnx_error_msg(true,"%s","[ERROR] cannot assign complex to real container.\n");
    }else cytnx_error_msg(true, "%s","[ERROR] try to get element from a void Storage.");
}
*/



PYBIND11_MODULE(torcyx,m){

    //m.attr("__version__") = "0.5.5a";
    //m.attr("__blasINTsize__") = cytnx::__blasINTsize__;

    py::add_ostream_redirect(m, "ostream_redirect");    


    py::class_<CyTensor>(m,"CyTensor")
                .def(py::init<>())
                .def(py::init<const torch::Tensor&, const cytnx_uint64&, const bool &>(),py::arg("Tin"),py::arg("rowrank"),py::arg("is_diag")=false)
                //.def(py::init<const std::vector<Bond> &, const std::vector<cytnx_int64> &, const cytnx_int64 &, const bool &, const torch::TensorOptions &>(),py::arg("bonds"),py::arg("labels")=std::vector<cytnx_int64>(),py::arg("rowrank")=(cytnx_int64)(-1),py::arg("is_diag")=false,py::arg("options")=torch::TensorOptions())
                .def("c_init",[](CyTensor &self,const std::vector<Bond> &bonds, const std::vector<cytnx_int64> & labels, const cytnx_int64 &rowrank, const bool &is_diag, const std::string &dtype, const std::string &device, const bool &requres_grad){
                                    //self.
                                    std::cout << dtype << std::endl;
                                    std::cout << device << std::endl;
                                 },py::arg("bonds"),py::arg("labels")=std::vector<cytnx_int64>(),py::arg("rowrank")=(cytnx_int64)(-1),py::arg("is_diag")=false, py::arg("dtype"),py::arg("device"),py::arg("requres_grad"));

                //.def("set_name",&UniTensor::set_name)
                //.def("set_label",&UniTensor::set_label,py::arg("idx"),py::arg("new_label"))
                //.def("set_labels",&UniTensor::set_labels,py::arg("new_labels"))
                //.def("set_rowrank",&UniTensor::set_rowrank, py::arg("new_rowrank"))

                //.def("rowrank",&UniTensor::rowrank)
                //.def("dtype",&UniTensor::dtype)
                //.def("dtype_str",&UniTensor::dtype_str)
                //.def("device",&UniTensor::device)
                //.def("device_str",&UniTensor::device_str)
                //.def("name",&UniTensor::name)
                /*
                .def("reshape",[](UniTensor &self, py::args args, py::kwargs kwargs)->UniTensor{
                    std::vector<cytnx::cytnx_int64> c_args = args.cast< std::vector<cytnx::cytnx_int64> >();
                    cytnx_uint64 rowrank = 0;
                   
                    if(kwargs){
                        if(kwargs.contains("rowrank")) rowrank = kwargs["rowrank"].cast<cytnx::cytnx_int64>();
                    }
 
                    return self.reshape(c_args,rowrank);
                })
                .def("reshape_",[](UniTensor &self, py::args args, py::kwargs kwargs){
                    std::vector<cytnx::cytnx_int64> c_args = args.cast< std::vector<cytnx::cytnx_int64> >();
                    cytnx_uint64 rowrank = 0;
                   
                    if(kwargs){
                        if(kwargs.contains("rowrank")) rowrank = kwargs["rowrank"].cast<cytnx::cytnx_int64>();
                    }
 
                    self.reshape_(c_args,rowrank);
                })
                .def("elem_exists",&UniTensor::elem_exists)
                .def("item",[](UniTensor &self){
                    py::object out;
                    if(self.dtype() == cytnx::Type.Double) 
                        out =  py::cast(self.item<cytnx::cytnx_double>());
                    else if(self.dtype() == cytnx::Type.Float) 
                        out = py::cast(self.item<cytnx::cytnx_float>());
                    else if(self.dtype() == cytnx::Type.ComplexDouble) 
                        out = py::cast(self.item<cytnx::cytnx_complex128>());
                    else if(self.dtype() == cytnx::Type.ComplexFloat) 
                        out = py::cast(self.item<cytnx::cytnx_complex64>());
                    else if(self.dtype() == cytnx::Type.Uint64) 
                        out = py::cast(self.item<cytnx::cytnx_uint64>());
                    else if(self.dtype() == cytnx::Type.Int64) 
                        out = py::cast(self.item<cytnx::cytnx_int64>());
                    else if(self.dtype() == cytnx::Type.Uint32) 
                        out = py::cast(self.item<cytnx::cytnx_uint32>());
                    else if(self.dtype() == cytnx::Type.Int32) 
                        out = py::cast(self.item<cytnx::cytnx_int32>());
                    else if(self.dtype() == cytnx::Type.Uint16) 
                        out = py::cast(self.item<cytnx::cytnx_uint16>());
                    else if(self.dtype() == cytnx::Type.Int16) 
                        out = py::cast(self.item<cytnx::cytnx_int16>());
                    else if(self.dtype() == cytnx::Type.Bool) 
                        out = py::cast(self.item<cytnx::cytnx_bool>());
                    else cytnx_error_msg(true, "%s","[ERROR] try to get element from a empty UniTensor.");
                    return out;
                 })

                .def("__getitem__",[](const UniTensor &self, py::object locators){
                    cytnx_error_msg(self.shape().size() == 0, "[ERROR] try to getitem from a empty UniTensor%s","\n");
                    cytnx_error_msg(self.uten_type() == UTenType.Sparse,"[ERROR] cannot get element using [] from SparseUniTensor. Use at() instead.%s","\n");
 
                    ssize_t start, stop, step, slicelength; 
                    std::vector<cytnx::Accessor> accessors;
                    if(py::isinstance<py::tuple>(locators)){
                        py::tuple Args = locators.cast<py::tuple>();
                        cytnx_uint64 cnt = 0;
                        // mixing of slice and ints
                        for(cytnx_uint32 axis=0;axis<Args.size();axis++){
                                cnt ++;
                                // check type:
                                if(py::isinstance<py::slice>(Args[axis])){
                                    py::slice sls = Args[axis].cast<py::slice>();
                                    if(!sls.compute((ssize_t)self.shape()[axis],&start,&stop,&step, &slicelength))
                                        throw py::error_already_set();
                                    //std::cout << start << " " << stop << " " << step << slicelength << std::endl;
                                    //if(slicelength == self.shape()[axis]) accessors.push_back(cytnx::Accessor::all());
                                    accessors.push_back(cytnx::Accessor::range(cytnx_int64(start),cytnx_int64(stop),cytnx_int64(step)));
                                }else{
                                    accessors.push_back(cytnx::Accessor(Args[axis].cast<cytnx_int64>()));
                                }
                            
                        }
                        while(cnt<self.shape().size()){
                            cnt++;
                            accessors.push_back(Accessor::all());
                        }
                    }else if(py::isinstance<py::slice>(locators)){
                        py::slice sls = locators.cast<py::slice>();
                        if(!sls.compute((ssize_t)self.shape()[0],&start,&stop,&step, &slicelength))
                            throw py::error_already_set();
                        //if(slicelength == self.shape()[0]) accessors.push_back(cytnx::Accessor::all());
                        accessors.push_back(cytnx::Accessor::range(start,stop,step));
                        for(cytnx_uint32 axis=1;axis<self.shape().size();axis++){
                            accessors.push_back(Accessor::all());
                        }


                    }else{
                        // only int
                        for(cytnx_uint32 i=0;i<self.shape().size();i++){
                            if(i==0) accessors.push_back(cytnx::Accessor(locators.cast<cytnx_int64>()));
                            else accessors.push_back(cytnx::Accessor::all());
                        }
                    }
                    

                    return self.get(accessors);
                    
                })
                .def("__setitem__",[](UniTensor &self, py::object locators, const cytnx::Tensor &rhs){
                    cytnx_error_msg(self.shape().size() == 0, "[ERROR] try to setelem to a empty UniTensor%s","\n");
                    cytnx_error_msg(self.uten_type() == UTenType.Sparse,"[ERROR] cannot set element using [] from SparseUniTensor. Use at() instead.%s","\n");
                    
                    ssize_t start, stop, step, slicelength; 
                    std::vector<cytnx::Accessor> accessors;
                    if(py::isinstance<py::tuple>(locators)){
                        py::tuple Args = locators.cast<py::tuple>();
                        cytnx_uint64 cnt = 0;
                        // mixing of slice and ints
                        for(cytnx_uint32 axis=0;axis<Args.size();axis++){
                                cnt ++;
                                // check type:
                                if(py::isinstance<py::slice>(Args[axis])){
                                    py::slice sls = Args[axis].cast<py::slice>();
                                    if(!sls.compute((ssize_t)self.shape()[axis],&start,&stop,&step, &slicelength))
                                        throw py::error_already_set();
                                    //std::cout << start << " " << stop << " " << step << slicelength << std::endl;
                                    //if(slicelength == self.shape()[axis]) accessors.push_back(cytnx::Accessor::all());
                                    accessors.push_back(cytnx::Accessor::range(cytnx_int64(start),cytnx_int64(stop),cytnx_int64(step)));
                                }else{
                                    accessors.push_back(cytnx::Accessor(Args[axis].cast<cytnx_int64>()));
                                }
                            
                        }
                        while(cnt<self.shape().size()){
                            cnt++;
                            accessors.push_back(Accessor::all());
                        }
                    }else if(py::isinstance<py::slice>(locators)){
                        py::slice sls = locators.cast<py::slice>();
                        if(!sls.compute((ssize_t)self.shape()[0],&start,&stop,&step, &slicelength))
                            throw py::error_already_set();
                        //if(slicelength == self.shape()[0]) accessors.push_back(cytnx::Accessor::all());
                        accessors.push_back(cytnx::Accessor::range(start,stop,step));
                        for(cytnx_uint32 axis=1;axis<self.shape().size();axis++){
                            accessors.push_back(Accessor::all());
                        }


                    }else{
                        // only int
                        for(cytnx_uint32 i=0;i<self.shape().size();i++){
                            if(i==0) accessors.push_back(cytnx::Accessor(locators.cast<cytnx_int64>()));
                            else accessors.push_back(cytnx::Accessor::all());
                        }
                    }
                    

                    self.set(accessors,rhs);
                    
                })
                .def("get_elem",[](UniTensor &self, const std::vector<cytnx_uint64> &locator){
                    py::object out;
                    if(self.dtype() == cytnx::Type.Double) 
                        out = py::cast(self.get_elem<cytnx::cytnx_double>(locator));
                    else if(self.dtype() == cytnx::Type.Float) 
                        out = py::cast(self.get_elem<cytnx::cytnx_float>(locator));
                    else if(self.dtype() == cytnx::Type.ComplexDouble) 
                        out = py::cast(self.get_elem<cytnx::cytnx_complex128>(locator));
                    else if(self.dtype() == cytnx::Type.ComplexFloat) 
                        out = py::cast(self.get_elem<cytnx::cytnx_complex64>(locator));
                    else cytnx_error_msg(true, "%s","[ERROR] try to get element from a void Storage.");
                    return out;
                })
                .def("set_elem",&f_UniTensor_setelem_scal_cd)
                .def("set_elem",&f_UniTensor_setelem_scal_cf)
                .def("set_elem",&f_UniTensor_setelem_scal_d)
                .def("set_elem",&f_UniTensor_setelem_scal_f)
                .def("set_elem",&f_UniTensor_setelem_scal_u64)
                .def("set_elem",&f_UniTensor_setelem_scal_i64)
                .def("set_elem",&f_UniTensor_setelem_scal_u32)
                .def("set_elem",&f_UniTensor_setelem_scal_i32)
                .def("set_elem",&f_UniTensor_setelem_scal_u16)
                .def("set_elem",&f_UniTensor_setelem_scal_i16)
                .def("set_elem",&f_UniTensor_setelem_scal_b)


                .def("is_contiguous", &UniTensor::is_contiguous)
                .def("is_diag",&UniTensor::is_diag)
                .def("is_tag" ,&UniTensor::is_tag)
                .def("is_braket_form",&UniTensor::is_braket_form)
                .def("labels",&UniTensor::labels)
                .def("bonds",[](UniTensor &self){
                    return self.bonds();
                    })
                .def("shape",&UniTensor::shape)
                .def("to_",&UniTensor::to_)
                .def("to_different_device" ,[](UniTensor &self,const cytnx_int64 &device){
                                                    cytnx_error_msg(self.device() == device, "[ERROR][pybind][to_diffferent_device] same device for to() should be handle in python side.%s","\n");
                                                    return self.to(device);
                                                } , py::arg("device"))
                .def("clone",&UniTensor::clone)
                .def("__copy__",&UniTensor::clone)
                .def("__deepcopy__",&UniTensor::clone)
                .def("Save",[](UniTensor &self, const std::string &fname){self.Save(fname);},py::arg("fname"))
                .def_static("Load",[](const std::string &fname){return UniTensor::Load(fname);},py::arg("fname"))
                //.def("permute",&UniTensor::permute,py::arg("mapper"),py::arg("rowrank")=(cytnx_int64)-1,py::arg("by_label")=false)
                //.def("permute_",&UniTensor::permute_,py::arg("mapper"),py::arg("rowrank")=(cytnx_int64)-1,py::arg("by_label")=false)

                .def("permute_",[](UniTensor &self, const std::vector<cytnx::cytnx_int64> &c_args, py::kwargs kwargs){
                    cytnx_int64 rowrank = -1;
                    bool by_label = false;
                    if(kwargs){
                        if(kwargs.contains("rowrank")){
                            rowrank = kwargs["rowrank"].cast<cytnx_int64>();
                        }
                        if(kwargs.contains("by_label")){ 
                            by_label = kwargs["by_label"].cast<bool>();
                        }
                    }
                    self.permute_(c_args,rowrank,by_label);
                })
                .def("permute",[](UniTensor &self,const std::vector<cytnx::cytnx_int64> &c_args, py::kwargs kwargs)->UniTensor{
                    cytnx_int64 rowrank = -1;
                    bool by_label = false;
                    if(kwargs){
                        if(kwargs.contains("rowrank")){
                            rowrank = kwargs["rowrank"].cast<cytnx_int64>();
                        }
                        if(kwargs.contains("by_label")){ 
                            by_label = kwargs["by_label"].cast<bool>();
                        }
                    }
                    return self.permute(c_args,rowrank,by_label);
                })

                .def("make_contiguous",&UniTensor::contiguous)
                .def("contiguous_",&UniTensor::contiguous_)
                .def("print_diagram",&UniTensor::print_diagram,py::arg("bond_info")=false,py::call_guard<py::scoped_ostream_redirect,
                     py::scoped_estream_redirect>())
                        
                .def("get_block", [](const UniTensor &self, const cytnx_uint64&idx){
                                        return self.get_block(idx);
                                  },py::arg("idx")=(cytnx_uint64)(0))

                .def("get_block", [](const UniTensor &self, const std::vector<cytnx_int64>&qnum){
                                        return self.get_block(qnum);
                                  },py::arg("qnum"))

                .def("get_block_",[](const UniTensor &self, const std::vector<cytnx_int64>&qnum){
                                        return self.get_block_(qnum);
                                  },py::arg("qnum"))
                .def("get_block_",[](UniTensor &self, const std::vector<cytnx_int64>&qnum){
                                        return self.get_block_(qnum);
                                  },py::arg("qnum"))
                .def("get_block_", [](const UniTensor &self, const cytnx_uint64&idx){
                                        return self.get_block_(idx);
                                  },py::arg("idx")=(cytnx_uint64)(0))
                .def("get_block_", [](UniTensor &self, const cytnx_uint64&idx){
                                        return self.get_block_(idx);
                                  },py::arg("idx")=(cytnx_uint64)(0))
                .def("get_blocks", [](const UniTensor &self){
                                        return self.get_blocks();
                                  })
                .def("get_blocks_", [](const UniTensor &self){
                                        return self.get_blocks_();
                                  })
                .def("get_blocks_", [](UniTensor &self){
                                        return self.get_blocks_();
                                  })
                .def("put_block", [](UniTensor &self, const cytnx::Tensor &in, const cytnx_uint64&idx){
                                        self.put_block(in,idx);
                                  },py::arg("in"),py::arg("idx")=(cytnx_uint64)(0))

                .def("put_block", [](UniTensor &self, const cytnx::Tensor &in, const std::vector<cytnx_int64>&qnum){
                                        self.put_block(in,qnum);
                                  },py::arg("in"),py::arg("qnum"))
                .def("put_block_", [](UniTensor &self, cytnx::Tensor &in, const cytnx_uint64&idx){
                                        self.put_block_(in,idx);
                                  },py::arg("in"),py::arg("idx")=(cytnx_uint64)(0))

                .def("put_block_", [](UniTensor &self, cytnx::Tensor &in, const std::vector<cytnx_int64>&qnum){
                                        self.put_block_(in,qnum);
                                  },py::arg("in"),py::arg("qnum"))
                .def("__repr__",[](UniTensor &self)->std::string{
                    std::cout << self << std::endl;
                    return std::string("");
                 },py::call_guard<py::scoped_ostream_redirect,
                     py::scoped_estream_redirect>()) 
                .def("to_dense",&UniTensor::to_dense)
                .def("to_dense_",&UniTensor::to_dense_)
                .def("combineBonds",&UniTensor::combineBonds,py::arg("indicators"),py::arg("permute_back")=true,py::arg("by_label")=true)
                .def("contract", &UniTensor::contract)
		
        		//arithmetic >>
                .def("__neg__",[](UniTensor &self){
                                    if(self.dtype() == Type.Double){
                                        return linalg::Mul(cytnx_double(-1),self);
                                    }else if(self.dtype()==Type.ComplexDouble){
                                        return linalg::Mul(cytnx_complex128(-1,0),self);
                                    }else if(self.dtype()==Type.Float){
                                        return linalg::Mul(cytnx_float(-1),self);
                                    }else if(self.dtype()==Type.ComplexFloat){
                                        return linalg::Mul(cytnx_complex64(-1,0),self);
                                    }else{
                                        return linalg::Mul(-1,self);
                                    }
                                  })
                .def("__pos__",[](UniTensor &self){return self;})
                .def("__add__",[](UniTensor &self, const UniTensor &rhs){return self.Add(rhs);})
                .def("__add__",[](UniTensor &self, const cytnx::cytnx_complex128&rhs){return self.Add(rhs);})
                .def("__add__",[](UniTensor &self, const cytnx::cytnx_complex64 &rhs){return self.Add(rhs);})
                .def("__add__",[](UniTensor &self, const cytnx::cytnx_double    &rhs){return self.Add(rhs);})
                .def("__add__",[](UniTensor &self, const cytnx::cytnx_float     &rhs){return self.Add(rhs);})
                .def("__add__",[](UniTensor &self, const cytnx::cytnx_int64     &rhs){return self.Add(rhs);})
                .def("__add__",[](UniTensor &self, const cytnx::cytnx_uint64    &rhs){return self.Add(rhs);})
                .def("__add__",[](UniTensor &self, const cytnx::cytnx_int32     &rhs){return self.Add(rhs);})
                .def("__add__",[](UniTensor &self, const cytnx::cytnx_uint32    &rhs){return self.Add(rhs);})
                .def("__add__",[](UniTensor &self, const cytnx::cytnx_int16     &rhs){return self.Add(rhs);})
                .def("__add__",[](UniTensor &self, const cytnx::cytnx_uint16    &rhs){return self.Add(rhs);})
                .def("__add__",[](UniTensor &self, const cytnx::cytnx_bool      &rhs){return self.Add(rhs);})
                
                .def("__radd__",[](UniTensor &self, const cytnx::cytnx_complex128&lhs){return linalg::Add(lhs,self);})
                .def("__radd__",[](UniTensor &self, const cytnx::cytnx_complex64 &lhs){return linalg::Add(lhs,self);})
                .def("__radd__",[](UniTensor &self, const cytnx::cytnx_double    &lhs){return linalg::Add(lhs,self);})
                .def("__radd__",[](UniTensor &self, const cytnx::cytnx_float     &lhs){return linalg::Add(lhs,self);})
                .def("__radd__",[](UniTensor &self, const cytnx::cytnx_int64     &lhs){return linalg::Add(lhs,self);})
                .def("__radd__",[](UniTensor &self, const cytnx::cytnx_uint64    &lhs){return linalg::Add(lhs,self);})
                .def("__radd__",[](UniTensor &self, const cytnx::cytnx_int32     &lhs){return linalg::Add(lhs,self);})
                .def("__radd__",[](UniTensor &self, const cytnx::cytnx_uint32    &lhs){return linalg::Add(lhs,self);})
                .def("__radd__",[](UniTensor &self, const cytnx::cytnx_int16     &lhs){return linalg::Add(lhs,self);})
                .def("__radd__",[](UniTensor &self, const cytnx::cytnx_uint16    &lhs){return linalg::Add(lhs,self);})
                .def("__radd__",[](UniTensor &self, const cytnx::cytnx_bool      &lhs){return linalg::Add(lhs,self);})
                
                .def("__iadd__",[](UniTensor &self, const UniTensor &rhs){return self.Add_(rhs);}) // these will return self!
                .def("__iadd__",[](UniTensor &self, const cytnx::cytnx_complex128&rhs){return self.Add_(rhs);})
                .def("__iadd__",[](UniTensor &self, const cytnx::cytnx_complex64 &rhs){return self.Add_(rhs);})
                .def("__iadd__",[](UniTensor &self, const cytnx::cytnx_double    &rhs){return self.Add_(rhs);})
                .def("__iadd__",[](UniTensor &self, const cytnx::cytnx_float     &rhs){return self.Add_(rhs);})
                .def("__iadd__",[](UniTensor &self, const cytnx::cytnx_int64     &rhs){return self.Add_(rhs);})
                .def("__iadd__",[](UniTensor &self, const cytnx::cytnx_uint64    &rhs){return self.Add_(rhs);})
                .def("__iadd__",[](UniTensor &self, const cytnx::cytnx_int32     &rhs){return self.Add_(rhs);})
                .def("__iadd__",[](UniTensor &self, const cytnx::cytnx_uint32    &rhs){return self.Add_(rhs);})
                .def("__iadd__",[](UniTensor &self, const cytnx::cytnx_int16     &rhs){return self.Add_(rhs);})
                .def("__iadd__",[](UniTensor &self, const cytnx::cytnx_uint16    &rhs){return self.Add_(rhs);})
                .def("__iadd__",[](UniTensor &self, const cytnx::cytnx_bool      &rhs){return self.Add_(rhs);})

                .def("__sub__",[](UniTensor &self, const UniTensor &rhs){return self.Sub(rhs);})
                .def("__sub__",[](UniTensor &self, const cytnx::cytnx_complex128&rhs){return self.Sub(rhs);})
                .def("__sub__",[](UniTensor &self, const cytnx::cytnx_complex64 &rhs){return self.Sub(rhs);})
                .def("__sub__",[](UniTensor &self, const cytnx::cytnx_double    &rhs){return self.Sub(rhs);})
                .def("__sub__",[](UniTensor &self, const cytnx::cytnx_float     &rhs){return self.Sub(rhs);})
                .def("__sub__",[](UniTensor &self, const cytnx::cytnx_int64     &rhs){return self.Sub(rhs);})
                .def("__sub__",[](UniTensor &self, const cytnx::cytnx_uint64    &rhs){return self.Sub(rhs);})
                .def("__sub__",[](UniTensor &self, const cytnx::cytnx_int32     &rhs){return self.Sub(rhs);})
                .def("__sub__",[](UniTensor &self, const cytnx::cytnx_uint32    &rhs){return self.Sub(rhs);})
                .def("__sub__",[](UniTensor &self, const cytnx::cytnx_int16     &rhs){return self.Sub(rhs);})
                .def("__sub__",[](UniTensor &self, const cytnx::cytnx_uint16    &rhs){return self.Sub(rhs);})
                .def("__sub__",[](UniTensor &self, const cytnx::cytnx_bool      &rhs){return self.Sub(rhs);})

                .def("__rsub__",[](UniTensor &self, const cytnx::cytnx_complex128&lhs){return linalg::Sub(lhs,self);})
                .def("__rsub__",[](UniTensor &self, const cytnx::cytnx_complex64 &lhs){return linalg::Sub(lhs,self);})
                .def("__rsub__",[](UniTensor &self, const cytnx::cytnx_double    &lhs){return linalg::Sub(lhs,self);})
                .def("__rsub__",[](UniTensor &self, const cytnx::cytnx_float     &lhs){return linalg::Sub(lhs,self);})
                .def("__rsub__",[](UniTensor &self, const cytnx::cytnx_int64     &lhs){return linalg::Sub(lhs,self);})
                .def("__rsub__",[](UniTensor &self, const cytnx::cytnx_uint64    &lhs){return linalg::Sub(lhs,self);})
                .def("__rsub__",[](UniTensor &self, const cytnx::cytnx_int32     &lhs){return linalg::Sub(lhs,self);})
                .def("__rsub__",[](UniTensor &self, const cytnx::cytnx_uint32    &lhs){return linalg::Sub(lhs,self);})
                .def("__rsub__",[](UniTensor &self, const cytnx::cytnx_int16     &lhs){return linalg::Sub(lhs,self);})
                .def("__rsub__",[](UniTensor &self, const cytnx::cytnx_uint16    &lhs){return linalg::Sub(lhs,self);})
                .def("__rsub__",[](UniTensor &self, const cytnx::cytnx_bool      &lhs){return linalg::Sub(lhs,self);})
 
                .def("__isub__",[](UniTensor &self, const UniTensor &rhs){return self.Sub_(rhs);}) // these will return self!
                .def("__isub__",[](UniTensor &self, const cytnx::cytnx_complex128&rhs){return self.Sub_(rhs);})
                .def("__isub__",[](UniTensor &self, const cytnx::cytnx_complex64 &rhs){return self.Sub_(rhs);})
                .def("__isub__",[](UniTensor &self, const cytnx::cytnx_double    &rhs){return self.Sub_(rhs);})
                .def("__isub__",[](UniTensor &self, const cytnx::cytnx_float     &rhs){return self.Sub_(rhs);})
                .def("__isub__",[](UniTensor &self, const cytnx::cytnx_int64     &rhs){return self.Sub_(rhs);})
                .def("__isub__",[](UniTensor &self, const cytnx::cytnx_uint64    &rhs){return self.Sub_(rhs);})
                .def("__isub__",[](UniTensor &self, const cytnx::cytnx_int32     &rhs){return self.Sub_(rhs);})
                .def("__isub__",[](UniTensor &self, const cytnx::cytnx_uint32    &rhs){return self.Sub_(rhs);})
                .def("__isub__",[](UniTensor &self, const cytnx::cytnx_int16     &rhs){return self.Sub_(rhs);})
                .def("__isub__",[](UniTensor &self, const cytnx::cytnx_uint16    &rhs){return self.Sub_(rhs);})
                .def("__isub__",[](UniTensor &self, const cytnx::cytnx_bool      &rhs){return self.Sub_(rhs);})

                .def("__mul__",[](UniTensor &self, const UniTensor &rhs){return self.Mul(rhs);})
                .def("__mul__",[](UniTensor &self, const cytnx::cytnx_complex128&rhs){return self.Mul(rhs);})
                .def("__mul__",[](UniTensor &self, const cytnx::cytnx_complex64 &rhs){return self.Mul(rhs);})
                .def("__mul__",[](UniTensor &self, const cytnx::cytnx_double    &rhs){return self.Mul(rhs);})
                .def("__mul__",[](UniTensor &self, const cytnx::cytnx_float     &rhs){return self.Mul(rhs);})
                .def("__mul__",[](UniTensor &self, const cytnx::cytnx_int64     &rhs){return self.Mul(rhs);})
                .def("__mul__",[](UniTensor &self, const cytnx::cytnx_uint64    &rhs){return self.Mul(rhs);})
                .def("__mul__",[](UniTensor &self, const cytnx::cytnx_int32     &rhs){return self.Mul(rhs);})
                .def("__mul__",[](UniTensor &self, const cytnx::cytnx_uint32    &rhs){return self.Mul(rhs);})
                .def("__mul__",[](UniTensor &self, const cytnx::cytnx_int16     &rhs){return self.Mul(rhs);})
                .def("__mul__",[](UniTensor &self, const cytnx::cytnx_uint16    &rhs){return self.Mul(rhs);})
                .def("__mul__",[](UniTensor &self, const cytnx::cytnx_bool    &rhs){return self.Mul(rhs);})

                .def("__rmul__",[](UniTensor &self, const cytnx::cytnx_complex128&lhs){return linalg::Mul(lhs,self);})
                .def("__rmul__",[](UniTensor &self, const cytnx::cytnx_complex64 &lhs){return linalg::Mul(lhs,self);})
                .def("__rmul__",[](UniTensor &self, const cytnx::cytnx_double    &lhs){return linalg::Mul(lhs,self);})
                .def("__rmul__",[](UniTensor &self, const cytnx::cytnx_float     &lhs){return linalg::Mul(lhs,self);})
                .def("__rmul__",[](UniTensor &self, const cytnx::cytnx_int64     &lhs){return linalg::Mul(lhs,self);})
                .def("__rmul__",[](UniTensor &self, const cytnx::cytnx_uint64    &lhs){return linalg::Mul(lhs,self);})
                .def("__rmul__",[](UniTensor &self, const cytnx::cytnx_int32     &lhs){return linalg::Mul(lhs,self);})
                .def("__rmul__",[](UniTensor &self, const cytnx::cytnx_uint32    &lhs){return linalg::Mul(lhs,self);})
                .def("__rmul__",[](UniTensor &self, const cytnx::cytnx_int16     &lhs){return linalg::Mul(lhs,self);})
                .def("__rmul__",[](UniTensor &self, const cytnx::cytnx_uint16    &lhs){return linalg::Mul(lhs,self);})
                .def("__rmul__",[](UniTensor &self, const cytnx::cytnx_bool      &lhs){return linalg::Mul(lhs,self);})

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

                .def("__rmod__",[](UniTensor &self, const cytnx::cytnx_complex128&lhs){return linalg::Mod(lhs,self);})
                .def("__rmod__",[](UniTensor &self, const cytnx::cytnx_complex64 &lhs){return linalg::Mod(lhs,self);})
                .def("__rmod__",[](UniTensor &self, const cytnx::cytnx_double    &lhs){return linalg::Mod(lhs,self);})
                .def("__rmod__",[](UniTensor &self, const cytnx::cytnx_float     &lhs){return linalg::Mod(lhs,self);})
                .def("__rmod__",[](UniTensor &self, const cytnx::cytnx_int64     &lhs){return linalg::Mod(lhs,self);})
                .def("__rmod__",[](UniTensor &self, const cytnx::cytnx_uint64    &lhs){return linalg::Mod(lhs,self);})
                .def("__rmod__",[](UniTensor &self, const cytnx::cytnx_int32     &lhs){return linalg::Mod(lhs,self);})
                .def("__rmod__",[](UniTensor &self, const cytnx::cytnx_uint32    &lhs){return linalg::Mod(lhs,self);})
                .def("__rmod__",[](UniTensor &self, const cytnx::cytnx_int16     &lhs){return linalg::Mod(lhs,self);})
                .def("__rmod__",[](UniTensor &self, const cytnx::cytnx_uint16    &lhs){return linalg::Mod(lhs,self);})
                .def("__rmod__",[](UniTensor &self, const cytnx::cytnx_bool      &lhs){return linalg::Mod(lhs,self);})
 
                .def("__imul__",[](UniTensor &self, const UniTensor &rhs){return self.Mul_(rhs);}) // these will return self!
                .def("__imul__",[](UniTensor &self, const cytnx::cytnx_complex128&rhs){return self.Mul_(rhs);})
                .def("__imul__",[](UniTensor &self, const cytnx::cytnx_complex64 &rhs){return self.Mul_(rhs);})
                .def("__imul__",[](UniTensor &self, const cytnx::cytnx_double    &rhs){return self.Mul_(rhs);})
                .def("__imul__",[](UniTensor &self, const cytnx::cytnx_float     &rhs){return self.Mul_(rhs);})
                .def("__imul__",[](UniTensor &self, const cytnx::cytnx_int64     &rhs){return self.Mul_(rhs);})
                .def("__imul__",[](UniTensor &self, const cytnx::cytnx_uint64    &rhs){return self.Mul_(rhs);})
                .def("__imul__",[](UniTensor &self, const cytnx::cytnx_int32     &rhs){return self.Mul_(rhs);})
                .def("__imul__",[](UniTensor &self, const cytnx::cytnx_uint32    &rhs){return self.Mul_(rhs);})
                .def("__imul__",[](UniTensor &self, const cytnx::cytnx_int16     &rhs){return self.Mul_(rhs);})
                .def("__imul__",[](UniTensor &self, const cytnx::cytnx_uint16    &rhs){return self.Mul_(rhs);})
                .def("__imul__",[](UniTensor &self, const cytnx::cytnx_bool      &rhs){return self.Mul_(rhs);})

                .def("__truediv__",[](UniTensor &self, const UniTensor &rhs){return self.Div(rhs);})
                .def("__truediv__",[](UniTensor &self, const cytnx::cytnx_complex128&rhs){return self.Div(rhs);})
                .def("__truediv__",[](UniTensor &self, const cytnx::cytnx_complex64 &rhs){return self.Div(rhs);})
                .def("__truediv__",[](UniTensor &self, const cytnx::cytnx_double    &rhs){return self.Div(rhs);})
                .def("__truediv__",[](UniTensor &self, const cytnx::cytnx_float     &rhs){return self.Div(rhs);})
                .def("__truediv__",[](UniTensor &self, const cytnx::cytnx_int64     &rhs){return self.Div(rhs);})
                .def("__truediv__",[](UniTensor &self, const cytnx::cytnx_uint64    &rhs){return self.Div(rhs);})
                .def("__truediv__",[](UniTensor &self, const cytnx::cytnx_int32     &rhs){return self.Div(rhs);})
                .def("__truediv__",[](UniTensor &self, const cytnx::cytnx_uint32    &rhs){return self.Div(rhs);})
                .def("__truediv__",[](UniTensor &self, const cytnx::cytnx_int16     &rhs){return self.Div(rhs);})
                .def("__truediv__",[](UniTensor &self, const cytnx::cytnx_uint16    &rhs){return self.Div(rhs);})
                .def("__truediv__",[](UniTensor &self, const cytnx::cytnx_bool      &rhs){return self.Div(rhs);})

                .def("__rtruediv__",[](UniTensor &self, const cytnx::cytnx_complex128&lhs){return linalg::Div(lhs,self);})
                .def("__rtruediv__",[](UniTensor &self, const cytnx::cytnx_complex64 &lhs){return linalg::Div(lhs,self);})
                .def("__rtruediv__",[](UniTensor &self, const cytnx::cytnx_double    &lhs){return linalg::Div(lhs,self);})
                .def("__rtruediv__",[](UniTensor &self, const cytnx::cytnx_float     &lhs){return linalg::Div(lhs,self);})
                .def("__rtruediv__",[](UniTensor &self, const cytnx::cytnx_int64     &lhs){return linalg::Div(lhs,self);})
                .def("__rtruediv__",[](UniTensor &self, const cytnx::cytnx_uint64    &lhs){return linalg::Div(lhs,self);})
                .def("__rtruediv__",[](UniTensor &self, const cytnx::cytnx_int32     &lhs){return linalg::Div(lhs,self);})
                .def("__rtruediv__",[](UniTensor &self, const cytnx::cytnx_uint32    &lhs){return linalg::Div(lhs,self);})
                .def("__rtruediv__",[](UniTensor &self, const cytnx::cytnx_int16     &lhs){return linalg::Div(lhs,self);})
                .def("__rtruediv__",[](UniTensor &self, const cytnx::cytnx_uint16    &lhs){return linalg::Div(lhs,self);})
                .def("__rtruediv__",[](UniTensor &self, const cytnx::cytnx_bool      &lhs){return linalg::Div(lhs,self);})
 
                .def("__itruediv__",[](UniTensor &self, const UniTensor &rhs){return self.Div_(rhs);}) // these will return self!
                .def("__itruediv__",[](UniTensor &self, const cytnx::cytnx_complex128&rhs){return self.Div_(rhs);})
                .def("__itruediv__",[](UniTensor &self, const cytnx::cytnx_complex64 &rhs){return self.Div_(rhs);})
                .def("__itruediv__",[](UniTensor &self, const cytnx::cytnx_double    &rhs){return self.Div_(rhs);})
                .def("__itruediv__",[](UniTensor &self, const cytnx::cytnx_float     &rhs){return self.Div_(rhs);})
                .def("__itruediv__",[](UniTensor &self, const cytnx::cytnx_int64     &rhs){return self.Div_(rhs);})
                .def("__itruediv__",[](UniTensor &self, const cytnx::cytnx_uint64    &rhs){return self.Div_(rhs);})
                .def("__itruediv__",[](UniTensor &self, const cytnx::cytnx_int32     &rhs){return self.Div_(rhs);})
                .def("__itruediv__",[](UniTensor &self, const cytnx::cytnx_uint32    &rhs){return self.Div_(rhs);})
                .def("__itruediv__",[](UniTensor &self, const cytnx::cytnx_int16     &rhs){return self.Div_(rhs);})
                .def("__itruediv__",[](UniTensor &self, const cytnx::cytnx_uint16    &rhs){return self.Div_(rhs);})
                .def("__itruediv__",[](UniTensor &self, const cytnx::cytnx_bool      &rhs){return self.Div_(rhs);})

                .def("__floordiv__",[](UniTensor &self, const UniTensor &rhs){return self.Div(rhs);})
                .def("__floordiv__",[](UniTensor &self, const cytnx::cytnx_complex128&rhs){return self.Div(rhs);})
                .def("__floordiv__",[](UniTensor &self, const cytnx::cytnx_complex64 &rhs){return self.Div(rhs);})
                .def("__floordiv__",[](UniTensor &self, const cytnx::cytnx_double    &rhs){return self.Div(rhs);})
                .def("__floordiv__",[](UniTensor &self, const cytnx::cytnx_float     &rhs){return self.Div(rhs);})
                .def("__floordiv__",[](UniTensor &self, const cytnx::cytnx_int64     &rhs){return self.Div(rhs);})
                .def("__floordiv__",[](UniTensor &self, const cytnx::cytnx_uint64    &rhs){return self.Div(rhs);})
                .def("__floordiv__",[](UniTensor &self, const cytnx::cytnx_int32     &rhs){return self.Div(rhs);})
                .def("__floordiv__",[](UniTensor &self, const cytnx::cytnx_uint32    &rhs){return self.Div(rhs);})
                .def("__floordiv__",[](UniTensor &self, const cytnx::cytnx_int16     &rhs){return self.Div(rhs);})
                .def("__floordiv__",[](UniTensor &self, const cytnx::cytnx_uint16    &rhs){return self.Div(rhs);})
                .def("__floordiv__",[](UniTensor &self, const cytnx::cytnx_bool      &rhs){return self.Div(rhs);})

                .def("__rfloordiv__",[](UniTensor &self, const cytnx::cytnx_complex128&lhs){return linalg::Div(lhs,self);})
                .def("__rfloordiv__",[](UniTensor &self, const cytnx::cytnx_complex64 &lhs){return linalg::Div(lhs,self);})
                .def("__rfloordiv__",[](UniTensor &self, const cytnx::cytnx_double    &lhs){return linalg::Div(lhs,self);})
                .def("__rfloordiv__",[](UniTensor &self, const cytnx::cytnx_float     &lhs){return linalg::Div(lhs,self);})
                .def("__rfloordiv__",[](UniTensor &self, const cytnx::cytnx_int64     &lhs){return linalg::Div(lhs,self);})
                .def("__rfloordiv__",[](UniTensor &self, const cytnx::cytnx_uint64    &lhs){return linalg::Div(lhs,self);})
                .def("__rfloordiv__",[](UniTensor &self, const cytnx::cytnx_int32     &lhs){return linalg::Div(lhs,self);})
                .def("__rfloordiv__",[](UniTensor &self, const cytnx::cytnx_uint32    &lhs){return linalg::Div(lhs,self);})
                .def("__rfloordiv__",[](UniTensor &self, const cytnx::cytnx_int16     &lhs){return linalg::Div(lhs,self);})
                .def("__rfloordiv__",[](UniTensor &self, const cytnx::cytnx_uint16    &lhs){return linalg::Div(lhs,self);})
                .def("__rfloordiv__",[](UniTensor &self, const cytnx::cytnx_bool      &lhs){return linalg::Div(lhs,self);})
 
                .def("__ifloordiv__",[](UniTensor &self, const UniTensor &rhs){return self.Div_(rhs);}) // these will return self!
                .def("__ifloordiv__",[](UniTensor &self, const cytnx::cytnx_complex128&rhs){return self.Div_(rhs);})
                .def("__ifloordiv__",[](UniTensor &self, const cytnx::cytnx_complex64 &rhs){return self.Div_(rhs);})
                .def("__ifloordiv__",[](UniTensor &self, const cytnx::cytnx_double    &rhs){return self.Div_(rhs);})
                .def("__ifloordiv__",[](UniTensor &self, const cytnx::cytnx_float     &rhs){return self.Div_(rhs);})
                .def("__ifloordiv__",[](UniTensor &self, const cytnx::cytnx_int64     &rhs){return self.Div_(rhs);})
                .def("__ifloordiv__",[](UniTensor &self, const cytnx::cytnx_uint64    &rhs){return self.Div_(rhs);})
                .def("__ifloordiv__",[](UniTensor &self, const cytnx::cytnx_int32     &rhs){return self.Div_(rhs);})
                .def("__ifloordiv__",[](UniTensor &self, const cytnx::cytnx_uint32    &rhs){return self.Div_(rhs);})
                .def("__ifloordiv__",[](UniTensor &self, const cytnx::cytnx_int16     &rhs){return self.Div_(rhs);})
                .def("__ifloordiv__",[](UniTensor &self, const cytnx::cytnx_uint16    &rhs){return self.Div_(rhs);})
                .def("__ifloordiv__",[](UniTensor &self, const cytnx::cytnx_bool      &rhs){return self.Div_(rhs);})
                .def("__pow__",[](UniTensor &self, const cytnx::cytnx_double &p){return self.Pow(p);})
                .def("c__ipow__",[](UniTensor &self, const cytnx::cytnx_double &p){self.Pow_(p);})
                .def("Pow",&UniTensor::Pow)
                .def("cPow_",&UniTensor::Pow_)
                .def("cConj_",&UniTensor::Conj_)
                .def("Conj",&UniTensor::Conj)
                .def("cTrace_",&UniTensor::Trace_,py::arg("a"),py::arg("b"),py::arg("by_label")=false)
                .def("Trace",&UniTensor::Trace,py::arg("a"),py::arg("b"),py::arg("by_label")=false)
                .def("cTranspose_",&UniTensor::Transpose_)
                .def("Transpose",&UniTensor::Transpose)
                .def("cDagger_",&UniTensor::Dagger_)
                .def("Dagger",&UniTensor::Dagger)
                .def("ctag",&UniTensor::tag)
                .def("truncate",&UniTensor::truncate,py::arg("bond_idx"),py::arg("dim"),py::arg("by_label")=false)
                .def("ctruncate_",&UniTensor::truncate_)
                */
                ;
    //m.def("Contract",Contract);
   

}

