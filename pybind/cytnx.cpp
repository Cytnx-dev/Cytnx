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
//#include "../include/cytnx_error.hpp"
#include "complex.h"




namespace py = pybind11;
using namespace pybind11::literals;
using namespace cytnx;

//ref: https://developer.lsst.io/v/DM-9089/coding/python_wrappers_for_cpp_with_pybind11.html
//ref: https://pybind11.readthedocs.io/en/stable/advanced/classes.html
//ref: https://block.arch.ethz.ch/blog/2016/07/adding-methods-to-python-classes/
//ref: https://medium.com/@mgarod/dynamically-add-a-method-to-a-class-in-python-c49204b85bd6

class PyLinOp: public LinOp{
    public:
        /* inherit constructor */
        using LinOp::LinOp;
        
        Tensor matvec(const Tensor& Tin) override {
            PYBIND11_OVERLOAD(
                Tensor, /* Return type */
                LinOp,      /* Parent class */
                matvec,          /* Name of function in C++ (must match Python name) */
                Tin      /* Argument(s) */
            );
        }

};





//=============================================
template<class T>
void f_Tensor_setitem_scal(cytnx::Tensor &self, py::object locators, const T &rc){
    cytnx_error_msg(self.shape().size() == 0, "[ERROR] try to setelem to a empty Tensor%s","\n");
    
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
    }else{
        // only int
        for(cytnx_uint32 i=0;i<self.shape().size();i++){
            if(i==0) accessors.push_back(cytnx::Accessor(locators.cast<cytnx_int64>()));
            else accessors.push_back(cytnx::Accessor::all());
        }
    }
    
    self.set(accessors,rc);
    
}

void f_CyTensor_setelem_scal_d(cytnx_extension::CyTensor &self, const std::vector<cytnx_uint64> &locator, const cytnx::cytnx_double&rc){
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

void f_CyTensor_setelem_scal_f(cytnx_extension::CyTensor &self, const std::vector<cytnx_uint64> &locator, const cytnx::cytnx_float&rc){
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
void f_CyTensor_setelem_scal_u64(cytnx_extension::CyTensor &self, const std::vector<cytnx_uint64> &locator, const cytnx::cytnx_uint64&rc){
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
void f_CyTensor_setelem_scal_i64(cytnx_extension::CyTensor &self, const std::vector<cytnx_uint64> &locator, const cytnx::cytnx_int64&rc){
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
void f_CyTensor_setelem_scal_u32(cytnx_extension::CyTensor &self, const std::vector<cytnx_uint64> &locator, const cytnx::cytnx_uint32&rc){
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
void f_CyTensor_setelem_scal_i32(cytnx_extension::CyTensor &self, const std::vector<cytnx_uint64> &locator, const cytnx::cytnx_int32&rc){
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
void f_CyTensor_setelem_scal_u16(cytnx_extension::CyTensor &self, const std::vector<cytnx_uint64> &locator, const cytnx::cytnx_uint16&rc){
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
void f_CyTensor_setelem_scal_i16(cytnx_extension::CyTensor &self, const std::vector<cytnx_uint64> &locator, const cytnx::cytnx_int16&rc){
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
void f_CyTensor_setelem_scal_b(cytnx_extension::CyTensor &self, const std::vector<cytnx_uint64> &locator, const cytnx::cytnx_bool&rc){
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
void f_CyTensor_setelem_scal_cd(cytnx_extension::CyTensor &self, const std::vector<cytnx_uint64> &locator, const cytnx::cytnx_complex128 &rc){
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
void f_CyTensor_setelem_scal_cf(cytnx_extension::CyTensor &self, const std::vector<cytnx_uint64> &locator, const cytnx::cytnx_complex64 &rc){
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




PYBIND11_MODULE(cytnx,m){

    m.attr("__version__") = "0.5.4a";
    m.attr("__blasINTsize__") = cytnx::__blasINTsize__;
    //global vars
    //m.attr("cytnxdevice") = cytnx::cytnxdevice;
    //m.attr("Type")   = py::cast(cytnx::Type);    
    //m.attr("redirect_output") = py::capsule(new py::scoped_ostream_redirect(...),
    //[](void *sor) { delete static_cast<py::scoped_ostream_redirect *>(sor); });
    py::add_ostream_redirect(m, "ostream_redirect");


    

    py::enum_<cytnx::__type::__pybind_type>(m,"Type")
        .value("Void", cytnx::__type::__pybind_type::Void)
        .value("ComplexDouble", cytnx::__type::__pybind_type::ComplexDouble)
		.value("ComplexFloat", cytnx::__type::__pybind_type::ComplexFloat )	
        .value("Double", cytnx::__type::__pybind_type::Double)
		.value("Float", cytnx::__type::__pybind_type::Float  )	
        .value("Uint64", cytnx::__type::__pybind_type::Uint64)
		.value("Int64", cytnx::__type::__pybind_type::Int64  ) 	
        .value("Uint32", cytnx::__type::__pybind_type::Uint32)
		.value("Int32", cytnx::__type::__pybind_type::Int32  ) 	
        .value("Uint16", cytnx::__type::__pybind_type::Uint16)
		.value("Int16", cytnx::__type::__pybind_type::Int16  ) 	
		.value("Bool", cytnx::__type::__pybind_type::Bool    ) 	
		.export_values();
    
    
    //py::enum_<cytnx::__device::__pybind_device>(m,"Device",py::arithmetic())
    //    .value("cpu", cytnx::__device::__pybind_device::cpu)
	//	.value("cuda", cytnx::__device::__pybind_device::cuda)	
	//	.export_values();

    

    //m.attr("Device") = py::module::import("enum").attr("IntEnum")
    //    ("Device", py::dict("cpu"_a=(cytnx_int64)cytnx::Device.cpu, "cuda"_a=(cytnx_int64)cytnx::Device.cuda)); 


    auto mdev = m.def_submodule("Device");
    mdev.attr("cpu")=(cytnx_int64)cytnx::Device.cpu;
    mdev.attr("cuda")=(cytnx_int64)cytnx::Device.cuda;
    //mdev.def("cudaDeviceSynchronize",[](){cytnx::Device.cudaDeviceSynchronize();});
    
    



    m.def("zeros",[](const cytnx_uint64 &Nelem, const unsigned int &dtype, const int &device)->Tensor{
                        return cytnx::zeros(Nelem,dtype,device);
                  },py::arg("size"),py::arg("dtype")=(unsigned int)(cytnx::Type.Double), py::arg("device")=(int)(cytnx::Device.cpu));

    m.def("zeros",[](py::object Nelem, const unsigned int &dtype, const int &device)->Tensor{
                        std::vector<cytnx_uint64> tmp = Nelem.cast<std::vector<cytnx_uint64> >();
                        return cytnx::zeros(tmp,dtype,device);
                  },py::arg("size"),py::arg("dtype")=(unsigned int)(cytnx::Type.Double), py::arg("device")=(int)(cytnx::Device.cpu));
    
    m.def("ones",[](const cytnx_uint64 &Nelem, const unsigned int &dtype, const int &device)->Tensor{
                        return cytnx::ones(Nelem,dtype,device);
                  },py::arg("size"),py::arg("dtype")=(unsigned int)(cytnx::Type.Double), py::arg("device")=(int)(cytnx::Device.cpu));

    m.def("ones",[](py::object Nelem, const unsigned int &dtype, const int &device)->Tensor{
                        std::vector<cytnx_uint64> tmp = Nelem.cast<std::vector<cytnx_uint64> >();
                        return cytnx::ones(tmp,dtype,device);
                  },py::arg("size"),py::arg("dtype")=(unsigned int)(cytnx::Type.Double), py::arg("device")=(int)(cytnx::Device.cpu));
    m.def("identity",&cytnx::identity
                  ,py::arg("Dim"),py::arg("dtype")=(unsigned int)(cytnx::Type.Double), py::arg("device")=(int)(cytnx::Device.cpu));
    m.def("eye",&cytnx::identity
                  ,py::arg("Dim"),py::arg("dtype")=(unsigned int)(cytnx::Type.Double), py::arg("device")=(int)(cytnx::Device.cpu));

    m.def("arange",[](const cytnx_uint64 &Nelem, const unsigned int &dtype, const int &device)->Tensor{
                        return cytnx::arange(Nelem,dtype,device);
                  },py::arg("size"),py::arg("dtype")=(unsigned int)(cytnx::Type.Double), py::arg("device")=(int)(cytnx::Device.cpu));

    m.def("arange",[](const cytnx_double &start, const cytnx_double &end, const cytnx_double &step, const unsigned int &dtype, const int &device)->Tensor{
                        return cytnx::arange(start,end,step,dtype,device);
                  },py::arg("start"),py::arg("end"),py::arg("step") = double(1), py::arg("dtype")=(unsigned int)(cytnx::Type.Double), py::arg("device")=(int)(cytnx::Device.cpu));


    m.def("_from_numpy", [](py::buffer b)-> Tensor{
        py::buffer_info info = b.request();

        //std::cout << PyBuffer_IsContiguous(info,'C') << std::endl;

        // check type:                    
        int dtype;
        std::vector<cytnx_uint64> shape(info.shape.begin(),info.shape.end());
        ssize_t Totbytes;

        if(info.format ==py::format_descriptor<cytnx_complex128>::format()){
            dtype=Type.ComplexDouble;
            Totbytes = info.strides[0]*info.shape[0];
        }else if(info.format ==py::format_descriptor<cytnx_complex64>::format()){
            dtype=Type.ComplexFloat;
            Totbytes = info.strides[0]*info.shape[0];
        }else if(info.format ==py::format_descriptor<cytnx_double>::format()){
            dtype=Type.Double;
            Totbytes = info.strides[0]*info.shape[0];
        }else if(info.format ==py::format_descriptor<cytnx_float>::format()){
            dtype=Type.Float;
            Totbytes = info.strides[0]*info.shape[0];
        }else if(info.format ==py::format_descriptor<uint64_t>::format()|| info.format=="L"){
            dtype=Type.Uint64;
            Totbytes = info.strides[0]*info.shape[0];
        }else if(info.format ==py::format_descriptor<int64_t>::format() || info.format=="l"){
            dtype=Type.Int64;
            Totbytes = info.strides[0]*info.shape[0];
        }else if(info.format ==py::format_descriptor<uint32_t>::format()){
            dtype=Type.Uint32;
            Totbytes = info.strides[0]*info.shape[0];
        }else if(info.format ==py::format_descriptor<int32_t>::format()){
            dtype=Type.Int32;
            Totbytes = info.strides[0]*info.shape[0];
        }else if(info.format ==py::format_descriptor<cytnx_bool>::format()){
            dtype=Type.Bool;
            Totbytes = info.strides[0]*info.shape[0];
        }else{
            cytnx_error_msg(true,"[ERROR] invalid type from numpy.ndarray to Tensor%s","\n");
        }
        /*        
        for( int i=0;i<info.strides.size();i++)
            std::cout << info.strides[i] << " ";
        std::cout << std::endl;
        for( int i=0;i<info.shape.size();i++)
            std::cout << info.shape[i] << " ";
        std::cout << std::endl;

        std::cout << Type.getname(dtype) << std::endl;
        std::cout << Totbytes << std::endl;
        */
        //Totbytes *= cytnx::Type.typeSize(dtype);
        
        Tensor m;
        m.Init(shape,dtype);
        memcpy(m.storage()._impl->Mem,info.ptr,Totbytes);
        return m;
     });
 

    py::class_<LinOp,PyLinOp>(m,"LinOp")
        .def(py::init<const std::string &, const cytnx_uint64 &, const int &, const int &, std::function<Tensor(const Tensor&)> >(),py::arg("type"),py::arg("nx"),py::arg("dtype")=(int)Type.Double,py::arg("device")=(int)Device.cpu,py::arg("custom_f")=nullptr)
        .def("set_func",&LinOp::set_func,py::arg("custom_f"),py::arg("dtype"),py::arg("device"))
        .def("matvec", &LinOp::matvec)
        .def("set_device", &LinOp::set_device)
        .def("set_dtype", &LinOp::set_dtype)
        .def("device", &LinOp::device)
        .def("dtype", &LinOp::dtype)
        .def("nx", &LinOp::nx)
        ;


    py::class_<cytnx::Storage>(m,"Storage")
                //construction
                .def(py::init<>())
                .def(py::init<const cytnx::Storage&>())
                .def(py::init<boost::intrusive_ptr<cytnx::Storage_base> >())
                .def(py::init<const unsigned long long &, const unsigned int&, int>(),py::arg("size"), py::arg("dtype")=(cytnx_uint64)Type.Double,py::arg("device")=-1)
                .def("Init", &cytnx::Storage::Init,py::arg("size"), py::arg("dtype")=(cytnx_uint64)Type.Double,py::arg("device")=-1)

                .def("dtype",&cytnx::Storage::dtype)
                .def("dtype_str",&cytnx::Storage::dtype_str)
                .def("device",&cytnx::Storage::device)
                .def("device_str",&cytnx::Storage::device_str)

                //[note] this is an interesting binding, since we want if new_type==self.dtype() to return self,
                //       the pybind cannot handle this. The direct binding will make a "new" instance in terms of 
                //       python's consideration. 
                //       The solution is to move the definition into python side. (see cytnx/Storage_conti.py)
                //.def("astype", &cytnx::Storage::astype,py::arg("new_type"))
                .def("astype_different_type",[](cytnx::Storage &self, const cytnx_uint64 &new_type){
                                    cytnx_error_msg(self.dtype()==new_type,"[ERROR][pybind][astype_diffferent_type] same type for astype() should be handle in python side.%s","\n");
                                    return self.astype(new_type);
                                },py::arg("new_type"))

                .def("__getitem__",[](cytnx::Storage &self, const unsigned long long &idx){
                    cytnx_error_msg(idx > self.size(),"idx exceed the size of storage.%s","\n");
                    py::object out;
                    if(self.dtype() == cytnx::Type.Double) 
                        out =  py::cast(self.at<cytnx::cytnx_double>(idx));
                    else if(self.dtype() == cytnx::Type.Float) 
                        out = py::cast(self.at<cytnx::cytnx_float>(idx));
                    else if(self.dtype() == cytnx::Type.ComplexDouble) 
                        out = py::cast(self.at<cytnx::cytnx_complex128>(idx));
                    else if(self.dtype() == cytnx::Type.ComplexFloat) 
                        out = py::cast(self.at<cytnx::cytnx_complex64>(idx));
                    else if(self.dtype() == cytnx::Type.Uint64) 
                        out = py::cast(self.at<cytnx::cytnx_uint64>(idx));
                    else if(self.dtype() == cytnx::Type.Int64) 
                        out = py::cast(self.at<cytnx::cytnx_int64>(idx));
                    else if(self.dtype() == cytnx::Type.Uint32) 
                        out = py::cast(self.at<cytnx::cytnx_uint32>(idx));
                    else if(self.dtype() == cytnx::Type.Int32) 
                        out = py::cast(self.at<cytnx::cytnx_int32>(idx));
                    else if(self.dtype() == cytnx::Type.Uint16) 
                        out = py::cast(self.at<cytnx::cytnx_uint16>(idx));
                    else if(self.dtype() == cytnx::Type.Int16) 
                        out = py::cast(self.at<cytnx::cytnx_int16>(idx));
                    else if(self.dtype() == cytnx::Type.Bool) 
                        out = py::cast(self.at<cytnx::cytnx_bool>(idx));
                    else cytnx_error_msg(true, "%s","[ERROR] try to get element from a void Storage.");

                    return out;
                 })
                .def("__setitem__",[](cytnx::Storage &self, const unsigned long long &idx, py::object in){
                    cytnx_error_msg(idx > self.size(),"idx exceed the size of storage.%s","\n");
                    py::object out;
                    if(self.dtype() == cytnx::Type.Double) 
                        self.at<cytnx::cytnx_double>(idx) = in.cast<cytnx::cytnx_double>();
                    else if(self.dtype() == cytnx::Type.Float) 
                        self.at<cytnx::cytnx_float>(idx) = in.cast<cytnx::cytnx_float>();
                    else if(self.dtype() == cytnx::Type.ComplexDouble) 
                        self.at<cytnx::cytnx_complex128>(idx) = in.cast<cytnx::cytnx_complex128>();
                    else if(self.dtype() == cytnx::Type.ComplexFloat) 
                        self.at<cytnx::cytnx_complex64>(idx) = in.cast<cytnx::cytnx_complex64>();
                    else if(self.dtype() == cytnx::Type.Uint64) 
                        self.at<cytnx::cytnx_uint64>(idx) = in.cast<cytnx::cytnx_uint64>();
                    else if(self.dtype() == cytnx::Type.Int64) 
                        self.at<cytnx::cytnx_int64>(idx) = in.cast<cytnx::cytnx_int64>();
                    else if(self.dtype() == cytnx::Type.Uint32) 
                        self.at<cytnx::cytnx_uint32>(idx) = in.cast<cytnx::cytnx_uint32>();
                    else if(self.dtype() == cytnx::Type.Int32) 
                        self.at<cytnx::cytnx_int32>(idx) = in.cast<cytnx::cytnx_int32>();
                    else if(self.dtype() == cytnx::Type.Uint16) 
                        self.at<cytnx::cytnx_uint16>(idx) = in.cast<cytnx::cytnx_uint16>();
                    else if(self.dtype() == cytnx::Type.Int16) 
                        self.at<cytnx::cytnx_int16>(idx) = in.cast<cytnx::cytnx_int16>();
                    else if(self.dtype() == cytnx::Type.Bool) 
                        self.at<cytnx::cytnx_bool>(idx) = in.cast<cytnx::cytnx_bool>();
                    else cytnx_error_msg(true, "%s","[ERROR] try to get element from a void Storage.");
                 })
                .def("__repr__",[](cytnx::Storage &self)->std::string{
                    std::cout << self << std::endl;
                    return std::string("");
                 },py::call_guard<py::scoped_ostream_redirect,
                     py::scoped_estream_redirect>())
                .def("__len__",[](cytnx::Storage &self)->cytnx::cytnx_uint64{return self.size();})
                
                .def("to_", &cytnx::Storage::to_, py::arg("device"))

                // handle same device from cytnx/Storage_conti.py
                .def("to_different_device" ,[](cytnx::Storage &self,const cytnx_int64 &device){
                                                    cytnx_error_msg(self.device() == device, "[ERROR][pybind][to_diffferent_device] same device for to() should be handle in python side.%s","\n");
                                                    return self.to(device);
                                                } , py::arg("device"))

                
                .def("resize", &cytnx::Storage::resize)
                .def("capacity", &cytnx::Storage::capacity)
                .def("clone", &cytnx::Storage::clone)
                .def("__copy__",&cytnx::Storage::clone)
                .def("__deepcopy__",&cytnx::Storage::clone)
                .def("size", &cytnx::Storage::size)
                .def("__len__",[](cytnx::Storage &self){return self.size();})
                .def("print_info", &cytnx::Storage::print_info,py::call_guard<py::scoped_ostream_redirect,
                     py::scoped_estream_redirect>())
                .def("set_zeros",  &cytnx::Storage::set_zeros)
                .def("__eq__",[](cytnx::Storage &self, const cytnx::Storage &rhs)->bool{return self == rhs;})

                .def("fill",&cytnx::Storage::fill<cytnx::cytnx_complex128>, py::arg("val"))
                .def("fill",&cytnx::Storage::fill<cytnx::cytnx_complex64>, py::arg("val"))
                .def("fill",&cytnx::Storage::fill<cytnx::cytnx_double   >, py::arg("val"))
                .def("fill",&cytnx::Storage::fill<cytnx::cytnx_float    >, py::arg("val"))
                .def("fill",&cytnx::Storage::fill<cytnx::cytnx_int64    >, py::arg("val"))
                .def("fill",&cytnx::Storage::fill<cytnx::cytnx_uint64   >, py::arg("val"))
                .def("fill",&cytnx::Storage::fill<cytnx::cytnx_int32    >, py::arg("val"))
                .def("fill",&cytnx::Storage::fill<cytnx::cytnx_uint32   >, py::arg("val"))
                .def("fill",&cytnx::Storage::fill<cytnx::cytnx_int16    >, py::arg("val"))
                .def("fill",&cytnx::Storage::fill<cytnx::cytnx_uint16   >, py::arg("val"))
                .def("fill",&cytnx::Storage::fill<cytnx::cytnx_bool     >, py::arg("val"))
                
                .def("append",&cytnx::Storage::append<cytnx::cytnx_complex128>, py::arg("val"))
                .def("append",&cytnx::Storage::append<cytnx::cytnx_complex64>, py::arg("val"))
                .def("append",&cytnx::Storage::append<cytnx::cytnx_double   >, py::arg("val"))
                .def("append",&cytnx::Storage::append<cytnx::cytnx_float    >, py::arg("val"))
                .def("append",&cytnx::Storage::append<cytnx::cytnx_int64    >, py::arg("val"))
                .def("append",&cytnx::Storage::append<cytnx::cytnx_uint64   >, py::arg("val"))
                .def("append",&cytnx::Storage::append<cytnx::cytnx_int32    >, py::arg("val"))
                .def("append",&cytnx::Storage::append<cytnx::cytnx_uint32   >, py::arg("val"))
                .def("append",&cytnx::Storage::append<cytnx::cytnx_int16    >, py::arg("val"))
                .def("append",&cytnx::Storage::append<cytnx::cytnx_uint16   >, py::arg("val"))
                .def("append",&cytnx::Storage::append<cytnx::cytnx_bool     >, py::arg("val"))
                .def("Save",[](cytnx::Storage &self, const std::string &fname){self.Save(fname);},py::arg("fname"))
                .def_static("Load",[](const std::string &fname){return cytnx::Storage::Load(fname);},py::arg("fname"))
                .def("real",&cytnx::Storage::real)
                .def("imag",&cytnx::Storage::imag)
                ;


    //entry.Tensor
    py::class_<cytnx::Tensor>(m,"Tensor")
                .def("numpy",[](Tensor &self)-> py::array {

                    //device on GPU? move to cpu:ref it;
                    Tensor tmpIN;
                    if(self.device() >= 0){
                        tmpIN = self.to(Device.cpu);
                    }else{
                        tmpIN = self;
                    }
                    if(tmpIN.is_contiguous())
                        tmpIN = self.clone();
                    else
                        tmpIN = self.contiguous();

                    
                    //calculate stride:
                    std::vector<ssize_t> stride(tmpIN.shape().size());
                    std::vector<ssize_t> shape(tmpIN.shape().begin(),tmpIN.shape().end());
                    ssize_t accu = 1;
                    for(int i=shape.size()-1;i>=0;i--){
                        stride[i] = accu*Type.typeSize(tmpIN.dtype());
                        accu*=shape[i];
                    }
                    py::buffer_info npbuf;
                    std::string chr_dtype;
                    if(tmpIN.dtype()==Type.ComplexDouble){ 
                        chr_dtype = py::format_descriptor<cytnx_complex128>::format();
                    }else if(tmpIN.dtype() == Type.ComplexFloat){
                        chr_dtype = py::format_descriptor<cytnx_complex64>::format();
                    }else if(tmpIN.dtype() == Type.Double){
                        chr_dtype = py::format_descriptor<cytnx_double>::format();
                    }else if(tmpIN.dtype() == Type.Float){
                        chr_dtype = py::format_descriptor<cytnx_float>::format();
                    }else if(tmpIN.dtype() == Type.Uint64){
                        chr_dtype = py::format_descriptor<cytnx_uint64>::format();
                    }else if(tmpIN.dtype() == Type.Int64){
                        chr_dtype = py::format_descriptor<cytnx_int64>::format();
                    }else if(tmpIN.dtype() == Type.Uint32){
                        chr_dtype = py::format_descriptor<cytnx_uint32>::format();
                    }else if(tmpIN.dtype() == Type.Int32){
                        chr_dtype = py::format_descriptor<cytnx_int32>::format();
                    }else if(tmpIN.dtype() == Type.Bool){
                        chr_dtype = py::format_descriptor<cytnx_bool>::format();
                    }else{
                        cytnx_error_msg(true,"[ERROR] Void Type Tensor cannot convert to numpy ndarray%s","\n");
                    }


                    npbuf = py::buffer_info(
                                tmpIN.storage()._impl->Mem,    //ptr
                                Type.typeSize(tmpIN.dtype()), //size of elem 
                                chr_dtype, //pss format
                                tmpIN.rank(),    //rank 
                                shape,       // shape
                                stride      // stride
                           );
                    py::array out(npbuf);
                    // delegate numpy array with it's ptr, and swap a auxiliary ptr for intrusive_ptr to free.
                    void* pswap = malloc(sizeof(bool));
                    tmpIN.storage()._impl->Mem = pswap;
                    return out;

                })
                //construction
                .def(py::init<>())
                .def(py::init<const cytnx::Tensor&>())
                .def(py::init<const std::vector<cytnx::cytnx_uint64>&, const unsigned int&, int>(),py::arg("shape"), py::arg("dtype")=(cytnx_uint64)cytnx::Type.Double,py::arg("device")=(int)cytnx::Device.cpu)
                .def("Init", &cytnx::Tensor::Init,py::arg("shape"), py::arg("dtype")=(cytnx_uint64)cytnx::Type.Double,py::arg("device")=(int)cytnx::Device.cpu)
                .def("dtype",&cytnx::Tensor::dtype)
                .def("dtype_str",&cytnx::Tensor::dtype_str)
                .def("device",&cytnx::Tensor::device)
                .def("device_str",&cytnx::Tensor::device_str)
                .def("shape",&cytnx::Tensor::shape)
                .def("rank",&cytnx::Tensor::rank) 
                .def("clone", &cytnx::Tensor::clone)
                .def("__copy__", &cytnx::Tensor::clone)
                .def("__deepcopy__", &cytnx::Tensor::clone)
                //.def("to", &cytnx::Tensor::to, py::arg("device"))
                // handle same device from cytnx/Tensor_conti.py
                .def("to_different_device" ,[](cytnx::Tensor &self,const cytnx_int64 &device){
                                                    cytnx_error_msg(self.device() == device, "[ERROR][pybind][to_diffferent_device] same device for to() should be handle in python side.%s","\n");
                                                    return self.to(device);
                                                } , py::arg("device"))

                .def("to_", &cytnx::Tensor::to_, py::arg("device"))
                .def("is_contiguous", &cytnx::Tensor::is_contiguous)
                .def("permute_",[](cytnx::Tensor &self, py::args args){
                    std::vector<cytnx::cytnx_uint64> c_args = args.cast< std::vector<cytnx::cytnx_uint64> >();
                    //std::cout << c_args.size() << std::endl;
                    self.permute_(c_args);
                })
                .def("permute",[](cytnx::Tensor &self, py::args args)->cytnx::Tensor{
                    std::vector<cytnx::cytnx_uint64> c_args = args.cast< std::vector<cytnx::cytnx_uint64> >();
                    //std::cout << c_args.size() << std::endl;
                    return self.permute(c_args);
                })
                .def("flatten",&cytnx::Tensor::flatten)
                .def("flatten_",&cytnx::Tensor::flatten_)
                .def("make_contiguous",&cytnx::Tensor::contiguous) // this will be rename by python side conti
                .def("contiguous_",&cytnx::Tensor::contiguous_)
                .def("reshape_",[](cytnx::Tensor &self, py::args args){
                    std::vector<cytnx::cytnx_int64> c_args = args.cast< std::vector<cytnx::cytnx_int64> >();
                    self.reshape_(c_args);
                })
                .def("reshape",[](cytnx::Tensor &self, py::args args)->cytnx::Tensor{
                    std::vector<cytnx::cytnx_int64> c_args = args.cast< std::vector<cytnx::cytnx_int64> >();
                    return self.reshape(c_args);
                })
                //.def("astype", &cytnx::Tensor::astype,py::arg("new_type"))
                .def("astype_different_dtype",[](cytnx::Tensor &self, const cytnx_uint64 &dtype){
                                                    cytnx_error_msg(self.dtype() == dtype, "[ERROR][pybind][astype_diffferent_device] same dtype for astype() should be handle in python side.%s","\n");
                                                    return self.astype(dtype);
                                                },py::arg("new_type"))

                .def("item",[](cytnx::Tensor &self){
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
                    else cytnx_error_msg(true, "%s","[ERROR] try to get element from a void Storage.");
                    return out;
                 })
                .def("storage",&cytnx::Tensor::storage) 
                .def("real", &cytnx::Tensor::real)
                .def("imag", &cytnx::Tensor::imag)
                .def("__repr__",[](cytnx::Tensor &self)->std::string{
                    std::cout << self << std::endl;
                    return std::string("");
                 },py::call_guard<py::scoped_ostream_redirect,
                     py::scoped_estream_redirect>())
                .def("fill",&cytnx::Tensor::fill<cytnx::cytnx_complex128>, py::arg("val"))
                .def("fill",&cytnx::Tensor::fill<cytnx::cytnx_complex64>, py::arg("val"))
                .def("fill",&cytnx::Tensor::fill<cytnx::cytnx_double   >, py::arg("val"))
                .def("fill",&cytnx::Tensor::fill<cytnx::cytnx_float    >, py::arg("val"))
                .def("fill",&cytnx::Tensor::fill<cytnx::cytnx_int64    >, py::arg("val"))
                .def("fill",&cytnx::Tensor::fill<cytnx::cytnx_uint64   >, py::arg("val"))
                .def("fill",&cytnx::Tensor::fill<cytnx::cytnx_int32    >, py::arg("val"))
                .def("fill",&cytnx::Tensor::fill<cytnx::cytnx_uint32   >, py::arg("val"))
                .def("fill",&cytnx::Tensor::fill<cytnx::cytnx_int16    >, py::arg("val"))
                .def("fill",&cytnx::Tensor::fill<cytnx::cytnx_uint16   >, py::arg("val"))
                .def("fill",&cytnx::Tensor::fill<cytnx::cytnx_bool     >, py::arg("val"))

                .def("append",&cytnx::Tensor::append<cytnx::cytnx_complex128>, py::arg("val"))
                .def("append",&cytnx::Tensor::append<cytnx::cytnx_complex64>, py::arg("val"))
                .def("append",&cytnx::Tensor::append<cytnx::cytnx_double   >, py::arg("val"))
                .def("append",&cytnx::Tensor::append<cytnx::cytnx_float    >, py::arg("val"))
                .def("append",&cytnx::Tensor::append<cytnx::cytnx_int64    >, py::arg("val"))
                .def("append",&cytnx::Tensor::append<cytnx::cytnx_uint64   >, py::arg("val"))
                .def("append",&cytnx::Tensor::append<cytnx::cytnx_int32    >, py::arg("val"))
                .def("append",&cytnx::Tensor::append<cytnx::cytnx_uint32   >, py::arg("val"))
                .def("append",&cytnx::Tensor::append<cytnx::cytnx_int16    >, py::arg("val"))
                .def("append",&cytnx::Tensor::append<cytnx::cytnx_uint16   >, py::arg("val"))
                .def("append",&cytnx::Tensor::append<cytnx::cytnx_bool     >, py::arg("val"))
                .def("append",[](cytnx::Tensor &self,const cytnx::Tensor &rhs){self.append(rhs);}, py::arg("val"))

                .def("Save",[](cytnx::Tensor &self, const std::string &fname){self.Save(fname);},py::arg("fname"))
                .def_static("Load",[](const std::string &fname){return cytnx::Tensor::Load(fname);},py::arg("fname"))

                .def("__len__",[](const cytnx::Tensor &self){
                    if(self.dtype()==Type.Void){
                        cytnx_error_msg(true,"[ERROR] uninitialize Tensor does not have len!%s","\n");
                        
                    }else{
                        return self.shape()[0];
                    }
                })
                .def("__getitem__",[](const cytnx::Tensor &self, py::object locators){
                    cytnx_error_msg(self.shape().size() == 0, "[ERROR] try to getitem from a empty Tensor%s","\n");
                    
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
                        //std::cout << "int locators" << std::endl;
                        //std::cout << locators.cast<cytnx_int64>() << std::endl;
                        // only int
                        for(cytnx_uint32 i=0;i<self.shape().size();i++){
                            if(i==0) accessors.push_back(cytnx::Accessor(locators.cast<cytnx_int64>()));
                            else accessors.push_back(cytnx::Accessor::all());
                        }
                    }
                        
                     
                    return self.get(accessors);
                    
                })

                .def("__setitem__",[](cytnx::Tensor &self, py::object locators, const cytnx::Tensor &rhs){
                    cytnx_error_msg(self.shape().size() == 0, "[ERROR] try to setelem to a empty Tensor%s","\n");
                    
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
                .def("__setitem__",&f_Tensor_setitem_scal<cytnx_complex128>) 
                .def("__setitem__",&f_Tensor_setitem_scal<cytnx_complex64> ) 
                .def("__setitem__",&f_Tensor_setitem_scal<cytnx_double>) 
                .def("__setitem__",&f_Tensor_setitem_scal<cytnx_float> ) 
                .def("__setitem__",&f_Tensor_setitem_scal<cytnx_int64>) 
                .def("__setitem__",&f_Tensor_setitem_scal<cytnx_uint64> ) 
                .def("__setitem__",&f_Tensor_setitem_scal<cytnx_int32>) 
                .def("__setitem__",&f_Tensor_setitem_scal<cytnx_uint32> )
                .def("__setitem__",&f_Tensor_setitem_scal<cytnx_int16>) 
                .def("__setitem__",&f_Tensor_setitem_scal<cytnx_uint16> )
                .def("__setitem__",&f_Tensor_setitem_scal<cytnx_bool> )


                //arithmetic >>
                .def("__neg__",[](cytnx::Tensor &self){
                                    if(self.dtype() == Type.Double){
                                        return cytnx::linalg::Mul(cytnx_double(-1),self);
                                    }else if(self.dtype()==Type.ComplexDouble){
                                        return cytnx::linalg::Mul(cytnx_complex128(-1,0),self);
                                    }else if(self.dtype()==Type.Float){
                                        return cytnx::linalg::Mul(cytnx_float(-1),self);
                                    }else if(self.dtype()==Type.ComplexFloat){
                                        return cytnx::linalg::Mul(cytnx_complex64(-1,0),self);
                                    }else{
                                        return cytnx::linalg::Mul(-1,self);
                                    }
                                  })
                .def("__pos__",[](cytnx::Tensor &self){return self;})
                .def("__add__",[](cytnx::Tensor &self, const cytnx::Tensor &rhs){return self.Add(rhs);})
                .def("__add__",[](cytnx::Tensor &self, const cytnx::cytnx_complex128&rhs){return self.Add(rhs);})
                .def("__add__",[](cytnx::Tensor &self, const cytnx::cytnx_complex64 &rhs){return self.Add(rhs);})
                .def("__add__",[](cytnx::Tensor &self, const cytnx::cytnx_double    &rhs){return self.Add(rhs);})
                .def("__add__",[](cytnx::Tensor &self, const cytnx::cytnx_float     &rhs){return self.Add(rhs);})
                .def("__add__",[](cytnx::Tensor &self, const cytnx::cytnx_int64     &rhs){return self.Add(rhs);})
                .def("__add__",[](cytnx::Tensor &self, const cytnx::cytnx_uint64    &rhs){return self.Add(rhs);})
                .def("__add__",[](cytnx::Tensor &self, const cytnx::cytnx_int32     &rhs){return self.Add(rhs);})
                .def("__add__",[](cytnx::Tensor &self, const cytnx::cytnx_uint32    &rhs){return self.Add(rhs);})
                .def("__add__",[](cytnx::Tensor &self, const cytnx::cytnx_int16     &rhs){return self.Add(rhs);})
                .def("__add__",[](cytnx::Tensor &self, const cytnx::cytnx_uint16    &rhs){return self.Add(rhs);})
                .def("__add__",[](cytnx::Tensor &self, const cytnx::cytnx_bool      &rhs){return self.Add(rhs);})
                
                .def("__radd__",[](cytnx::Tensor &self, const cytnx::cytnx_complex128&lhs){return cytnx::linalg::Add(lhs,self);})
                .def("__radd__",[](cytnx::Tensor &self, const cytnx::cytnx_complex64 &lhs){return cytnx::linalg::Add(lhs,self);})
                .def("__radd__",[](cytnx::Tensor &self, const cytnx::cytnx_double    &lhs){return cytnx::linalg::Add(lhs,self);})
                .def("__radd__",[](cytnx::Tensor &self, const cytnx::cytnx_float     &lhs){return cytnx::linalg::Add(lhs,self);})
                .def("__radd__",[](cytnx::Tensor &self, const cytnx::cytnx_int64     &lhs){return cytnx::linalg::Add(lhs,self);})
                .def("__radd__",[](cytnx::Tensor &self, const cytnx::cytnx_uint64    &lhs){return cytnx::linalg::Add(lhs,self);})
                .def("__radd__",[](cytnx::Tensor &self, const cytnx::cytnx_int32     &lhs){return cytnx::linalg::Add(lhs,self);})
                .def("__radd__",[](cytnx::Tensor &self, const cytnx::cytnx_uint32    &lhs){return cytnx::linalg::Add(lhs,self);})
                .def("__radd__",[](cytnx::Tensor &self, const cytnx::cytnx_int16     &lhs){return cytnx::linalg::Add(lhs,self);})
                .def("__radd__",[](cytnx::Tensor &self, const cytnx::cytnx_uint16    &lhs){return cytnx::linalg::Add(lhs,self);})
                .def("__radd__",[](cytnx::Tensor &self, const cytnx::cytnx_bool      &lhs){return cytnx::linalg::Add(lhs,self);})
                
                .def("c__iadd__",[](cytnx::Tensor &self, const cytnx::Tensor &rhs){return self.Add_(rhs);}) // these will return self!
                .def("c__iadd__",[](cytnx::Tensor &self, const cytnx::cytnx_complex128&rhs){return self.Add_(rhs);})
                .def("c__iadd__",[](cytnx::Tensor &self, const cytnx::cytnx_complex64 &rhs){return self.Add_(rhs);})
                .def("c__iadd__",[](cytnx::Tensor &self, const cytnx::cytnx_double    &rhs){return self.Add_(rhs);})
                .def("c__iadd__",[](cytnx::Tensor &self, const cytnx::cytnx_float     &rhs){return self.Add_(rhs);})
                .def("c__iadd__",[](cytnx::Tensor &self, const cytnx::cytnx_int64     &rhs){return self.Add_(rhs);})
                .def("c__iadd__",[](cytnx::Tensor &self, const cytnx::cytnx_uint64    &rhs){return self.Add_(rhs);})
                .def("c__iadd__",[](cytnx::Tensor &self, const cytnx::cytnx_int32     &rhs){return self.Add_(rhs);})
                .def("c__iadd__",[](cytnx::Tensor &self, const cytnx::cytnx_uint32    &rhs){return self.Add_(rhs);})
                .def("c__iadd__",[](cytnx::Tensor &self, const cytnx::cytnx_int16     &rhs){return self.Add_(rhs);})
                .def("c__iadd__",[](cytnx::Tensor &self, const cytnx::cytnx_uint16    &rhs){return self.Add_(rhs);})
                .def("c__iadd__",[](cytnx::Tensor &self, const cytnx::cytnx_bool      &rhs){return self.Add_(rhs);})

                .def("__sub__",[](cytnx::Tensor &self, const cytnx::Tensor &rhs){return self.Sub(rhs);})
                .def("__sub__",[](cytnx::Tensor &self, const cytnx::cytnx_complex128&rhs){return self.Sub(rhs);})
                .def("__sub__",[](cytnx::Tensor &self, const cytnx::cytnx_complex64 &rhs){return self.Sub(rhs);})
                .def("__sub__",[](cytnx::Tensor &self, const cytnx::cytnx_double    &rhs){return self.Sub(rhs);})
                .def("__sub__",[](cytnx::Tensor &self, const cytnx::cytnx_float     &rhs){return self.Sub(rhs);})
                .def("__sub__",[](cytnx::Tensor &self, const cytnx::cytnx_int64     &rhs){return self.Sub(rhs);})
                .def("__sub__",[](cytnx::Tensor &self, const cytnx::cytnx_uint64    &rhs){return self.Sub(rhs);})
                .def("__sub__",[](cytnx::Tensor &self, const cytnx::cytnx_int32     &rhs){return self.Sub(rhs);})
                .def("__sub__",[](cytnx::Tensor &self, const cytnx::cytnx_uint32    &rhs){return self.Sub(rhs);})
                .def("__sub__",[](cytnx::Tensor &self, const cytnx::cytnx_int16     &rhs){return self.Sub(rhs);})
                .def("__sub__",[](cytnx::Tensor &self, const cytnx::cytnx_uint16    &rhs){return self.Sub(rhs);})
                .def("__sub__",[](cytnx::Tensor &self, const cytnx::cytnx_bool      &rhs){return self.Sub(rhs);})

                .def("__rsub__",[](cytnx::Tensor &self, const cytnx::cytnx_complex128&lhs){return cytnx::linalg::Sub(lhs,self);})
                .def("__rsub__",[](cytnx::Tensor &self, const cytnx::cytnx_complex64 &lhs){return cytnx::linalg::Sub(lhs,self);})
                .def("__rsub__",[](cytnx::Tensor &self, const cytnx::cytnx_double    &lhs){return cytnx::linalg::Sub(lhs,self);})
                .def("__rsub__",[](cytnx::Tensor &self, const cytnx::cytnx_float     &lhs){return cytnx::linalg::Sub(lhs,self);})
                .def("__rsub__",[](cytnx::Tensor &self, const cytnx::cytnx_int64     &lhs){return cytnx::linalg::Sub(lhs,self);})
                .def("__rsub__",[](cytnx::Tensor &self, const cytnx::cytnx_uint64    &lhs){return cytnx::linalg::Sub(lhs,self);})
                .def("__rsub__",[](cytnx::Tensor &self, const cytnx::cytnx_int32     &lhs){return cytnx::linalg::Sub(lhs,self);})
                .def("__rsub__",[](cytnx::Tensor &self, const cytnx::cytnx_uint32    &lhs){return cytnx::linalg::Sub(lhs,self);})
                .def("__rsub__",[](cytnx::Tensor &self, const cytnx::cytnx_int16     &lhs){return cytnx::linalg::Sub(lhs,self);})
                .def("__rsub__",[](cytnx::Tensor &self, const cytnx::cytnx_uint16    &lhs){return cytnx::linalg::Sub(lhs,self);})
                .def("__rsub__",[](cytnx::Tensor &self, const cytnx::cytnx_bool      &lhs){return cytnx::linalg::Sub(lhs,self);})
 
                .def("c__isub__",[](cytnx::Tensor &self, const cytnx::Tensor &rhs){return self.Sub_(rhs);}) // these will return self!
                .def("c__isub__",[](cytnx::Tensor &self, const cytnx::cytnx_complex128&rhs){return self.Sub_(rhs);})
                .def("c__isub__",[](cytnx::Tensor &self, const cytnx::cytnx_complex64 &rhs){return self.Sub_(rhs);})
                .def("c__isub__",[](cytnx::Tensor &self, const cytnx::cytnx_double    &rhs){return self.Sub_(rhs);})
                .def("c__isub__",[](cytnx::Tensor &self, const cytnx::cytnx_float     &rhs){return self.Sub_(rhs);})
                .def("c__isub__",[](cytnx::Tensor &self, const cytnx::cytnx_int64     &rhs){return self.Sub_(rhs);})
                .def("c__isub__",[](cytnx::Tensor &self, const cytnx::cytnx_uint64    &rhs){return self.Sub_(rhs);})
                .def("c__isub__",[](cytnx::Tensor &self, const cytnx::cytnx_int32     &rhs){return self.Sub_(rhs);})
                .def("c__isub__",[](cytnx::Tensor &self, const cytnx::cytnx_uint32    &rhs){return self.Sub_(rhs);})
                .def("c__isub__",[](cytnx::Tensor &self, const cytnx::cytnx_int16     &rhs){return self.Sub_(rhs);})
                .def("c__isub__",[](cytnx::Tensor &self, const cytnx::cytnx_uint16    &rhs){return self.Sub_(rhs);})
                .def("c__isub__",[](cytnx::Tensor &self, const cytnx::cytnx_bool      &rhs){return self.Sub_(rhs);})

                .def("__mul__",[](cytnx::Tensor &self, const cytnx::Tensor &rhs){return self.Mul(rhs);})
                .def("__mul__",[](cytnx::Tensor &self, const cytnx::cytnx_complex128&rhs){return self.Mul(rhs);})
                .def("__mul__",[](cytnx::Tensor &self, const cytnx::cytnx_complex64 &rhs){return self.Mul(rhs);})
                .def("__mul__",[](cytnx::Tensor &self, const cytnx::cytnx_double    &rhs){return self.Mul(rhs);})
                .def("__mul__",[](cytnx::Tensor &self, const cytnx::cytnx_float     &rhs){return self.Mul(rhs);})
                .def("__mul__",[](cytnx::Tensor &self, const cytnx::cytnx_int64     &rhs){return self.Mul(rhs);})
                .def("__mul__",[](cytnx::Tensor &self, const cytnx::cytnx_uint64    &rhs){return self.Mul(rhs);})
                .def("__mul__",[](cytnx::Tensor &self, const cytnx::cytnx_int32     &rhs){return self.Mul(rhs);})
                .def("__mul__",[](cytnx::Tensor &self, const cytnx::cytnx_uint32    &rhs){return self.Mul(rhs);})
                .def("__mul__",[](cytnx::Tensor &self, const cytnx::cytnx_int16     &rhs){return self.Mul(rhs);})
                .def("__mul__",[](cytnx::Tensor &self, const cytnx::cytnx_uint16    &rhs){return self.Mul(rhs);})
                .def("__mul__",[](cytnx::Tensor &self, const cytnx::cytnx_bool    &rhs){return self.Mul(rhs);})

                .def("__rmul__",[](cytnx::Tensor &self, const cytnx::cytnx_complex128&lhs){return cytnx::linalg::Mul(lhs,self);})
                .def("__rmul__",[](cytnx::Tensor &self, const cytnx::cytnx_complex64 &lhs){return cytnx::linalg::Mul(lhs,self);})
                .def("__rmul__",[](cytnx::Tensor &self, const cytnx::cytnx_double    &lhs){return cytnx::linalg::Mul(lhs,self);})
                .def("__rmul__",[](cytnx::Tensor &self, const cytnx::cytnx_float     &lhs){return cytnx::linalg::Mul(lhs,self);})
                .def("__rmul__",[](cytnx::Tensor &self, const cytnx::cytnx_int64     &lhs){return cytnx::linalg::Mul(lhs,self);})
                .def("__rmul__",[](cytnx::Tensor &self, const cytnx::cytnx_uint64    &lhs){return cytnx::linalg::Mul(lhs,self);})
                .def("__rmul__",[](cytnx::Tensor &self, const cytnx::cytnx_int32     &lhs){return cytnx::linalg::Mul(lhs,self);})
                .def("__rmul__",[](cytnx::Tensor &self, const cytnx::cytnx_uint32    &lhs){return cytnx::linalg::Mul(lhs,self);})
                .def("__rmul__",[](cytnx::Tensor &self, const cytnx::cytnx_int16     &lhs){return cytnx::linalg::Mul(lhs,self);})
                .def("__rmul__",[](cytnx::Tensor &self, const cytnx::cytnx_uint16    &lhs){return cytnx::linalg::Mul(lhs,self);})
                .def("__rmul__",[](cytnx::Tensor &self, const cytnx::cytnx_bool      &lhs){return cytnx::linalg::Mul(lhs,self);})
 
                .def("c__imul__",[](cytnx::Tensor &self, const cytnx::Tensor &rhs){return self.Mul_(rhs);}) // these will return self!
                .def("c__imul__",[](cytnx::Tensor &self, const cytnx::cytnx_complex128&rhs){return self.Mul_(rhs);})
                .def("c__imul__",[](cytnx::Tensor &self, const cytnx::cytnx_complex64 &rhs){return self.Mul_(rhs);})
                .def("c__imul__",[](cytnx::Tensor &self, const cytnx::cytnx_double    &rhs){return self.Mul_(rhs);})
                .def("c__imul__",[](cytnx::Tensor &self, const cytnx::cytnx_float     &rhs){return self.Mul_(rhs);})
                .def("c__imul__",[](cytnx::Tensor &self, const cytnx::cytnx_int64     &rhs){return self.Mul_(rhs);})
                .def("c__imul__",[](cytnx::Tensor &self, const cytnx::cytnx_uint64    &rhs){return self.Mul_(rhs);})
                .def("c__imul__",[](cytnx::Tensor &self, const cytnx::cytnx_int32     &rhs){return self.Mul_(rhs);})
                .def("c__imul__",[](cytnx::Tensor &self, const cytnx::cytnx_uint32    &rhs){return self.Mul_(rhs);})
                .def("c__imul__",[](cytnx::Tensor &self, const cytnx::cytnx_int16     &rhs){return self.Mul_(rhs);})
                .def("c__imul__",[](cytnx::Tensor &self, const cytnx::cytnx_uint16    &rhs){return self.Mul_(rhs);})
                .def("c__imul__",[](cytnx::Tensor &self, const cytnx::cytnx_bool      &rhs){return self.Mul_(rhs);})

                .def("__truediv__",[](cytnx::Tensor &self, const cytnx::Tensor &rhs){return self.Div(rhs);})
                .def("__truediv__",[](cytnx::Tensor &self, const cytnx::cytnx_complex128&rhs){return self.Div(rhs);})
                .def("__truediv__",[](cytnx::Tensor &self, const cytnx::cytnx_complex64 &rhs){return self.Div(rhs);})
                .def("__truediv__",[](cytnx::Tensor &self, const cytnx::cytnx_double    &rhs){return self.Div(rhs);})
                .def("__truediv__",[](cytnx::Tensor &self, const cytnx::cytnx_float     &rhs){return self.Div(rhs);})
                .def("__truediv__",[](cytnx::Tensor &self, const cytnx::cytnx_int64     &rhs){return self.Div(rhs);})
                .def("__truediv__",[](cytnx::Tensor &self, const cytnx::cytnx_uint64    &rhs){return self.Div(rhs);})
                .def("__truediv__",[](cytnx::Tensor &self, const cytnx::cytnx_int32     &rhs){return self.Div(rhs);})
                .def("__truediv__",[](cytnx::Tensor &self, const cytnx::cytnx_uint32    &rhs){return self.Div(rhs);})
                .def("__truediv__",[](cytnx::Tensor &self, const cytnx::cytnx_int16     &rhs){return self.Div(rhs);})
                .def("__truediv__",[](cytnx::Tensor &self, const cytnx::cytnx_uint16    &rhs){return self.Div(rhs);})
                .def("__truediv__",[](cytnx::Tensor &self, const cytnx::cytnx_bool      &rhs){return self.Div(rhs);})

                .def("__rtruediv__",[](cytnx::Tensor &self, const cytnx::cytnx_complex128&lhs){return cytnx::linalg::Div(lhs,self);})
                .def("__rtruediv__",[](cytnx::Tensor &self, const cytnx::cytnx_complex64 &lhs){return cytnx::linalg::Div(lhs,self);})
                .def("__rtruediv__",[](cytnx::Tensor &self, const cytnx::cytnx_double    &lhs){return cytnx::linalg::Div(lhs,self);})
                .def("__rtruediv__",[](cytnx::Tensor &self, const cytnx::cytnx_float     &lhs){return cytnx::linalg::Div(lhs,self);})
                .def("__rtruediv__",[](cytnx::Tensor &self, const cytnx::cytnx_int64     &lhs){return cytnx::linalg::Div(lhs,self);})
                .def("__rtruediv__",[](cytnx::Tensor &self, const cytnx::cytnx_uint64    &lhs){return cytnx::linalg::Div(lhs,self);})
                .def("__rtruediv__",[](cytnx::Tensor &self, const cytnx::cytnx_int32     &lhs){return cytnx::linalg::Div(lhs,self);})
                .def("__rtruediv__",[](cytnx::Tensor &self, const cytnx::cytnx_uint32    &lhs){return cytnx::linalg::Div(lhs,self);})
                .def("__rtruediv__",[](cytnx::Tensor &self, const cytnx::cytnx_int16     &lhs){return cytnx::linalg::Div(lhs,self);})
                .def("__rtruediv__",[](cytnx::Tensor &self, const cytnx::cytnx_uint16    &lhs){return cytnx::linalg::Div(lhs,self);})
                .def("__rtruediv__",[](cytnx::Tensor &self, const cytnx::cytnx_bool      &lhs){return cytnx::linalg::Div(lhs,self);})
 
                .def("c__itruediv__",[](cytnx::Tensor &self, const cytnx::Tensor &rhs){return self.Div_(rhs);}) // these will return self!
                .def("c__itruediv__",[](cytnx::Tensor &self, const cytnx::cytnx_complex128&rhs){return self.Div_(rhs);})
                .def("c__itruediv__",[](cytnx::Tensor &self, const cytnx::cytnx_complex64 &rhs){return self.Div_(rhs);})
                .def("c__itruediv__",[](cytnx::Tensor &self, const cytnx::cytnx_double    &rhs){return self.Div_(rhs);})
                .def("c__itruediv__",[](cytnx::Tensor &self, const cytnx::cytnx_float     &rhs){return self.Div_(rhs);})
                .def("c__itruediv__",[](cytnx::Tensor &self, const cytnx::cytnx_int64     &rhs){return self.Div_(rhs);})
                .def("c__itruediv__",[](cytnx::Tensor &self, const cytnx::cytnx_uint64    &rhs){return self.Div_(rhs);})
                .def("c__itruediv__",[](cytnx::Tensor &self, const cytnx::cytnx_int32     &rhs){return self.Div_(rhs);})
                .def("c__itruediv__",[](cytnx::Tensor &self, const cytnx::cytnx_uint32    &rhs){return self.Div_(rhs);})
                .def("c__itruediv__",[](cytnx::Tensor &self, const cytnx::cytnx_int16     &rhs){return self.Div_(rhs);})
                .def("c__itruediv__",[](cytnx::Tensor &self, const cytnx::cytnx_uint16    &rhs){return self.Div_(rhs);})
                .def("c__itruediv__",[](cytnx::Tensor &self, const cytnx::cytnx_bool      &rhs){return self.Div_(rhs);})

                .def("__floordiv__",[](cytnx::Tensor &self, const cytnx::Tensor &rhs){return self.Div(rhs);})
                .def("__floordiv__",[](cytnx::Tensor &self, const cytnx::cytnx_complex128&rhs){return self.Div(rhs);})
                .def("__floordiv__",[](cytnx::Tensor &self, const cytnx::cytnx_complex64 &rhs){return self.Div(rhs);})
                .def("__floordiv__",[](cytnx::Tensor &self, const cytnx::cytnx_double    &rhs){return self.Div(rhs);})
                .def("__floordiv__",[](cytnx::Tensor &self, const cytnx::cytnx_float     &rhs){return self.Div(rhs);})
                .def("__floordiv__",[](cytnx::Tensor &self, const cytnx::cytnx_int64     &rhs){return self.Div(rhs);})
                .def("__floordiv__",[](cytnx::Tensor &self, const cytnx::cytnx_uint64    &rhs){return self.Div(rhs);})
                .def("__floordiv__",[](cytnx::Tensor &self, const cytnx::cytnx_int32     &rhs){return self.Div(rhs);})
                .def("__floordiv__",[](cytnx::Tensor &self, const cytnx::cytnx_uint32    &rhs){return self.Div(rhs);})
                .def("__floordiv__",[](cytnx::Tensor &self, const cytnx::cytnx_int16     &rhs){return self.Div(rhs);})
                .def("__floordiv__",[](cytnx::Tensor &self, const cytnx::cytnx_uint16    &rhs){return self.Div(rhs);})
                .def("__floordiv__",[](cytnx::Tensor &self, const cytnx::cytnx_bool      &rhs){return self.Div(rhs);})

                .def("__rfloordiv__",[](cytnx::Tensor &self, const cytnx::cytnx_complex128&lhs){return cytnx::linalg::Div(lhs,self);})
                .def("__rfloordiv__",[](cytnx::Tensor &self, const cytnx::cytnx_complex64 &lhs){return cytnx::linalg::Div(lhs,self);})
                .def("__rfloordiv__",[](cytnx::Tensor &self, const cytnx::cytnx_double    &lhs){return cytnx::linalg::Div(lhs,self);})
                .def("__rfloordiv__",[](cytnx::Tensor &self, const cytnx::cytnx_float     &lhs){return cytnx::linalg::Div(lhs,self);})
                .def("__rfloordiv__",[](cytnx::Tensor &self, const cytnx::cytnx_int64     &lhs){return cytnx::linalg::Div(lhs,self);})
                .def("__rfloordiv__",[](cytnx::Tensor &self, const cytnx::cytnx_uint64    &lhs){return cytnx::linalg::Div(lhs,self);})
                .def("__rfloordiv__",[](cytnx::Tensor &self, const cytnx::cytnx_int32     &lhs){return cytnx::linalg::Div(lhs,self);})
                .def("__rfloordiv__",[](cytnx::Tensor &self, const cytnx::cytnx_uint32    &lhs){return cytnx::linalg::Div(lhs,self);})
                .def("__rfloordiv__",[](cytnx::Tensor &self, const cytnx::cytnx_int16     &lhs){return cytnx::linalg::Div(lhs,self);})
                .def("__rfloordiv__",[](cytnx::Tensor &self, const cytnx::cytnx_uint16    &lhs){return cytnx::linalg::Div(lhs,self);})
                .def("__rfloordiv__",[](cytnx::Tensor &self, const cytnx::cytnx_bool      &lhs){return cytnx::linalg::Div(lhs,self);})
 
                .def("c__ifloordiv__",[](cytnx::Tensor &self, const cytnx::Tensor &rhs){return self.Div_(rhs);}) // these will return self!
                .def("c__ifloordiv__",[](cytnx::Tensor &self, const cytnx::cytnx_complex128&rhs){return self.Div_(rhs);})
                .def("c__ifloordiv__",[](cytnx::Tensor &self, const cytnx::cytnx_complex64 &rhs){return self.Div_(rhs);})
                .def("c__ifloordiv__",[](cytnx::Tensor &self, const cytnx::cytnx_double    &rhs){return self.Div_(rhs);})
                .def("c__ifloordiv__",[](cytnx::Tensor &self, const cytnx::cytnx_float     &rhs){return self.Div_(rhs);})
                .def("c__ifloordiv__",[](cytnx::Tensor &self, const cytnx::cytnx_int64     &rhs){return self.Div_(rhs);})
                .def("c__ifloordiv__",[](cytnx::Tensor &self, const cytnx::cytnx_uint64    &rhs){return self.Div_(rhs);})
                .def("c__ifloordiv__",[](cytnx::Tensor &self, const cytnx::cytnx_int32     &rhs){return self.Div_(rhs);})
                .def("c__ifloordiv__",[](cytnx::Tensor &self, const cytnx::cytnx_uint32    &rhs){return self.Div_(rhs);})
                .def("c__ifloordiv__",[](cytnx::Tensor &self, const cytnx::cytnx_int16     &rhs){return self.Div_(rhs);})
                .def("c__ifloordiv__",[](cytnx::Tensor &self, const cytnx::cytnx_uint16    &rhs){return self.Div_(rhs);})
                .def("c__ifloordiv__",[](cytnx::Tensor &self, const cytnx::cytnx_bool      &rhs){return self.Div_(rhs);})

                .def("__eq__",[](cytnx::Tensor &self, const cytnx::Tensor &rhs){return self == rhs;})
                .def("__eq__",[](cytnx::Tensor &self, const cytnx::cytnx_complex128&rhs){return self == rhs;})
                .def("__eq__",[](cytnx::Tensor &self, const cytnx::cytnx_complex64 &rhs){return self == rhs;})
                .def("__eq__",[](cytnx::Tensor &self, const cytnx::cytnx_double    &rhs){return self == rhs;})
                .def("__eq__",[](cytnx::Tensor &self, const cytnx::cytnx_float     &rhs){return self == rhs;})
                .def("__eq__",[](cytnx::Tensor &self, const cytnx::cytnx_int64     &rhs){return self == rhs;})
                .def("__eq__",[](cytnx::Tensor &self, const cytnx::cytnx_uint64    &rhs){return self == rhs;})
                .def("__eq__",[](cytnx::Tensor &self, const cytnx::cytnx_int32     &rhs){return self == rhs;})
                .def("__eq__",[](cytnx::Tensor &self, const cytnx::cytnx_uint32    &rhs){return self == rhs;})
                .def("__eq__",[](cytnx::Tensor &self, const cytnx::cytnx_int16     &rhs){return self == rhs;})
                .def("__eq__",[](cytnx::Tensor &self, const cytnx::cytnx_uint16    &rhs){return self == rhs;})
                .def("__eq__",[](cytnx::Tensor &self, const cytnx::cytnx_bool      &rhs){return self == rhs;})

                .def("__pow__",[](cytnx::Tensor &self, const cytnx::cytnx_double      &p){return cytnx::linalg::Pow(self,p);})
                .def("c__ipow__",[](cytnx::Tensor &self, const cytnx::cytnx_double      &p){cytnx::linalg::Pow_(self,p);})
                .def("__matmul__",[](cytnx::Tensor &self, const cytnx::Tensor &rhs){return cytnx::linalg::Dot(self,rhs);})
                .def("c__imatmul__",[](cytnx::Tensor &self, const cytnx::Tensor &rhs){self = cytnx::linalg::Dot(self,rhs); return self;})

                //linalg >>
                .def("Svd",&cytnx::Tensor::Svd, py::arg("is_U"), py::arg("is_vT"))
                .def("Eigh",&cytnx::Tensor::Eigh, py::arg("is_V")=true,py::arg("row_v")=false)
                .def("cInvM_",&cytnx::Tensor::InvM_)
                .def("InvM",&cytnx::Tensor::InvM)
                .def("cInv_",&cytnx::Tensor::Inv_,py::arg("clip"))
                .def("Inv",&cytnx::Tensor::Inv,py::arg("clip"))
                .def("cConj_",&cytnx::Tensor::Conj_)
                .def("Conj",&cytnx::Tensor::Conj)
                .def("cExp_",&cytnx::Tensor::Exp_)
                .def("Exp",&cytnx::Tensor::Exp)
                .def("Pow",&cytnx::Tensor::Pow)
                .def("cPow_",&cytnx::Tensor::Pow_)
                .def("Abs",&cytnx::Tensor::Abs)
                .def("cAbs_",&cytnx::Tensor::Abs_)
                .def("Max",&cytnx::Tensor::Max)
                .def("Min",&cytnx::Tensor::Min)
                .def("Norm",&cytnx::Tensor::Norm)
                .def("Trace",&cytnx::Tensor::Trace)
                ;


    auto mext = m.def_submodule("cytnx_extension_c");
    py::enum_<cytnx_extension::__sym::__stype>(mext,"SymType")
        .value("Z",cytnx_extension::__sym::__stype::Z)
        .value("U",cytnx_extension::__sym::__stype::U)
        .export_values();

    py::enum_<cytnx_extension::__ntwk::__nttype>(mext,"NtType")
        .value("Regular",cytnx_extension::__ntwk::__nttype::Regular)
        .value("Fermion",cytnx_extension::__ntwk::__nttype::Fermion)
        .value("Void"   ,cytnx_extension::__ntwk::__nttype::Void)
        .export_values();
    
    py::enum_<cytnx_extension::bondType>(mext,"bondType")
        .value("BD_BRA", cytnx_extension::bondType::BD_BRA)
        .value("BD_KET", cytnx_extension::bondType::BD_KET)
        .value("BD_REG", cytnx_extension::bondType::BD_REG)
        .export_values();



    py::class_<cytnx_extension::Network>(mext,"Network")
                .def(py::init<>())
                .def(py::init<const std::string &, const int&>(),py::arg("fname"),py::arg("network_type")=(int)cytnx_extension::NtType.Regular)
                .def("_cget_tn_names",[](cytnx_extension::Network &self){
                    return self._impl->names;
                })
                .def("_cget_tn_labels",[](cytnx_extension::Network &self){
                    return self._impl->label_arr;
                })
                .def("_cget_tn_out_labels",[](cytnx_extension::Network &self){
                    return self._impl->TOUT_labels;
                })
                .def("isLoad",[](cytnx_extension::Network &self)->bool{
                    return self._impl->tensors.size()==0?false:true;
                })
                .def("isAllset",[](cytnx_extension::Network &self)->bool{
                    bool out = true;
                    for(int i=0;i<self._impl->tensors.size();i++){
                        if(self._impl->tensors[i].uten_type()==cytnx_extension::UTenType.Void)
                            out = false;
                    }
                    return out;
                })
                .def("_cget_filename",[](cytnx_extension::Network &self){
                    return self._impl->filename;
                })
                .def("Fromfile",&cytnx_extension::Network::Fromfile,py::arg("fname"),py::arg("network_type")=(int)cytnx_extension::NtType.Regular)
                .def("Savefile",&cytnx_extension::Network::Savefile,py::arg("fname"))
                .def("PutCyTensor",[](cytnx_extension::Network &self,const std::string &name, const cytnx_extension::CyTensor &utensor, const bool &is_clone){
                                                self.PutCyTensor(name,utensor,is_clone);
                                        },py::arg("name"),py::arg("utensor"),py::arg("is_clone")=true)
                .def("PutCyTensor",[](cytnx_extension::Network &self,const cytnx_uint64 &idx, const cytnx_extension::CyTensor &utensor, const bool &is_clone){
                                                self.PutCyTensor(idx,utensor,is_clone);
                                        },py::arg("idx"),py::arg("utensor"),py::arg("is_clone")=true)
                .def("PutCyTensors",[](cytnx_extension::Network &self,const std::vector<std::string> &names, const std::vector<cytnx_extension::CyTensor> &utensors, const bool &is_clone){
                                                self.PutCyTensors(names,utensors,is_clone);
                                        },py::arg("names"),py::arg("utensors"),py::arg("is_clone")=true)
                .def("Launch",&cytnx_extension::Network::Launch,py::arg("optimal")=false)
                .def("clear",&cytnx_extension::Network::clear)
                .def("clone",&cytnx_extension::Network::clone)
                .def("__copy__",&cytnx_extension::Network::clone)
                .def("__deepcopy__",&cytnx_extension::Network::clone)
                .def("__repr__",[](cytnx_extension::Network &self)->std::string{
                    self.PrintNet();
                    return std::string("");
                 },py::call_guard<py::scoped_ostream_redirect,
                     py::scoped_estream_redirect>())
                .def("PrintNet",&cytnx_extension::Network::PrintNet)
                ;



    py::class_<cytnx_extension::Symmetry>(mext,"Symmetry")
                //construction
                .def(py::init<>())
                //.def(py::init<const int &, const int&>())
                .def("U1",&cytnx_extension::Symmetry::U1)
                .def("Zn",&cytnx_extension::Symmetry::Zn)
                .def("clone",&cytnx_extension::Symmetry::clone)
                .def("stype", &cytnx_extension::Symmetry::stype)
                .def("stype_str", &cytnx_extension::Symmetry::stype_str)
                .def("n",&cytnx_extension::Symmetry::n)
                .def("clone",&cytnx_extension::Symmetry::clone)
                .def("__copy__",&cytnx_extension::Symmetry::clone)
                .def("__deepcopy__",&cytnx_extension::Symmetry::clone)
                .def("__eq__",&cytnx_extension::Symmetry::operator==)
                .def("Save",[](cytnx_extension::Symmetry &self, const std::string &fname){self.Save(fname);},py::arg("fname"))
                .def_static("Load",[](const std::string &fname){return cytnx_extension::Symmetry::Load(fname);},py::arg("fname"))
                //.def("combine_rule",&cytnx_extension::Symmetry::combine_rule,py::arg("qnums_1"),py::arg("qnums_2"))
                //.def("combine_rule_",&cytnx_extension::Symmetry::combine_rule_,py::arg("qnums_l"),py::arg("qnums_r"))
                //.def("check_qnum", &cytnx_extension::Symmetry::check_qnum,py::arg("qnum"))
                //.def("check_qnums", &cytnx_extension::Symmetry::check_qnums, py::arg("qnums"))
                ;
    py::class_<cytnx_extension::Bond>(mext,"Bond")
            //construction
            .def(py::init<>())
            .def(py::init<const cytnx_uint64 &, const cytnx_extension::bondType &, const std::vector<std::vector<cytnx_int64> > &, const std::vector<cytnx_extension::Symmetry>& >(),py::arg("dim"),py::arg("bond_type")=cytnx_extension::bondType::BD_REG,py::arg("qnums")=std::vector<std::vector<cytnx_int64> >(),py::arg("symmetries")=std::vector<cytnx_extension::Symmetry>())
            .def("Init",&cytnx_extension::Bond::Init,py::arg("dim"),py::arg("bond_type")=cytnx_extension::bondType::BD_REG,py::arg("qnums")=std::vector<std::vector<cytnx_int64> >(),py::arg("symmetries")=std::vector<cytnx_extension::Symmetry>())

            .def("__repr__",[](cytnx_extension::Bond &self){
                std::cout << self << std::endl;
                return std::string("");
             },py::call_guard<py::scoped_ostream_redirect,
                     py::scoped_estream_redirect>())
            .def("__eq__",&cytnx_extension::Bond::operator==)
            .def("type",&cytnx_extension::Bond::type)
            .def("qnums",[](cytnx_extension::Bond &self){return self.qnums();})
            .def("qnums_clone",[](cytnx_extension::Bond &self){return self.qnums_clone();})
            .def("dim", &cytnx_extension::Bond::dim)
            .def("Nsym", &cytnx_extension::Bond::Nsym)
            .def("syms", [](cytnx_extension::Bond &self){return self.syms();})
            .def("syms_clone", [](cytnx_extension::Bond &self){return self.syms_clone();})
            .def("set_type", &cytnx_extension::Bond::set_type)
            .def("clear_type", &cytnx_extension::Bond::clear_type)
            .def("clone", &cytnx_extension::Bond::clone)
            .def("__copy__",&cytnx_extension::Bond::clone)
            .def("__deepcopy__",&cytnx_extension::Bond::clone)
            .def("combineBond", &cytnx_extension::Bond::combineBond)
            .def("combineBond_",&cytnx_extension::Bond::combineBond_)
            .def("combineBonds",&cytnx_extension::Bond::combineBonds)
            .def("combineBonds_", &cytnx_extension::Bond::combineBonds_)
            .def("Save",[](cytnx_extension::Bond &self, const std::string &fname){self.Save(fname);},py::arg("fname"))
            .def_static("Load",[](const std::string &fname){return cytnx_extension::Bond::Load(fname);},py::arg("fname"))
            ;

    //entry.CyTenor
    py::class_<cytnx_extension::CyTensor>(mext,"CyTensor")
                .def(py::init<>())
                .def(py::init<const cytnx::Tensor&, const cytnx_uint64&, const bool &>(),py::arg("Tin"),py::arg("rowrank"),py::arg("is_diag")=false)
                .def(py::init<const std::vector<cytnx_extension::Bond> &, const std::vector<cytnx_int64> &, const cytnx_int64 &, const unsigned int &,const int &, const bool &>(),py::arg("bonds"),py::arg("labels")=std::vector<cytnx_int64>(),py::arg("rowrank")=(cytnx_int64)(-1),py::arg("dtype")=(unsigned int)(cytnx::Type.Double),py::arg("device")=(int)cytnx::Device.cpu,py::arg("is_diag")=false)
                .def("set_name",&cytnx_extension::CyTensor::set_name)
                .def("set_label",&cytnx_extension::CyTensor::set_label,py::arg("idx"),py::arg("new_label"))
                .def("set_labels",&cytnx_extension::CyTensor::set_labels,py::arg("new_labels"))
                .def("set_rowrank",&cytnx_extension::CyTensor::set_rowrank, py::arg("new_rowrank"))

                .def("rowrank",&cytnx_extension::CyTensor::rowrank)
                .def("dtype",&cytnx_extension::CyTensor::dtype)
                .def("dtype_str",&cytnx_extension::CyTensor::dtype_str)
                .def("device",&cytnx_extension::CyTensor::device)
                .def("device_str",&cytnx_extension::CyTensor::device_str)
                .def("name",&cytnx_extension::CyTensor::name)

                .def("reshape",[](cytnx_extension::CyTensor &self, py::args args, py::kwargs kwargs)->cytnx_extension::CyTensor{
                    std::vector<cytnx::cytnx_int64> c_args = args.cast< std::vector<cytnx::cytnx_int64> >();
                    cytnx_uint64 rowrank = 0;
                   
                    if(kwargs){
                        if(kwargs.contains("rowrank")) rowrank = kwargs["rowrank"].cast<cytnx::cytnx_int64>();
                    }
 
                    return self.reshape(c_args,rowrank);
                })
                .def("reshape_",[](cytnx_extension::CyTensor &self, py::args args, py::kwargs kwargs){
                    std::vector<cytnx::cytnx_int64> c_args = args.cast< std::vector<cytnx::cytnx_int64> >();
                    cytnx_uint64 rowrank = 0;
                   
                    if(kwargs){
                        if(kwargs.contains("rowrank")) rowrank = kwargs["rowrank"].cast<cytnx::cytnx_int64>();
                    }
 
                    self.reshape_(c_args,rowrank);
                })
                .def("elem_exists",&cytnx_extension::CyTensor::elem_exists)
                .def("item",[](cytnx_extension::CyTensor &self){
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
                    else cytnx_error_msg(true, "%s","[ERROR] try to get element from a empty CyTensor.");
                    return out;
                 })

                .def("__getitem__",[](const cytnx_extension::CyTensor &self, py::object locators){
                    cytnx_error_msg(self.shape().size() == 0, "[ERROR] try to getitem from a empty CyTensor%s","\n");
                    cytnx_error_msg(self.uten_type() == cytnx_extension::UTenType.Sparse,"[ERROR] cannot get element using [] from SparseCyTensor. Use at() instead.%s","\n");
 
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
                .def("__setitem__",[](cytnx_extension::CyTensor &self, py::object locators, const cytnx::Tensor &rhs){
                    cytnx_error_msg(self.shape().size() == 0, "[ERROR] try to setelem to a empty CyTensor%s","\n");
                    cytnx_error_msg(self.uten_type() == cytnx_extension::UTenType.Sparse,"[ERROR] cannot set element using [] from SparseCyTensor. Use at() instead.%s","\n");
                    
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
                .def("get_elem",[](cytnx_extension::CyTensor &self, const std::vector<cytnx_uint64> &locator){
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
                .def("set_elem",&f_CyTensor_setelem_scal_cd)
                .def("set_elem",&f_CyTensor_setelem_scal_cf)
                .def("set_elem",&f_CyTensor_setelem_scal_d)
                .def("set_elem",&f_CyTensor_setelem_scal_f)
                .def("set_elem",&f_CyTensor_setelem_scal_u64)
                .def("set_elem",&f_CyTensor_setelem_scal_i64)
                .def("set_elem",&f_CyTensor_setelem_scal_u32)
                .def("set_elem",&f_CyTensor_setelem_scal_i32)
                .def("set_elem",&f_CyTensor_setelem_scal_u16)
                .def("set_elem",&f_CyTensor_setelem_scal_i16)
                .def("set_elem",&f_CyTensor_setelem_scal_b)


                .def("is_contiguous", &cytnx_extension::CyTensor::is_contiguous)
                .def("is_diag",&cytnx_extension::CyTensor::is_diag)
                .def("is_tag" ,&cytnx_extension::CyTensor::is_tag)
                .def("is_braket_form",&cytnx_extension::CyTensor::is_braket_form)
                .def("labels",&cytnx_extension::CyTensor::labels)
                .def("bonds",[](cytnx_extension::CyTensor &self){
                    return self.bonds();
                    })
                .def("shape",&cytnx_extension::CyTensor::shape)
                .def("to_",&cytnx_extension::CyTensor::to_)
                .def("to_different_device" ,[](cytnx_extension::CyTensor &self,const cytnx_int64 &device){
                                                    cytnx_error_msg(self.device() == device, "[ERROR][pybind][to_diffferent_device] same device for to() should be handle in python side.%s","\n");
                                                    return self.to(device);
                                                } , py::arg("device"))
                .def("clone",&cytnx_extension::CyTensor::clone)
                .def("__copy__",&cytnx_extension::CyTensor::clone)
                .def("__deepcopy__",&cytnx_extension::CyTensor::clone)
                .def("Save",[](cytnx_extension::CyTensor &self, const std::string &fname){self.Save(fname);},py::arg("fname"))
                .def_static("Load",[](const std::string &fname){return cytnx_extension::CyTensor::Load(fname);},py::arg("fname"))
                //.def("permute",&cytnx_extension::CyTensor::permute,py::arg("mapper"),py::arg("rowrank")=(cytnx_int64)-1,py::arg("by_label")=false)
                //.def("permute_",&cytnx_extension::CyTensor::permute_,py::arg("mapper"),py::arg("rowrank")=(cytnx_int64)-1,py::arg("by_label")=false)

                .def("permute_",[](cytnx_extension::CyTensor &self, const std::vector<cytnx::cytnx_int64> &c_args, py::kwargs kwargs){
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
                .def("permute",[](cytnx_extension::CyTensor &self,const std::vector<cytnx::cytnx_int64> &c_args, py::kwargs kwargs)->cytnx_extension::CyTensor{
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

                .def("make_contiguous",&cytnx_extension::CyTensor::contiguous)
                .def("contiguous_",&cytnx_extension::CyTensor::contiguous_)
                .def("print_diagram",&cytnx_extension::CyTensor::print_diagram,py::arg("bond_info")=false,py::call_guard<py::scoped_ostream_redirect,
                     py::scoped_estream_redirect>())
                        
                .def("get_block", [](const cytnx_extension::CyTensor &self, const cytnx_uint64&idx){
                                        return self.get_block(idx);
                                  },py::arg("idx")=(cytnx_uint64)(0))

                .def("get_block", [](const cytnx_extension::CyTensor &self, const std::vector<cytnx_int64>&qnum){
                                        return self.get_block(qnum);
                                  },py::arg("qnum"))

                .def("get_block_",[](const cytnx_extension::CyTensor &self, const std::vector<cytnx_int64>&qnum){
                                        return self.get_block_(qnum);
                                  },py::arg("qnum"))
                .def("get_block_",[](cytnx_extension::CyTensor &self, const std::vector<cytnx_int64>&qnum){
                                        return self.get_block_(qnum);
                                  },py::arg("qnum"))
                .def("get_block_", [](const cytnx_extension::CyTensor &self, const cytnx_uint64&idx){
                                        return self.get_block_(idx);
                                  },py::arg("idx")=(cytnx_uint64)(0))
                .def("get_block_", [](cytnx_extension::CyTensor &self, const cytnx_uint64&idx){
                                        return self.get_block_(idx);
                                  },py::arg("idx")=(cytnx_uint64)(0))
                .def("get_blocks", [](const cytnx_extension::CyTensor &self){
                                        return self.get_blocks();
                                  })
                .def("get_blocks_", [](const cytnx_extension::CyTensor &self){
                                        return self.get_blocks_();
                                  })
                .def("get_blocks_", [](cytnx_extension::CyTensor &self){
                                        return self.get_blocks_();
                                  })
                .def("put_block", [](cytnx_extension::CyTensor &self, const cytnx::Tensor &in, const cytnx_uint64&idx){
                                        self.put_block(in,idx);
                                  },py::arg("in"),py::arg("idx")=(cytnx_uint64)(0))

                .def("put_block", [](cytnx_extension::CyTensor &self, const cytnx::Tensor &in, const std::vector<cytnx_int64>&qnum){
                                        self.put_block(in,qnum);
                                  },py::arg("in"),py::arg("qnum"))
                .def("put_block_", [](cytnx_extension::CyTensor &self, cytnx::Tensor &in, const cytnx_uint64&idx){
                                        self.put_block_(in,idx);
                                  },py::arg("in"),py::arg("idx")=(cytnx_uint64)(0))

                .def("put_block_", [](cytnx_extension::CyTensor &self, cytnx::Tensor &in, const std::vector<cytnx_int64>&qnum){
                                        self.put_block_(in,qnum);
                                  },py::arg("in"),py::arg("qnum"))
                .def("__repr__",[](cytnx_extension::CyTensor &self)->std::string{
                    std::cout << self << std::endl;
                    return std::string("");
                 },py::call_guard<py::scoped_ostream_redirect,
                     py::scoped_estream_redirect>()) 
                .def("to_dense",&cytnx_extension::CyTensor::to_dense)
                .def("to_dense_",&cytnx_extension::CyTensor::to_dense_)
                .def("combineBonds",&cytnx_extension::CyTensor::combineBonds,py::arg("indicators"),py::arg("permute_back")=true,py::arg("by_label")=true)
                .def("contract", &cytnx_extension::CyTensor::contract)
		
        		//arithmetic >>
                .def("__neg__",[](cytnx_extension::CyTensor &self){
                                    if(self.dtype() == Type.Double){
                                        return cytnx_extension::xlinalg::Mul(cytnx_double(-1),self);
                                    }else if(self.dtype()==Type.ComplexDouble){
                                        return cytnx_extension::xlinalg::Mul(cytnx_complex128(-1,0),self);
                                    }else if(self.dtype()==Type.Float){
                                        return cytnx_extension::xlinalg::Mul(cytnx_float(-1),self);
                                    }else if(self.dtype()==Type.ComplexFloat){
                                        return cytnx_extension::xlinalg::Mul(cytnx_complex64(-1,0),self);
                                    }else{
                                        return cytnx_extension::xlinalg::Mul(-1,self);
                                    }
                                  })
                .def("__pos__",[](cytnx_extension::CyTensor &self){return self;})
                .def("__add__",[](cytnx_extension::CyTensor &self, const cytnx_extension::CyTensor &rhs){return self.Add(rhs);})
                .def("__add__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_complex128&rhs){return self.Add(rhs);})
                .def("__add__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_complex64 &rhs){return self.Add(rhs);})
                .def("__add__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_double    &rhs){return self.Add(rhs);})
                .def("__add__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_float     &rhs){return self.Add(rhs);})
                .def("__add__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_int64     &rhs){return self.Add(rhs);})
                .def("__add__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_uint64    &rhs){return self.Add(rhs);})
                .def("__add__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_int32     &rhs){return self.Add(rhs);})
                .def("__add__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_uint32    &rhs){return self.Add(rhs);})
                .def("__add__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_int16     &rhs){return self.Add(rhs);})
                .def("__add__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_uint16    &rhs){return self.Add(rhs);})
                .def("__add__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_bool      &rhs){return self.Add(rhs);})
                
                .def("__radd__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_complex128&lhs){return cytnx_extension::xlinalg::Add(lhs,self);})
                .def("__radd__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_complex64 &lhs){return cytnx_extension::xlinalg::Add(lhs,self);})
                .def("__radd__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_double    &lhs){return cytnx_extension::xlinalg::Add(lhs,self);})
                .def("__radd__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_float     &lhs){return cytnx_extension::xlinalg::Add(lhs,self);})
                .def("__radd__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_int64     &lhs){return cytnx_extension::xlinalg::Add(lhs,self);})
                .def("__radd__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_uint64    &lhs){return cytnx_extension::xlinalg::Add(lhs,self);})
                .def("__radd__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_int32     &lhs){return cytnx_extension::xlinalg::Add(lhs,self);})
                .def("__radd__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_uint32    &lhs){return cytnx_extension::xlinalg::Add(lhs,self);})
                .def("__radd__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_int16     &lhs){return cytnx_extension::xlinalg::Add(lhs,self);})
                .def("__radd__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_uint16    &lhs){return cytnx_extension::xlinalg::Add(lhs,self);})
                .def("__radd__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_bool      &lhs){return cytnx_extension::xlinalg::Add(lhs,self);})
                
                .def("__iadd__",[](cytnx_extension::CyTensor &self, const cytnx_extension::CyTensor &rhs){return self.Add_(rhs);}) // these will return self!
                .def("__iadd__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_complex128&rhs){return self.Add_(rhs);})
                .def("__iadd__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_complex64 &rhs){return self.Add_(rhs);})
                .def("__iadd__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_double    &rhs){return self.Add_(rhs);})
                .def("__iadd__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_float     &rhs){return self.Add_(rhs);})
                .def("__iadd__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_int64     &rhs){return self.Add_(rhs);})
                .def("__iadd__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_uint64    &rhs){return self.Add_(rhs);})
                .def("__iadd__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_int32     &rhs){return self.Add_(rhs);})
                .def("__iadd__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_uint32    &rhs){return self.Add_(rhs);})
                .def("__iadd__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_int16     &rhs){return self.Add_(rhs);})
                .def("__iadd__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_uint16    &rhs){return self.Add_(rhs);})
                .def("__iadd__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_bool      &rhs){return self.Add_(rhs);})

                .def("__sub__",[](cytnx_extension::CyTensor &self, const cytnx_extension::CyTensor &rhs){return self.Sub(rhs);})
                .def("__sub__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_complex128&rhs){return self.Sub(rhs);})
                .def("__sub__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_complex64 &rhs){return self.Sub(rhs);})
                .def("__sub__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_double    &rhs){return self.Sub(rhs);})
                .def("__sub__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_float     &rhs){return self.Sub(rhs);})
                .def("__sub__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_int64     &rhs){return self.Sub(rhs);})
                .def("__sub__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_uint64    &rhs){return self.Sub(rhs);})
                .def("__sub__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_int32     &rhs){return self.Sub(rhs);})
                .def("__sub__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_uint32    &rhs){return self.Sub(rhs);})
                .def("__sub__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_int16     &rhs){return self.Sub(rhs);})
                .def("__sub__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_uint16    &rhs){return self.Sub(rhs);})
                .def("__sub__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_bool      &rhs){return self.Sub(rhs);})

                .def("__rsub__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_complex128&lhs){return cytnx_extension::xlinalg::Sub(lhs,self);})
                .def("__rsub__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_complex64 &lhs){return cytnx_extension::xlinalg::Sub(lhs,self);})
                .def("__rsub__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_double    &lhs){return cytnx_extension::xlinalg::Sub(lhs,self);})
                .def("__rsub__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_float     &lhs){return cytnx_extension::xlinalg::Sub(lhs,self);})
                .def("__rsub__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_int64     &lhs){return cytnx_extension::xlinalg::Sub(lhs,self);})
                .def("__rsub__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_uint64    &lhs){return cytnx_extension::xlinalg::Sub(lhs,self);})
                .def("__rsub__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_int32     &lhs){return cytnx_extension::xlinalg::Sub(lhs,self);})
                .def("__rsub__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_uint32    &lhs){return cytnx_extension::xlinalg::Sub(lhs,self);})
                .def("__rsub__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_int16     &lhs){return cytnx_extension::xlinalg::Sub(lhs,self);})
                .def("__rsub__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_uint16    &lhs){return cytnx_extension::xlinalg::Sub(lhs,self);})
                .def("__rsub__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_bool      &lhs){return cytnx_extension::xlinalg::Sub(lhs,self);})
 
                .def("__isub__",[](cytnx_extension::CyTensor &self, const cytnx_extension::CyTensor &rhs){return self.Sub_(rhs);}) // these will return self!
                .def("__isub__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_complex128&rhs){return self.Sub_(rhs);})
                .def("__isub__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_complex64 &rhs){return self.Sub_(rhs);})
                .def("__isub__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_double    &rhs){return self.Sub_(rhs);})
                .def("__isub__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_float     &rhs){return self.Sub_(rhs);})
                .def("__isub__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_int64     &rhs){return self.Sub_(rhs);})
                .def("__isub__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_uint64    &rhs){return self.Sub_(rhs);})
                .def("__isub__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_int32     &rhs){return self.Sub_(rhs);})
                .def("__isub__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_uint32    &rhs){return self.Sub_(rhs);})
                .def("__isub__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_int16     &rhs){return self.Sub_(rhs);})
                .def("__isub__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_uint16    &rhs){return self.Sub_(rhs);})
                .def("__isub__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_bool      &rhs){return self.Sub_(rhs);})

                .def("__mul__",[](cytnx_extension::CyTensor &self, const cytnx_extension::CyTensor &rhs){return self.Mul(rhs);})
                .def("__mul__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_complex128&rhs){return self.Mul(rhs);})
                .def("__mul__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_complex64 &rhs){return self.Mul(rhs);})
                .def("__mul__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_double    &rhs){return self.Mul(rhs);})
                .def("__mul__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_float     &rhs){return self.Mul(rhs);})
                .def("__mul__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_int64     &rhs){return self.Mul(rhs);})
                .def("__mul__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_uint64    &rhs){return self.Mul(rhs);})
                .def("__mul__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_int32     &rhs){return self.Mul(rhs);})
                .def("__mul__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_uint32    &rhs){return self.Mul(rhs);})
                .def("__mul__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_int16     &rhs){return self.Mul(rhs);})
                .def("__mul__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_uint16    &rhs){return self.Mul(rhs);})
                .def("__mul__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_bool    &rhs){return self.Mul(rhs);})

                .def("__rmul__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_complex128&lhs){return cytnx_extension::xlinalg::Mul(lhs,self);})
                .def("__rmul__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_complex64 &lhs){return cytnx_extension::xlinalg::Mul(lhs,self);})
                .def("__rmul__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_double    &lhs){return cytnx_extension::xlinalg::Mul(lhs,self);})
                .def("__rmul__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_float     &lhs){return cytnx_extension::xlinalg::Mul(lhs,self);})
                .def("__rmul__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_int64     &lhs){return cytnx_extension::xlinalg::Mul(lhs,self);})
                .def("__rmul__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_uint64    &lhs){return cytnx_extension::xlinalg::Mul(lhs,self);})
                .def("__rmul__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_int32     &lhs){return cytnx_extension::xlinalg::Mul(lhs,self);})
                .def("__rmul__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_uint32    &lhs){return cytnx_extension::xlinalg::Mul(lhs,self);})
                .def("__rmul__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_int16     &lhs){return cytnx_extension::xlinalg::Mul(lhs,self);})
                .def("__rmul__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_uint16    &lhs){return cytnx_extension::xlinalg::Mul(lhs,self);})
                .def("__rmul__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_bool      &lhs){return cytnx_extension::xlinalg::Mul(lhs,self);})
 
                .def("__imul__",[](cytnx_extension::CyTensor &self, const cytnx_extension::CyTensor &rhs){return self.Mul_(rhs);}) // these will return self!
                .def("__imul__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_complex128&rhs){return self.Mul_(rhs);})
                .def("__imul__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_complex64 &rhs){return self.Mul_(rhs);})
                .def("__imul__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_double    &rhs){return self.Mul_(rhs);})
                .def("__imul__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_float     &rhs){return self.Mul_(rhs);})
                .def("__imul__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_int64     &rhs){return self.Mul_(rhs);})
                .def("__imul__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_uint64    &rhs){return self.Mul_(rhs);})
                .def("__imul__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_int32     &rhs){return self.Mul_(rhs);})
                .def("__imul__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_uint32    &rhs){return self.Mul_(rhs);})
                .def("__imul__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_int16     &rhs){return self.Mul_(rhs);})
                .def("__imul__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_uint16    &rhs){return self.Mul_(rhs);})
                .def("__imul__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_bool      &rhs){return self.Mul_(rhs);})

                .def("__truediv__",[](cytnx_extension::CyTensor &self, const cytnx_extension::CyTensor &rhs){return self.Div(rhs);})
                .def("__truediv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_complex128&rhs){return self.Div(rhs);})
                .def("__truediv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_complex64 &rhs){return self.Div(rhs);})
                .def("__truediv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_double    &rhs){return self.Div(rhs);})
                .def("__truediv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_float     &rhs){return self.Div(rhs);})
                .def("__truediv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_int64     &rhs){return self.Div(rhs);})
                .def("__truediv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_uint64    &rhs){return self.Div(rhs);})
                .def("__truediv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_int32     &rhs){return self.Div(rhs);})
                .def("__truediv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_uint32    &rhs){return self.Div(rhs);})
                .def("__truediv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_int16     &rhs){return self.Div(rhs);})
                .def("__truediv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_uint16    &rhs){return self.Div(rhs);})
                .def("__truediv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_bool      &rhs){return self.Div(rhs);})

                .def("__rtruediv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_complex128&lhs){return cytnx_extension::xlinalg::Div(lhs,self);})
                .def("__rtruediv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_complex64 &lhs){return cytnx_extension::xlinalg::Div(lhs,self);})
                .def("__rtruediv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_double    &lhs){return cytnx_extension::xlinalg::Div(lhs,self);})
                .def("__rtruediv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_float     &lhs){return cytnx_extension::xlinalg::Div(lhs,self);})
                .def("__rtruediv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_int64     &lhs){return cytnx_extension::xlinalg::Div(lhs,self);})
                .def("__rtruediv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_uint64    &lhs){return cytnx_extension::xlinalg::Div(lhs,self);})
                .def("__rtruediv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_int32     &lhs){return cytnx_extension::xlinalg::Div(lhs,self);})
                .def("__rtruediv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_uint32    &lhs){return cytnx_extension::xlinalg::Div(lhs,self);})
                .def("__rtruediv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_int16     &lhs){return cytnx_extension::xlinalg::Div(lhs,self);})
                .def("__rtruediv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_uint16    &lhs){return cytnx_extension::xlinalg::Div(lhs,self);})
                .def("__rtruediv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_bool      &lhs){return cytnx_extension::xlinalg::Div(lhs,self);})
 
                .def("__itruediv__",[](cytnx_extension::CyTensor &self, const cytnx_extension::CyTensor &rhs){return self.Div_(rhs);}) // these will return self!
                .def("__itruediv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_complex128&rhs){return self.Div_(rhs);})
                .def("__itruediv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_complex64 &rhs){return self.Div_(rhs);})
                .def("__itruediv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_double    &rhs){return self.Div_(rhs);})
                .def("__itruediv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_float     &rhs){return self.Div_(rhs);})
                .def("__itruediv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_int64     &rhs){return self.Div_(rhs);})
                .def("__itruediv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_uint64    &rhs){return self.Div_(rhs);})
                .def("__itruediv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_int32     &rhs){return self.Div_(rhs);})
                .def("__itruediv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_uint32    &rhs){return self.Div_(rhs);})
                .def("__itruediv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_int16     &rhs){return self.Div_(rhs);})
                .def("__itruediv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_uint16    &rhs){return self.Div_(rhs);})
                .def("__itruediv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_bool      &rhs){return self.Div_(rhs);})

                .def("__floordiv__",[](cytnx_extension::CyTensor &self, const cytnx_extension::CyTensor &rhs){return self.Div(rhs);})
                .def("__floordiv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_complex128&rhs){return self.Div(rhs);})
                .def("__floordiv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_complex64 &rhs){return self.Div(rhs);})
                .def("__floordiv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_double    &rhs){return self.Div(rhs);})
                .def("__floordiv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_float     &rhs){return self.Div(rhs);})
                .def("__floordiv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_int64     &rhs){return self.Div(rhs);})
                .def("__floordiv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_uint64    &rhs){return self.Div(rhs);})
                .def("__floordiv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_int32     &rhs){return self.Div(rhs);})
                .def("__floordiv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_uint32    &rhs){return self.Div(rhs);})
                .def("__floordiv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_int16     &rhs){return self.Div(rhs);})
                .def("__floordiv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_uint16    &rhs){return self.Div(rhs);})
                .def("__floordiv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_bool      &rhs){return self.Div(rhs);})

                .def("__rfloordiv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_complex128&lhs){return cytnx_extension::xlinalg::Div(lhs,self);})
                .def("__rfloordiv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_complex64 &lhs){return cytnx_extension::xlinalg::Div(lhs,self);})
                .def("__rfloordiv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_double    &lhs){return cytnx_extension::xlinalg::Div(lhs,self);})
                .def("__rfloordiv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_float     &lhs){return cytnx_extension::xlinalg::Div(lhs,self);})
                .def("__rfloordiv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_int64     &lhs){return cytnx_extension::xlinalg::Div(lhs,self);})
                .def("__rfloordiv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_uint64    &lhs){return cytnx_extension::xlinalg::Div(lhs,self);})
                .def("__rfloordiv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_int32     &lhs){return cytnx_extension::xlinalg::Div(lhs,self);})
                .def("__rfloordiv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_uint32    &lhs){return cytnx_extension::xlinalg::Div(lhs,self);})
                .def("__rfloordiv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_int16     &lhs){return cytnx_extension::xlinalg::Div(lhs,self);})
                .def("__rfloordiv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_uint16    &lhs){return cytnx_extension::xlinalg::Div(lhs,self);})
                .def("__rfloordiv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_bool      &lhs){return cytnx_extension::xlinalg::Div(lhs,self);})
 
                .def("__ifloordiv__",[](cytnx_extension::CyTensor &self, const cytnx_extension::CyTensor &rhs){return self.Div_(rhs);}) // these will return self!
                .def("__ifloordiv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_complex128&rhs){return self.Div_(rhs);})
                .def("__ifloordiv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_complex64 &rhs){return self.Div_(rhs);})
                .def("__ifloordiv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_double    &rhs){return self.Div_(rhs);})
                .def("__ifloordiv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_float     &rhs){return self.Div_(rhs);})
                .def("__ifloordiv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_int64     &rhs){return self.Div_(rhs);})
                .def("__ifloordiv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_uint64    &rhs){return self.Div_(rhs);})
                .def("__ifloordiv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_int32     &rhs){return self.Div_(rhs);})
                .def("__ifloordiv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_uint32    &rhs){return self.Div_(rhs);})
                .def("__ifloordiv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_int16     &rhs){return self.Div_(rhs);})
                .def("__ifloordiv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_uint16    &rhs){return self.Div_(rhs);})
                .def("__ifloordiv__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_bool      &rhs){return self.Div_(rhs);})
                .def("__pow__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_double &p){return self.Pow(p);})
                .def("c__ipow__",[](cytnx_extension::CyTensor &self, const cytnx::cytnx_double &p){self.Pow_(p);})
                .def("Pow",&cytnx_extension::CyTensor::Pow)
                .def("cPow_",&cytnx_extension::CyTensor::Pow_)
                .def("cConj_",&cytnx_extension::CyTensor::Conj_)
                .def("Conj",&cytnx_extension::CyTensor::Conj)
                .def("cTrace_",&cytnx_extension::CyTensor::Trace_,py::arg("a"),py::arg("b"),py::arg("by_label")=false)
                .def("Trace",&cytnx_extension::CyTensor::Trace,py::arg("a"),py::arg("b"),py::arg("by_label")=false)
                .def("cTranspose_",&cytnx_extension::CyTensor::Transpose_)
                .def("Transpose",&cytnx_extension::CyTensor::Transpose)
                .def("cDagger_",&cytnx_extension::CyTensor::Dagger_)
                .def("Dagger",&cytnx_extension::CyTensor::Dagger)
                .def("ctag",&cytnx_extension::CyTensor::tag)
                .def("truncate",&cytnx_extension::CyTensor::truncate,py::arg("bond_idx"),py::arg("dim"),py::arg("by_label")=false)
                .def("ctruncate_",&cytnx_extension::CyTensor::truncate_)
                ;
    mext.def("Contract",cytnx_extension::Contract);
   
    // [Submodule linalg]
    pybind11::module mext_xlinalg = mext.def_submodule("xlinalg","linear algebra for cytnx_extension.");
    mext_xlinalg.def("Svd",&cytnx_extension::xlinalg::Svd,py::arg("Tin"),py::arg("is_U")=true,py::arg("is_vT")=true);
    mext_xlinalg.def("Svd_truncate",&cytnx_extension::xlinalg::Svd_truncate,py::arg("Tin"),py::arg("keepdim"),py::arg("is_U")=true,py::arg("is_vT")=true);
    mext_xlinalg.def("ExpH",&cytnx_extension::xlinalg::ExpH,py::arg("Tin"),py::arg("a")=1.,py::arg("b")=0.);
    mext_xlinalg.def("ExpM",&cytnx_extension::xlinalg::ExpM,py::arg("Tin"),py::arg("a")=1.,py::arg("b")=0.);
    mext_xlinalg.def("Trace",&cytnx_extension::xlinalg::Trace,py::arg("Tin"),py::arg("a"),py::arg("b"),py::arg("by_label")=false);
    mext_xlinalg.def("Hosvd",&cytnx_extension::xlinalg::Hosvd, py::arg("Tin"),py::arg("mode"),py::arg("is_core")=true,py::arg("is_Ls")=false,py::arg("truncate_dim")=std::vector<cytnx_int64>());
    mext_xlinalg.def("Pow",&cytnx_extension::xlinalg::Pow,py::arg("Tin"),py::arg("p"));
    mext_xlinalg.def("Pow_",&cytnx_extension::xlinalg::Pow_,py::arg("Tin"),py::arg("p"));
    mext_xlinalg.def("QR",&cytnx_extension::xlinalg::QR,py::arg("Tin"),py::arg("is_tau")=false);

    // [Submodule linalg] 
    pybind11::module m_linalg = m.def_submodule("linalg","linear algebra related.");

    m_linalg.def("Svd",&cytnx::linalg::Svd,py::arg("Tin"),py::arg("is_U")=true,py::arg("is_vT")=true);
    m_linalg.def("Eigh",&cytnx::linalg::Eigh,py::arg("Tin"),py::arg("is_V")=true,py::arg("row_v")=false);
    m_linalg.def("Eig",&cytnx::linalg::Eig,py::arg("Tin"),py::arg("is_V")=true,py::arg("row_v")=false);
    m_linalg.def("Exp",&cytnx::linalg::Exp,py::arg("Tin"));
    m_linalg.def("Exp_",&cytnx::linalg::Exp_,py::arg("Tio"));
    m_linalg.def("Expf_",&cytnx::linalg::Expf_,py::arg("Tio"));
    m_linalg.def("Expf",&cytnx::linalg::Expf,py::arg("Tio"));
    m_linalg.def("ExpH",&cytnx::linalg::ExpH,py::arg("Tio"),py::arg("a")=1.0,py::arg("b")=0);
    m_linalg.def("ExpM",&cytnx::linalg::ExpM,py::arg("Tio"),py::arg("a")=1.0,py::arg("b")=0);
    m_linalg.def("QR",&cytnx::linalg::QR,py::arg("Tio"),py::arg("is_tau")=false);
    m_linalg.def("InvM",&cytnx::linalg::InvM,py::arg("Tin"));
    m_linalg.def("InvM_",&cytnx::linalg::InvM_,py::arg("Tio"));
    m_linalg.def("Inv_",&cytnx::linalg::Inv_,py::arg("Tio"),py::arg("clip"));
    m_linalg.def("Inv",&cytnx::linalg::Inv,py::arg("Tio"),py::arg("clip"));
    m_linalg.def("Conj",&cytnx::linalg::Conj,py::arg("Tin"));
    m_linalg.def("Conj_",&cytnx::linalg::Conj_,py::arg("Tio"));
    m_linalg.def("Matmul",&cytnx::linalg::Matmul,py::arg("T1"),py::arg("T2"));
    m_linalg.def("Diag",&cytnx::linalg::Diag, py::arg("Tin"));
    m_linalg.def("Det",&cytnx::linalg::Det, py::arg("Tin"));
    m_linalg.def("Tensordot",&cytnx::linalg::Tensordot, py::arg("T1"),py::arg("T2"),py::arg("indices_1"),py::arg("indices_2"));
    m_linalg.def("Outer",&cytnx::linalg::Outer, py::arg("T1"),py::arg("T2"));
    m_linalg.def("Kron",&cytnx::linalg::Kron, py::arg("T1"),py::arg("T2"),py::arg("Tl_pad_left")=false,py::arg("Tr_pad_left")=false);
    m_linalg.def("Vectordot",&cytnx::linalg::Vectordot, py::arg("T1"),py::arg("T2"),py::arg("is_conj")=false);
    m_linalg.def("Norm",&cytnx::linalg::Norm, py::arg("T1"));
    m_linalg.def("Dot",&cytnx::linalg::Dot, py::arg("T1"),py::arg("T2"));
    m_linalg.def("Trace",&cytnx::linalg::Trace, py::arg("Tn"),py::arg("axisA")=0,py::arg("axisB")=1);
    m_linalg.def("Pow",&cytnx::linalg::Pow, py::arg("Tn"),py::arg("p"));
    m_linalg.def("Pow_",&cytnx::linalg::Pow_, py::arg("Tn"),py::arg("p"));
    m_linalg.def("Abs",&cytnx::linalg::Abs, py::arg("Tn"));
    m_linalg.def("Abs_",&cytnx::linalg::Abs_, py::arg("Tn"));
    m_linalg.def("Max",&cytnx::linalg::Max, py::arg("Tn"));
    m_linalg.def("Min",&cytnx::linalg::Min, py::arg("Tn"));
    m_linalg.def("Sum",&cytnx::linalg::Sum, py::arg("Tn"));
    m_linalg.def("Hosvd",&cytnx::linalg::Hosvd, py::arg("Tn"),py::arg("mode"),py::arg("is_core")=true,py::arg("is_Ls")=false,py::arg("truncate_dim")=std::vector<cytnx_int64>());
    m_linalg.def("c_Lanczos_ER",[](LinOp *Hop, const cytnx_uint64&k, const bool &is_V, const cytnx_uint64 &maxiter, const double &CvgCrit, const bool &is_row, const Tensor &Tin, const cytnx_uint32 &max_krydim){
                                    return cytnx::linalg::Lanczos_ER(Hop,k,is_V,maxiter,CvgCrit,is_row,Tin,max_krydim);
                                });

    // [Submodule physics]
    pybind11::module m_physics = m.def_submodule("physics","physics related.");
    m_physics.def("spin",[](const cytnx_double &S, const std::string &Comp, const int &device)->Tensor{
                                return cytnx::physics::spin(S,Comp,device);
                            },py::arg("S"),py::arg("Comp"),py::arg("device")=(int)cytnx::Device.cpu);
    m_physics.def("pauli",[](const std::string &Comp, const int &device)->Tensor{
                                return cytnx::physics::pauli(Comp,device);
                            } ,py::arg("Comp"),py::arg("device")=(int)cytnx::Device.cpu);

    // [Submodule random]
    pybind11::module m_random = m.def_submodule("random","random related.");
   
    m_random.def("Make_normal", [](cytnx::Tensor &Tin, const double &mean, const double &std, const long long &seed){
                                            cytnx::random::Make_normal(Tin,mean,std,seed);
                                  },py::arg("Tin"),py::arg("mean"),py::arg("std"),py::arg("seed")=std::random_device()());

    m_random.def("Make_normal", [](cytnx::Storage &Sin, const double &mean, const double &std, const long long &seed){
                                            cytnx::random::Make_normal(Sin,mean,std,seed);
                                  },py::arg("Sin"),py::arg("mean"),py::arg("std"),py::arg("seed")=std::random_device()());
    
    m_random.def("Make_uniform", [](cytnx::Tensor &Tin, const double &low, const double &high, const long long &seed){
                                            cytnx::random::Make_uniform(Tin,low,high,seed);
                                  },py::arg("Tin"),py::arg("low")=double(0),py::arg("high")=double(1.0),py::arg("seed")=std::random_device()());

    m_random.def("Make_uniform", [](cytnx::Storage &Sin, const double &low, const double &high, const long long &seed){
                                            cytnx::random::Make_uniform(Sin,low,high,seed);
                                  },py::arg("Sin"),py::arg("low")=double(0),py::arg("high")=double(1.0),py::arg("seed")=std::random_device()());


    
    m_random.def("normal", [](const cytnx_uint64& Nelem,const double &mean, const double &std, const int&device, const unsigned int &seed){
                                   return cytnx::random::normal(Nelem,mean,std,device,seed);
                                  },py::arg("Nelem"),py::arg("mean"),py::arg("std"),py::arg("device")=-1,py::arg("seed")=std::random_device()());
    m_random.def("normal", [](const std::vector<cytnx_uint64>& Nelem,const double &mean, const double &std, const int&device, const unsigned int &seed){
                                   return cytnx::random::normal(Nelem,mean,std,device,seed);
                                  },py::arg("Nelem"),py::arg("mean"),py::arg("std"),py::arg("device")=-1,py::arg("seed")=std::random_device()());
    m_random.def("uniform", [](const cytnx_uint64& Nelem,const double &low, const double &high, const int&device, const unsigned int &seed){
                                   return cytnx::random::uniform(Nelem,low,high,device,seed);
                                  },py::arg("Nelem"),py::arg("low"),py::arg("high"),py::arg("device")=-1,py::arg("seed")=std::random_device()());
    m_random.def("uniform", [](const std::vector<cytnx_uint64>& Nelem,const double &low, const double &high, const int&device, const unsigned int &seed){
                                   return cytnx::random::uniform(Nelem,low,high,device,seed);
                                  },py::arg("Nelem"),py::arg("low"),py::arg("high"),py::arg("device")=-1,py::arg("seed")=std::random_device()());

}

