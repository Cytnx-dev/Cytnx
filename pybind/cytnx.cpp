#include <vector>
#include <map>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/buffer_info.h>

#include "../include/cytnx.hpp"
//#include "../include/cytnx_error.hpp"

namespace py = pybind11;
using namespace pybind11::literals;


PYBIND11_MODULE(cytnx,m){

    m.attr("__version__") = "0.0.0";

    //global vars
    //m.attr("cytnxdevice") = cytnx::cytnxdevice;
    //m.attr("Type")   = py::cast(cytnx::Type);    

    py::enum_<cytnx::__type::__pybind_type>(m,"Type")
        .value("ComplexDouble", cytnx::__type::__pybind_type::ComplexDouble)
		.value("ComplexFloat", cytnx::__type::__pybind_type::ComplexFloat )	
        .value("Double", cytnx::__type::__pybind_type::Double)
		.value("Float", cytnx::__type::__pybind_type::Float  )	
        .value("Uint64", cytnx::__type::__pybind_type::Uint64)
		.value("Int64", cytnx::__type::__pybind_type::Int64  ) 	
        .value("Uint32", cytnx::__type::__pybind_type::Uint32)
		.value("Int32", cytnx::__type::__pybind_type::Int32  ) 	
		.export_values();

    py::enum_<cytnx::__device::__pybind_device>(m,"Device")
        .value("cpu", cytnx::__device::__pybind_device::cpu)
		.value("cuda", cytnx::__device::__pybind_device::cuda)	
		.export_values();


    py::class_<cytnx::Storage>(m,"Storage")
                //construction
                .def(py::init<>())
                .def(py::init<const cytnx::Storage&>())
                .def(py::init<boost::intrusive_ptr<cytnx::Storage_base> >())
                .def(py::init<const unsigned long long &, const unsigned int&, int>(),py::arg("size"), py::arg("dtype"),py::arg("device")=-1)

                .def_property_readonly("dtype",&cytnx::Storage::dtype)
                .def_property_readonly("dtype_str",&cytnx::Storage::dtype_str)
                .def_property_readonly("device",&cytnx::Storage::device)
                .def_property_readonly("dtype_str",&cytnx::Storage::device_str)


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
                    else cytnx_error_msg(true, "%s","[ERROR] try to get element from a void Storage.");
                 });


}

