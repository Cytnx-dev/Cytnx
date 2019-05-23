#include <vector>
#include <map>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/buffer_info.h>

#include "../include/tor10.hpp"
//#include "../include/tor10_error.hpp"

namespace py = pybind11;
using namespace pybind11::literals;


PYBIND11_MODULE(pytor10,m){

    m.attr("__version__") = "0.0.0";

    //global vars
    //m.attr("tor10device") = tor10::tor10device;
    //m.attr("tor10type")   = py::cast(tor10::tor10type);    

    py::enum_<tor10::__type::__pybind_type>(m,"tor10type")
        .value("ComplexDouble", tor10::__type::__pybind_type::ComplexDouble)
		.value("ComplexFloat", tor10::__type::__pybind_type::ComplexFloat )	
        .value("Double", tor10::__type::__pybind_type::Double)
		.value("Float", tor10::__type::__pybind_type::Float  )	
        .value("Uint64", tor10::__type::__pybind_type::Uint64)
		.value("Int64", tor10::__type::__pybind_type::Int64  ) 	
        .value("Uint32", tor10::__type::__pybind_type::Uint32)
		.value("Int32", tor10::__type::__pybind_type::Int32  ) 	
		.export_values();


    py::class_<tor10::Storage>(m,"Storage")
                //construction
                .def(py::init<>())
                .def(py::init<const tor10::Storage&>())
                .def(py::init<boost::intrusive_ptr<tor10::Storage_base> >())
                .def(py::init<const unsigned long long &, const unsigned int&, int>(),py::arg("size"), py::arg("dtype"),py::arg("device")=-1)

                .def_property_readonly("dtype",&tor10::Storage::dtype)
                .def_property_readonly("dtype_str",&tor10::Storage::dtype_str)

                .def("__getitem__",[](tor10::Storage &self, const unsigned long long &idx){
                    tor10_error_msg(idx > self.size(),"idx exceed the size of storage.%s","\n");
                    py::object out;
                    if(self.dtype() == tor10::tor10type.Double) 
                        out =  py::cast(self.at<tor10::tor10_double>(idx));
                    else if(self.dtype() == tor10::tor10type.Float) 
                        out = py::cast(self.at<tor10::tor10_float>(idx));
                    else if(self.dtype() == tor10::tor10type.ComplexDouble) 
                        out = py::cast(self.at<tor10::tor10_complex128>(idx));
                    else if(self.dtype() == tor10::tor10type.ComplexFloat) 
                        out = py::cast(self.at<tor10::tor10_complex64>(idx));
                    else if(self.dtype() == tor10::tor10type.Uint64) 
                        out = py::cast(self.at<tor10::tor10_uint64>(idx));
                    else if(self.dtype() == tor10::tor10type.Int64) 
                        out = py::cast(self.at<tor10::tor10_int64>(idx));
                    else if(self.dtype() == tor10::tor10type.Uint32) 
                        out = py::cast(self.at<tor10::tor10_uint32>(idx));
                    else if(self.dtype() == tor10::tor10type.Int32) 
                        out = py::cast(self.at<tor10::tor10_int32>(idx));
                    else tor10_error_msg(true, "%s","[ERROR] try to get element from a void Storage.");

                    return out;
                 })
                .def("__setitem__",[](tor10::Storage &self, const unsigned long long &idx, py::object in){
                    tor10_error_msg(idx > self.size(),"idx exceed the size of storage.%s","\n");
                    py::object out;
                    if(self.dtype() == tor10::tor10type.Double) 
                        self.at<tor10::tor10_double>(idx) = in.cast<tor10::tor10_double>();
                    else if(self.dtype() == tor10::tor10type.Float) 
                        self.at<tor10::tor10_float>(idx) = in.cast<tor10::tor10_float>();
                    else if(self.dtype() == tor10::tor10type.ComplexDouble) 
                        self.at<tor10::tor10_complex128>(idx) = in.cast<tor10::tor10_complex128>();
                    else if(self.dtype() == tor10::tor10type.ComplexFloat) 
                        self.at<tor10::tor10_complex64>(idx) = in.cast<tor10::tor10_complex64>();
                    else if(self.dtype() == tor10::tor10type.Uint64) 
                        self.at<tor10::tor10_uint64>(idx) = in.cast<tor10::tor10_uint64>();
                    else if(self.dtype() == tor10::tor10type.Int64) 
                        self.at<tor10::tor10_int64>(idx) = in.cast<tor10::tor10_int64>();
                    else if(self.dtype() == tor10::tor10type.Uint32) 
                        self.at<tor10::tor10_uint32>(idx) = in.cast<tor10::tor10_uint32>();
                    else if(self.dtype() == tor10::tor10type.Int32) 
                        self.at<tor10::tor10_int32>(idx) = in.cast<tor10::tor10_int32>();
                    else tor10_error_msg(true, "%s","[ERROR] try to get element from a void Storage.");
                 });


}

