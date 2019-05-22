#include <vector>
#include <map>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/buffer_info.h>

#include "../include/tor10.hpp"


namespace py = pybind11;
using namespace pybind11::literals;


PYBIND11_MODULE(pytor10,m){
    m.attr("__version__") = "0.0.0";

    py::class_<tor10::Storage>(m,"Storage")
                //construction
                .def(py::init<>())
                .def(py::init<const tor10::Storage&>())
                .def(py::init<boost::intrusive_ptr<tor10::Storage_base> >())
                .def(py::init<const unsigned long long &, const unsigned int&, int>(),py::arg("size"), py::arg("dtype"),py::arg("device")=-1)
                .def_property_readonly("dtype",&tor10::Storage::dtype)
                .def_property_readonly("dtype_str",&tor10::Storage::dtype_str);
}

