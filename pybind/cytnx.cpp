#include <vector>
#include <map>
#include <random>

  #include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/buffer_info.h>

#include "cytnx.hpp"
//#include "../include/cytnx_error.hpp"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace cytnx;

//ref: https://developer.lsst.io/v/DM-9089/coding/python_wrappers_for_cpp_with_pybind11.html
//ref: https://pybind11.readthedocs.io/en/stable/advanced/classes.html
//ref: https://block.arch.ethz.ch/blog/2016/07/adding-methods-to-python-classes/
//ref: https://medium.com/@mgarod/dynamically-add-a-method-to-a-class-in-python-c49204b85bd6



template<class T>
void f_Tensor_setitem_scal(cytnx::Tensor &self, py::object locators, const T &rc){
    cytnx_error_msg(self.shape().size() == 0, "[ERROR] try to setelem to a empty Tensor%s","\n");
    
    size_t start, stop, step, slicelength; 
    std::vector<cytnx::Accessor> accessors;
    if(py::isinstance<py::tuple>(locators)){
        py::tuple Args = locators.cast<py::tuple>();
        // mixing of slice and ints
        for(cytnx_uint32 axis=0;axis<self.shape().size();axis++){
            if(axis >= Args.size()){accessors.push_back(Accessor::all());}
            else{ 
                // check type:
                if(py::isinstance<py::slice>(Args[axis])){
                    py::slice sls = Args[axis].cast<py::slice>();
                    if(!sls.compute(self.shape()[axis],&start,&stop,&step, &slicelength))
                        throw py::error_already_set();
                    if(slicelength == self.shape()[axis]) accessors.push_back(cytnx::Accessor::all());
                    else accessors.push_back(cytnx::Accessor::range(start,stop,step));
                }else{
                    accessors.push_back(cytnx::Accessor(Args[axis].cast<cytnx_int64>()));
                }
            }
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


PYBIND11_MODULE(cytnx,m){

    m.attr("__version__") = "0.4.0";

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

    m.attr("Device") = py::module::import("enum").attr("IntEnum")
        ("Device", py::dict("cpu"_a=(cytnx_int64)cytnx::Device.cpu, "cuda"_a=(cytnx_int64)cytnx::Device.cuda)); 





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

    m.def("arange",[](const cytnx_uint64 &Nelem, const unsigned int &dtype, const int &device)->Tensor{
                        return cytnx::arange(Nelem,dtype,device);
                  },py::arg("size"),py::arg("dtype")=(unsigned int)(cytnx::Type.Double), py::arg("device")=(int)(cytnx::Device.cpu));

    m.def("arange",[](const cytnx_double &start, const cytnx_double &end, const cytnx_double &step, const unsigned int &dtype, const int &device)->Tensor{
                        return cytnx::arange(start,end,step,dtype,device);
                  },py::arg("start"),py::arg("end"),py::arg("step") = double(1), py::arg("dtype")=(unsigned int)(cytnx::Type.Double), py::arg("device")=(int)(cytnx::Device.cpu));


    m.def("from_numpy", [](py::buffer b)-> Tensor{
        py::buffer_info info = b.request();

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
        Tensor m;
        m.Init(shape,dtype);
        memcpy(m.storage()._impl->Mem,info.ptr,Totbytes);
        return m;
     });
 


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
                .def("Load",[](cytnx::Storage &self, const std::string &fname){self.Load(fname);},py::arg("fname"))
                .def("real",&cytnx::Storage::real)
                .def("imag",&cytnx::Storage::imag)
                ;

    py::class_<cytnx::Tensor>(m,"Tensor",py::buffer_protocol())
                .def_buffer([](Tensor &m) -> py::buffer_info
                {
                    int dtype = m.dtype();
                    if(dtype == Type.Void) cytnx_error_msg(true,"[ERROR] Void Type Tensor cannot convert to numpy ndarray%s","\n");

                    //calculate stride:
                    std::vector<ssize_t> stride(m.shape().size());
                    std::vector<ssize_t> shape(m.shape().begin(),m.shape().end());
                    ssize_t accu = 1;
                    for(int i=shape.size()-1;i>=0;i--){
                        stride[i] = accu;
                        accu*=shape[i];
                    }

                    if(dtype==Type.ComplexDouble){ //--------------------------------
                        for(int i=0;i<shape.size();i++){
                            stride[i]*=sizeof(cytnx_complex128);
                        }
                        return py::buffer_info(
                                    m.storage()._impl->Mem,    //ptr
                                    sizeof(cytnx_complex128), //size of elem 
                                    py::format_descriptor<std::complex<double> >::format(), //pss format
                                    m.rank(),    //rank 
                                    shape,       // shape
                                    stride      // stride
                               );
                    }else if(dtype==Type.ComplexFloat){ //--------------------------------
                        for(int i=0;i<shape.size();i++){
                            stride[i]*=sizeof(cytnx_complex64);
                        }
                        return py::buffer_info(
                                    m.storage()._impl->Mem,    //ptr
                                    sizeof(cytnx_complex64), //size of elem 
                                    py::format_descriptor<std::complex<float> >::format(), //pss format
                                    m.rank(),    //rank 
                                    shape,       // shape
                                    stride      // stride
                               );
                    }else if(dtype==Type.Double){ //--------------------------------
                        for(int i=0;i<shape.size();i++){
                            stride[i]*=sizeof(cytnx_double);
                        }
                        return py::buffer_info(
                                    m.storage()._impl->Mem,    //ptr
                                    sizeof(cytnx_double), //size of elem 
                                    py::format_descriptor<double>::format(), //pss format
                                    m.rank(),    //rank 
                                    shape,       // shape
                                    stride      // stride
                               );
                    }else if(dtype==Type.Float){ //--------------------------------
                        for(int i=0;i<shape.size();i++){
                            stride[i]*=sizeof(cytnx_float);
                        }
                        return py::buffer_info(
                                    m.storage()._impl->Mem,    //ptr
                                    sizeof(cytnx_float), //size of elem 
                                    py::format_descriptor<float>::format(), //pss format
                                    m.rank(),    //rank 
                                    shape,       // shape
                                    stride      // stride
                               );
                    }else if(dtype==Type.Uint64){ //--------------------------------
                        for(int i=0;i<shape.size();i++){
                            stride[i]*=sizeof(cytnx_uint64);
                        }
                        return py::buffer_info(
                                    m.storage()._impl->Mem,    //ptr
                                    sizeof(cytnx_uint64), //size of elem 
                                    py::format_descriptor<uint64_t>::format(), //pss format
                                    m.rank(),    //rank 
                                    shape,       // shape
                                    stride      // stride
                               );
                    }else if(dtype==Type.Int64){ //--------------------------------
                        for(int i=0;i<shape.size();i++){
                            stride[i]*=sizeof(cytnx_int64);
                        }
                        return py::buffer_info(
                                    m.storage()._impl->Mem,    //ptr
                                    sizeof(cytnx_int64), //size of elem 
                                    py::format_descriptor<int64_t>::format(), //pss format
                                    m.rank(),    //rank 
                                    shape,       // shape
                                    stride      // stride
                               );
                    }else if(dtype==Type.Uint32){ //--------------------------------
                        for(int i=0;i<shape.size();i++){
                            stride[i]*=sizeof(cytnx_uint32);
                        }
                        return py::buffer_info( 
                                    m.storage()._impl->Mem,    //ptr
                                    sizeof(cytnx_uint32), //size of elem 
                                    py::format_descriptor<uint32_t>::format(), //pss format
                                    m.rank(),    //rank 
                                    shape,       // shape
                                    stride      // stride
                               );
                    }else if(dtype==Type.Int32){ //--------------------------------
                        for(int i=0;i<shape.size();i++){
                            stride[i]*=sizeof(cytnx_int32);
                        }
                        return py::buffer_info(
                                    m.storage()._impl->Mem,    //ptr
                                    sizeof(cytnx_int32), //size of elem 
                                    py::format_descriptor<int32_t>::format(), //pss format
                                    m.rank(),    //rank 
                                    shape,       // shape
                                    stride      // stride
                               );
                    }else if(dtype==Type.Bool){ //--------------------------------
                        for(int i=0;i<shape.size();i++){
                            stride[i]*=sizeof(cytnx_bool);
                        }
                        return py::buffer_info(
                                    m.storage()._impl->Mem,    //ptr
                                    sizeof(cytnx_bool), //size of elem 
                                    py::format_descriptor<cytnx_bool>::format(), //pss format
                                    m.rank(),    //rank 
                                    shape,       // shape
                                    stride      // stride
                               );
                    }else{
                        cytnx_error_msg(true,"[ERROR] Void Type Tensor cannot convert to numpy ndarray%s","\n");
                    }


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
                .def("make_contiguous",&cytnx::Tensor::contiguous)
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

                .def("Save",[](cytnx::Tensor &self, const std::string &fname){self.Save(fname);},py::arg("fname"))
                .def("Load",[](cytnx::Tensor &self, const std::string &fname){self.Load(fname);},py::arg("fname"))


                .def("__getitem__",[](const cytnx::Tensor &self, py::object locators){
                    cytnx_error_msg(self.shape().size() == 0, "[ERROR] try to getitem from a empty Tensor%s","\n");
                    
                    size_t start, stop, step, slicelength; 
                    std::vector<cytnx::Accessor> accessors;
                    if(py::isinstance<py::tuple>(locators)){
                        py::tuple Args = locators.cast<py::tuple>();
                        // mixing of slice and ints
                        for(cytnx_uint32 axis=0;axis<self.shape().size();axis++){
                            if(axis >= Args.size()){accessors.push_back(Accessor::all());}
                            else{ 
                                // check type:
                                if(py::isinstance<py::slice>(Args[axis])){
                                    py::slice sls = Args[axis].cast<py::slice>();
                                    if(!sls.compute(self.shape()[axis],&start,&stop,&step, &slicelength))
                                        throw py::error_already_set();
                                    if(slicelength == self.shape()[axis]) accessors.push_back(cytnx::Accessor::all());
                                    else accessors.push_back(cytnx::Accessor::range(start,stop,step));
                                }else{
                                    accessors.push_back(cytnx::Accessor(Args[axis].cast<cytnx_int64>()));
                                }
                            }
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

                .def("__setitem__",[](cytnx::Tensor &self, py::object locators, const cytnx::Tensor &rhs){
                    cytnx_error_msg(self.shape().size() == 0, "[ERROR] try to setelem to a empty Tensor%s","\n");
                    
                    size_t start, stop, step, slicelength; 
                    std::vector<cytnx::Accessor> accessors;
                    if(py::isinstance<py::tuple>(locators)){
                        py::tuple Args = locators.cast<py::tuple>();
                        // mixing of slice and ints
                        for(cytnx_uint32 axis=0;axis<self.shape().size();axis++){
                            if(axis >= Args.size()){accessors.push_back(Accessor::all());}
                            else{ 
                                // check type:
                                if(py::isinstance<py::slice>(Args[axis])){
                                    py::slice sls = Args[axis].cast<py::slice>();
                                    if(!sls.compute(self.shape()[axis],&start,&stop,&step, &slicelength))
                                        throw py::error_already_set();
                                    if(slicelength == self.shape()[axis]) accessors.push_back(cytnx::Accessor::all());
                                    else accessors.push_back(cytnx::Accessor::range(start,stop,step));
                                }else{
                                    accessors.push_back(cytnx::Accessor(Args[axis].cast<cytnx_int64>()));
                                }
                            }
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
                
                .def("__iadd__",[](cytnx::Tensor &self, const cytnx::Tensor &rhs){return self.Add_(rhs);}) // these will return self!
                .def("__iadd__",[](cytnx::Tensor &self, const cytnx::cytnx_complex128&rhs){return self.Add_(rhs);})
                .def("__iadd__",[](cytnx::Tensor &self, const cytnx::cytnx_complex64 &rhs){return self.Add_(rhs);})
                .def("__iadd__",[](cytnx::Tensor &self, const cytnx::cytnx_double    &rhs){return self.Add_(rhs);})
                .def("__iadd__",[](cytnx::Tensor &self, const cytnx::cytnx_float     &rhs){return self.Add_(rhs);})
                .def("__iadd__",[](cytnx::Tensor &self, const cytnx::cytnx_int64     &rhs){return self.Add_(rhs);})
                .def("__iadd__",[](cytnx::Tensor &self, const cytnx::cytnx_uint64    &rhs){return self.Add_(rhs);})
                .def("__iadd__",[](cytnx::Tensor &self, const cytnx::cytnx_int32     &rhs){return self.Add_(rhs);})
                .def("__iadd__",[](cytnx::Tensor &self, const cytnx::cytnx_uint32    &rhs){return self.Add_(rhs);})
                .def("__iadd__",[](cytnx::Tensor &self, const cytnx::cytnx_int16     &rhs){return self.Add_(rhs);})
                .def("__iadd__",[](cytnx::Tensor &self, const cytnx::cytnx_uint16    &rhs){return self.Add_(rhs);})
                .def("__iadd__",[](cytnx::Tensor &self, const cytnx::cytnx_bool      &rhs){return self.Add_(rhs);})

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
 
                .def("__isub__",[](cytnx::Tensor &self, const cytnx::Tensor &rhs){return self.Sub_(rhs);}) // these will return self!
                .def("__isub__",[](cytnx::Tensor &self, const cytnx::cytnx_complex128&rhs){return self.Sub_(rhs);})
                .def("__isub__",[](cytnx::Tensor &self, const cytnx::cytnx_complex64 &rhs){return self.Sub_(rhs);})
                .def("__isub__",[](cytnx::Tensor &self, const cytnx::cytnx_double    &rhs){return self.Sub_(rhs);})
                .def("__isub__",[](cytnx::Tensor &self, const cytnx::cytnx_float     &rhs){return self.Sub_(rhs);})
                .def("__isub__",[](cytnx::Tensor &self, const cytnx::cytnx_int64     &rhs){return self.Sub_(rhs);})
                .def("__isub__",[](cytnx::Tensor &self, const cytnx::cytnx_uint64    &rhs){return self.Sub_(rhs);})
                .def("__isub__",[](cytnx::Tensor &self, const cytnx::cytnx_int32     &rhs){return self.Sub_(rhs);})
                .def("__isub__",[](cytnx::Tensor &self, const cytnx::cytnx_uint32    &rhs){return self.Sub_(rhs);})
                .def("__isub__",[](cytnx::Tensor &self, const cytnx::cytnx_int16     &rhs){return self.Sub_(rhs);})
                .def("__isub__",[](cytnx::Tensor &self, const cytnx::cytnx_uint16    &rhs){return self.Sub_(rhs);})
                .def("__isub__",[](cytnx::Tensor &self, const cytnx::cytnx_bool      &rhs){return self.Sub_(rhs);})

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
 
                .def("__imul__",[](cytnx::Tensor &self, const cytnx::Tensor &rhs){return self.Mul_(rhs);}) // these will return self!
                .def("__imul__",[](cytnx::Tensor &self, const cytnx::cytnx_complex128&rhs){return self.Mul_(rhs);})
                .def("__imul__",[](cytnx::Tensor &self, const cytnx::cytnx_complex64 &rhs){return self.Mul_(rhs);})
                .def("__imul__",[](cytnx::Tensor &self, const cytnx::cytnx_double    &rhs){return self.Mul_(rhs);})
                .def("__imul__",[](cytnx::Tensor &self, const cytnx::cytnx_float     &rhs){return self.Mul_(rhs);})
                .def("__imul__",[](cytnx::Tensor &self, const cytnx::cytnx_int64     &rhs){return self.Mul_(rhs);})
                .def("__imul__",[](cytnx::Tensor &self, const cytnx::cytnx_uint64    &rhs){return self.Mul_(rhs);})
                .def("__imul__",[](cytnx::Tensor &self, const cytnx::cytnx_int32     &rhs){return self.Mul_(rhs);})
                .def("__imul__",[](cytnx::Tensor &self, const cytnx::cytnx_uint32    &rhs){return self.Mul_(rhs);})
                .def("__imul__",[](cytnx::Tensor &self, const cytnx::cytnx_int16     &rhs){return self.Mul_(rhs);})
                .def("__imul__",[](cytnx::Tensor &self, const cytnx::cytnx_uint16    &rhs){return self.Mul_(rhs);})
                .def("__imul__",[](cytnx::Tensor &self, const cytnx::cytnx_bool      &rhs){return self.Mul_(rhs);})

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
 
                .def("__itruediv__",[](cytnx::Tensor &self, const cytnx::Tensor &rhs){return self.Div_(rhs);}) // these will return self!
                .def("__itruediv__",[](cytnx::Tensor &self, const cytnx::cytnx_complex128&rhs){return self.Div_(rhs);})
                .def("__itruediv__",[](cytnx::Tensor &self, const cytnx::cytnx_complex64 &rhs){return self.Div_(rhs);})
                .def("__itruediv__",[](cytnx::Tensor &self, const cytnx::cytnx_double    &rhs){return self.Div_(rhs);})
                .def("__itruediv__",[](cytnx::Tensor &self, const cytnx::cytnx_float     &rhs){return self.Div_(rhs);})
                .def("__itruediv__",[](cytnx::Tensor &self, const cytnx::cytnx_int64     &rhs){return self.Div_(rhs);})
                .def("__itruediv__",[](cytnx::Tensor &self, const cytnx::cytnx_uint64    &rhs){return self.Div_(rhs);})
                .def("__itruediv__",[](cytnx::Tensor &self, const cytnx::cytnx_int32     &rhs){return self.Div_(rhs);})
                .def("__itruediv__",[](cytnx::Tensor &self, const cytnx::cytnx_uint32    &rhs){return self.Div_(rhs);})
                .def("__itruediv__",[](cytnx::Tensor &self, const cytnx::cytnx_int16     &rhs){return self.Div_(rhs);})
                .def("__itruediv__",[](cytnx::Tensor &self, const cytnx::cytnx_uint16    &rhs){return self.Div_(rhs);})
                .def("__itruediv__",[](cytnx::Tensor &self, const cytnx::cytnx_bool      &rhs){return self.Div_(rhs);})

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
 
                .def("__ifloordiv__",[](cytnx::Tensor &self, const cytnx::Tensor &rhs){return self.Div_(rhs);}) // these will return self!
                .def("__ifloordiv__",[](cytnx::Tensor &self, const cytnx::cytnx_complex128&rhs){return self.Div_(rhs);})
                .def("__ifloordiv__",[](cytnx::Tensor &self, const cytnx::cytnx_complex64 &rhs){return self.Div_(rhs);})
                .def("__ifloordiv__",[](cytnx::Tensor &self, const cytnx::cytnx_double    &rhs){return self.Div_(rhs);})
                .def("__ifloordiv__",[](cytnx::Tensor &self, const cytnx::cytnx_float     &rhs){return self.Div_(rhs);})
                .def("__ifloordiv__",[](cytnx::Tensor &self, const cytnx::cytnx_int64     &rhs){return self.Div_(rhs);})
                .def("__ifloordiv__",[](cytnx::Tensor &self, const cytnx::cytnx_uint64    &rhs){return self.Div_(rhs);})
                .def("__ifloordiv__",[](cytnx::Tensor &self, const cytnx::cytnx_int32     &rhs){return self.Div_(rhs);})
                .def("__ifloordiv__",[](cytnx::Tensor &self, const cytnx::cytnx_uint32    &rhs){return self.Div_(rhs);})
                .def("__ifloordiv__",[](cytnx::Tensor &self, const cytnx::cytnx_int16     &rhs){return self.Div_(rhs);})
                .def("__ifloordiv__",[](cytnx::Tensor &self, const cytnx::cytnx_uint16    &rhs){return self.Div_(rhs);})
                .def("__ifloordiv__",[](cytnx::Tensor &self, const cytnx::cytnx_bool      &rhs){return self.Div_(rhs);})

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

                //linalg >>
                .def("Svd",&cytnx::Tensor::Svd, py::arg("is_U"), py::arg("is_vT"))
                .def("Eigh",&cytnx::Tensor::Eigh, py::arg("is_V")=true,py::arg("row_v")=false)
                .def("Inv_",&cytnx::Tensor::Inv_)
                .def("Inv",&cytnx::Tensor::Inv_)
                .def("Conj_",&cytnx::Tensor::Conj_)
                .def("Conj",&cytnx::Tensor::Conj_)
                .def("Exp_",&cytnx::Tensor::Exp_)
                .def("Exp",&cytnx::Tensor::Exp)
                .def("Norm",&cytnx::Tensor::Norm)
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
                .def("Fromfile",&cytnx_extension::Network::Fromfile,py::arg("fname"),py::arg("network_type")=(int)cytnx_extension::NtType.Regular)
                .def("PutCyTensor",[](cytnx_extension::Network &self,const std::string &name, const cytnx_extension::CyTensor &utensor, const bool &is_clone){
                                                self.PutCyTensor(name,utensor,is_clone);
                                        },py::arg("name"),py::arg("utensor"),py::arg("is_clone")=true)
                .def("PutCyTensor",[](cytnx_extension::Network &self,const cytnx_uint64 &idx, const cytnx_extension::CyTensor &utensor, const bool &is_clone){
                                                self.PutCyTensor(idx,utensor,is_clone);
                                        },py::arg("idx"),py::arg("utensor"),py::arg("is_clone")=true)
                .def("Launch",&cytnx_extension::Network::Launch)
                .def("Clear",&cytnx_extension::Network::Clear)
                .def("clone",&cytnx_extension::Network::clone)
                .def("__copy__",&cytnx_extension::Network::clone)
                .def("__deepcopy__",&cytnx_extension::Network::clone)
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
            ;


    py::class_<cytnx_extension::CyTensor>(mext,"CyTensor")
                .def(py::init<>())
                .def(py::init<const cytnx::Tensor&, const cytnx_uint64&>())
                .def(py::init<const std::vector<cytnx_extension::Bond> &, const std::vector<cytnx_int64> &, const cytnx_int64 &, const unsigned int &,const int &, const bool &>(),py::arg("bonds"),py::arg("labels")=std::vector<cytnx_int64>(),py::arg("rowrank")=(cytnx_int64)(-1),py::arg("dtype")=(unsigned int)(cytnx::Type.Double),py::arg("device")=(int)cytnx::Device.cpu,py::arg("is_diag")=false)
                .def("set_name",&cytnx_extension::CyTensor::set_name)
                .def("set_label",&cytnx_extension::CyTensor::set_label,py::arg("idx"),py::arg("new_label"))
                .def("set_labels",&cytnx_extension::CyTensor::set_labels,py::arg("new_labels"))
                .def("set_Rowrank",&cytnx_extension::CyTensor::set_Rowrank, py::arg("new_Rowrank"))

                .def("Rowrank",&cytnx_extension::CyTensor::Rowrank)
                .def("dtype",&cytnx_extension::CyTensor::dtype)
                .def("dtype_str",&cytnx_extension::CyTensor::dtype_str)
                .def("device",&cytnx_extension::CyTensor::device)
                .def("device_str",&cytnx_extension::CyTensor::device_str)
                .def("name",&cytnx_extension::CyTensor::name)

                .def("reshape",[](cytnx_extension::CyTensor &self, py::args args, py::kwargs kwargs)->cytnx_extension::CyTensor{
                    std::vector<cytnx::cytnx_int64> c_args = args.cast< std::vector<cytnx::cytnx_int64> >();
                    cytnx_uint64 Rowrank = 0;
                   
                    if(kwargs){
                        if(kwargs.contains("Rowrank")) Rowrank = kwargs["Rowrank"].cast<cytnx::cytnx_int64>();
                    }
 
                    return self.reshape(c_args,Rowrank);
                })
                .def("reshape_",[](cytnx_extension::CyTensor &self, py::args args, py::kwargs kwargs){
                    std::vector<cytnx::cytnx_int64> c_args = args.cast< std::vector<cytnx::cytnx_int64> >();
                    cytnx_uint64 Rowrank = 0;
                   
                    if(kwargs){
                        if(kwargs.contains("Rowrank")) Rowrank = kwargs["Rowrank"].cast<cytnx::cytnx_int64>();
                    }
 
                    self.reshape_(c_args,Rowrank);
                })


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
                    
                    size_t start, stop, step, slicelength; 
                    std::vector<cytnx::Accessor> accessors;
                    if(py::isinstance<py::tuple>(locators)){
                        py::tuple Args = locators.cast<py::tuple>();
                        // mixing of slice and ints
                        for(cytnx_uint32 axis=0;axis<self.shape().size();axis++){
                            if(axis >= Args.size()){accessors.push_back(Accessor::all());}
                            else{ 
                                // check type:
                                if(py::isinstance<py::slice>(Args[axis])){
                                    py::slice sls = Args[axis].cast<py::slice>();
                                    if(!sls.compute(self.shape()[axis],&start,&stop,&step, &slicelength))
                                        throw py::error_already_set();
                                    if(slicelength == self.shape()[axis]) accessors.push_back(cytnx::Accessor::all());
                                    else accessors.push_back(cytnx::Accessor::range(start,stop,step));
                                }else{
                                    accessors.push_back(cytnx::Accessor(Args[axis].cast<cytnx_int64>()));
                                }
                            }
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
                    
                    size_t start, stop, step, slicelength; 
                    std::vector<cytnx::Accessor> accessors;
                    if(py::isinstance<py::tuple>(locators)){
                        py::tuple Args = locators.cast<py::tuple>();
                        // mixing of slice and ints
                        for(cytnx_uint32 axis=0;axis<self.shape().size();axis++){
                            if(axis >= Args.size()){accessors.push_back(Accessor::all());}
                            else{ 
                                // check type:
                                if(py::isinstance<py::slice>(Args[axis])){
                                    py::slice sls = Args[axis].cast<py::slice>();
                                    if(!sls.compute(self.shape()[axis],&start,&stop,&step, &slicelength))
                                        throw py::error_already_set();
                                    if(slicelength == self.shape()[axis]) accessors.push_back(cytnx::Accessor::all());
                                    else accessors.push_back(cytnx::Accessor::range(start,stop,step));
                                }else{
                                    accessors.push_back(cytnx::Accessor(Args[axis].cast<cytnx_int64>()));
                                }
                            }
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

                .def("is_contiguous", &cytnx_extension::CyTensor::is_contiguous)
                .def("is_diag",&cytnx_extension::CyTensor::is_diag)
                .def("is_tag" ,&cytnx_extension::CyTensor::is_tag)
                .def("is_braket_form",&cytnx_extension::CyTensor::is_braket_form)
                .def("labels",&cytnx_extension::CyTensor::labels)
                .def("bonds",&cytnx_extension::CyTensor::bonds)
                .def("shape",&cytnx_extension::CyTensor::shape)
                .def("to_",&cytnx_extension::CyTensor::to_)
                .def("to_different_device" ,[](cytnx_extension::CyTensor &self,const cytnx_int64 &device){
                                                    cytnx_error_msg(self.device() == device, "[ERROR][pybind][to_diffferent_device] same device for to() should be handle in python side.%s","\n");
                                                    return self.to(device);
                                                } , py::arg("device"))
                .def("clone",&cytnx_extension::CyTensor::clone)
                .def("__copy__",&cytnx_extension::CyTensor::clone)
                .def("__deepcopy__",&cytnx_extension::CyTensor::clone)
                //.def("permute",&cytnx_extension::CyTensor::permute,py::arg("mapper"),py::arg("Rowrank")=(cytnx_int64)-1,py::arg("by_label")=false)
                //.def("permute_",&cytnx_extension::CyTensor::permute_,py::arg("mapper"),py::arg("Rowrank")=(cytnx_int64)-1,py::arg("by_label")=false)

                .def("permute_",[](cytnx_extension::CyTensor &self, const std::vector<cytnx::cytnx_int64> &c_args, py::kwargs kwargs){
                    cytnx_int64 Rowrank = -1;
                    bool by_label = false;
                    if(kwargs){
                        if(kwargs.contains("Rowrank")){
                            Rowrank = kwargs["Rowrank"].cast<cytnx_int64>();
                        }
                        if(kwargs.contains("by_label")){ 
                            by_label = kwargs["by_label"].cast<bool>();
                        }
                    }
                    self.permute_(c_args,Rowrank,by_label);
                })
                .def("permute",[](cytnx_extension::CyTensor &self,const std::vector<cytnx::cytnx_int64> &c_args, py::kwargs kwargs)->cytnx_extension::CyTensor{
                    cytnx_int64 Rowrank = -1;
                    bool by_label = false;
                    if(kwargs){
                        if(kwargs.contains("Rowrank")){
                            Rowrank = kwargs["Rowrank"].cast<cytnx_int64>();
                        }
                        if(kwargs.contains("by_label")){ 
                            by_label = kwargs["by_label"].cast<bool>();
                        }
                    }
                    return self.permute(c_args,Rowrank,by_label);
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
                .def("put_block_", [](cytnx_extension::CyTensor &self, const cytnx::Tensor &in, const cytnx_uint64&idx){
                                        self.put_block_(in,idx);
                                  },py::arg("in"),py::arg("idx")=(cytnx_uint64)(0))

                .def("put_block_", [](cytnx_extension::CyTensor &self, const cytnx::Tensor &in, const std::vector<cytnx_int64>&qnum){
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

                ;
    mext.def("Contract",cytnx_extension::Contract);
   
    // [Submodule linalg]
    pybind11::module mext_xlinalg = mext.def_submodule("xlinalg","linear algebra for cytnx_extension.");
    mext_xlinalg.def("Svd",&cytnx_extension::xlinalg::Svd,py::arg("Tin"),py::arg("is_U")=true,py::arg("is_vT")=true);
    mext_xlinalg.def("Svd_truncate",&cytnx_extension::xlinalg::Svd_truncate,py::arg("Tin"),py::arg("keepdim"),py::arg("is_U")=true,py::arg("is_vT")=true);

    // [Submodule linalg] 
    pybind11::module m_linalg = m.def_submodule("linalg","linear algebra related.");

    m_linalg.def("Svd",&cytnx::linalg::Svd,py::arg("Tin"),py::arg("is_U")=true,py::arg("is_vT")=true);
    m_linalg.def("Eigh",&cytnx::linalg::Eigh,py::arg("Tin"),py::arg("is_V")=true,py::arg("row_v")=false);
    m_linalg.def("Exp",&cytnx::linalg::Exp,py::arg("Tin"));
    m_linalg.def("Exp_",&cytnx::linalg::Exp_,py::arg("Tio"));
    m_linalg.def("Expf_",&cytnx::linalg::Expf_,py::arg("Tio"));
    m_linalg.def("Expf",&cytnx::linalg::Expf,py::arg("Tio"));
    m_linalg.def("ExpH",&cytnx::linalg::ExpH,py::arg("Tio"),py::arg("a")=1.0,py::arg("b")=0);
    m_linalg.def("Inv",&cytnx::linalg::Inv,py::arg("Tin"));
    m_linalg.def("Inv_",&cytnx::linalg::Inv_,py::arg("Tio"));
    m_linalg.def("Conj",&cytnx::linalg::Inv,py::arg("Tin"));
    m_linalg.def("Conj_",&cytnx::linalg::Inv_,py::arg("Tio"));
    m_linalg.def("Matmul",&cytnx::linalg::Matmul,py::arg("T1"),py::arg("T2"));
    m_linalg.def("Diag",&cytnx::linalg::Diag, py::arg("Tin"));
    m_linalg.def("Tensordot",&cytnx::linalg::Tensordot, py::arg("T1"),py::arg("T2"),py::arg("indices_1"),py::arg("indices_2"));
    m_linalg.def("Outer",&cytnx::linalg::Outer, py::arg("T1"),py::arg("T2"));
    m_linalg.def("Kron",&cytnx::linalg::Kron, py::arg("T1"),py::arg("T2"));
    m_linalg.def("Vectordot",&cytnx::linalg::Vectordot, py::arg("T1"),py::arg("T2"),py::arg("is_conj")=false);
    m_linalg.def("Norm",&cytnx::linalg::Norm, py::arg("T1"));
    m_linalg.def("Dot",&cytnx::linalg::Dot, py::arg("T1"),py::arg("T2"));

    // [Submodule random]
    pybind11::module m_random = m.def_submodule("random","random related.");
   
    m_random.def("Make_normal", [](cytnx::Tensor &Tin, const double &mean, const double &std, const long long &seed){
                                        if(seed<0)
                                            cytnx::random::Make_normal(Tin,mean,std);
                                        else
                                            cytnx::random::Make_normal(Tin,mean,std,seed);
                                  },py::arg("Tin"),py::arg("mean"),py::arg("std"),py::arg("seed")=std::random_device()());

    m_random.def("Make_normal", [](cytnx::Storage &Sin, const double &mean, const double &std, const long long &seed){
                                        if(seed<0)
                                            cytnx::random::Make_normal(Sin,mean,std);
                                        else
                                            cytnx::random::Make_normal(Sin,mean,std,seed);
                                  },py::arg("Sin"),py::arg("mean"),py::arg("std"),py::arg("seed")=std::random_device()());
    
    m_random.def("normal", [](const cytnx_uint64& Nelem,const double &mean, const double &std, const int&device, const unsigned int &seed){
                                   return cytnx::random::normal(Nelem,mean,std,device,seed);
                                  },py::arg("Nelem"),py::arg("mean"),py::arg("std"),py::arg("device")=-1,py::arg("seed")=std::random_device()());
    m_random.def("normal", [](const std::vector<cytnx_uint64>& Nelem,const double &mean, const double &std, const int&device, const unsigned int &seed){
                                   return cytnx::random::normal(Nelem,mean,std,device,seed);
                                  },py::arg("Nelem"),py::arg("mean"),py::arg("std"),py::arg("device")=-1,py::arg("seed")=std::random_device()());
}

