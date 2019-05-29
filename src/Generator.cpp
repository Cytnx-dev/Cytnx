#include "Generator.hpp"
#include "utils/utils_internal_interface.hpp"
#include <cfloat>
namespace cytnx{

    Tensor zeros(const cytnx_uint64 &Nelem, const unsigned int &dtype, const int &device){
        Tensor out({Nelem},dtype,device); // the default
        out._impl->storage().set_zeros(); 
        return out;
    }
    Tensor zeros(const std::vector<cytnx_uint64> &Nelem, const unsigned int &dtype, const int &device){
        Tensor out(Nelem,dtype,device);
        out._impl->storage().set_zeros(); 
        return out;
    }
    Tensor zeros(const std::initializer_list<cytnx_uint64> &Nelem, const unsigned int &dtype, const int &device){
        Tensor out(Nelem,dtype,device);
        out._impl->storage().set_zeros(); 
        return out;
    }
    //-----------------
    Tensor ones(const cytnx_uint64 &Nelem, const unsigned int &dtype, const int &device){
        Tensor out({Nelem},dtype,device); // the default
        out._impl->storage().fill(1); 
        return out;
    }
    Tensor ones(const std::vector<cytnx_uint64> &Nelem, const unsigned int &dtype, const int &device){
        Tensor out(Nelem,dtype,device);
        out._impl->storage().fill(1); 
        return out;
    }
    Tensor ones(const std::initializer_list<cytnx_uint64> &Nelem, const unsigned int &dtype, const int &device){
        Tensor out(Nelem,dtype,device);
        out._impl->storage().fill(1); 
        return out;
    }
    //-----------------
    Tensor arange(const cytnx_double &start, const cytnx_double &end, const cytnx_double &step, const unsigned int &dtype, const int &device){
        cytnx_uint64 Nelem; 
        Tensor out;
        if(start < end){
            Nelem = cytnx_uint64((end - start)/step);
            if(fmod((end - start),step) > 0) Nelem += 1;
            cytnx_error_msg(Nelem == 0,"[ERROR] arange(start,end,step)%s","Nelem cannot be zero! check the range!\n");
            out.Init({Nelem},dtype,device);
            if(device == cytnxdevice.cpu){
                utils_internal::uii.SetArange_ii[dtype](out._impl->storage()._impl, start, end, step, Nelem,0); // 0, end > start
            }else{
                #ifdef UNI_GPU
                    utils_internal::uii.cuSetArange_ii[dtype](out._impl->storage()._impl, start, end, step, Nelem,0); 
                #else
                    cytnx_error_msg(true,"[ERROR] fatal internal, %s"," [arange] the container is on gpu without CUDA support!%s","\n")
                #endif
            }        
           
        }else{
            Nelem = cytnx_uint64((start - end)/(-step));
            if(fmod((start - end),(-step)) > 0) Nelem += 1;
            cytnx_error_msg(Nelem == 0,"[ERROR] arange(start,end,step)%s","Nelem cannot be zero! check the range!\n");
            out.Init({Nelem},dtype,device);
            if(device == cytnxdevice.cpu){
                utils_internal::uii.SetArange_ii[dtype](out._impl->storage()._impl, start, end, step, Nelem,1);
            }else{
                #ifdef UNI_GPU
                    utils_internal::uii.cuSetArange_ii[dtype](out._impl->storage()._impl, start, end, step, Nelem,1); //1, start > end
                #else
                    cytnx_error_msg(true,"[ERROR] fatal internal, %s"," [arange] the container is on gpu without CUDA support!%s","\n")
                #endif
            }

        } 
        return out;    
    }
    Tensor arange(const cytnx_int64 &Nelem, const unsigned int &dtype, const int &device){
        cytnx_error_msg(Nelem <= 0, "[ERROR] arange(Nelem) , %s","Nelem must be integer > 0");
        return arange(0,Nelem,1,dtype,device);
    }


}
