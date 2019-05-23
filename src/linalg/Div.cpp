#include "linalg/linalg.hpp"


namespace tor10{

    Tensor Div(const Tensor &Lt, const Tensor &Rt){
        
        tor10_error_msg(Lt.shape() != Rt.shape(),"[Div] error, the two tensor does not have the same type.%s","\n");
        tor10_error_msg(Lt.device() != Rt.device(),"[Div] error, two tensor cannot on different devices.%s","\n");

        Tensor out(Lt.shape(),Lt.dtype() < Rt.dtype()?Lt.dtype():Rt.dtype(),Lt.device());

        if(Lt.device() == tor10device.cpu){
            linalg_internal::lii.Ari_iicpu[Lt.dtype()][Rt.dtype()](out._impl->_get_storage(),Lt._impl->_get_storage(),Rt._impl->_get_storage(),Lt._impl->_get_storage()->size(),3);
        }else{
            #ifdef UNI_GPU
                linalg_internal::lii.Ari_iigpu[Lt.dtype()][Rt.dtype()](out._impl->_get_storage(),Lt._impl->_get_storage(),Rt._impl->_get_storage(),Rt._impl->_get_storage()->size(),3);
            #else
                tor10_error_msg(true,"[Div] fatal error, the tensor is on GPU without CUDA support.%s","\n"); 
            #endif
        }

        return out;

    }

    //-----------------------------------------------------------------------------------
    template<>
    Tensor Div<tor10_complex128>(const tor10_complex128 &lc, const Tensor &Rt){
        Storage Cnst(1,tor10type.ComplexDouble);
        Cnst.at<tor10_complex128>(0) = lc;

        Tensor out(Rt.shape(),tor10type.ComplexDouble,Rt.device());

        if(Rt.device()==tor10device.cpu){
            linalg_internal::lii.Ari_iicpu[tor10type.ComplexDouble][Rt.dtype()](out._impl->_get_storage(),Cnst._impl,Rt._impl->_get_storage(),Rt._impl->_get_storage()->size(),3);
        }else{
            #ifdef UNI_GPU
                linalg_internal::lii.Ari_iigpu[tor10type.ComplexDouble][Rt.dtype()](out._impl->_get_storage(),Cnst._impl,Rt._impl->_get_storage(),Rt._impl->_get_storage()->size(),3);
            #else
                tor10_error_msg(true,"[Div] fatal error, the tensor is on GPU without CUDA support.%s","\n"); 
            #endif 
        }        

        return out;
    }

    template<>
    Tensor Div<tor10_complex64>(const tor10_complex64 &lc, const Tensor &Rt){
        Storage Cnst(1,tor10type.ComplexFloat);
        Cnst.at<tor10_complex64>(0) = lc;

        Tensor out(Rt.shape(),tor10type.ComplexFloat < Rt.dtype()?tor10type.ComplexFloat:Rt.dtype(),Rt.device());

        if(Rt.device()==tor10device.cpu){
            linalg_internal::lii.Ari_iicpu[tor10type.ComplexFloat][Rt.dtype()](out._impl->_get_storage(),Cnst._impl,Rt._impl->_get_storage(),Rt._impl->_get_storage()->size(),3);
        }else{
            #ifdef UNI_GPU
                linalg_internal::lii.Ari_iigpu[tor10type.ComplexFloat][Rt.dtype()](out._impl->_get_storage(),Cnst._impl,Rt._impl->_get_storage(),Rt._impl->_get_storage()->size(),3);
            #else
                tor10_error_msg(true,"[Div] fatal error, the tensor is on GPU without CUDA support.%s","\n"); 
            #endif 
        }        

        return out;
    }
    
    template<>
    Tensor Div<tor10_double>(const tor10_double &lc, const Tensor &Rt){
        Storage Cnst(1,tor10type.Double);
        Cnst.at<tor10_double>(0) = lc;

        Tensor out(Rt.shape(),tor10type.Double < Rt.dtype()?tor10type.Double:Rt.dtype(),Rt.device());

        if(Rt.device()==tor10device.cpu){
            linalg_internal::lii.Ari_iicpu[tor10type.Double][Rt.dtype()](out._impl->_get_storage(),Cnst._impl,Rt._impl->_get_storage(),Rt._impl->_get_storage()->size(),3);
        }else{
            #ifdef UNI_GPU
                linalg_internal::lii.Ari_iigpu[tor10type.Double][Rt.dtype()](out._impl->_get_storage(),Cnst._impl,Rt._impl->_get_storage(),Rt._impl->_get_storage()->size(),3);
            #else
                tor10_error_msg(true,"[Div] fatal error, the tensor is on GPU without CUDA support.%s","\n"); 
            #endif 
        }        

        return out;
    }

    template<>
    Tensor Div<tor10_float>(const tor10_float &lc, const Tensor &Rt){
        Storage Cnst(1,tor10type.Float);
        Cnst.at<tor10_float>(0) = lc;

        Tensor out(Rt.shape(),tor10type.Float < Rt.dtype()?tor10type.Float:Rt.dtype(),Rt.device());

        if(Rt.device()==tor10device.cpu){
            linalg_internal::lii.Ari_iicpu[tor10type.Float][Rt.dtype()](out._impl->_get_storage(),Cnst._impl,Rt._impl->_get_storage(),Rt._impl->_get_storage()->size(),3);
        }else{
            #ifdef UNI_GPU
                linalg_internal::lii.Ari_iigpu[tor10type.Float][Rt.dtype()](out._impl->_get_storage(),Cnst._impl,Rt._impl->_get_storage(),Rt._impl->_get_storage()->size(),3);
            #else
                tor10_error_msg(true,"[Div] fatal error, the tensor is on GPU without CUDA support.%s","\n"); 
            #endif 
        }        

        return out;
    }

    template<>
    Tensor Div<tor10_int64>(const tor10_int64 &lc, const Tensor &Rt){
        Storage Cnst(1,tor10type.Int64);
        Cnst.at<tor10_int64>(0) = lc;

        Tensor out(Rt.shape(),tor10type.Int64 < Rt.dtype()?tor10type.Int64:Rt.dtype(),Rt.device());

        if(Rt.device()==tor10device.cpu){
            linalg_internal::lii.Ari_iicpu[tor10type.Int64][Rt.dtype()](out._impl->_get_storage(),Cnst._impl,Rt._impl->_get_storage(),Rt._impl->_get_storage()->size(),3);
        }else{
            #ifdef UNI_GPU
                linalg_internal::lii.Ari_iigpu[tor10type.Int64][Rt.dtype()](out._impl->_get_storage(),Cnst._impl,Rt._impl->_get_storage(),Rt._impl->_get_storage()->size(),3);
            #else
                tor10_error_msg(true,"[Div] fatal error, the tensor is on GPU without CUDA support.%s","\n"); 
            #endif 
        }        

        return out;
    }

    template<>
    Tensor Div<tor10_uint64>(const tor10_uint64 &lc, const Tensor &Rt){
        Storage Cnst(1,tor10type.Uint64);
        Cnst.at<tor10_uint64>(0) = lc;

        Tensor out(Rt.shape(),tor10type.Uint64 < Rt.dtype()?tor10type.Uint64:Rt.dtype(),Rt.device());

        if(Rt.device()==tor10device.cpu){
            linalg_internal::lii.Ari_iicpu[tor10type.Uint64][Rt.dtype()](out._impl->_get_storage(),Cnst._impl,Rt._impl->_get_storage(),Rt._impl->_get_storage()->size(),3);
        }else{
            #ifdef UNI_GPU
                linalg_internal::lii.Ari_iigpu[tor10type.Uint64][Rt.dtype()](out._impl->_get_storage(),Cnst._impl,Rt._impl->_get_storage(),Rt._impl->_get_storage()->size(),3);
            #else
                tor10_error_msg(true,"[Div] fatal error, the tensor is on GPU without CUDA support.%s","\n"); 
            #endif 
        }        

        return out;
    }

    template<>
    Tensor Div<tor10_int32>(const tor10_int32 &lc, const Tensor &Rt){
        Storage Cnst(1,tor10type.Int32);
        Cnst.at<tor10_int32>(0) = lc;

        Tensor out(Rt.shape(),tor10type.Int32 < Rt.dtype()?tor10type.Int32:Rt.dtype(),Rt.device());

        if(Rt.device()==tor10device.cpu){
            linalg_internal::lii.Ari_iicpu[tor10type.Int32][Rt.dtype()](out._impl->_get_storage(),Cnst._impl,Rt._impl->_get_storage(),Rt._impl->_get_storage()->size(),3);
        }else{
            #ifdef UNI_GPU
                linalg_internal::lii.Ari_iigpu[tor10type.Int32][Rt.dtype()](out._impl->_get_storage(),Cnst._impl,Rt._impl->_get_storage(),Rt._impl->_get_storage()->size(),3);
            #else
                tor10_error_msg(true,"[Div] fatal error, the tensor is on GPU without CUDA support.%s","\n"); 
            #endif 
        }        

        return out;
    }

    template<>
    Tensor Div<tor10_uint32>(const tor10_uint32 &lc, const Tensor &Rt){
        Storage Cnst(1,tor10type.Uint32);
        Cnst.at<tor10_uint32>(0) = lc;

        Tensor out(Rt.shape(),tor10type.Uint32 < Rt.dtype()?tor10type.Uint32:Rt.dtype(),Rt.device());

        if(Rt.device()==tor10device.cpu){
            linalg_internal::lii.Ari_iicpu[tor10type.Uint32][Rt.dtype()](out._impl->_get_storage(),Cnst._impl,Rt._impl->_get_storage(),Rt._impl->_get_storage()->size(),3);
        }else{
            #ifdef UNI_GPU
                linalg_internal::lii.Ari_iigpu[tor10type.Uint32][Rt.dtype()](out._impl->_get_storage(),Cnst._impl,Rt._impl->_get_storage(),Rt._impl->_get_storage()->size(),3);
            #else
                tor10_error_msg(true,"[Div] fatal error, the tensor is on GPU without CUDA support.%s","\n"); 
            #endif 
        }        

        return out;
    }
    
    //-----------------------------------------------------------------------------------
    template<>
    Tensor Div<tor10_complex128>(const Tensor &Lt, const tor10_complex128 &rc){
        Storage Cnst(1,tor10type.ComplexDouble);
        Cnst.at<tor10_complex128>(0) = rc;

        Tensor out(Lt.shape(),tor10type.ComplexDouble,Lt.device());

        if(Lt.device()==tor10device.cpu){
            linalg_internal::lii.Ari_iicpu[Lt.dtype()][tor10type.ComplexDouble](out._impl->_get_storage(),Lt._impl->_get_storage(),Cnst._impl,Lt._impl->_get_storage()->size(),3);
        }else{
            #ifdef UNI_GPU
            linalg_internal::lii.Ari_iigpu[Lt.dtype()][tor10type.ComplexDouble](out._impl->_get_storage(),Lt._impl->_get_storage(),Cnst._impl,Lt._impl->_get_storage()->size(),3);
            #else
                tor10_error_msg(true,"[Div] fatal error, the tensor is on GPU without CUDA support.%s","\n"); 
            #endif 
        }        

        return out;
    }
    template<>
    Tensor Div<tor10_complex64>(const Tensor &Lt, const tor10_complex64 &rc){
        Storage Cnst(1,tor10type.ComplexFloat);
        Cnst.at<tor10_complex64>(0) = rc;

        Tensor out(Lt.shape(),tor10type.ComplexFloat < Lt.dtype()?tor10type.ComplexFloat:Lt.dtype(),Lt.device());

        if(Lt.device()==tor10device.cpu){
            linalg_internal::lii.Ari_iicpu[Lt.dtype()][tor10type.ComplexFloat](out._impl->_get_storage(),Lt._impl->_get_storage(),Cnst._impl,Lt._impl->_get_storage()->size(),3);
        }else{
            #ifdef UNI_GPU
            linalg_internal::lii.Ari_iigpu[Lt.dtype()][tor10type.ComplexFloat](out._impl->_get_storage(),Lt._impl->_get_storage(),Cnst._impl,Lt._impl->_get_storage()->size(),3);
            #else
                tor10_error_msg(true,"[Div] fatal error, the tensor is on GPU without CUDA support.%s","\n"); 
            #endif 
        }        

        return out;
    }
    template<>
    Tensor Div<tor10_double>(const Tensor &Lt, const tor10_double &rc){
        Storage Cnst(1,tor10type.Double);
        Cnst.at<tor10_double>(0) = rc;

        Tensor out(Lt.shape(),tor10type.Double < Lt.dtype()?tor10type.Double:Lt.dtype(),Lt.device());

        if(Lt.device()==tor10device.cpu){
            linalg_internal::lii.Ari_iicpu[Lt.dtype()][tor10type.Double](out._impl->_get_storage(),Lt._impl->_get_storage(),Cnst._impl,Lt._impl->_get_storage()->size(),3);
        }else{
            #ifdef UNI_GPU
            linalg_internal::lii.Ari_iigpu[Lt.dtype()][tor10type.Double](out._impl->_get_storage(),Lt._impl->_get_storage(),Cnst._impl,Lt._impl->_get_storage()->size(),3);
            #else
                tor10_error_msg(true,"[Div] fatal error, the tensor is on GPU without CUDA support.%s","\n"); 
            #endif 
        }        

        return out;
    }
    template<>
    Tensor Div<tor10_float>(const Tensor &Lt, const tor10_float &rc){
        Storage Cnst(1,tor10type.Float);
        Cnst.at<tor10_float>(0) = rc;

        Tensor out(Lt.shape(),tor10type.Float < Lt.dtype()?tor10type.Float:Lt.dtype(),Lt.device());

        if(Lt.device()==tor10device.cpu){
            linalg_internal::lii.Ari_iicpu[Lt.dtype()][tor10type.Float](out._impl->_get_storage(),Lt._impl->_get_storage(),Cnst._impl,Lt._impl->_get_storage()->size(),3);
        }else{
            #ifdef UNI_GPU
            linalg_internal::lii.Ari_iigpu[Lt.dtype()][tor10type.Float](out._impl->_get_storage(),Lt._impl->_get_storage(),Cnst._impl,Lt._impl->_get_storage()->size(),3);
            #else
                tor10_error_msg(true,"[Div] fatal error, the tensor is on GPU without CUDA support.%s","\n"); 
            #endif 
        }        

        return out;
    }
    template<>
    Tensor Div<tor10_int64>(const Tensor &Lt, const tor10_int64 &rc){
        Storage Cnst(1,tor10type.Int64);
        Cnst.at<tor10_int64>(0) = rc;

        Tensor out(Lt.shape(),tor10type.Int64 < Lt.dtype()?tor10type.Int64:Lt.dtype(),Lt.device());

        if(Lt.device()==tor10device.cpu){
            linalg_internal::lii.Ari_iicpu[Lt.dtype()][tor10type.Int64](out._impl->_get_storage(),Lt._impl->_get_storage(),Cnst._impl,Lt._impl->_get_storage()->size(),3);
        }else{
            #ifdef UNI_GPU
            linalg_internal::lii.Ari_iigpu[Lt.dtype()][tor10type.Int64](out._impl->_get_storage(),Lt._impl->_get_storage(),Cnst._impl,Lt._impl->_get_storage()->size(),3);
            #else
                tor10_error_msg(true,"[Div] fatal error, the tensor is on GPU without CUDA support.%s","\n"); 
            #endif 
        }        

        return out;
    }
    template<>
    Tensor Div<tor10_uint64>(const Tensor &Lt, const tor10_uint64 &rc){
        Storage Cnst(1,tor10type.Uint64);
        Cnst.at<tor10_uint64>(0) = rc;

        Tensor out(Lt.shape(),tor10type.Uint64 < Lt.dtype()?tor10type.Uint64:Lt.dtype(),Lt.device());

        if(Lt.device()==tor10device.cpu){
            linalg_internal::lii.Ari_iicpu[Lt.dtype()][tor10type.Uint64](out._impl->_get_storage(),Lt._impl->_get_storage(),Cnst._impl,Lt._impl->_get_storage()->size(),3);
        }else{
            #ifdef UNI_GPU
            linalg_internal::lii.Ari_iigpu[Lt.dtype()][tor10type.Uint64](out._impl->_get_storage(),Lt._impl->_get_storage(),Cnst._impl,Lt._impl->_get_storage()->size(),3);
            #else
                tor10_error_msg(true,"[Div] fatal error, the tensor is on GPU without CUDA support.%s","\n"); 
            #endif 
        }        

        return out;
    }
    template<>
    Tensor Div<tor10_int32>(const Tensor &Lt, const tor10_int32 &rc){
        Storage Cnst(1,tor10type.Int32);
        Cnst.at<tor10_int32>(0) = rc;

        Tensor out(Lt.shape(),tor10type.Int32 < Lt.dtype()?tor10type.Int32:Lt.dtype(),Lt.device());

        if(Lt.device()==tor10device.cpu){
            linalg_internal::lii.Ari_iicpu[Lt.dtype()][tor10type.Int32](out._impl->_get_storage(),Lt._impl->_get_storage(),Cnst._impl,Lt._impl->_get_storage()->size(),3);
        }else{
            #ifdef UNI_GPU
            linalg_internal::lii.Ari_iigpu[Lt.dtype()][tor10type.Int32](out._impl->_get_storage(),Lt._impl->_get_storage(),Cnst._impl,Lt._impl->_get_storage()->size(),3);
            #else
                tor10_error_msg(true,"[Div] fatal error, the tensor is on GPU without CUDA support.%s","\n"); 
            #endif 
        }        

        return out;
    }
    template<>
    Tensor Div<tor10_uint32>(const Tensor &Lt, const tor10_uint32 &rc){
        Storage Cnst(1,tor10type.Uint32);
        Cnst.at<tor10_uint32>(0) = rc;

        Tensor out(Lt.shape(),tor10type.Uint32 < Lt.dtype()?tor10type.Uint32:Lt.dtype(),Lt.device());

        if(Lt.device()==tor10device.cpu){
            linalg_internal::lii.Ari_iicpu[Lt.dtype()][tor10type.Uint32](out._impl->_get_storage(),Lt._impl->_get_storage(),Cnst._impl,Lt._impl->_get_storage()->size(),3);
        }else{
            #ifdef UNI_GPU
            linalg_internal::lii.Ari_iigpu[Lt.dtype()][tor10type.Uint32](out._impl->_get_storage(),Lt._impl->_get_storage(),Cnst._impl,Lt._impl->_get_storage()->size(),3);
            #else
                tor10_error_msg(true,"[Div] fatal error, the tensor is on GPU without CUDA support.%s","\n"); 
            #endif 
        }        

        return out;
    }

    Tensor operator/(const Tensor &Lt,const Tensor &Rt){
        return Div(Lt,Rt);
    }

}


