#include "linalg/linalg.hpp"


namespace cytnx{
    namespace linalg{
        Tensor Mul(const Tensor &Lt, const Tensor &Rt){
            
            cytnx_error_msg(Lt.shape() != Rt.shape(),"[Mul] error, the two tensor does not have the same type.%s","\n");
            cytnx_error_msg(Lt.device() != Rt.device(),"[Mul] error, two tensor cannot on different devices.%s","\n");
            cytnx_error_msg(!(Lt.is_contiguous() && Rt.is_contiguous()), "[Mul] error two tensors must be contiguous. Call Contiguous_() or Contiguous() first%s","\n");
            Tensor out(Lt.shape(),Lt.dtype() < Rt.dtype()?Lt.dtype():Rt.dtype(),Lt.device());

            if(Lt.device() == cytnxdevice.cpu){
                cytnx::linalg_internal::lii.Ari_ii[Lt.dtype()][Rt.dtype()](out._impl->storage()._impl,Lt._impl->storage()._impl,Rt._impl->storage()._impl,Lt._impl->storage()._impl->size(),1);
            }else{
                #ifdef UNI_GPU
                    cytnx::linalg_internal::lii.cuAri_ii[Lt.dtype()][Rt.dtype()](out._impl->storage()._impl,Lt._impl->storage()._impl,Rt._impl->storage()._impl,Rt._impl->storage()._impl->size(),1);
                #else
                    cytnx_error_msg(true,"[Mul] fatal error, the tensor is on GPU without CUDA support.%s","\n"); 
                #endif
            }

            return out;

        }

        //-----------------------------------------------------------------------------------
        template<>
        Tensor Mul<cytnx_complex128>(const cytnx_complex128 &lc, const Tensor &Rt){
            Storage Cnst(1,cytnxtype.ComplexDouble);
            Cnst.at<cytnx_complex128>(0) = lc;

            Tensor out(Rt.shape(),cytnxtype.ComplexDouble,Rt.device());

            if(Rt.device()==cytnxdevice.cpu){
                cytnx::linalg_internal::lii.Ari_ii[cytnxtype.ComplexDouble][Rt.dtype()](out._impl->storage()._impl,Cnst._impl,Rt._impl->storage()._impl,Rt._impl->storage()._impl->size(),1);
            }else{
                #ifdef UNI_GPU
                    cytnx::linalg_internal::lii.cuAri_ii[cytnxtype.ComplexDouble][Rt.dtype()](out._impl->storage()._impl,Cnst._impl,Rt._impl->storage()._impl,Rt._impl->storage()._impl->size(),1);
                #else
                    cytnx_error_msg(true,"[Mul] fatal error, the tensor is on GPU without CUDA support.%s","\n"); 
                #endif 
            }        

            return out;
        }

        template<>
        Tensor Mul<cytnx_complex64>(const cytnx_complex64 &lc, const Tensor &Rt){
            Storage Cnst(1,cytnxtype.ComplexFloat);
            Cnst.at<cytnx_complex64>(0) = lc;

            Tensor out(Rt.shape(),cytnxtype.ComplexFloat < Rt.dtype()?cytnxtype.ComplexFloat:Rt.dtype(),Rt.device());

            if(Rt.device()==cytnxdevice.cpu){
                cytnx::linalg_internal::lii.Ari_ii[cytnxtype.ComplexFloat][Rt.dtype()](out._impl->storage()._impl,Cnst._impl,Rt._impl->storage()._impl,Rt._impl->storage()._impl->size(),1);
            }else{
                #ifdef UNI_GPU
                    cytnx::linalg_internal::lii.cuAri_ii[cytnxtype.ComplexFloat][Rt.dtype()](out._impl->storage()._impl,Cnst._impl,Rt._impl->storage()._impl,Rt._impl->storage()._impl->size(),1);
                #else
                    cytnx_error_msg(true,"[Mul] fatal error, the tensor is on GPU without CUDA support.%s","\n"); 
                #endif 
            }        

            return out;
        }
        
        template<>
        Tensor Mul<cytnx_double>(const cytnx_double &lc, const Tensor &Rt){
            Storage Cnst(1,cytnxtype.Double);
            Cnst.at<cytnx_double>(0) = lc;

            Tensor out(Rt.shape(),cytnxtype.Double < Rt.dtype()?cytnxtype.Double:Rt.dtype(),Rt.device());

            if(Rt.device()==cytnxdevice.cpu){
                cytnx::linalg_internal::lii.Ari_ii[cytnxtype.Double][Rt.dtype()](out._impl->storage()._impl,Cnst._impl,Rt._impl->storage()._impl,Rt._impl->storage()._impl->size(),1);
            }else{
                #ifdef UNI_GPU
                    cytnx::linalg_internal::lii.cuAri_ii[cytnxtype.Double][Rt.dtype()](out._impl->storage()._impl,Cnst._impl,Rt._impl->storage()._impl,Rt._impl->storage()._impl->size(),1);
                #else
                    cytnx_error_msg(true,"[Mul] fatal error, the tensor is on GPU without CUDA support.%s","\n"); 
                #endif 
            }        

            return out;
        }

        template<>
        Tensor Mul<cytnx_float>(const cytnx_float &lc, const Tensor &Rt){
            Storage Cnst(1,cytnxtype.Float);
            Cnst.at<cytnx_float>(0) = lc;

            Tensor out(Rt.shape(),cytnxtype.Float < Rt.dtype()?cytnxtype.Float:Rt.dtype(),Rt.device());

            if(Rt.device()==cytnxdevice.cpu){
                cytnx::linalg_internal::lii.Ari_ii[cytnxtype.Float][Rt.dtype()](out._impl->storage()._impl,Cnst._impl,Rt._impl->storage()._impl,Rt._impl->storage()._impl->size(),1);
            }else{
                #ifdef UNI_GPU
                    cytnx::linalg_internal::lii.cuAri_ii[cytnxtype.Float][Rt.dtype()](out._impl->storage()._impl,Cnst._impl,Rt._impl->storage()._impl,Rt._impl->storage()._impl->size(),1);
                #else
                    cytnx_error_msg(true,"[Mul] fatal error, the tensor is on GPU without CUDA support.%s","\n"); 
                #endif 
            }        

            return out;
        }

        template<>
        Tensor Mul<cytnx_int64>(const cytnx_int64 &lc, const Tensor &Rt){
            Storage Cnst(1,cytnxtype.Int64);
            Cnst.at<cytnx_int64>(0) = lc;

            Tensor out(Rt.shape(),cytnxtype.Int64 < Rt.dtype()?cytnxtype.Int64:Rt.dtype(),Rt.device());

            if(Rt.device()==cytnxdevice.cpu){
                cytnx::linalg_internal::lii.Ari_ii[cytnxtype.Int64][Rt.dtype()](out._impl->storage()._impl,Cnst._impl,Rt._impl->storage()._impl,Rt._impl->storage()._impl->size(),1);
            }else{
                #ifdef UNI_GPU
                    cytnx::linalg_internal::lii.cuAri_ii[cytnxtype.Int64][Rt.dtype()](out._impl->storage()._impl,Cnst._impl,Rt._impl->storage()._impl,Rt._impl->storage()._impl->size(),1);
                #else
                    cytnx_error_msg(true,"[Mul] fatal error, the tensor is on GPU without CUDA support.%s","\n"); 
                #endif 
            }        

            return out;
        }

        template<>
        Tensor Mul<cytnx_uint64>(const cytnx_uint64 &lc, const Tensor &Rt){
            Storage Cnst(1,cytnxtype.Uint64);
            Cnst.at<cytnx_uint64>(0) = lc;

            Tensor out(Rt.shape(),cytnxtype.Uint64 < Rt.dtype()?cytnxtype.Uint64:Rt.dtype(),Rt.device());

            if(Rt.device()==cytnxdevice.cpu){
                cytnx::linalg_internal::lii.Ari_ii[cytnxtype.Uint64][Rt.dtype()](out._impl->storage()._impl,Cnst._impl,Rt._impl->storage()._impl,Rt._impl->storage()._impl->size(),1);
            }else{
                #ifdef UNI_GPU
                    cytnx::linalg_internal::lii.cuAri_ii[cytnxtype.Uint64][Rt.dtype()](out._impl->storage()._impl,Cnst._impl,Rt._impl->storage()._impl,Rt._impl->storage()._impl->size(),1);
                #else
                    cytnx_error_msg(true,"[Mul] fatal error, the tensor is on GPU without CUDA support.%s","\n"); 
                #endif 
            }        

            return out;
        }

        template<>
        Tensor Mul<cytnx_int32>(const cytnx_int32 &lc, const Tensor &Rt){
            Storage Cnst(1,cytnxtype.Int32);
            Cnst.at<cytnx_int32>(0) = lc;

            Tensor out(Rt.shape(),cytnxtype.Int32 < Rt.dtype()?cytnxtype.Int32:Rt.dtype(),Rt.device());

            if(Rt.device()==cytnxdevice.cpu){
                cytnx::linalg_internal::lii.Ari_ii[cytnxtype.Int32][Rt.dtype()](out._impl->storage()._impl,Cnst._impl,Rt._impl->storage()._impl,Rt._impl->storage()._impl->size(),1);
            }else{
                #ifdef UNI_GPU
                    cytnx::linalg_internal::lii.cuAri_ii[cytnxtype.Int32][Rt.dtype()](out._impl->storage()._impl,Cnst._impl,Rt._impl->storage()._impl,Rt._impl->storage()._impl->size(),1);
                #else
                    cytnx_error_msg(true,"[Mul] fatal error, the tensor is on GPU without CUDA support.%s","\n"); 
                #endif 
            }        

            return out;
        }

        template<>
        Tensor Mul<cytnx_uint32>(const cytnx_uint32 &lc, const Tensor &Rt){
            Storage Cnst(1,cytnxtype.Uint32);
            Cnst.at<cytnx_uint32>(0) = lc;

            Tensor out(Rt.shape(),cytnxtype.Uint32 < Rt.dtype()?cytnxtype.Uint32:Rt.dtype(),Rt.device());

            if(Rt.device()==cytnxdevice.cpu){
                cytnx::linalg_internal::lii.Ari_ii[cytnxtype.Uint32][Rt.dtype()](out._impl->storage()._impl,Cnst._impl,Rt._impl->storage()._impl,Rt._impl->storage()._impl->size(),1);
            }else{
                #ifdef UNI_GPU
                    cytnx::linalg_internal::lii.cuAri_ii[cytnxtype.Uint32][Rt.dtype()](out._impl->storage()._impl,Cnst._impl,Rt._impl->storage()._impl,Rt._impl->storage()._impl->size(),1);
                #else
                    cytnx_error_msg(true,"[Mul] fatal error, the tensor is on GPU without CUDA support.%s","\n"); 
                #endif 
            }        

            return out;
        }
        
        //-----------------------------------------------------------------------------------
        template<>
        Tensor Mul<cytnx_complex128>(const Tensor &Lc, const cytnx_complex128 &rc){
            return Mul(rc,Lc);
        }
        template<>
        Tensor Mul<cytnx_complex64>(const Tensor &Lc, const cytnx_complex64 &rc){
            return Mul(rc,Lc);
        }
        template<>
        Tensor Mul<cytnx_double>(const Tensor &Lc, const cytnx_double &rc){
            return Mul(rc,Lc);
        }
        template<>
        Tensor Mul<cytnx_float>(const Tensor &Lc, const cytnx_float &rc){
            return Mul(rc,Lc);
        }
        template<>
        Tensor Mul<cytnx_int64>(const Tensor &Lc, const cytnx_int64 &rc){
            return Mul(rc,Lc);
        }
        template<>
        Tensor Mul<cytnx_uint64>(const Tensor &Lc, const cytnx_uint64 &rc){
            return Mul(rc,Lc);
        }
        template<>
        Tensor Mul<cytnx_int32>(const Tensor &Lc, const cytnx_int32 &rc){
            return Mul(rc,Lc);
        }
        template<>
        Tensor Mul<cytnx_uint32>(const Tensor &Lc, const cytnx_uint32 &rc){
            return Mul(rc,Lc);
        }

    }//linalg

    Tensor operator*(const Tensor &Lt, const Tensor &Rt){
        return cytnx::linalg::Mul(Lt,Rt);
    }
    template<>
    Tensor operator*<cytnx_complex128>(const cytnx_complex128 &lc, const Tensor &Rt){
        return cytnx::linalg::Mul(lc,Rt);
    }
    template<>
    Tensor operator*<cytnx_complex64>(const cytnx_complex64 &lc, const Tensor &Rt){
        return cytnx::linalg::Mul(lc,Rt);
    }
    template<>
    Tensor operator*<cytnx_double>(const cytnx_double &lc, const Tensor &Rt){
        return cytnx::linalg::Mul(lc,Rt);
    }
    template<>
    Tensor operator*<cytnx_float>(const cytnx_float &lc, const Tensor &Rt){
        return cytnx::linalg::Mul(lc,Rt);
    }
    template<>
    Tensor operator*<cytnx_int64>(const cytnx_int64 &lc, const Tensor &Rt){
        return cytnx::linalg::Mul(lc,Rt);
    }
    template<>
    Tensor operator*<cytnx_uint64>(const cytnx_uint64 &lc, const Tensor &Rt){
        return cytnx::linalg::Mul(lc,Rt);
    }
    template<>
    Tensor operator*<cytnx_int32>(const cytnx_int32 &lc, const Tensor &Rt){
        return cytnx::linalg::Mul(lc,Rt);
    }
    template<>
    Tensor operator*<cytnx_uint32>(const cytnx_uint32 &lc, const Tensor &Rt){
        return cytnx::linalg::Mul(lc,Rt);
    }

    template<>
    Tensor operator*<cytnx_complex128>(const Tensor &Lt, const cytnx_complex128 &rc){
       return cytnx::linalg::Mul(Lt,rc);
    }
    template<>
    Tensor operator*<cytnx_complex64>(const Tensor &Lt, const cytnx_complex64 &rc){
       return cytnx::linalg::Mul(Lt,rc);
    }
    template<>
    Tensor operator*<cytnx_double>(const Tensor &Lt, const cytnx_double &rc){
       return cytnx::linalg::Mul(Lt,rc);
    }
    template<>
    Tensor operator*<cytnx_float>(const Tensor &Lt, const cytnx_float &rc){
       return cytnx::linalg::Mul(Lt,rc);
    }
    template<>
    Tensor operator*<cytnx_int64>(const Tensor &Lt, const cytnx_int64 &rc){
       return cytnx::linalg::Mul(Lt,rc);
    }
    template<>
    Tensor operator*<cytnx_uint64>(const Tensor &Lt, const cytnx_uint64 &rc){
       return cytnx::linalg::Mul(Lt,rc);
    }
    template<>
    Tensor operator*<cytnx_int32>(const Tensor &Lt, const cytnx_int32 &rc){
       return cytnx::linalg::Mul(Lt,rc);
    }
    template<>
    Tensor operator*<cytnx_uint32>(const Tensor &Lt, const cytnx_uint32 &rc){
       return cytnx::linalg::Mul(Lt,rc);
    }


    
    template<> Tensor operator*<cytnx_complex128>(const Tensor &, const cytnx_complex128&);
    template<> Tensor operator*<cytnx_complex64>(const Tensor &, const cytnx_complex64&);
    template<> Tensor operator*<cytnx_double>(const Tensor &, const cytnx_double&);
    template<> Tensor operator*<cytnx_float>(const Tensor &, const cytnx_float&);
    template<> Tensor operator*<cytnx_int64>(const Tensor &, const cytnx_int64&);
    template<> Tensor operator*<cytnx_uint64>(const Tensor &, const cytnx_uint64&);
    template<> Tensor operator*<cytnx_int32>(const Tensor &, const cytnx_int32&);
    template<> Tensor operator*<cytnx_uint32>(const Tensor &, const cytnx_uint32&);

    template<> Tensor operator*<cytnx_complex128>( const cytnx_complex128&,const Tensor &);
    template<> Tensor operator*<cytnx_complex64>( const cytnx_complex64&,const Tensor &);
    template<> Tensor operator*<cytnx_double>( const cytnx_double&,const Tensor &);
    template<> Tensor operator*<cytnx_float>( const cytnx_float&,const Tensor &);
    template<> Tensor operator*<cytnx_int64>( const cytnx_int64&,const Tensor &);
    template<> Tensor operator*<cytnx_uint64>( const cytnx_uint64&,const Tensor &);
    template<> Tensor operator*<cytnx_int32>( const cytnx_int32&,const Tensor &);
    template<> Tensor operator*<cytnx_uint32>( const cytnx_uint32&,const Tensor &);

}//cytnx


