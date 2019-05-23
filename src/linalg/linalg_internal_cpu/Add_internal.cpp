#include "linalg/linalg_internal_cpu/Add_internal.hpp"
#include "utils/utils_internal.hpp"

#ifdef UNI_OMP
    #include <omp.h>
#endif

namespace tor10{

    namespace linalg_internal{

        /// Add
        void Add_internal_cdtcd(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
            tor10_complex128 *_out = (tor10_complex128*)out->Mem;
            tor10_complex128 *_Lin = (tor10_complex128*)Lin->Mem;
            tor10_complex128 *_Rin = (tor10_complex128*)Rin->Mem;
            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] + _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[0];
                    }
            }else{
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[i];
                    }
            }
        }
        void Add_internal_cdtcf(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
            tor10_complex128 *_out = (tor10_complex128*)out->Mem;
            tor10_complex128 *_Lin = (tor10_complex128*)Lin->Mem;
            tor10_complex64 *_Rin = (tor10_complex64*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] + _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[0];
                    }
            }else{
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[i];
                    }
            }

        }
        void Add_internal_cdtd(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
            tor10_complex128 *_out = (tor10_complex128*)out->Mem;
            tor10_complex128 *_Lin = (tor10_complex128*)Lin->Mem;
            tor10_double *_Rin = (tor10_double*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] + _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[0];
                    }
            }else{
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[i];
                    }
            }


        }
        void Add_internal_cdtf(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
            tor10_complex128 *_out = (tor10_complex128*)out->Mem;
            tor10_complex128 *_Lin = (tor10_complex128*)Lin->Mem;
            tor10_float *_Rin = (tor10_float*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] + _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[0];
                    }
            }else{
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[i];
                    }
            }



        }
        void Add_internal_cdtu64(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
            tor10_complex128 *_out = (tor10_complex128*)out->Mem;
            tor10_complex128 *_Lin = (tor10_complex128*)Lin->Mem;
            tor10_uint64 *_Rin = (tor10_uint64*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] + _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[0];
                    }
            }else{
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[i];
                    }
            }



        }
        void Add_internal_cdtu32(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
            tor10_complex128 *_out = (tor10_complex128*)out->Mem;
            tor10_complex128 *_Lin = (tor10_complex128*)Lin->Mem;
            tor10_uint32 *_Rin = (tor10_uint32*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] + _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[0];
                    }
            }else{
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[i];
                    }
            }




        }
        void Add_internal_cdti64(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
            tor10_complex128 *_out = (tor10_complex128*)out->Mem;
            tor10_complex128 *_Lin = (tor10_complex128*)Lin->Mem;
            tor10_int64 *_Rin = (tor10_int64*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] + _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[0];
                    }
            }else{
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[i];
                    }
            }


        }
        void Add_internal_cdti32(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
            tor10_complex128 *_out = (tor10_complex128*)out->Mem;
            tor10_complex128 *_Lin = (tor10_complex128*)Lin->Mem;
            tor10_int32 *_Rin = (tor10_int32*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] + _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[0];
                    }
            }else{
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[i];
                    }
            }


        }

        void Add_internal_cftcd(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){

		Add_internal_cdtcf(out,Rin,Lin,len);

	}
        void Add_internal_cftcf(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
            tor10_complex64 *_out = (tor10_complex64*)out->Mem;
            tor10_complex64 *_Lin = (tor10_complex64*)Lin->Mem;
            tor10_complex64 *_Rin = (tor10_complex64*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] + _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[0];
                    }
            }else{
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[i];
                    }
            }

        }
        void Add_internal_cftd(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
            tor10_complex64 *_out = (tor10_complex64*)out->Mem;
            tor10_complex64 *_Lin = (tor10_complex64*)Lin->Mem;
            tor10_double *_Rin = (tor10_double*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] + _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[0];
                    }
            }else{
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[i];
                    }
            }


        }
        void Add_internal_cftf(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
            tor10_complex64 *_out = (tor10_complex64*)out->Mem;
            tor10_complex64 *_Lin = (tor10_complex64*)Lin->Mem;
            tor10_float *_Rin = (tor10_float*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] + _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[0];
                    }
            }else{
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[i];
                    }
            }

        }
        void Add_internal_cftu64(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
            tor10_complex64 *_out = (tor10_complex64*)out->Mem;
            tor10_complex64 *_Lin = (tor10_complex64*)Lin->Mem;
            tor10_uint64 *_Rin = (tor10_uint64*)Rin->Mem;
            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] + _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[0];
                    }
            }else{
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[i];
                    }
            }


        }
        void Add_internal_cftu32(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
            tor10_complex64 *_out = (tor10_complex64*)out->Mem;
            tor10_complex64 *_Lin = (tor10_complex64*)Lin->Mem;
            tor10_uint32 *_Rin = (tor10_uint32*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] + _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[0];
                    }
            }else{
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[i];
                    }
            }


        }
        void Add_internal_cfti64(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
            tor10_complex64 *_out = (tor10_complex64*)out->Mem;
            tor10_complex64 *_Lin = (tor10_complex64*)Lin->Mem;
            tor10_int64 *_Rin = (tor10_int64*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] + _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[0];
                    }
            }else{
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[i];
                    }
            }

        }
        void Add_internal_cfti32(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
            tor10_complex64 *_out = (tor10_complex64*)out->Mem;
            tor10_complex64 *_Lin = (tor10_complex64*)Lin->Mem;
            tor10_int32 *_Rin = (tor10_int32*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] + _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[0];
                    }
            }else{
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[i];
                    }
            }

        }

        void Add_internal_dtcd(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
             Add_internal_cdtd(out,Rin,Lin,len);
        }
        void Add_internal_dtcf(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
             Add_internal_cftd(out,Rin,Lin,len);
        }
        void Add_internal_dtd(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
            tor10_double *_out = (tor10_double*)out->Mem;
            tor10_double *_Lin = (tor10_double*)Lin->Mem;
            tor10_double *_Rin = (tor10_double*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] + _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[0];
                    }
            }else{
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[i];
                    }
            }


        }
        void Add_internal_dtf(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
            tor10_double *_out = (tor10_double*)out->Mem;
            tor10_double *_Lin = (tor10_double*)Lin->Mem;
            tor10_float *_Rin = (tor10_float*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] + _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[0];
                    }
            }else{
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[i];
                    }
            }
        }
        void Add_internal_dtu64(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){

            tor10_double *_out = (tor10_double*)out->Mem;
            tor10_double *_Lin = (tor10_double*)Lin->Mem;
            tor10_uint64 *_Rin = (tor10_uint64*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] + _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[0];
                    }
            }else{
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[i];
                    }
            }

        }
        void Add_internal_dtu32(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){

            tor10_double *_out = (tor10_double*)out->Mem;
            tor10_double *_Lin = (tor10_double*)Lin->Mem;
            tor10_uint32 *_Rin = (tor10_uint32*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] + _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[0];
                    }
            }else{
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[i];
                    }
            }
        }
        void Add_internal_dti64(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){

            tor10_double *_out = (tor10_double*)out->Mem;
            tor10_double *_Lin = (tor10_double*)Lin->Mem;
            tor10_int64 *_Rin = (tor10_int64*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] + _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[0];
                    }
            }else{
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[i];
                    }
            }

        }
        void Add_internal_dti32(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){

            tor10_double *_out = (tor10_double*)out->Mem;
            tor10_double *_Lin = (tor10_double*)Lin->Mem;
            tor10_int32 *_Rin = (tor10_int32*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] + _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[0];
                    }
            }else{
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[i];
                    }
            }

        }

        void Add_internal_ftcd(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
             Add_internal_cdtf(out,Rin,Lin,len);
        }
        void Add_internal_ftcf(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
             Add_internal_cftf(out,Rin,Lin,len);
        }
        void Add_internal_ftd(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
             Add_internal_dtf(out,Rin,Lin,len);
        }
        void Add_internal_ftf(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
            tor10_float *_out = (tor10_float*)out->Mem;
            tor10_float *_Lin = (tor10_float*)Lin->Mem;
            tor10_float *_Rin = (tor10_float*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] + _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[0];
                    }
            }else{
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[i];
                    }
            }

        }
        void Add_internal_ftu64(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
            tor10_float *_out = (tor10_float*)out->Mem;
            tor10_float *_Lin = (tor10_float*)Lin->Mem;
            tor10_uint64 *_Rin = (tor10_uint64*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] + _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[0];
                    }
            }else{
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[i];
                    }
            }

        }
        void Add_internal_ftu32(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
            tor10_float *_out = (tor10_float*)out->Mem;
            tor10_float *_Lin = (tor10_float*)Lin->Mem;
            tor10_uint32 *_Rin = (tor10_uint32*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] + _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[0];
                    }
            }else{
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[i];
                    }
            }
        }
        void Add_internal_fti64(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
            tor10_float *_out = (tor10_float*)out->Mem;
            tor10_float *_Lin = (tor10_float*)Lin->Mem;
            tor10_int64 *_Rin = (tor10_int64*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] + _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[0];
                    }
            }else{
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[i];
                    }
            }

        }
        void Add_internal_fti32(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
            tor10_float *_out = (tor10_float*)out->Mem;
            tor10_float *_Lin = (tor10_float*)Lin->Mem;
            tor10_int32 *_Rin = (tor10_int32*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] + _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[0];
                    }
            }else{
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[i];
                    }
            }

        }


        void Add_internal_i64tcd(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
             Add_internal_cdti64(out,Rin,Lin,len);
        }
        void Add_internal_i64tcf(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
             Add_internal_cfti64(out,Rin,Lin,len);
        }
        void Add_internal_i64td(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
             Add_internal_dti64(out,Rin,Lin,len);
        }
        void Add_internal_i64tf(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
             Add_internal_fti64(out,Rin,Lin,len);
        }
        void Add_internal_i64ti64(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
            tor10_int64 *_out = (tor10_int64*)out->Mem;
            tor10_int64 *_Lin = (tor10_int64*)Lin->Mem;
            tor10_int64 *_Rin = (tor10_int64*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] + _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[0];
                    }
            }else{
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[i];
                    }
            }

        }
        void Add_internal_i64tu64(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
            tor10_int64 *_out = (tor10_int64*)out->Mem;
            tor10_int64 *_Lin = (tor10_int64*)Lin->Mem;
            tor10_uint64 *_Rin = (tor10_uint64*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] + _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[0];
                    }
            }else{
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[i];
                    }
            }
        }
        void Add_internal_i64ti32(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
            tor10_int64 *_out = (tor10_int64*)out->Mem;
            tor10_int64 *_Lin = (tor10_int64*)Lin->Mem;
            tor10_int32 *_Rin = (tor10_int32*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] + _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[0];
                    }
            }else{
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[i];
                    }
            }
        }
        void Add_internal_i64tu32(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
            tor10_int64 *_out = (tor10_int64*)out->Mem;
            tor10_int64 *_Lin = (tor10_int64*)Lin->Mem;
            tor10_uint32 *_Rin = (tor10_uint32*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] + _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[0];
                    }
            }else{
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[i];
                    }
            }

        }


        void Add_internal_u64tcd(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
             Add_internal_cdtu64(out,Rin,Lin,len);
        }
        void Add_internal_u64tcf(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
             Add_internal_cftu64(out,Rin,Lin,len);
        }
        void Add_internal_u64td(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
             Add_internal_dtu64(out,Rin,Lin,len);
        }
        void Add_internal_u64tf(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
             Add_internal_ftu64(out,Rin,Lin,len);
        }
        void Add_internal_u64ti64(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
             Add_internal_i64tu64(out,Rin,Lin,len);
        }
        void Add_internal_u64tu64(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
            tor10_uint64 *_out = (tor10_uint64*)out->Mem;
            tor10_uint64 *_Lin = (tor10_uint64*)Lin->Mem;
            tor10_uint64 *_Rin = (tor10_uint64*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] + _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[0];
                    }
            }else{
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[i];
                    }
            }
        }
        void Add_internal_u64ti32(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
            tor10_uint64 *_out = (tor10_uint64*)out->Mem;
            tor10_uint64 *_Lin = (tor10_uint64*)Lin->Mem;
            tor10_int32 *_Rin = (tor10_int32*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] + _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[0];
                    }
            }else{
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[i];
                    }
            }
        }
        void Add_internal_u64tu32(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
            tor10_uint64 *_out = (tor10_uint64*)out->Mem;
            tor10_uint64 *_Lin = (tor10_uint64*)Lin->Mem;
            tor10_uint32 *_Rin = (tor10_uint32*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] + _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[0];
                    }
            }else{
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[i];
                    }
            }

        }

        void Add_internal_i32tcd(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
             Add_internal_cdti32(out,Rin,Lin,len);

        }
        void Add_internal_i32tcf(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
             Add_internal_cfti32(out,Rin,Lin,len);

        }
        void Add_internal_i32td(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
             Add_internal_dti32(out,Rin,Lin,len);

        }
        void Add_internal_i32tf(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
             Add_internal_fti32(out,Rin,Lin,len);

        }
        void Add_internal_i32ti64(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
             Add_internal_i64ti32(out,Rin,Lin,len);

        }
        void Add_internal_i32tu64(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
             Add_internal_u64ti32(out,Rin,Lin,len);

        }
        void Add_internal_i32ti32(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
            tor10_int32 *_out = (tor10_int32*)out->Mem;
            tor10_int32 *_Lin = (tor10_int32*)Lin->Mem;
            tor10_int32 *_Rin = (tor10_int32*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] + _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[0];
                    }
            }else{
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[i];
                    }
            }


        }
        void Add_internal_i32tu32(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
            tor10_int32 *_out = (tor10_int32*)out->Mem;
            tor10_int32 *_Lin = (tor10_int32*)Lin->Mem;
            tor10_uint32 *_Rin = (tor10_uint32*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] + _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[0];
                    }
            }else{
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[i];
                    }
            }

        }


        void Add_internal_u32tcd(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
             Add_internal_cdtu32(out,Rin,Lin,len);

        }
        void Add_internal_u32tcf(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
             Add_internal_cftu32(out,Rin,Lin,len);

        }
        void Add_internal_u32td(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
             Add_internal_dtu32(out,Rin,Lin,len);

        }
        void Add_internal_u32tf(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
             Add_internal_ftu32(out,Rin,Lin,len);

        }
        void Add_internal_u32ti64(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
             Add_internal_i64tu32(out,Rin,Lin,len);

        }
        void Add_internal_u32tu64(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
             Add_internal_u64tu32(out,Rin,Lin,len);

        }
        void Add_internal_u32ti32(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
             Add_internal_i32tu32(out,Rin,Lin,len);

        }
        void Add_internal_u32tu32(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len){
            tor10_uint32 *_out = (tor10_uint32*)out->Mem;
            tor10_uint32 *_Lin = (tor10_uint32*)Lin->Mem;
            tor10_uint32 *_Rin = (tor10_uint32*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] + _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[0];
                    }
            }else{
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] + _Rin[i];
                    }
            }
        }





    }//namespace linalg_internal
}//namespace tor10


