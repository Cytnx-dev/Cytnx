#include "Outer_internal.hpp"
#include "utils/utils_internal_interface.hpp"
#include "utils/complex_arithmetic.hpp"
#include "lapack_wrapper.hpp"

#ifdef UNI_OMP
    #include <omp.h>
#endif

namespace cytnx{

    namespace linalg_internal{

        /// Outer
        void Outer_internal_cdtcd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_complex128 *_Lin = (cytnx_complex128*)Lin->Mem;
            cytnx_complex128 *_Rin = (cytnx_complex128*)Rin->Mem;
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }

            

        }
        void Outer_internal_cdtcf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_complex128 *_Lin = (cytnx_complex128*)Lin->Mem;
            cytnx_complex64 *_Rin = (cytnx_complex64*)Rin->Mem;

            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }


        }
        void Outer_internal_cdtd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_complex128 *_Lin = (cytnx_complex128*)Lin->Mem;
            cytnx_double *_Rin = (cytnx_double*)Rin->Mem;

            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }


        }
        void Outer_internal_cdtf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_complex128 *_Lin = (cytnx_complex128*)Lin->Mem;
            cytnx_float *_Rin = (cytnx_float*)Rin->Mem;

            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }



        }
        void Outer_internal_cdtu64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_complex128 *_Lin = (cytnx_complex128*)Lin->Mem;
            cytnx_uint64 *_Rin = (cytnx_uint64*)Rin->Mem;

            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }




        }
        void Outer_internal_cdtu32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_complex128 *_Lin = (cytnx_complex128*)Lin->Mem;
            cytnx_uint32 *_Rin = (cytnx_uint32*)Rin->Mem;

            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }




        }
        void Outer_internal_cdti64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_complex128 *_Lin = (cytnx_complex128*)Lin->Mem;
            cytnx_int64 *_Rin = (cytnx_int64*)Rin->Mem;

            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }




        }
        void Outer_internal_cdti32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_complex128 *_Lin = (cytnx_complex128*)Lin->Mem;
            cytnx_int32 *_Rin = (cytnx_int32*)Rin->Mem;

            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }



        }
        void Outer_internal_cdti16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_complex128 *_Lin = (cytnx_complex128*)Lin->Mem;
            cytnx_int16 *_Rin = (cytnx_int16*)Rin->Mem;

            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }



        }
        void Outer_internal_cdtu16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_complex128 *_Lin = (cytnx_complex128*)Lin->Mem;
            cytnx_uint16 *_Rin = (cytnx_uint16*)Rin->Mem;

            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }



        }
        void Outer_internal_cdtb(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_complex128 *_Lin = (cytnx_complex128*)Lin->Mem;
            cytnx_bool *_Rin = (cytnx_bool*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*cytnx_complex128(_Rin[(i%i2)*j2+(r%j2)],0);
                    }
                }


        }
//----------------------------------------
        void Outer_internal_cftcd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){

            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_complex64 *_Lin  = (cytnx_complex64*)Lin->Mem;
            cytnx_complex128 *_Rin = (cytnx_complex128*)Rin->Mem;

            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }

	}
        void Outer_internal_cftcf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_complex64 *_out = (cytnx_complex64*)out->Mem;
            cytnx_complex64 *_Lin = (cytnx_complex64*)Lin->Mem;
            cytnx_complex64 *_Rin = (cytnx_complex64*)Rin->Mem;

            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }
        }
        void Outer_internal_cftd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_complex64 *_out = (cytnx_complex64*)out->Mem;
            cytnx_complex64 *_Lin = (cytnx_complex64*)Lin->Mem;
            cytnx_double *_Rin = (cytnx_double*)Rin->Mem;

            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }


        }
        void Outer_internal_cftf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_complex64 *_out = (cytnx_complex64*)out->Mem;
            cytnx_complex64 *_Lin = (cytnx_complex64*)Lin->Mem;
            cytnx_float *_Rin = (cytnx_float*)Rin->Mem;

            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }

        }
        void Outer_internal_cftu64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_complex64 *_out = (cytnx_complex64*)out->Mem;
            cytnx_complex64 *_Lin = (cytnx_complex64*)Lin->Mem;
            cytnx_uint64 *_Rin = (cytnx_uint64*)Rin->Mem;
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }

        }
        void Outer_internal_cftu32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_complex64 *_out = (cytnx_complex64*)out->Mem;
            cytnx_complex64 *_Lin = (cytnx_complex64*)Lin->Mem;
            cytnx_uint32 *_Rin = (cytnx_uint32*)Rin->Mem;

            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }


        }
        void Outer_internal_cfti64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_complex64 *_out = (cytnx_complex64*)out->Mem;
            cytnx_complex64 *_Lin = (cytnx_complex64*)Lin->Mem;
            cytnx_int64 *_Rin = (cytnx_int64*)Rin->Mem;

            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }

        }
        void Outer_internal_cfti32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_complex64 *_out = (cytnx_complex64*)out->Mem;
            cytnx_complex64 *_Lin = (cytnx_complex64*)Lin->Mem;
            cytnx_int32 *_Rin = (cytnx_int32*)Rin->Mem;

            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }

        }
        void Outer_internal_cfti16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_complex64 *_out = (cytnx_complex64*)out->Mem;
            cytnx_complex64 *_Lin = (cytnx_complex64*)Lin->Mem;
            cytnx_int16 *_Rin = (cytnx_int16*)Rin->Mem;

            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }

        }
        void Outer_internal_cftu16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_complex64 *_out = (cytnx_complex64*)out->Mem;
            cytnx_complex64 *_Lin = (cytnx_complex64*)Lin->Mem;
            cytnx_uint16 *_Rin = (cytnx_uint16*)Rin->Mem;

            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }
        }
        void Outer_internal_cftb(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_complex64 *_out = (cytnx_complex64*)out->Mem;
            cytnx_complex64 *_Lin = (cytnx_complex64*)Lin->Mem;
            cytnx_bool *_Rin = (cytnx_bool*)Rin->Mem;

            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*cytnx_complex64(_Rin[(i%i2)*j2+(r%j2)],0);
                    }
                }
        }

//-------------------------
        void Outer_internal_dtcd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_double *_Lin = (cytnx_double*)Lin->Mem;
            cytnx_complex128 *_Rin = (cytnx_complex128*)Rin->Mem;

            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }
        }
        void Outer_internal_dtcf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_complex64 *_out = (cytnx_complex64*)out->Mem;
            cytnx_double *_Lin = (cytnx_double*)Lin->Mem;
            cytnx_complex64 *_Rin = (cytnx_complex64*)Rin->Mem;

            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }
        }
        void Outer_internal_dtd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_double *_Lin = (cytnx_double*)Lin->Mem;
            cytnx_double *_Rin = (cytnx_double*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }

        }
        void Outer_internal_dtf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_double *_Lin = (cytnx_double*)Lin->Mem;
            cytnx_float *_Rin = (cytnx_float*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }
        }
        void Outer_internal_dtu64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){

            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_double *_Lin = (cytnx_double*)Lin->Mem;
            cytnx_uint64 *_Rin = (cytnx_uint64*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }

        }
        void Outer_internal_dtu32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){

            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_double *_Lin = (cytnx_double*)Lin->Mem;
            cytnx_uint32 *_Rin = (cytnx_uint32*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }
        }
        void Outer_internal_dti64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){

            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_double *_Lin = (cytnx_double*)Lin->Mem;
            cytnx_int64 *_Rin = (cytnx_int64*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }

        }
        void Outer_internal_dti32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){

            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_double *_Lin = (cytnx_double*)Lin->Mem;
            cytnx_int32 *_Rin = (cytnx_int32*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }

        }
        void Outer_internal_dti16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){

            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_double *_Lin = (cytnx_double*)Lin->Mem;
            cytnx_int16 *_Rin = (cytnx_int16*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }

        }
        void Outer_internal_dtu16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){

            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_double *_Lin = (cytnx_double*)Lin->Mem;
            cytnx_uint16 *_Rin = (cytnx_uint16*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }

        }
        void Outer_internal_dtb(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){

            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_double *_Lin = (cytnx_double*)Lin->Mem;
            cytnx_bool *_Rin = (cytnx_bool*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*double(_Rin[(i%i2)*j2+(r%j2)]);
                    }
                }

        }
//-------------------------------
        void Outer_internal_ftcd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_float *_Lin = (cytnx_float*)Lin->Mem;
            cytnx_complex128 *_Rin = (cytnx_complex128*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }
        }
        void Outer_internal_ftcf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_complex64 *_out = (cytnx_complex64*)out->Mem;
            cytnx_float *_Lin = (cytnx_float*)Lin->Mem;
            cytnx_complex64 *_Rin = (cytnx_complex64*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }
        }
        void Outer_internal_ftd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_float *_Lin = (cytnx_float*)Lin->Mem;
            cytnx_double *_Rin = (cytnx_double*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }
        }
        void Outer_internal_ftf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_float *_Lin = (cytnx_float*)Lin->Mem;
            cytnx_float *_Rin = (cytnx_float*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }

        }
        void Outer_internal_ftu64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_float *_Lin = (cytnx_float*)Lin->Mem;
            cytnx_uint64 *_Rin = (cytnx_uint64*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }

        }
        void Outer_internal_ftu32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_float *_Lin = (cytnx_float*)Lin->Mem;
            cytnx_uint32 *_Rin = (cytnx_uint32*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }
        }
        void Outer_internal_fti64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_float *_Lin = (cytnx_float*)Lin->Mem;
            cytnx_int64 *_Rin = (cytnx_int64*)Rin->Mem;

            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<out->len;i++){
                    _out[i] = _Lin[cytnx_uint64(i/Rin->len)]*_Rin[i%Rin->len];
                }

        }
        void Outer_internal_fti32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_float *_Lin = (cytnx_float*)Lin->Mem;
            cytnx_int32 *_Rin = (cytnx_int32*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }

        }
        void Outer_internal_fti16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_float *_Lin = (cytnx_float*)Lin->Mem;
            cytnx_int16 *_Rin = (cytnx_int16*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }

        }
        void Outer_internal_ftu16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_float *_Lin = (cytnx_float*)Lin->Mem;
            cytnx_uint16 *_Rin = (cytnx_uint16*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }

        }
        void Outer_internal_ftb(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_float *_Lin = (cytnx_float*)Lin->Mem;
            cytnx_bool *_Rin = (cytnx_bool*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*float(_Rin[(i%i2)*j2+(r%j2)]);
                    }
                }

        }

//----------------------------------------
        void Outer_internal_i64tcd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_int64 *_Lin = (cytnx_int64*)Lin->Mem;
            cytnx_complex128 *_Rin = (cytnx_complex128*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }
        }
        void Outer_internal_i64tcf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_complex64 *_out = (cytnx_complex64*)out->Mem;
            cytnx_int64 *_Lin = (cytnx_int64*)Lin->Mem;
            cytnx_complex64 *_Rin = (cytnx_complex64*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }
        }
        void Outer_internal_i64td(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_int64 *_Lin = (cytnx_int64*)Lin->Mem;
            cytnx_double *_Rin = (cytnx_double*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }
        }
        void Outer_internal_i64tf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_int64 *_Lin = (cytnx_int64*)Lin->Mem;
            cytnx_float *_Rin = (cytnx_float*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }
        }
        void Outer_internal_i64ti64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_int64 *_out = (cytnx_int64*)out->Mem;
            cytnx_int64 *_Lin = (cytnx_int64*)Lin->Mem;
            cytnx_int64 *_Rin = (cytnx_int64*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }

        }
        void Outer_internal_i64tu64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_int64 *_out = (cytnx_int64*)out->Mem;
            cytnx_int64 *_Lin = (cytnx_int64*)Lin->Mem;
            cytnx_uint64 *_Rin = (cytnx_uint64*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }
        }
        void Outer_internal_i64ti32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_int64 *_out = (cytnx_int64*)out->Mem;
            cytnx_int64 *_Lin = (cytnx_int64*)Lin->Mem;
            cytnx_int32 *_Rin = (cytnx_int32*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }
        }
        void Outer_internal_i64tu32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_int64 *_out = (cytnx_int64*)out->Mem;
            cytnx_int64 *_Lin = (cytnx_int64*)Lin->Mem;
            cytnx_uint32 *_Rin = (cytnx_uint32*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }

        }
        void Outer_internal_i64ti16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_int64 *_out = (cytnx_int64*)out->Mem;
            cytnx_int64 *_Lin = (cytnx_int64*)Lin->Mem;
            cytnx_int16 *_Rin = (cytnx_int16*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }
        }
        void Outer_internal_i64tu16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_int64 *_out = (cytnx_int64*)out->Mem;
            cytnx_int64 *_Lin = (cytnx_int64*)Lin->Mem;
            cytnx_uint16 *_Rin = (cytnx_uint16*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }

        }
        void Outer_internal_i64tb(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_int64 *_out = (cytnx_int64*)out->Mem;
            cytnx_int64 *_Lin = (cytnx_int64*)Lin->Mem;
            cytnx_bool *_Rin = (cytnx_bool*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*cytnx_int64(_Rin[(i%i2)*j2+(r%j2)]);
                    }
                }

        }
//-----------------------------------
        void Outer_internal_u64tcd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_uint64 *_Lin = (cytnx_uint64*)Lin->Mem;
            cytnx_complex128 *_Rin = (cytnx_complex128*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }
        }
        void Outer_internal_u64tcf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_complex64 *_out = (cytnx_complex64*)out->Mem;
            cytnx_uint64 *_Lin = (cytnx_uint64*)Lin->Mem;
            cytnx_complex64 *_Rin = (cytnx_complex64*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }
        }
        void Outer_internal_u64td(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_uint64 *_Lin = (cytnx_uint64*)Lin->Mem;
            cytnx_double *_Rin = (cytnx_double*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }
        }
        void Outer_internal_u64tf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_uint64 *_Lin = (cytnx_uint64*)Lin->Mem;
            cytnx_float *_Rin = (cytnx_float*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }
        }
        void Outer_internal_u64ti64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_int64 *_out = (cytnx_int64*)out->Mem;
            cytnx_uint64 *_Lin = (cytnx_uint64*)Lin->Mem;
            cytnx_int64 *_Rin = (cytnx_int64*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }
        }
        void Outer_internal_u64tu64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_uint64 *_out = (cytnx_uint64*)out->Mem;
            cytnx_uint64 *_Lin = (cytnx_uint64*)Lin->Mem;
            cytnx_uint64 *_Rin = (cytnx_uint64*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }
        }
        void Outer_internal_u64ti32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_uint64 *_out = (cytnx_uint64*)out->Mem;
            cytnx_uint64 *_Lin = (cytnx_uint64*)Lin->Mem;
            cytnx_int32 *_Rin = (cytnx_int32*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }
        }
        void Outer_internal_u64tu32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_uint64 *_out = (cytnx_uint64*)out->Mem;
            cytnx_uint64 *_Lin = (cytnx_uint64*)Lin->Mem;
            cytnx_uint32 *_Rin = (cytnx_uint32*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }

        }
        void Outer_internal_u64ti16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_uint64 *_out = (cytnx_uint64*)out->Mem;
            cytnx_uint64 *_Lin = (cytnx_uint64*)Lin->Mem;
            cytnx_int16 *_Rin = (cytnx_int16*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }
        }
        void Outer_internal_u64tu16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_uint64 *_out = (cytnx_uint64*)out->Mem;
            cytnx_uint64 *_Lin = (cytnx_uint64*)Lin->Mem;
            cytnx_uint16 *_Rin = (cytnx_uint16*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }

        }
        void Outer_internal_u64tb(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_uint64 *_out = (cytnx_uint64*)out->Mem;
            cytnx_uint64 *_Lin = (cytnx_uint64*)Lin->Mem;
            cytnx_bool *_Rin = (cytnx_bool*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*cytnx_uint64(_Rin[(i%i2)*j2+(r%j2)]);
                    }
                }

        }
//-------------------------------------
        void Outer_internal_i32tcd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_int32 *_Lin = (cytnx_int32*)Lin->Mem;
            cytnx_complex128 *_Rin = (cytnx_complex128*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }

        }
        void Outer_internal_i32tcf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_complex64 *_out = (cytnx_complex64*)out->Mem;
            cytnx_int32 *_Lin = (cytnx_int32*)Lin->Mem;
            cytnx_complex64 *_Rin = (cytnx_complex64*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }

        }
        void Outer_internal_i32td(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_int32 *_Lin = (cytnx_int32*)Lin->Mem;
            cytnx_double *_Rin = (cytnx_double*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }


        }
        void Outer_internal_i32tf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_int32 *_Lin = (cytnx_int32*)Lin->Mem;
            cytnx_float *_Rin = (cytnx_float*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }


        }
        void Outer_internal_i32ti64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_int64 *_out = (cytnx_int64*)out->Mem;
            cytnx_int32 *_Lin = (cytnx_int32*)Lin->Mem;
            cytnx_int64 *_Rin = (cytnx_int64*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }


        }
        void Outer_internal_i32tu64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_uint64 *_out = (cytnx_uint64*)out->Mem;
            cytnx_int32 *_Lin = (cytnx_int32*)Lin->Mem;
            cytnx_uint64 *_Rin = (cytnx_uint64*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }

        }
        void Outer_internal_i32ti32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_int32 *_out = (cytnx_int32*)out->Mem;
            cytnx_int32 *_Lin = (cytnx_int32*)Lin->Mem;
            cytnx_int32 *_Rin = (cytnx_int32*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }

        }
        void Outer_internal_i32tu32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_int32 *_out = (cytnx_int32*)out->Mem;
            cytnx_int32 *_Lin = (cytnx_int32*)Lin->Mem;
            cytnx_uint32 *_Rin = (cytnx_uint32*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }

        }
        void Outer_internal_i32ti16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_int32 *_out = (cytnx_int32*)out->Mem;
            cytnx_int32 *_Lin = (cytnx_int32*)Lin->Mem;
            cytnx_int16 *_Rin = (cytnx_int16*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }


        }
        void Outer_internal_i32tu16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_int32 *_out = (cytnx_int32*)out->Mem;
            cytnx_int32 *_Lin = (cytnx_int32*)Lin->Mem;
            cytnx_uint16 *_Rin = (cytnx_uint16*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }
        }
        void Outer_internal_i32tb(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_int32 *_out = (cytnx_int32*)out->Mem;
            cytnx_int32 *_Lin = (cytnx_int32*)Lin->Mem;
            cytnx_bool *_Rin = (cytnx_bool*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*cytnx_int32(_Rin[(i%i2)*j2+(r%j2)]);
                    }
                }

        }

//----------------------------------------
        void Outer_internal_u32tcd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_uint32 *_Lin = (cytnx_uint32*)Lin->Mem;
            cytnx_complex128 *_Rin = (cytnx_complex128*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }


        }
        void Outer_internal_u32tcf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_complex64 *_out = (cytnx_complex64*)out->Mem;
            cytnx_uint32 *_Lin = (cytnx_uint32*)Lin->Mem;
            cytnx_complex64 *_Rin = (cytnx_complex64*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }
        }
        void Outer_internal_u32td(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_uint32 *_Lin = (cytnx_uint32*)Lin->Mem;
            cytnx_double *_Rin = (cytnx_double*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }

        }
        void Outer_internal_u32tf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_uint32 *_Lin = (cytnx_uint32*)Lin->Mem;
            cytnx_float *_Rin = (cytnx_float*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }

        }
        void Outer_internal_u32ti64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_int64 *_out = (cytnx_int64*)out->Mem;
            cytnx_uint32 *_Lin = (cytnx_uint32*)Lin->Mem;
            cytnx_int64 *_Rin = (cytnx_int64*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }

        }
        void Outer_internal_u32tu64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_uint64 *_out = (cytnx_uint64*)out->Mem;
            cytnx_uint32 *_Lin = (cytnx_uint32*)Lin->Mem;
            cytnx_uint64 *_Rin = (cytnx_uint64*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }

        }
        void Outer_internal_u32ti32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_int32 *_out = (cytnx_int32*)out->Mem;
            cytnx_uint32 *_Lin = (cytnx_uint32*)Lin->Mem;
            cytnx_int32 *_Rin = (cytnx_int32*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }

        }
        void Outer_internal_u32tu32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_uint32 *_out = (cytnx_uint32*)out->Mem;
            cytnx_uint32 *_Lin = (cytnx_uint32*)Lin->Mem;
            cytnx_uint32 *_Rin = (cytnx_uint32*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }
        }
        void Outer_internal_u32ti16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_uint32 *_out = (cytnx_uint32*)out->Mem;
            cytnx_uint32 *_Lin = (cytnx_uint32*)Lin->Mem;
            cytnx_int16 *_Rin = (cytnx_int16*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }


        }
        void Outer_internal_u32tu16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_uint32 *_out = (cytnx_uint32*)out->Mem;
            cytnx_uint32 *_Lin = (cytnx_uint32*)Lin->Mem;
            cytnx_uint16 *_Rin = (cytnx_uint16*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }
        }
        void Outer_internal_u32tb(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_uint32 *_out = (cytnx_uint32*)out->Mem;
            cytnx_uint32 *_Lin = (cytnx_uint32*)Lin->Mem;
            cytnx_bool *_Rin = (cytnx_bool*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*cytnx_uint32(_Rin[(i%i2)*j2+(r%j2)]);
                    }
                }
        }

//----------------------------------------
        void Outer_internal_i16tcd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_int16 *_Lin = (cytnx_int16*)Lin->Mem;
            cytnx_complex128 *_Rin = (cytnx_complex128*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }

        }
        void Outer_internal_i16tcf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_complex64 *_out = (cytnx_complex64*)out->Mem;
            cytnx_int16 *_Lin = (cytnx_int16*)Lin->Mem;
            cytnx_complex64 *_Rin = (cytnx_complex64*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }

        }
        void Outer_internal_i16td(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_int16 *_Lin = (cytnx_int16*)Lin->Mem;
            cytnx_double *_Rin = (cytnx_double*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }

        }
        void Outer_internal_i16tf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_int16 *_Lin = (cytnx_int16*)Lin->Mem;
            cytnx_float *_Rin = (cytnx_float*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }

        }
        void Outer_internal_i16ti64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_int64 *_out = (cytnx_int64*)out->Mem;
            cytnx_int16 *_Lin = (cytnx_int16*)Lin->Mem;
            cytnx_int64 *_Rin = (cytnx_int64*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }

        }
        void Outer_internal_i16tu64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_uint64 *_out = (cytnx_uint64*)out->Mem;
            cytnx_int16 *_Lin = (cytnx_int16*)Lin->Mem;
            cytnx_uint64 *_Rin = (cytnx_uint64*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }

        }
        void Outer_internal_i16ti32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_int32 *_out = (cytnx_int32*)out->Mem;
            cytnx_int16 *_Lin = (cytnx_int16*)Lin->Mem;
            cytnx_int32 *_Rin = (cytnx_int32*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }


        }
        void Outer_internal_i16tu32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_uint32 *_out = (cytnx_uint32*)out->Mem;
            cytnx_int16 *_Lin = (cytnx_int16*)Lin->Mem;
            cytnx_uint32 *_Rin = (cytnx_uint32*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }
        }
        void Outer_internal_i16ti16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_int16 *_out = (cytnx_int16*)out->Mem;
            cytnx_int16 *_Lin = (cytnx_int16*)Lin->Mem;
            cytnx_int16 *_Rin = (cytnx_int16*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }


        }
        void Outer_internal_i16tu16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_int16 *_out = (cytnx_int16*)out->Mem;
            cytnx_int16 *_Lin = (cytnx_int16*)Lin->Mem;
            cytnx_uint16 *_Rin = (cytnx_uint16*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }
        }
        void Outer_internal_i16tb(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_int16 *_out = (cytnx_int16*)out->Mem;
            cytnx_int16 *_Lin = (cytnx_int16*)Lin->Mem;
            cytnx_bool *_Rin = (cytnx_bool*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*cytnx_int16(_Rin[(i%i2)*j2+(r%j2)]);
                    }
                }
        }

//----------------------------------------
        void Outer_internal_u16tcd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_uint16 *_Lin = (cytnx_uint16*)Lin->Mem;
            cytnx_complex128 *_Rin = (cytnx_complex128*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }


        }
        void Outer_internal_u16tcf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_complex64 *_out = (cytnx_complex64*)out->Mem;
            cytnx_uint16 *_Lin = (cytnx_uint16*)Lin->Mem;
            cytnx_complex64 *_Rin = (cytnx_complex64*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }
        }
        void Outer_internal_u16td(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_uint16 *_Lin = (cytnx_uint16*)Lin->Mem;
            cytnx_double *_Rin = (cytnx_double*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }
        }
        void Outer_internal_u16tf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_uint16 *_Lin = (cytnx_uint16*)Lin->Mem;
            cytnx_float *_Rin = (cytnx_float*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }

        }
        void Outer_internal_u16ti64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_int64 *_out = (cytnx_int64*)out->Mem;
            cytnx_uint16 *_Lin = (cytnx_uint16*)Lin->Mem;
            cytnx_int64 *_Rin = (cytnx_int64*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }

        }
        void Outer_internal_u16tu64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_uint64 *_out = (cytnx_uint64*)out->Mem;
            cytnx_uint16 *_Lin = (cytnx_uint16*)Lin->Mem;
            cytnx_uint64 *_Rin = (cytnx_uint64*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }


        }
        void Outer_internal_u16ti32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_int32 *_out = (cytnx_int32*)out->Mem;
            cytnx_uint16 *_Lin = (cytnx_uint16*)Lin->Mem;
            cytnx_int32 *_Rin = (cytnx_int32*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }


        }
        void Outer_internal_u16tu32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_uint32 *_out = (cytnx_uint32*)out->Mem;
            cytnx_uint16 *_Lin = (cytnx_uint16*)Lin->Mem;
            cytnx_uint32 *_Rin = (cytnx_uint32*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }
        }
        void Outer_internal_u16ti16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_int16 *_out = (cytnx_int16*)out->Mem;
            cytnx_uint16 *_Lin = (cytnx_uint16*)Lin->Mem;
            cytnx_int16 *_Rin = (cytnx_int16*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }


        }
        void Outer_internal_u16tu16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_uint16 *_out = (cytnx_uint16*)out->Mem;
            cytnx_uint16 *_Lin = (cytnx_uint16*)Lin->Mem;
            cytnx_uint16 *_Rin = (cytnx_uint16*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }
        }
        void Outer_internal_u16tb(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_uint16 *_out = (cytnx_uint16*)out->Mem;
            cytnx_uint16 *_Lin = (cytnx_uint16*)Lin->Mem;
            cytnx_bool *_Rin = (cytnx_bool*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*cytnx_uint16(_Rin[(i%i2)*j2+(r%j2)]);
                    }
                }
        }

//----------------------------------------
        void Outer_internal_btcd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_bool *_Lin = (cytnx_bool*)Lin->Mem;
            cytnx_complex128 *_Rin = (cytnx_complex128*)Rin->Mem;


            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = cytnx_complex128(_Lin[cytnx_uint64(i/i2)*j1+(r/j2)],0)*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }

        }
        void Outer_internal_btcf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_complex64 *_out = (cytnx_complex64*)out->Mem;
            cytnx_bool *_Lin = (cytnx_bool*)Lin->Mem;
            cytnx_complex64 *_Rin = (cytnx_complex64*)Rin->Mem;

            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = cytnx_complex64(_Lin[cytnx_uint64(i/i2)*j1+(r/j2)],0)*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }

        }
        void Outer_internal_btd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_bool *_Lin = (cytnx_bool*)Lin->Mem;
            cytnx_double *_Rin = (cytnx_double*)Rin->Mem;

            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = double(_Lin[cytnx_uint64(i/i2)*j1+(r/j2)])*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }

        }
        void Outer_internal_btf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_bool *_Lin = (cytnx_bool*)Lin->Mem;
            cytnx_float *_Rin = (cytnx_float*)Rin->Mem;

            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = float(_Lin[cytnx_uint64(i/i2)*j1+(r/j2)])*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }

        }
        void Outer_internal_bti64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_int64 *_out = (cytnx_int64*)out->Mem;
            cytnx_bool *_Lin = (cytnx_bool*)Lin->Mem;
            cytnx_int64 *_Rin = (cytnx_int64*)Rin->Mem;

            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = cytnx_int64(_Lin[cytnx_uint64(i/i2)*j1+(r/j2)])*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }

        }
        void Outer_internal_btu64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_uint64 *_out = (cytnx_uint64*)out->Mem;
            cytnx_bool *_Lin = (cytnx_bool*)Lin->Mem;
            cytnx_uint64 *_Rin = (cytnx_uint64*)Rin->Mem;

            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = cytnx_uint64(_Lin[cytnx_uint64(i/i2)*j1+(r/j2)])*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }

        }
        void Outer_internal_bti32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_int32 *_out = (cytnx_int32*)out->Mem;
            cytnx_bool *_Lin = (cytnx_bool*)Lin->Mem;
            cytnx_int32 *_Rin = (cytnx_int32*)Rin->Mem;

            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = cytnx_int32(_Lin[cytnx_uint64(i/i2)*j1+(r/j2)])*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }


        }
        void Outer_internal_btu32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_uint32 *_out = (cytnx_uint32*)out->Mem;
            cytnx_bool *_Lin = (cytnx_bool*)Lin->Mem;
            cytnx_uint32 *_Rin = (cytnx_uint32*)Rin->Mem;

            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = cytnx_uint32(_Lin[cytnx_uint64(i/i2)*j1+(r/j2)])*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }
        }
        void Outer_internal_bti16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_int16 *_out = (cytnx_int16*)out->Mem;
            cytnx_bool *_Lin = (cytnx_bool*)Lin->Mem;
            cytnx_int16 *_Rin = (cytnx_int16*)Rin->Mem;

            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = cytnx_int16(_Lin[cytnx_uint64(i/i2)*j1+(r/j2)])*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }

        }
        void Outer_internal_btu16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_uint16 *_out = (cytnx_uint16*)out->Mem;
            cytnx_bool *_Lin = (cytnx_bool*)Lin->Mem;
            cytnx_uint16 *_Rin = (cytnx_uint16*)Rin->Mem;

            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = cytnx_uint16(_Lin[cytnx_uint64(i/i2)*j1+(r/j2)])*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }
        }
        void Outer_internal_btb(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_bool *_out = (cytnx_bool*)out->Mem;
            cytnx_bool *_Lin = (cytnx_bool*)Lin->Mem;
            cytnx_bool *_Rin = (cytnx_bool*)Rin->Mem;

            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<i1*i2;i++){
                    for(unsigned long long r=0;r<j1*j2;r++){
                        _out[i*(j1*j2)+r] = _Lin[cytnx_uint64(i/i2)*j1+(r/j2)]*_Rin[(i%i2)*j2+(r%j2)];
                    }
                }
        }

    }//namespace linalg_internal
}//namespace cytnx


