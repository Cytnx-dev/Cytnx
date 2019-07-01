#include "linalg/linalg_internal_cpu/Sub_internal.hpp"
#include "utils/utils_internal_interface.hpp"
#include "utils/utils.hpp"
#include <iostream>
#ifdef UNI_OMP
    #include <omp.h>
#endif

namespace cytnx{

    namespace linalg_internal{

        /// Sub
        void Sub_internal_cdtcd(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_complex128 *_Lin = (cytnx_complex128*)Lin->Mem;
            cytnx_complex128 *_Rin = (cytnx_complex128*)Rin->Mem;
            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] - _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] - _Rin[0];
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Lin[i] - _Rin[i];
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)] 
                                    - _Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)];
                        }

                    
                }
            }
        }
        void Sub_internal_cdtcf(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_complex128 *_Lin = (cytnx_complex128*)Lin->Mem;
            cytnx_complex64 *_Rin = (cytnx_complex64*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] - _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] - _Rin[0];
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Lin[i] - _Rin[i];
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)] 
                                    - _Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)];
                        }

                    
                }
            }

        }
        void Sub_internal_cdtd(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_complex128 *_Lin = (cytnx_complex128*)Lin->Mem;
            cytnx_double *_Rin = (cytnx_double*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0]; 
                        _out[i].real(_out[i].real()-_Rin[i]);
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i];
                        _out[i].real(_out[i].real()-_Rin[0]);
                    }
            }else{



                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Lin[i];
                            _out[i].real(_out[i].real()-_Rin[i]);
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)];
                            _out[i].real(_out[i].real()-_Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)]);
                        }

                    
                }


            }

        }
        void Sub_internal_cdtf(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_complex128 *_Lin = (cytnx_complex128*)Lin->Mem;
            cytnx_float *_Rin = (cytnx_float*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] ;
                        _out[i].real(_out[i].real()-_Rin[i]);
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] ;
                        _out[i].real(_out[i].real()-_Rin[0]);
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Lin[i];
                            _out[i].real(_out[i].real()-_Rin[i]);
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)];
                            _out[i].real(_out[i].real()-_Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)]);
                        }

                    
                }
            }


        }
        void Sub_internal_cdtu64(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_complex128 *_Lin = (cytnx_complex128*)Lin->Mem;
            cytnx_uint64 *_Rin = (cytnx_uint64*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] ;
                        _out[i].real(_out[i].real()-_Rin[i]);
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] ;
                        _out[i].real(_out[i].real()-_Rin[0]);
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Lin[i];
                            _out[i].real(_out[i].real()-_Rin[i]);
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)];
                            _out[i].real(_out[i].real()-_Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)]);
                        }

                    
                }
            }



        }
        void Sub_internal_cdtu32(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_complex128 *_Lin = (cytnx_complex128*)Lin->Mem;
            cytnx_uint32 *_Rin = (cytnx_uint32*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] ;
                        _out[i].real(_out[i].real()-_Rin[i]);
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] ;
                        _out[i].real(_out[i].real()-_Rin[0]);
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Lin[i];
                            _out[i].real(_out[i].real()-_Rin[i]);
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)];
                            _out[i].real(_out[i].real()-_Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)]);
                        }

                    
                }
            }



        }
        void Sub_internal_cdti64(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_complex128 *_Lin = (cytnx_complex128*)Lin->Mem;
            cytnx_int64 *_Rin = (cytnx_int64*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] ;
                        _out[i].real(_out[i].real()-_Rin[i]);
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] ;
                        _out[i].real(_out[i].real()-_Rin[0]);
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Lin[i];
                            _out[i].real(_out[i].real()-_Rin[i]);
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)];
                            _out[i].real(_out[i].real()-_Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)]);
                        }

                    
                }
            }


        }
        void Sub_internal_cdti32(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_complex128 *_Lin = (cytnx_complex128*)Lin->Mem;
            cytnx_int32 *_Rin = (cytnx_int32*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] ;
                        _out[i].real(_out[i].real()-_Rin[i]);
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] ;
                        _out[i].real(_out[i].real()-_Rin[0]);
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Lin[i];
                            _out[i].real(_out[i].real()-_Rin[i]);
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)];
                            _out[i].real(_out[i].real()-_Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)]);
                        }

                    
                }
            }

        }

        void Sub_internal_cftcd(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_complex64  *_Lin = (cytnx_complex64*)Lin->Mem;
            cytnx_complex128 *_Rin = (cytnx_complex128*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] - _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] - _Rin[0];
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Lin[i] - _Rin[i];
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)] 
                                    - _Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)];
                        }

                    
                }
            }

	}
        void Sub_internal_cftcf(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_complex64 *_out = (cytnx_complex64*)out->Mem;
            cytnx_complex64 *_Lin = (cytnx_complex64*)Lin->Mem;
            cytnx_complex64 *_Rin = (cytnx_complex64*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] - _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] - _Rin[0];
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Lin[i] - _Rin[i];
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)] 
                                    - _Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)];
                        }

                    
                }
            }

        }
        void Sub_internal_cftd(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_complex64 *_out = (cytnx_complex64*)out->Mem;
            cytnx_complex64 *_Lin = (cytnx_complex64*)Lin->Mem;
            cytnx_double *_Rin = (cytnx_double*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] ;
                        _out[i].real(_out[i].real()-_Rin[i]);
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] ;
                        _out[i].real(_out[i].real()-_Rin[0]);
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Lin[i];
                            _out[i].real(_out[i].real()-_Rin[i]);
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)];
                            _out[i].real(_out[i].real()-_Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)]);
                        }

                    
                }
            }

        }
        void Sub_internal_cftf(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_complex64 *_out = (cytnx_complex64*)out->Mem;
            cytnx_complex64 *_Lin = (cytnx_complex64*)Lin->Mem;
            cytnx_float *_Rin = (cytnx_float*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] ;
                        _out[i].real(_out[i].real()-_Rin[i]);
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] ;
                        _out[i].real(_out[i].real()-_Rin[0]);
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Lin[i];
                            _out[i].real(_out[i].real()-_Rin[i]);
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)];
                            _out[i].real(_out[i].real()-_Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)]);
                        }

                    
                }
            }
        }
        void Sub_internal_cftu64(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_complex64 *_out = (cytnx_complex64*)out->Mem;
            cytnx_complex64 *_Lin = (cytnx_complex64*)Lin->Mem;
            cytnx_uint64 *_Rin = (cytnx_uint64*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] ;
                        _out[i].real(_out[i].real()-_Rin[i]);
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] ;
                        _out[i].real(_out[i].real()-_Rin[0]);
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Lin[i];
                            _out[i].real(_out[i].real()-_Rin[i]);
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)];
                            _out[i].real(_out[i].real()-_Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)]);
                        }

                    
                }
            }

        }
        void Sub_internal_cftu32(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_complex64 *_out = (cytnx_complex64*)out->Mem;
            cytnx_complex64 *_Lin = (cytnx_complex64*)Lin->Mem;
            cytnx_uint32 *_Rin = (cytnx_uint32*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] ;
                        _out[i].real(_out[i].real()-_Rin[i]);
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] ;
                        _out[i].real(_out[i].real()-_Rin[0]);
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Lin[i];
                            _out[i].real(_out[i].real()-_Rin[i]);
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)];
                            _out[i].real(_out[i].real()-_Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)]);
                        }

                    
                }
            }

        }
        void Sub_internal_cfti64(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_complex64 *_out = (cytnx_complex64*)out->Mem;
            cytnx_complex64 *_Lin = (cytnx_complex64*)Lin->Mem;
            cytnx_int64 *_Rin = (cytnx_int64*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] ;
                        _out[i].real(_out[i].real()-_Rin[i]);
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] ;
                        _out[i].real(_out[i].real()-_Rin[0]);
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Lin[i];
                            _out[i].real(_out[i].real()-_Rin[i]);
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)];
                            _out[i].real(_out[i].real()-_Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)]);
                        }

                    
                }
            }
        }
        void Sub_internal_cfti32(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_complex64 *_out = (cytnx_complex64*)out->Mem;
            cytnx_complex64 *_Lin = (cytnx_complex64*)Lin->Mem;
            cytnx_int32 *_Rin = (cytnx_int32*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] ;
                        _out[i].real(_out[i].real()-_Rin[i]);
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] ;
                        _out[i].real(_out[i].real()-_Rin[0]);
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Lin[i];
                            _out[i].real(_out[i].real()-_Rin[i]);
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)];
                            _out[i].real(_out[i].real()-_Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)]);
                        }

                    
                }
            }
        }

        void Sub_internal_dtcd(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_double *_Lin   = (cytnx_double*)Lin->Mem;
            cytnx_complex128 *_Rin  = (cytnx_complex128*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Rin[i] ;
                        _out[i].real(_Lin[0]-_out[i].real());
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Rin[0] ;
                        _out[i].real(_Lin[i] - _out[i].real());
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Rin[i];
                            _out[i].real(_Lin[i] - _out[i].real());
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)];
                            _out[i].real(_Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)] - _out[i].real());
                        }

                    
                }
            }

        }
        void Sub_internal_dtcf(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_complex64 *_out = (cytnx_complex64*)out->Mem;
            cytnx_double *_Lin   = (cytnx_double*)Lin->Mem;
            cytnx_complex64 *_Rin  = (cytnx_complex64*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Rin[i] ;
                        _out[i].real(_Lin[0]-_out[i].real());
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Rin[0] ;
                        _out[i].real(_Lin[i] - _out[i].real());
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Rin[i];
                            _out[i].real(_Lin[i] - _out[i].real());
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)];
                            _out[i].real(_Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)] - _out[i].real());
                        }

                    
                }
            }
        }
        void Sub_internal_dtd(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_double *_Lin = (cytnx_double*)Lin->Mem;
            cytnx_double *_Rin = (cytnx_double*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] - _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] - _Rin[0];
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Lin[i] - _Rin[i];
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)] 
                                    - _Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)];
                        }

                    
                }
            }


        }
        void Sub_internal_dtf(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_double *_Lin = (cytnx_double*)Lin->Mem;
            cytnx_float *_Rin = (cytnx_float*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] - _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] - _Rin[0];
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Lin[i] - _Rin[i];
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)] 
                                    - _Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)];
                        }

                    
                }
            }
        }
        void Sub_internal_dtu64(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){

            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_double *_Lin = (cytnx_double*)Lin->Mem;
            cytnx_uint64 *_Rin = (cytnx_uint64*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] - _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] - _Rin[0];
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Lin[i] - _Rin[i];
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)] 
                                    - _Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)];
                        }

                    
                }
            }

        }
        void Sub_internal_dtu32(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){

            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_double *_Lin = (cytnx_double*)Lin->Mem;
            cytnx_uint32 *_Rin = (cytnx_uint32*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] - _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] - _Rin[0];
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Lin[i] - _Rin[i];
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)] 
                                    - _Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)];
                        }

                    
                }
            }
        }
        void Sub_internal_dti64(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){

            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_double *_Lin = (cytnx_double*)Lin->Mem;
            cytnx_int64 *_Rin = (cytnx_int64*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] - _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] - _Rin[0];
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Lin[i] - _Rin[i];
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)] 
                                    - _Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)];
                        }

                    
                }
            }

        }
        void Sub_internal_dti32(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){

            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_double *_Lin = (cytnx_double*)Lin->Mem;
            cytnx_int32 *_Rin = (cytnx_int32*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] - _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] - _Rin[0];
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Lin[i] - _Rin[i];
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)] 
                                    - _Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)];
                        }

                    
                }
            }

        }

        void Sub_internal_ftcd(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_float*_Lin   = (cytnx_float*)Lin->Mem;
            cytnx_complex128 *_Rin  = (cytnx_complex128*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Rin[i] ;
                        _out[i].real(_Lin[0]-_out[i].real());
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Rin[0] ;
                        _out[i].real(_Lin[i] - _out[i].real());
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Lin[i] - _Rin[i];
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)] 
                                    - _Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)];
                        }

                    
                }
            }
        }
        void Sub_internal_ftcf(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_complex64 *_out = (cytnx_complex64*)out->Mem;
            cytnx_float*_Lin   = (cytnx_float*)Lin->Mem;
            cytnx_complex64 *_Rin  = (cytnx_complex64*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Rin[i] ;
                        _out[i].real(_Lin[0]-_out[i].real());
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Rin[0] ;
                        _out[i].real(_Lin[i] - _out[i].real());
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Lin[i] - _Rin[i];
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)] 
                                    - _Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)];
                        }

                    
                }
            }
        }
        void Sub_internal_ftd(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_float*_Lin   = (cytnx_float*)Lin->Mem;
            cytnx_double *_Rin  = (cytnx_double*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] - _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] - _Rin[0];
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Lin[i] - _Rin[i];
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)] 
                                    - _Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)];
                        }

                    
                }
            }
        }
        void Sub_internal_ftf(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_float *_Lin = (cytnx_float*)Lin->Mem;
            cytnx_float *_Rin = (cytnx_float*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] - _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] - _Rin[0];
                    }
            }else{
                 if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Lin[i] - _Rin[i];
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)] 
                                    - _Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)];
                        }

                    
                }
            }

        }
        void Sub_internal_ftu64(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_float *_Lin = (cytnx_float*)Lin->Mem;
            cytnx_uint64 *_Rin = (cytnx_uint64*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] - _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] - _Rin[0];
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Lin[i] - _Rin[i];
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)] 
                                    - _Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)];
                        }

                    
                }
            }

        }
        void Sub_internal_ftu32(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_float *_Lin = (cytnx_float*)Lin->Mem;
            cytnx_uint32 *_Rin = (cytnx_uint32*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] - _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] - _Rin[0];
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Lin[i] - _Rin[i];
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)] 
                                    - _Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)];
                        }

                    
                }
            }
        }
        void Sub_internal_fti64(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_float *_Lin = (cytnx_float*)Lin->Mem;
            cytnx_int64 *_Rin = (cytnx_int64*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] - _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] - _Rin[0];
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Lin[i] - _Rin[i];
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)] 
                                    - _Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)];
                        }

                    
                }
            }

        }
        void Sub_internal_fti32(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_float *_Lin = (cytnx_float*)Lin->Mem;
            cytnx_int32 *_Rin = (cytnx_int32*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] - _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] - _Rin[0];
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Lin[i] - _Rin[i];
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)] 
                                    - _Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)];
                        }

                    
                }
            }

        }


        void Sub_internal_i64tcd(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_int64*_Lin   = (cytnx_int64*)Lin->Mem;
            cytnx_complex128 *_Rin  = (cytnx_complex128*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Rin[i] ;
                        _out[i].real(_Lin[0]-_out[i].real());
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Rin[0] ;
                        _out[i].real(_Lin[i] - _out[i].real());
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Rin[i];
                            _out[i].real(_Lin[i] - _out[i].real());
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)];
                            _out[i].real(_Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)] - _out[i].real());
                        }

                    
                }
            }
        }
        void Sub_internal_i64tcf(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_complex64 *_out = (cytnx_complex64*)out->Mem;
            cytnx_int64*_Lin   = (cytnx_int64*)Lin->Mem;
            cytnx_complex64 *_Rin  = (cytnx_complex64*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Rin[i] ;
                        _out[i].real(_Lin[0]-_out[i].real());
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Rin[0] ;
                        _out[i].real(_Lin[i] - _out[i].real());
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Rin[i];
                            _out[i].real(_Lin[i] - _out[i].real());
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)];
                            _out[i].real(_Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)] - _out[i].real());
                        }

                    
                }
            }
        }
        void Sub_internal_i64td(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_int64*_Lin   = (cytnx_int64*)Lin->Mem;
            cytnx_double *_Rin  = (cytnx_double*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] - _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] - _Rin[0];
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Lin[i] - _Rin[i];
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)] 
                                    - _Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)];
                        }

                    
                }
            }
        }
        void Sub_internal_i64tf(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_int64*_Lin   = (cytnx_int64*)Lin->Mem;
            cytnx_float *_Rin  = (cytnx_float*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] - _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] - _Rin[0];
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Lin[i] - _Rin[i];
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)] 
                                    - _Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)];
                        }

                    
                }
            }
        }
        void Sub_internal_i64ti64(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_int64 *_out = (cytnx_int64*)out->Mem;
            cytnx_int64 *_Lin = (cytnx_int64*)Lin->Mem;
            cytnx_int64 *_Rin = (cytnx_int64*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] - _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] - _Rin[0];
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Lin[i] - _Rin[i];
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)] 
                                    - _Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)];
                        }

                    
                }
            }

        }
        void Sub_internal_i64tu64(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_int64 *_out = (cytnx_int64*)out->Mem;
            cytnx_int64 *_Lin = (cytnx_int64*)Lin->Mem;
            cytnx_uint64 *_Rin = (cytnx_uint64*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] - _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] - _Rin[0];
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Lin[i] - _Rin[i];
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)] 
                                    - _Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)];
                        }

                    
                }
            }
        }
        void Sub_internal_i64ti32(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_int64 *_out = (cytnx_int64*)out->Mem;
            cytnx_int64 *_Lin = (cytnx_int64*)Lin->Mem;
            cytnx_int32 *_Rin = (cytnx_int32*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] - _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] - _Rin[0];
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Lin[i] - _Rin[i];
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)] 
                                    - _Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)];
                        }

                    
                }
            }
        }
        void Sub_internal_i64tu32(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_int64 *_out = (cytnx_int64*)out->Mem;
            cytnx_int64 *_Lin = (cytnx_int64*)Lin->Mem;
            cytnx_uint32 *_Rin = (cytnx_uint32*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] - _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] - _Rin[0];
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Lin[i] - _Rin[i];
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)] 
                                    - _Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)];
                        }

                    
                }
            }

        }


        void Sub_internal_u64tcd(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_uint64*_Lin   = (cytnx_uint64*)Lin->Mem;
            cytnx_complex128 *_Rin  = (cytnx_complex128*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Rin[i] ;
                        _out[i].real(_Lin[0]-_out[i].real());
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Rin[0] ;
                        _out[i].real(_Lin[i] - _out[i].real());
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Rin[i];
                            _out[i].real(_Lin[i] - _out[i].real());
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)];
                            _out[i].real(_Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)] - _out[i].real());
                        }

                    
                }
            }
        }
        void Sub_internal_u64tcf(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_complex64 *_out = (cytnx_complex64*)out->Mem;
            cytnx_uint64*_Lin   = (cytnx_uint64*)Lin->Mem;
            cytnx_complex64 *_Rin  = (cytnx_complex64*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Rin[i] ;
                        _out[i].real(_Lin[0]-_out[i].real());
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Rin[0] ;
                        _out[i].real(_Lin[i] - _out[i].real());
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Rin[i];
                            _out[i].real(_Lin[i] - _out[i].real());
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)];
                            _out[i].real(_Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)] - _out[i].real());
                        }

                    
                }
            }
        }
        void Sub_internal_u64td(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_uint64 *_Lin = (cytnx_uint64*)Lin->Mem;
            cytnx_double *_Rin = (cytnx_double*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] - _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] - _Rin[0];
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Lin[i] - _Rin[i];
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)] 
                                    - _Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)];
                        }

                    
                }
            }

        }
        void Sub_internal_u64tf(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_uint64 *_Lin = (cytnx_uint64*)Lin->Mem;
            cytnx_float *_Rin = (cytnx_float*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] - _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] - _Rin[0];
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Lin[i] - _Rin[i];
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)] 
                                    - _Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)];
                        }

                    
                }
            }

        }
        void Sub_internal_u64ti64(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_int64 *_out = (cytnx_int64*)out->Mem;
            cytnx_uint64 *_Lin = (cytnx_uint64*)Lin->Mem;
            cytnx_int64 *_Rin = (cytnx_int64*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] - _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] - _Rin[0];
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Lin[i] - _Rin[i];
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)] 
                                    - _Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)];
                        }

                    
                }
            }

        }
        void Sub_internal_u64tu64(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_uint64 *_out = (cytnx_uint64*)out->Mem;
            cytnx_uint64 *_Lin = (cytnx_uint64*)Lin->Mem;
            cytnx_uint64 *_Rin = (cytnx_uint64*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] - _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] - _Rin[0];
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Lin[i] - _Rin[i];
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)] 
                                    - _Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)];
                        }

                    
                }
            }
        }
        void Sub_internal_u64ti32(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_uint64 *_out = (cytnx_uint64*)out->Mem;
            cytnx_uint64 *_Lin = (cytnx_uint64*)Lin->Mem;
            cytnx_int32 *_Rin = (cytnx_int32*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] - _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] - _Rin[0];
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Lin[i] - _Rin[i];
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)] 
                                    - _Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)];
                        }

                    
                }
            }
        }
        void Sub_internal_u64tu32(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_uint64 *_out = (cytnx_uint64*)out->Mem;
            cytnx_uint64 *_Lin = (cytnx_uint64*)Lin->Mem;
            cytnx_uint32 *_Rin = (cytnx_uint32*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] - _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] - _Rin[0];
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Lin[i] - _Rin[i];
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)] 
                                    - _Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)];
                        }

                    
                }
            }

        }

        void Sub_internal_i32tcd(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_int32*_Lin   = (cytnx_int32*)Lin->Mem;
            cytnx_complex128 *_Rin  = (cytnx_complex128*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Rin[i] ;
                        _out[i].real(_Lin[0]-_out[i].real());
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Rin[0] ;
                        _out[i].real(_Lin[i] - _out[i].real());
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Rin[i];
                            _out[i].real(_Lin[i] - _out[i].real());
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)];
                            _out[i].real(_Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)] - _out[i].real());
                        }

                    
                }
            }

        }
        void Sub_internal_i32tcf(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_complex64 *_out = (cytnx_complex64*)out->Mem;
            cytnx_int32*_Lin   = (cytnx_int32*)Lin->Mem;
            cytnx_complex64 *_Rin  = (cytnx_complex64*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Rin[i] ;
                        _out[i].real(_Lin[0]-_out[i].real());
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Rin[0] ;
                        _out[i].real(_Lin[i] - _out[i].real());
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Rin[i];
                            _out[i].real(_Lin[i] - _out[i].real());
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)];
                            _out[i].real(_Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)] - _out[i].real());
                        }

                    
                }
            }

        }
        void Sub_internal_i32td(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_int64 *_Lin = (cytnx_int64*)Lin->Mem;
            cytnx_double *_Rin = (cytnx_double*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] - _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] - _Rin[0];
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Lin[i] - _Rin[i];
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)] 
                                    - _Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)];
                        }

                    
                }
            }

        }
        void Sub_internal_i32tf(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_int32 *_Lin = (cytnx_int32*)Lin->Mem;
            cytnx_float *_Rin = (cytnx_float*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] - _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] - _Rin[0];
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Lin[i] - _Rin[i];
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)] 
                                    - _Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)];
                        }

                    
                }
            }

        }
        void Sub_internal_i32ti64(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_int64 *_out = (cytnx_int64*)out->Mem;
            cytnx_int32 *_Lin = (cytnx_int32*)Lin->Mem;
            cytnx_int64 *_Rin = (cytnx_int64*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] - _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] - _Rin[0];
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Lin[i] - _Rin[i];
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)] 
                                    - _Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)];
                        }

                    
                }
            }

        }
        void Sub_internal_i32tu64(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_uint64 *_out = (cytnx_uint64*)out->Mem;
            cytnx_int32 *_Lin = (cytnx_int32*)Lin->Mem;
            cytnx_uint64 *_Rin = (cytnx_uint64*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] - _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] - _Rin[0];
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Lin[i] - _Rin[i];
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)] 
                                    - _Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)];
                        }

                    
                }
            }

        }
        void Sub_internal_i32ti32(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_int32 *_out = (cytnx_int32*)out->Mem;
            cytnx_int32 *_Lin = (cytnx_int32*)Lin->Mem;
            cytnx_int32 *_Rin = (cytnx_int32*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] - _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] - _Rin[0];
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Lin[i] - _Rin[i];
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)] 
                                    - _Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)];
                        }

                    
                }
            }


        }
        void Sub_internal_i32tu32(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_int32 *_out = (cytnx_int32*)out->Mem;
            cytnx_int32 *_Lin = (cytnx_int32*)Lin->Mem;
            cytnx_uint32 *_Rin = (cytnx_uint32*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] - _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] - _Rin[0];
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Lin[i] - _Rin[i];
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)] 
                                    - _Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)];
                        }

                    
                }
            }

        }


        void Sub_internal_u32tcd(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_uint32*_Lin   = (cytnx_uint32*)Lin->Mem;
            cytnx_complex128 *_Rin  = (cytnx_complex128*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Rin[i] ;
                        _out[i].real(_Lin[0]-_out[i].real());
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Rin[0] ;
                        _out[i].real(_Lin[i] - _out[i].real());
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Rin[i];
                            _out[i].real(_Lin[i] - _out[i].real());
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)];
                            _out[i].real(_Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)] - _out[i].real());
                        }

                    
                }
            }

        }
        void Sub_internal_u32tcf(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_complex64 *_out = (cytnx_complex64*)out->Mem;
            cytnx_uint32*_Lin   = (cytnx_uint32*)Lin->Mem;
            cytnx_complex64 *_Rin  = (cytnx_complex64*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Rin[i] ;
                        _out[i].real(_Lin[0]-_out[i].real());
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Rin[0] ;
                        _out[i].real(_Lin[i] - _out[i].real());
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Rin[i];
                            _out[i].real(_Lin[i]-_out[i].real());
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)];
                            _out[i].real(_Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)] - _out[i].real());
                        }

                    
                }
            }

        }
        void Sub_internal_u32td(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_uint32 *_Lin = (cytnx_uint32*)Lin->Mem;
            cytnx_double *_Rin = (cytnx_double*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] - _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] - _Rin[0];
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Lin[i] - _Rin[i];
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)] 
                                    - _Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)];
                        }

                    
                }
            }

        }
        void Sub_internal_u32tf(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_uint32 *_Lin = (cytnx_uint32*)Lin->Mem;
            cytnx_float *_Rin = (cytnx_float*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] - _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] - _Rin[0];
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Lin[i] - _Rin[i];
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)] 
                                    - _Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)];
                        }

                    
                }
            }


        }
        void Sub_internal_u32ti64(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_int64 *_out = (cytnx_int64*)out->Mem;
            cytnx_uint32 *_Lin = (cytnx_uint32*)Lin->Mem;
            cytnx_int64 *_Rin = (cytnx_int64*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] - _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] - _Rin[0];
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Lin[i] - _Rin[i];
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)] 
                                    - _Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)];
                        }

                    
                }
            }

        }
        void Sub_internal_u32tu64(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_uint64 *_out = (cytnx_uint64*)out->Mem;
            cytnx_uint32 *_Lin = (cytnx_uint32*)Lin->Mem;
            cytnx_uint64 *_Rin = (cytnx_uint64*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] - _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] - _Rin[0];
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Lin[i] - _Rin[i];
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)] 
                                    - _Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)];
                        }

                    
                }
            }

        }
        void Sub_internal_u32ti32(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_int32 *_out = (cytnx_int32*)out->Mem;
            cytnx_uint32 *_Lin = (cytnx_uint32*)Lin->Mem;
            cytnx_int32 *_Rin = (cytnx_int32*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] - _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] - _Rin[0];
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Lin[i] - _Rin[i];
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)] 
                                    - _Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)];
                        }

                    
                }
            }

        }
        void Sub_internal_u32tu32(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L, const std::vector<cytnx_uint64> &invmapper_R){
            cytnx_uint32 *_out = (cytnx_uint32*)out->Mem;
            cytnx_uint32 *_Lin = (cytnx_uint32*)Lin->Mem;
            cytnx_uint32 *_Rin = (cytnx_uint32*)Rin->Mem;

            if(Lin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[0] - _Rin[i];
                    }
            }else if(Rin->size()==1){
                #ifdef UNI_OMP
                    #pragma omp parallel for schedule(dynamic) 
                #endif
                    for(unsigned long long i=0;i<len;i++){
                        _out[i] = _Lin[i] - _Rin[0];
                    }
            }else{
                if(shape.size()==0){
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(unsigned long long i=0;i<len;i++){
                            _out[i] = _Lin[i] - _Rin[i];
                        }
                }else{

                    ///handle non-contiguous:
                    std::vector<cytnx_uint64> accu_shape(shape.size());
                    std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
                    cytnx_uint64 tmp1=1,tmp2=1,tmp3=1;
                    for(cytnx_uint64 i=0;i<shape.size();i++){
                        accu_shape[shape.size()-1-i] = tmp1; 
                        tmp1*=shape[shape.size()-1-i];
                        
                        old_accu_shapeL[shape.size()-1-i] = tmp2; 
                        tmp2*=shape[invmapper_L[shape.size()-1-i]];
                        
                        old_accu_shapeR[shape.size()-1-i] = tmp3; 
                        tmp3*=shape[invmapper_R[shape.size()-1-i]];
                    }

                    // handle non-contiguous
                    #ifdef UNI_OMP
                        #pragma omp parallel for schedule(dynamic) 
                    #endif
                        for(cytnx_uint64 i=0;i<len;i++){
                            std::vector<cytnx_uint64> tmpv = c2cartesian(i,accu_shape);
                            _out[i] = _Lin[cartesian2c(vec_map(tmpv,invmapper_L),old_accu_shapeL)] 
                                    - _Rin[cartesian2c(vec_map(tmpv,invmapper_R),old_accu_shapeR)];
                        }

                    
                }
            }
        }





    }//namespace linalg_internal
}//namespace cytnx



