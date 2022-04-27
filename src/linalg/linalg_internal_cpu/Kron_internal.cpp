#include "linalg/linalg_internal_cpu/Kron_internal.hpp"
#include "utils/utils_internal_interface.hpp"
#include "utils/complex_arithmetic.hpp"
//#include "lapack_wrapper.hpp"
#ifdef UNI_OMP
    #include <omp.h>
#endif

namespace cytnx{

    namespace linalg_internal{

        /// Kron
        void Kron_internal_cdtcd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_complex128 *_Lin = (cytnx_complex128*)Lin->Mem;
            cytnx_complex128 *_Rin = (cytnx_complex128*)Rin->Mem;
            
            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }

            

        }
        void Kron_internal_cdtcf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_complex128 *_Lin = (cytnx_complex128*)Lin->Mem;
            cytnx_complex64 *_Rin = (cytnx_complex64*)Rin->Mem;

            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }

            


        }
        void Kron_internal_cdtd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_complex128 *_Lin = (cytnx_complex128*)Lin->Mem;
            cytnx_double *_Rin = (cytnx_double*)Rin->Mem;

            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }


        }
        void Kron_internal_cdtf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_complex128 *_Lin = (cytnx_complex128*)Lin->Mem;
            cytnx_float *_Rin = (cytnx_float*)Rin->Mem;

            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }


        }
        void Kron_internal_cdtu64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_complex128 *_Lin = (cytnx_complex128*)Lin->Mem;
            cytnx_uint64 *_Rin = (cytnx_uint64*)Rin->Mem;

            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }




        }
        void Kron_internal_cdtu32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_complex128 *_Lin = (cytnx_complex128*)Lin->Mem;
            cytnx_uint32 *_Rin = (cytnx_uint32*)Rin->Mem;

            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }




        }
        void Kron_internal_cdti64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_complex128 *_Lin = (cytnx_complex128*)Lin->Mem;
            cytnx_int64 *_Rin = (cytnx_int64*)Rin->Mem;

            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }



        }
        void Kron_internal_cdti32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_complex128 *_Lin = (cytnx_complex128*)Lin->Mem;
            cytnx_int32 *_Rin = (cytnx_int32*)Rin->Mem;

            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }



        }
        void Kron_internal_cdti16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_complex128 *_Lin = (cytnx_complex128*)Lin->Mem;
            cytnx_int16 *_Rin = (cytnx_int16*)Rin->Mem;

            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }



        }
        void Kron_internal_cdtu16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_complex128 *_Lin = (cytnx_complex128*)Lin->Mem;
            cytnx_uint16 *_Rin = (cytnx_uint16*)Rin->Mem;

            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }



        }
        void Kron_internal_cdtb(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_complex128 *_Lin = (cytnx_complex128*)Lin->Mem;
            cytnx_bool *_Rin = (cytnx_bool*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*cytnx_complex128(_Rin[tmp2],0);
                }


        }
//----------------------------------------
        void Kron_internal_cftcd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){

            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_complex64 *_Lin  = (cytnx_complex64*)Lin->Mem;
            cytnx_complex128 *_Rin = (cytnx_complex128*)Rin->Mem;

            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }

	}
        void Kron_internal_cftcf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_complex64 *_out = (cytnx_complex64*)out->Mem;
            cytnx_complex64 *_Lin = (cytnx_complex64*)Lin->Mem;
            cytnx_complex64 *_Rin = (cytnx_complex64*)Rin->Mem;

            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }


        }
        void Kron_internal_cftd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_complex64 *_out = (cytnx_complex64*)out->Mem;
            cytnx_complex64 *_Lin = (cytnx_complex64*)Lin->Mem;
            cytnx_double *_Rin = (cytnx_double*)Rin->Mem;

            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }


        }
        void Kron_internal_cftf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_complex64 *_out = (cytnx_complex64*)out->Mem;
            cytnx_complex64 *_Lin = (cytnx_complex64*)Lin->Mem;
            cytnx_float *_Rin = (cytnx_float*)Rin->Mem;

            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }

        }
        void Kron_internal_cftu64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_complex64 *_out = (cytnx_complex64*)out->Mem;
            cytnx_complex64 *_Lin = (cytnx_complex64*)Lin->Mem;
            cytnx_uint64 *_Rin = (cytnx_uint64*)Rin->Mem;
            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }

        }
        void Kron_internal_cftu32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_complex64 *_out = (cytnx_complex64*)out->Mem;
            cytnx_complex64 *_Lin = (cytnx_complex64*)Lin->Mem;
            cytnx_uint32 *_Rin = (cytnx_uint32*)Rin->Mem;

            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }


        }
        void Kron_internal_cfti64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_complex64 *_out = (cytnx_complex64*)out->Mem;
            cytnx_complex64 *_Lin = (cytnx_complex64*)Lin->Mem;
            cytnx_int64 *_Rin = (cytnx_int64*)Rin->Mem;

            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }

        }
        void Kron_internal_cfti32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_complex64 *_out = (cytnx_complex64*)out->Mem;
            cytnx_complex64 *_Lin = (cytnx_complex64*)Lin->Mem;
            cytnx_int32 *_Rin = (cytnx_int32*)Rin->Mem;

            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }

        }
        void Kron_internal_cfti16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_complex64 *_out = (cytnx_complex64*)out->Mem;
            cytnx_complex64 *_Lin = (cytnx_complex64*)Lin->Mem;
            cytnx_int16 *_Rin = (cytnx_int16*)Rin->Mem;

            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }

        }
        void Kron_internal_cftu16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_complex64 *_out = (cytnx_complex64*)out->Mem;
            cytnx_complex64 *_Lin = (cytnx_complex64*)Lin->Mem;
            cytnx_uint16 *_Rin = (cytnx_uint16*)Rin->Mem;

            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }


        }
        void Kron_internal_cftb(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_complex64 *_out = (cytnx_complex64*)out->Mem;
            cytnx_complex64 *_Lin = (cytnx_complex64*)Lin->Mem;
            cytnx_bool *_Rin = (cytnx_bool*)Rin->Mem;

            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*cytnx_complex64(_Rin[tmp2],0);
                }


        }

//-------------------------
        void Kron_internal_dtcd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_double *_Lin = (cytnx_double*)Lin->Mem;
            cytnx_complex128 *_Rin = (cytnx_complex128*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }


        }
        void Kron_internal_dtcf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_complex64 *_out = (cytnx_complex64*)out->Mem;
            cytnx_double *_Lin = (cytnx_double*)Lin->Mem;
            cytnx_complex64 *_Rin = (cytnx_complex64*)Rin->Mem;

            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }


        }
        void Kron_internal_dtd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_double *_Lin = (cytnx_double*)Lin->Mem;
            cytnx_double *_Rin = (cytnx_double*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        cytnx_uint64 tmp = i, tmp2;
                        cytnx_uint64 x=0,y=0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            tmp2 = tmp/new_shape_acc[j];
                            tmp %= new_shape_acc[j];
                            x += cytnx_uint64(tmp2/shape2[j])*shape1_acc[j];
                            y += cytnx_uint64(tmp2%shape2[j])*shape2_acc[j];
                        }
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }

        }
        void Kron_internal_dtf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_double *_Lin = (cytnx_double*)Lin->Mem;
            cytnx_float *_Rin = (cytnx_float*)Rin->Mem;

            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }



        }
        void Kron_internal_dtu64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){

            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_double *_Lin = (cytnx_double*)Lin->Mem;
            cytnx_uint64 *_Rin = (cytnx_uint64*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }

        }
        void Kron_internal_dtu32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){

            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_double *_Lin = (cytnx_double*)Lin->Mem;
            cytnx_uint32 *_Rin = (cytnx_uint32*)Rin->Mem;

            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }

        }
        void Kron_internal_dti64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){

            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_double *_Lin = (cytnx_double*)Lin->Mem;
            cytnx_int64 *_Rin = (cytnx_int64*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }

        }
        void Kron_internal_dti32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){

            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_double *_Lin = (cytnx_double*)Lin->Mem;
            cytnx_int32 *_Rin = (cytnx_int32*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }

        }
        void Kron_internal_dti16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){

            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_double *_Lin = (cytnx_double*)Lin->Mem;
            cytnx_int16 *_Rin = (cytnx_int16*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }

        }
        void Kron_internal_dtu16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){

            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_double *_Lin = (cytnx_double*)Lin->Mem;
            cytnx_uint16 *_Rin = (cytnx_uint16*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }

        }
        void Kron_internal_dtb(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){

            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_double *_Lin = (cytnx_double*)Lin->Mem;
            cytnx_bool *_Rin = (cytnx_bool*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*double(_Rin[tmp2]);
                }

        }
//-------------------------------
        void Kron_internal_ftcd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_float *_Lin = (cytnx_float*)Lin->Mem;
            cytnx_complex128 *_Rin = (cytnx_complex128*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }

    
        }
        void Kron_internal_ftcf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_complex64 *_out = (cytnx_complex64*)out->Mem;
            cytnx_float *_Lin = (cytnx_float*)Lin->Mem;
            cytnx_complex64 *_Rin = (cytnx_complex64*)Rin->Mem;

            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }


        }
        void Kron_internal_ftd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_float *_Lin = (cytnx_float*)Lin->Mem;
            cytnx_double *_Rin = (cytnx_double*)Rin->Mem;

            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }



        }
        void Kron_internal_ftf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_float *_Lin = (cytnx_float*)Lin->Mem;
            cytnx_float *_Rin = (cytnx_float*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }

        }
        void Kron_internal_ftu64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_float *_Lin = (cytnx_float*)Lin->Mem;
            cytnx_uint64 *_Rin = (cytnx_uint64*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }

        }
        void Kron_internal_ftu32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_float *_Lin = (cytnx_float*)Lin->Mem;
            cytnx_uint32 *_Rin = (cytnx_uint32*)Rin->Mem;

            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }



        }
        void Kron_internal_fti64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_float *_Lin = (cytnx_float*)Lin->Mem;
            cytnx_int64 *_Rin = (cytnx_int64*)Rin->Mem;

            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }

        }
        void Kron_internal_fti32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_float *_Lin = (cytnx_float*)Lin->Mem;
            cytnx_int32 *_Rin = (cytnx_int32*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }

        }
        void Kron_internal_fti16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_float *_Lin = (cytnx_float*)Lin->Mem;
            cytnx_int16 *_Rin = (cytnx_int16*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }

        }
        void Kron_internal_ftu16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_float *_Lin = (cytnx_float*)Lin->Mem;
            cytnx_uint16 *_Rin = (cytnx_uint16*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }

        }
        void Kron_internal_ftb(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_float *_Lin = (cytnx_float*)Lin->Mem;
            cytnx_bool *_Rin = (cytnx_bool*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*float(_Rin[tmp2]);
                }

        }

//----------------------------------------
        void Kron_internal_i64tcd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_int64 *_Lin = (cytnx_int64*)Lin->Mem;
            cytnx_complex128 *_Rin = (cytnx_complex128*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }


        }
        void Kron_internal_i64tcf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_complex64 *_out = (cytnx_complex64*)out->Mem;
            cytnx_int64 *_Lin = (cytnx_int64*)Lin->Mem;
            cytnx_complex64 *_Rin = (cytnx_complex64*)Rin->Mem;

            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }



        }
        void Kron_internal_i64td(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_int64 *_Lin = (cytnx_int64*)Lin->Mem;
            cytnx_double *_Rin = (cytnx_double*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }


        }
        void Kron_internal_i64tf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_int64 *_Lin = (cytnx_int64*)Lin->Mem;
            cytnx_float *_Rin = (cytnx_float*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }


        }
        void Kron_internal_i64ti64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_int64 *_out = (cytnx_int64*)out->Mem;
            cytnx_int64 *_Lin = (cytnx_int64*)Lin->Mem;
            cytnx_int64 *_Rin = (cytnx_int64*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }

        }
        void Kron_internal_i64tu64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_int64 *_out = (cytnx_int64*)out->Mem;
            cytnx_int64 *_Lin = (cytnx_int64*)Lin->Mem;
            cytnx_uint64 *_Rin = (cytnx_uint64*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }


        }
        void Kron_internal_i64ti32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_int64 *_out = (cytnx_int64*)out->Mem;
            cytnx_int64 *_Lin = (cytnx_int64*)Lin->Mem;
            cytnx_int32 *_Rin = (cytnx_int32*)Rin->Mem;

            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }

        }
        void Kron_internal_i64tu32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_int64 *_out = (cytnx_int64*)out->Mem;
            cytnx_int64 *_Lin = (cytnx_int64*)Lin->Mem;
            cytnx_uint32 *_Rin = (cytnx_uint32*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }

        }
        void Kron_internal_i64ti16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_int64 *_out = (cytnx_int64*)out->Mem;
            cytnx_int64 *_Lin = (cytnx_int64*)Lin->Mem;
            cytnx_int16 *_Rin = (cytnx_int16*)Rin->Mem;

            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }

        }
        void Kron_internal_i64tu16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_int64 *_out = (cytnx_int64*)out->Mem;
            cytnx_int64 *_Lin = (cytnx_int64*)Lin->Mem;
            cytnx_uint16 *_Rin = (cytnx_uint16*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }

        }
        void Kron_internal_i64tb(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_int64 *_out = (cytnx_int64*)out->Mem;
            cytnx_int64 *_Lin = (cytnx_int64*)Lin->Mem;
            cytnx_bool *_Rin = (cytnx_bool*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*cytnx_int64(_Rin[tmp2]);
                }

        }
//-----------------------------------
        void Kron_internal_u64tcd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_uint64 *_Lin = (cytnx_uint64*)Lin->Mem;
            cytnx_complex128 *_Rin = (cytnx_complex128*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }


        }
        void Kron_internal_u64tcf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_complex64 *_out = (cytnx_complex64*)out->Mem;
            cytnx_uint64 *_Lin = (cytnx_uint64*)Lin->Mem;
            cytnx_complex64 *_Rin = (cytnx_complex64*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }


        }
        void Kron_internal_u64td(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_uint64 *_Lin = (cytnx_uint64*)Lin->Mem;
            cytnx_double *_Rin = (cytnx_double*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }


        }
        void Kron_internal_u64tf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_uint64 *_Lin = (cytnx_uint64*)Lin->Mem;
            cytnx_float *_Rin = (cytnx_float*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }


        }
        void Kron_internal_u64ti64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_int64 *_out = (cytnx_int64*)out->Mem;
            cytnx_uint64 *_Lin = (cytnx_uint64*)Lin->Mem;
            cytnx_int64 *_Rin = (cytnx_int64*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }


        }
        void Kron_internal_u64tu64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_uint64 *_out = (cytnx_uint64*)out->Mem;
            cytnx_uint64 *_Lin = (cytnx_uint64*)Lin->Mem;
            cytnx_uint64 *_Rin = (cytnx_uint64*)Rin->Mem;

            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }

        }
        void Kron_internal_u64ti32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_uint64 *_out = (cytnx_uint64*)out->Mem;
            cytnx_uint64 *_Lin = (cytnx_uint64*)Lin->Mem;
            cytnx_int32 *_Rin = (cytnx_int32*)Rin->Mem;
            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }


        }
        void Kron_internal_u64tu32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_uint64 *_out = (cytnx_uint64*)out->Mem;
            cytnx_uint64 *_Lin = (cytnx_uint64*)Lin->Mem;
            cytnx_uint32 *_Rin = (cytnx_uint32*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }

        }
        void Kron_internal_u64ti16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_uint64 *_out = (cytnx_uint64*)out->Mem;
            cytnx_uint64 *_Lin = (cytnx_uint64*)Lin->Mem;
            cytnx_int16 *_Rin = (cytnx_int16*)Rin->Mem;

            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }


        }
        void Kron_internal_u64tu16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_uint64 *_out = (cytnx_uint64*)out->Mem;
            cytnx_uint64 *_Lin = (cytnx_uint64*)Lin->Mem;
            cytnx_uint16 *_Rin = (cytnx_uint16*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }

        }
        void Kron_internal_u64tb(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_uint64 *_out = (cytnx_uint64*)out->Mem;
            cytnx_uint64 *_Lin = (cytnx_uint64*)Lin->Mem;
            cytnx_bool *_Rin = (cytnx_bool*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*cytnx_uint64(_Rin[tmp2]);
                }

        }
//-------------------------------------
        void Kron_internal_i32tcd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_int32 *_Lin = (cytnx_int32*)Lin->Mem;
            cytnx_complex128 *_Rin = (cytnx_complex128*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }

        }
        void Kron_internal_i32tcf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_complex64 *_out = (cytnx_complex64*)out->Mem;
            cytnx_int32 *_Lin = (cytnx_int32*)Lin->Mem;
            cytnx_complex64 *_Rin = (cytnx_complex64*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }

        }
        void Kron_internal_i32td(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_int32 *_Lin = (cytnx_int32*)Lin->Mem;
            cytnx_double *_Rin = (cytnx_double*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }


        }
        void Kron_internal_i32tf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_int32 *_Lin = (cytnx_int32*)Lin->Mem;
            cytnx_float *_Rin = (cytnx_float*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }


        }
        void Kron_internal_i32ti64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_int64 *_out = (cytnx_int64*)out->Mem;
            cytnx_int32 *_Lin = (cytnx_int32*)Lin->Mem;
            cytnx_int64 *_Rin = (cytnx_int64*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }


        }
        void Kron_internal_i32tu64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_uint64 *_out = (cytnx_uint64*)out->Mem;
            cytnx_int32 *_Lin = (cytnx_int32*)Lin->Mem;
            cytnx_uint64 *_Rin = (cytnx_uint64*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }

        }
        void Kron_internal_i32ti32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_int32 *_out = (cytnx_int32*)out->Mem;
            cytnx_int32 *_Lin = (cytnx_int32*)Lin->Mem;
            cytnx_int32 *_Rin = (cytnx_int32*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }

        }
        void Kron_internal_i32tu32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_int32 *_out = (cytnx_int32*)out->Mem;
            cytnx_int32 *_Lin = (cytnx_int32*)Lin->Mem;
            cytnx_uint32 *_Rin = (cytnx_uint32*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }

        }
        void Kron_internal_i32ti16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_int32 *_out = (cytnx_int32*)out->Mem;
            cytnx_int32 *_Lin = (cytnx_int32*)Lin->Mem;
            cytnx_int16 *_Rin = (cytnx_int16*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }


        }
        void Kron_internal_i32tu16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_int32 *_out = (cytnx_int32*)out->Mem;
            cytnx_int32 *_Lin = (cytnx_int32*)Lin->Mem;
            cytnx_uint16 *_Rin = (cytnx_uint16*)Rin->Mem;

            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }


        }
        void Kron_internal_i32tb(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_int32 *_out = (cytnx_int32*)out->Mem;
            cytnx_int32 *_Lin = (cytnx_int32*)Lin->Mem;
            cytnx_bool *_Rin = (cytnx_bool*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*cytnx_int32(_Rin[tmp2]);
                }

        }

//----------------------------------------
        void Kron_internal_u32tcd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_uint32 *_Lin = (cytnx_uint32*)Lin->Mem;
            cytnx_complex128 *_Rin = (cytnx_complex128*)Rin->Mem;

            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }


        }
        void Kron_internal_u32tcf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_complex64 *_out = (cytnx_complex64*)out->Mem;
            cytnx_uint32 *_Lin = (cytnx_uint32*)Lin->Mem;
            cytnx_complex64 *_Rin = (cytnx_complex64*)Rin->Mem;

            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }


        }
        void Kron_internal_u32td(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_uint32 *_Lin = (cytnx_uint32*)Lin->Mem;
            cytnx_double *_Rin = (cytnx_double*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }

        }
        void Kron_internal_u32tf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_uint32 *_Lin = (cytnx_uint32*)Lin->Mem;
            cytnx_float *_Rin = (cytnx_float*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }

        }
        void Kron_internal_u32ti64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_int64 *_out = (cytnx_int64*)out->Mem;
            cytnx_uint32 *_Lin = (cytnx_uint32*)Lin->Mem;
            cytnx_int64 *_Rin = (cytnx_int64*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }

        }
        void Kron_internal_u32tu64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_uint64 *_out = (cytnx_uint64*)out->Mem;
            cytnx_uint32 *_Lin = (cytnx_uint32*)Lin->Mem;
            cytnx_uint64 *_Rin = (cytnx_uint64*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }

        }
        void Kron_internal_u32ti32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_int32 *_out = (cytnx_int32*)out->Mem;
            cytnx_uint32 *_Lin = (cytnx_uint32*)Lin->Mem;
            cytnx_int32 *_Rin = (cytnx_int32*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }

        }
        void Kron_internal_u32tu32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_uint32 *_out = (cytnx_uint32*)out->Mem;
            cytnx_uint32 *_Lin = (cytnx_uint32*)Lin->Mem;
            cytnx_uint32 *_Rin = (cytnx_uint32*)Rin->Mem;

            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }

                
        }
        void Kron_internal_u32ti16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_uint32 *_out = (cytnx_uint32*)out->Mem;
            cytnx_uint32 *_Lin = (cytnx_uint32*)Lin->Mem;
            cytnx_int16 *_Rin = (cytnx_int16*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }


        }
        void Kron_internal_u32tu16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_uint32 *_out = (cytnx_uint32*)out->Mem;
            cytnx_uint32 *_Lin = (cytnx_uint32*)Lin->Mem;
            cytnx_uint16 *_Rin = (cytnx_uint16*)Rin->Mem;

            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }


        }
        void Kron_internal_u32tb(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_uint32 *_out = (cytnx_uint32*)out->Mem;
            cytnx_uint32 *_Lin = (cytnx_uint32*)Lin->Mem;
            cytnx_bool *_Rin = (cytnx_bool*)Rin->Mem;

            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*cytnx_uint32(_Rin[tmp2]);
                }



        }

//----------------------------------------
        void Kron_internal_i16tcd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_int16 *_Lin = (cytnx_int16*)Lin->Mem;
            cytnx_complex128 *_Rin = (cytnx_complex128*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }

        }
        void Kron_internal_i16tcf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_complex64 *_out = (cytnx_complex64*)out->Mem;
            cytnx_int16 *_Lin = (cytnx_int16*)Lin->Mem;
            cytnx_complex64 *_Rin = (cytnx_complex64*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }

        }
        void Kron_internal_i16td(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_int16 *_Lin = (cytnx_int16*)Lin->Mem;
            cytnx_double *_Rin = (cytnx_double*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }

        }
        void Kron_internal_i16tf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_int16 *_Lin = (cytnx_int16*)Lin->Mem;
            cytnx_float *_Rin = (cytnx_float*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }

        }
        void Kron_internal_i16ti64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_int64 *_out = (cytnx_int64*)out->Mem;
            cytnx_int16 *_Lin = (cytnx_int16*)Lin->Mem;
            cytnx_int64 *_Rin = (cytnx_int64*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }

        }
        void Kron_internal_i16tu64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_uint64 *_out = (cytnx_uint64*)out->Mem;
            cytnx_int16 *_Lin = (cytnx_int16*)Lin->Mem;
            cytnx_uint64 *_Rin = (cytnx_uint64*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }

        }
        void Kron_internal_i16ti32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_int32 *_out = (cytnx_int32*)out->Mem;
            cytnx_int16 *_Lin = (cytnx_int16*)Lin->Mem;
            cytnx_int32 *_Rin = (cytnx_int32*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }


        }
        void Kron_internal_i16tu32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_uint32 *_out = (cytnx_uint32*)out->Mem;
            cytnx_int16 *_Lin = (cytnx_int16*)Lin->Mem;
            cytnx_uint32 *_Rin = (cytnx_uint32*)Rin->Mem;

            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }


        }
        void Kron_internal_i16ti16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_int16 *_out = (cytnx_int16*)out->Mem;
            cytnx_int16 *_Lin = (cytnx_int16*)Lin->Mem;
            cytnx_int16 *_Rin = (cytnx_int16*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }


        }
        void Kron_internal_i16tu16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_int16 *_out = (cytnx_int16*)out->Mem;
            cytnx_int16 *_Lin = (cytnx_int16*)Lin->Mem;
            cytnx_uint16 *_Rin = (cytnx_uint16*)Rin->Mem;

            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }

            
        }
        void Kron_internal_i16tb(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_int16 *_out = (cytnx_int16*)out->Mem;
            cytnx_int16 *_Lin = (cytnx_int16*)Lin->Mem;
            cytnx_bool *_Rin = (cytnx_bool*)Rin->Mem;

            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*cytnx_int16(_Rin[tmp2]);
                }

        }

//----------------------------------------
        void Kron_internal_u16tcd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_uint16 *_Lin = (cytnx_uint16*)Lin->Mem;
            cytnx_complex128 *_Rin = (cytnx_complex128*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }

        }
        void Kron_internal_u16tcf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_complex64 *_out = (cytnx_complex64*)out->Mem;
            cytnx_uint16 *_Lin = (cytnx_uint16*)Lin->Mem;
            cytnx_complex64 *_Rin = (cytnx_complex64*)Rin->Mem;
            
            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }


        }
        void Kron_internal_u16td(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_uint16 *_Lin = (cytnx_uint16*)Lin->Mem;
            cytnx_double *_Rin = (cytnx_double*)Rin->Mem;

            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }



        }
        void Kron_internal_u16tf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_uint16 *_Lin = (cytnx_uint16*)Lin->Mem;
            cytnx_float *_Rin = (cytnx_float*)Rin->Mem;
            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }


        }
        void Kron_internal_u16ti64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_int64 *_out = (cytnx_int64*)out->Mem;
            cytnx_uint16 *_Lin = (cytnx_uint16*)Lin->Mem;
            cytnx_int64 *_Rin = (cytnx_int64*)Rin->Mem;
            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }


        }
        void Kron_internal_u16tu64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_uint64 *_out = (cytnx_uint64*)out->Mem;
            cytnx_uint16 *_Lin = (cytnx_uint16*)Lin->Mem;
            cytnx_uint64 *_Rin = (cytnx_uint64*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }

        }
        void Kron_internal_u16ti32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_int32 *_out = (cytnx_int32*)out->Mem;
            cytnx_uint16 *_Lin = (cytnx_uint16*)Lin->Mem;
            cytnx_int32 *_Rin = (cytnx_int32*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }

        }
        void Kron_internal_u16tu32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_uint32 *_out = (cytnx_uint32*)out->Mem;
            cytnx_uint16 *_Lin = (cytnx_uint16*)Lin->Mem;
            cytnx_uint32 *_Rin = (cytnx_uint32*)Rin->Mem;
            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }

        }
        void Kron_internal_u16ti16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_int16 *_out = (cytnx_int16*)out->Mem;
            cytnx_uint16 *_Lin = (cytnx_uint16*)Lin->Mem;
            cytnx_int16 *_Rin = (cytnx_int16*)Rin->Mem;

            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }



        }
        void Kron_internal_u16tu16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_uint16 *_out = (cytnx_uint16*)out->Mem;
            cytnx_uint16 *_Lin = (cytnx_uint16*)Lin->Mem;
            cytnx_uint16 *_Rin = (cytnx_uint16*)Rin->Mem;
            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }

        }

        void Kron_internal_u16tb(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_uint16 *_out = (cytnx_uint16*)out->Mem;
            cytnx_uint16 *_Lin = (cytnx_uint16*)Lin->Mem;
            cytnx_bool *_Rin = (cytnx_bool*)Rin->Mem;

            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*cytnx_uint16(_Rin[tmp2]);
                }


        }

//----------------------------------------
        void Kron_internal_btcd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_complex128 *_out = (cytnx_complex128*)out->Mem;
            cytnx_bool *_Lin = (cytnx_bool*)Lin->Mem;
            cytnx_complex128 *_Rin = (cytnx_complex128*)Rin->Mem;


            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = cytnx_complex128(_Lin[tmp],0)*_Rin[tmp2];
                }

                
        }
        void Kron_internal_btcf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_complex64 *_out = (cytnx_complex64*)out->Mem;
            cytnx_bool *_Lin = (cytnx_bool*)Lin->Mem;
            cytnx_complex64 *_Rin = (cytnx_complex64*)Rin->Mem;

            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = cytnx_complex64(_Lin[tmp],0)*_Rin[tmp2];
                }

        }
        void Kron_internal_btd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_bool *_Lin = (cytnx_bool*)Lin->Mem;
            cytnx_double *_Rin = (cytnx_double*)Rin->Mem;

            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = cytnx_double(_Lin[tmp])*_Rin[tmp2];
                }

        }
        void Kron_internal_btf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_bool *_Lin = (cytnx_bool*)Lin->Mem;
            cytnx_float *_Rin = (cytnx_float*)Rin->Mem;

            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = cytnx_float(_Lin[tmp])*_Rin[tmp2];
                }

        }
        void Kron_internal_bti64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_int64 *_out = (cytnx_int64*)out->Mem;
            cytnx_bool *_Lin = (cytnx_bool*)Lin->Mem;
            cytnx_int64 *_Rin = (cytnx_int64*)Rin->Mem;

            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = cytnx_int64(_Lin[tmp])*_Rin[tmp2];
                }

        }
        void Kron_internal_btu64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_uint64 *_out = (cytnx_uint64*)out->Mem;
            cytnx_bool *_Lin = (cytnx_bool*)Lin->Mem;
            cytnx_uint64 *_Rin = (cytnx_uint64*)Rin->Mem;

            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = cytnx_uint64(_Lin[tmp])*_Rin[tmp2];
                }

        }
        void Kron_internal_bti32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_int32 *_out = (cytnx_int32*)out->Mem;
            cytnx_bool *_Lin = (cytnx_bool*)Lin->Mem;
            cytnx_int32 *_Rin = (cytnx_int32*)Rin->Mem;

            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = cytnx_int32(_Lin[tmp])*_Rin[tmp2];
                }


        }
        void Kron_internal_btu32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_uint32 *_out = (cytnx_uint32*)out->Mem;
            cytnx_bool *_Lin = (cytnx_bool*)Lin->Mem;
            cytnx_uint32 *_Rin = (cytnx_uint32*)Rin->Mem;

            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = cytnx_uint32(_Lin[tmp])*_Rin[tmp2];
                }


        }
        void Kron_internal_bti16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_int16 *_out = (cytnx_int16*)out->Mem;
            cytnx_bool *_Lin = (cytnx_bool*)Lin->Mem;
            cytnx_int16 *_Rin = (cytnx_int16*)Rin->Mem;

            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = cytnx_int16(_Lin[tmp])*_Rin[tmp2];
                }

        }
        void Kron_internal_btu16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_uint16 *_out = (cytnx_uint16*)out->Mem;
            cytnx_bool *_Lin = (cytnx_bool*)Lin->Mem;
            cytnx_uint16 *_Rin = (cytnx_uint16*)Rin->Mem;

            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = cytnx_uint16(_Lin[tmp])*_Rin[tmp2];
                }
 
        }
        void Kron_internal_btb(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const std::vector<cytnx_uint64> &shape1, const std::vector<cytnx_uint64> &shape2){
            cytnx_bool *_out = (cytnx_bool*)out->Mem;
            cytnx_bool *_Lin = (cytnx_bool*)Lin->Mem;
            cytnx_bool *_Rin = (cytnx_bool*)Rin->Mem;

            cytnx_error_msg(shape1.size()!=shape2.size(),"[ERROR][Internal Kron] T1 rank != T2 rank %s","\n");
            cytnx_uint64 TotalElem = shape1[0]*shape2[0];
            std::vector<cytnx_uint64> new_shape_acc(shape1.size());
            std::vector<cytnx_uint64> shape1_acc(shape1.size());
            std::vector<cytnx_uint64> shape2_acc(shape1.size());
            new_shape_acc.back() = 1;
            shape1_acc.back() = 1;
            shape2_acc.back() = 1;

            for(unsigned long long i=1;i<new_shape_acc.size();i++){
                new_shape_acc[new_shape_acc.size()-1-i] = new_shape_acc[new_shape_acc.size()-i]*shape1[new_shape_acc.size()-i]*shape2[new_shape_acc.size()-i];
                TotalElem*=shape1[i]*shape2[i];
                shape1_acc[shape1_acc.size()-1-i] = shape1_acc[shape1_acc.size()-i]*shape1[shape1_acc.size()-i];
                shape2_acc[shape2_acc.size()-1-i] = shape2_acc[shape2_acc.size()-i]*shape2[shape2_acc.size()-i];
            }

            
            
            #ifdef UNI_OMP
                #pragma omp parallel for schedule(dynamic) 
            #endif
                for(unsigned long long i=0;i<TotalElem;i++){
                        std::vector<cytnx_uint64> idd;
                        cytnx_uint64 tmp = i;
                        cytnx_uint64 tmp2;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++)
                        {
                            idd.push_back(tmp/new_shape_acc[j]);
                            tmp= tmp%new_shape_acc[j];
                        }
                        //using idd to calculate add of Lin and Rin
                        tmp = tmp2 = 0;
                        for(unsigned long long j=0;j<new_shape_acc.size();j++){
                            tmp += cytnx_uint64(idd[j]/shape2[j])*shape1_acc[j];
                            tmp2 += cytnx_uint64(idd[j]%shape2[j])*shape2_acc[j];
                        }
        
                        _out[i] = _Lin[tmp]*_Rin[tmp2];
                }


        }

    }//namespace linalg_internal
}//namespace cytnx


