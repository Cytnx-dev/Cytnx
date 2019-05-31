#include "utils/utils_internal_cpu/SetElems_cpu.hpp"
#include "utils/utils_internal_interface.hpp"
#ifdef UNI_OMP
#include <omp.h>
#endif

namespace cytnx{
    namespace utils_internal{
    
        template<class T1, class T2>
        void SetElems_cpu_impl(void *in, void *out, const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem){

            //Start copy elem:
            T1* new_elem_ptr_ = static_cast<T1*>(in); 
            T2* elem_ptr_     = static_cast<T2*>(out);

            #ifdef UNI_OMP 
            #pragma omp parallel for schedule(dynamic)
            #endif
            for(cytnx_uint64 n=0;n < TotalElem; n++){
                //map from mem loc of new tensor to old tensor
                cytnx_uint64 Loc=0;
                cytnx_uint64 tmpn = n;
                for(cytnx_uint32 r=0;r < offj.size();r++){
                    if(locators[r].size()) Loc += locators[r][tmpn/new_offj[r]]*offj[r];
                    else Loc += cytnx_uint64(tmpn/new_offj[r])*offj[r];
                    tmpn %= new_offj[r];
                }
                elem_ptr_[Loc] = new_elem_ptr_[n];
            }

        }

        template<class T1, class T2>
        void SetElems_cpu_scal_impl(void *in, void *out, const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem){

            //Start copy elem:
            T1 new_elem_ = *(static_cast<T1*>(in)); 
            T2* elem_ptr_     = static_cast<T2*>(out);

            #ifdef UNI_OMP 
            #pragma omp parallel for schedule(dynamic)
            #endif
            for(cytnx_uint64 n=0;n < TotalElem; n++){
                //map from mem loc of new tensor to old tensor
                cytnx_uint64 Loc=0;
                cytnx_uint64 tmpn = n;
                for(cytnx_uint32 r=0;r < offj.size();r++){
                    if(locators[r].size()) Loc += locators[r][tmpn/new_offj[r]]*offj[r];
                    else Loc += cytnx_uint64(tmpn/new_offj[r])*offj[r];
                    tmpn %= new_offj[r];
                }
                elem_ptr_[Loc] = new_elem_;
            }

        }


        // out is the target Tensor, in is the rhs
        void SetElems_cpu_cdtcd(void* in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar){
            if(is_scalar) SetElems_cpu_scal_impl<cytnx_complex128,cytnx_complex128>(in,out,offj,new_offj,locators,TotalElem);
            else          SetElems_cpu_impl<cytnx_complex128,cytnx_complex128>(in,out,offj,new_offj,locators,TotalElem);
            
        }
        void SetElems_cpu_cdtcf(void* in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar){
            if(is_scalar) SetElems_cpu_scal_impl<cytnx_complex128,cytnx_complex64>(in,out,offj,new_offj,locators,TotalElem);
            else          SetElems_cpu_impl<cytnx_complex128,cytnx_complex64>(in,out,offj,new_offj,locators,TotalElem);
        }

        //----
        void SetElems_cpu_cftcd(void* in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar){
            if(is_scalar) SetElems_cpu_scal_impl<cytnx_complex64,cytnx_complex128>(in,out,offj,new_offj,locators,TotalElem);
            else          SetElems_cpu_impl<cytnx_complex64,cytnx_complex128>(in,out,offj,new_offj,locators,TotalElem);
        }
        void SetElems_cpu_cftcf(void* in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar){
            if(is_scalar) SetElems_cpu_scal_impl<cytnx_complex64,cytnx_complex64>(in,out,offj,new_offj,locators,TotalElem);
            else SetElems_cpu_impl<cytnx_complex64,cytnx_complex64>(in,out,offj,new_offj,locators,TotalElem);
        }

        //----
        void SetElems_cpu_dtcd(void* in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar){
            if(is_scalar) SetElems_cpu_scal_impl<cytnx_double,cytnx_complex128>(in,out,offj,new_offj,locators,TotalElem);
            else SetElems_cpu_impl<cytnx_double,cytnx_complex128>(in,out,offj,new_offj,locators,TotalElem);
        }
        void SetElems_cpu_dtcf(void* in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar){
            if(is_scalar) SetElems_cpu_scal_impl<cytnx_double,cytnx_complex64>(in,out,offj,new_offj,locators,TotalElem);
            else SetElems_cpu_impl<cytnx_double,cytnx_complex64>(in,out,offj,new_offj,locators,TotalElem);
        }
        void SetElems_cpu_dtd(void* in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar){
            if(is_scalar) SetElems_cpu_scal_impl<cytnx_double,cytnx_double>(in,out,offj,new_offj,locators,TotalElem);
            else SetElems_cpu_impl<cytnx_double,cytnx_double>(in,out,offj,new_offj,locators,TotalElem);
        }
        void SetElems_cpu_dtf(void* in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar){
            if(is_scalar) SetElems_cpu_scal_impl<cytnx_double,cytnx_float>(in,out,offj,new_offj,locators,TotalElem);
            else SetElems_cpu_impl<cytnx_double,cytnx_float>(in,out,offj,new_offj,locators,TotalElem);
        }
        void SetElems_cpu_dti64(void* in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar){
            if(is_scalar) SetElems_cpu_scal_impl<cytnx_double,cytnx_int64>(in,out,offj,new_offj,locators,TotalElem);
            else SetElems_cpu_impl<cytnx_double,cytnx_int64>(in,out,offj,new_offj,locators,TotalElem);
        }
        void SetElems_cpu_dtu64(void* in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar){
            if(is_scalar) SetElems_cpu_scal_impl<cytnx_double,cytnx_uint64>(in,out,offj,new_offj,locators,TotalElem);
            else SetElems_cpu_impl<cytnx_double,cytnx_uint64>(in,out,offj,new_offj,locators,TotalElem);
        }
        void SetElems_cpu_dti32(void* in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar){
            if(is_scalar) SetElems_cpu_scal_impl<cytnx_double,cytnx_int32>(in,out,offj,new_offj,locators,TotalElem);
            else SetElems_cpu_impl<cytnx_double,cytnx_int32>(in,out,offj,new_offj,locators,TotalElem);
        }
        void SetElems_cpu_dtu32(void* in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar){
            if(is_scalar) SetElems_cpu_scal_impl<cytnx_double,cytnx_uint32>(in,out,offj,new_offj,locators,TotalElem);
            else SetElems_cpu_impl<cytnx_double,cytnx_uint32>(in,out,offj,new_offj,locators,TotalElem);
        }

        //----
        void SetElems_cpu_ftcd(void* in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar){
            if(is_scalar) SetElems_cpu_scal_impl<cytnx_float,cytnx_complex128>(in,out,offj,new_offj,locators,TotalElem);
            else SetElems_cpu_impl<cytnx_float,cytnx_complex128>(in,out,offj,new_offj,locators,TotalElem);
        }
        void SetElems_cpu_ftcf(void* in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar){
            if(is_scalar) SetElems_cpu_scal_impl<cytnx_float,cytnx_complex64>(in,out,offj,new_offj,locators,TotalElem);
            else SetElems_cpu_impl<cytnx_float,cytnx_complex64>(in,out,offj,new_offj,locators,TotalElem);
        }
        void SetElems_cpu_ftd(void* in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar){
            if(is_scalar) SetElems_cpu_scal_impl<cytnx_float,cytnx_double>(in,out,offj,new_offj,locators,TotalElem);
            else SetElems_cpu_impl<cytnx_float,cytnx_double>(in,out,offj,new_offj,locators,TotalElem);
        }
        void SetElems_cpu_ftf(void* in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar){
            if(is_scalar) SetElems_cpu_scal_impl<cytnx_float,cytnx_float>(in,out,offj,new_offj,locators,TotalElem);
            else SetElems_cpu_impl<cytnx_float,cytnx_float>(in,out,offj,new_offj,locators,TotalElem);
        }
        void SetElems_cpu_fti64(void* in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar){
            if(is_scalar) SetElems_cpu_scal_impl<cytnx_float,cytnx_int64>(in,out,offj,new_offj,locators,TotalElem);
            else SetElems_cpu_impl<cytnx_float,cytnx_int64>(in,out,offj,new_offj,locators,TotalElem);
        }
        void SetElems_cpu_ftu64(void* in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar){
            if(is_scalar) SetElems_cpu_scal_impl<cytnx_float,cytnx_uint64>(in,out,offj,new_offj,locators,TotalElem);
            else SetElems_cpu_impl<cytnx_float,cytnx_uint64>(in,out,offj,new_offj,locators,TotalElem);
        }
        void SetElems_cpu_fti32(void* in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar){
            if(is_scalar) SetElems_cpu_scal_impl<cytnx_float,cytnx_int32>(in,out,offj,new_offj,locators,TotalElem);
            else SetElems_cpu_impl<cytnx_float,cytnx_int32>(in,out,offj,new_offj,locators,TotalElem);
        }
        void SetElems_cpu_ftu32(void* in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar){
            if(is_scalar) SetElems_cpu_scal_impl<cytnx_float,cytnx_uint32>(in,out,offj,new_offj,locators,TotalElem);
            else SetElems_cpu_impl<cytnx_float,cytnx_uint32>(in,out,offj,new_offj,locators,TotalElem);
        }
        
        //----
        void SetElems_cpu_i64tcd(void* in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar){
            if(is_scalar) SetElems_cpu_scal_impl<cytnx_int64,cytnx_complex128>(in,out,offj,new_offj,locators,TotalElem);
            else SetElems_cpu_impl<cytnx_int64,cytnx_complex128>(in,out,offj,new_offj,locators,TotalElem);
        }
        void SetElems_cpu_i64tcf(void* in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar){
            if(is_scalar) SetElems_cpu_scal_impl<cytnx_int64,cytnx_complex64>(in,out,offj,new_offj,locators,TotalElem);
            else SetElems_cpu_impl<cytnx_int64,cytnx_complex64>(in,out,offj,new_offj,locators,TotalElem);
        }
        void SetElems_cpu_i64td(void* in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar){
            if(is_scalar) SetElems_cpu_scal_impl<cytnx_int64,cytnx_double>(in,out,offj,new_offj,locators,TotalElem);
            else SetElems_cpu_impl<cytnx_int64,cytnx_double>(in,out,offj,new_offj,locators,TotalElem);
        }
        void SetElems_cpu_i64tf(void* in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar){
            if(is_scalar) SetElems_cpu_scal_impl<cytnx_int64,cytnx_float>(in,out,offj,new_offj,locators,TotalElem);
            else SetElems_cpu_impl<cytnx_int64,cytnx_float>(in,out,offj,new_offj,locators,TotalElem);
        }
        void SetElems_cpu_i64ti64(void* in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar){
            if(is_scalar) SetElems_cpu_scal_impl<cytnx_int64,cytnx_int64>(in,out,offj,new_offj,locators,TotalElem);
            else SetElems_cpu_impl<cytnx_int64,cytnx_int64>(in,out,offj,new_offj,locators,TotalElem);
        }
        void SetElems_cpu_i64tu64(void* in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar){
            if(is_scalar) SetElems_cpu_scal_impl<cytnx_int64,cytnx_uint64>(in,out,offj,new_offj,locators,TotalElem);
            else SetElems_cpu_impl<cytnx_int64,cytnx_uint64>(in,out,offj,new_offj,locators,TotalElem);
        }
        void SetElems_cpu_i64ti32(void* in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar){
            if(is_scalar) SetElems_cpu_scal_impl<cytnx_int64,cytnx_int32>(in,out,offj,new_offj,locators,TotalElem);
            else SetElems_cpu_impl<cytnx_int64,cytnx_int32>(in,out,offj,new_offj,locators,TotalElem);
        }
        void SetElems_cpu_i64tu32(void* in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar){
            if(is_scalar) SetElems_cpu_scal_impl<cytnx_int64,cytnx_uint32>(in,out,offj,new_offj,locators,TotalElem);
            else SetElems_cpu_impl<cytnx_int64,cytnx_uint32>(in,out,offj,new_offj,locators,TotalElem);
        }

        //----
        void SetElems_cpu_u64tcd(void* in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar){
            if(is_scalar) SetElems_cpu_scal_impl<cytnx_uint64,cytnx_complex128>(in,out,offj,new_offj,locators,TotalElem);
            else SetElems_cpu_impl<cytnx_uint64,cytnx_complex128>(in,out,offj,new_offj,locators,TotalElem);
        }
        void SetElems_cpu_u64tcf(void* in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar){
            if(is_scalar) SetElems_cpu_scal_impl<cytnx_uint64,cytnx_complex64>(in,out,offj,new_offj,locators,TotalElem);
            else SetElems_cpu_impl<cytnx_uint64,cytnx_complex64>(in,out,offj,new_offj,locators,TotalElem);
        }
        void SetElems_cpu_u64td(void* in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar){
            if(is_scalar) SetElems_cpu_scal_impl<cytnx_uint64,cytnx_double>(in,out,offj,new_offj,locators,TotalElem);
            else SetElems_cpu_impl<cytnx_uint64,cytnx_double>(in,out,offj,new_offj,locators,TotalElem);
        }
        void SetElems_cpu_u64tf(void* in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar){
            if(is_scalar) SetElems_cpu_scal_impl<cytnx_uint64,cytnx_float>(in,out,offj,new_offj,locators,TotalElem);
            else SetElems_cpu_impl<cytnx_uint64,cytnx_float>(in,out,offj,new_offj,locators,TotalElem);
        }
        void SetElems_cpu_u64ti64(void* in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar){
            if(is_scalar) SetElems_cpu_scal_impl<cytnx_uint64,cytnx_int64>(in,out,offj,new_offj,locators,TotalElem);
            else SetElems_cpu_impl<cytnx_uint64,cytnx_int64>(in,out,offj,new_offj,locators,TotalElem);
        }
        void SetElems_cpu_u64tu64(void* in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar){
            if(is_scalar) SetElems_cpu_scal_impl<cytnx_uint64,cytnx_uint64>(in,out,offj,new_offj,locators,TotalElem);
            else SetElems_cpu_impl<cytnx_uint64,cytnx_uint64>(in,out,offj,new_offj,locators,TotalElem);
        }
        void SetElems_cpu_u64ti32(void* in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar){
            if(is_scalar) SetElems_cpu_scal_impl<cytnx_uint64,cytnx_int32>(in,out,offj,new_offj,locators,TotalElem);
            else SetElems_cpu_impl<cytnx_uint64,cytnx_int32>(in,out,offj,new_offj,locators,TotalElem);
        }
        void SetElems_cpu_u64tu32(void* in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar){
            if(is_scalar) SetElems_cpu_scal_impl<cytnx_uint64,cytnx_uint32>(in,out,offj,new_offj,locators,TotalElem);
            else SetElems_cpu_impl<cytnx_uint64,cytnx_uint32>(in,out,offj,new_offj,locators,TotalElem);
        }

        //----
        void SetElems_cpu_i32tcd(void* in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar){
            if(is_scalar) SetElems_cpu_scal_impl<cytnx_int32,cytnx_complex128>(in,out,offj,new_offj,locators,TotalElem);
            else SetElems_cpu_impl<cytnx_int32,cytnx_complex128>(in,out,offj,new_offj,locators,TotalElem);
        }
        void SetElems_cpu_i32tcf(void* in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar){
            if(is_scalar) SetElems_cpu_scal_impl<cytnx_int32,cytnx_complex64>(in,out,offj,new_offj,locators,TotalElem);
            else SetElems_cpu_impl<cytnx_int32,cytnx_complex64>(in,out,offj,new_offj,locators,TotalElem);
        }
        void SetElems_cpu_i32td(void* in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar){
            if(is_scalar) SetElems_cpu_scal_impl<cytnx_int32,cytnx_double>(in,out,offj,new_offj,locators,TotalElem);
            else SetElems_cpu_impl<cytnx_int32,cytnx_double>(in,out,offj,new_offj,locators,TotalElem);
        } 
        void SetElems_cpu_i32tf(void* in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar){
            if(is_scalar) SetElems_cpu_scal_impl<cytnx_int32,cytnx_float>(in,out,offj,new_offj,locators,TotalElem);
            else SetElems_cpu_impl<cytnx_int32,cytnx_float>(in,out,offj,new_offj,locators,TotalElem);
        }
        void SetElems_cpu_i32ti64(void* in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar){
            if(is_scalar) SetElems_cpu_scal_impl<cytnx_int32,cytnx_int64>(in,out,offj,new_offj,locators,TotalElem);
            else SetElems_cpu_impl<cytnx_int32,cytnx_int64>(in,out,offj,new_offj,locators,TotalElem);
        }
        void SetElems_cpu_i32tu64(void* in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar){
            if(is_scalar) SetElems_cpu_scal_impl<cytnx_int32,cytnx_uint64>(in,out,offj,new_offj,locators,TotalElem);
            else SetElems_cpu_impl<cytnx_int32,cytnx_uint64>(in,out,offj,new_offj,locators,TotalElem);
        }
        void SetElems_cpu_i32ti32(void* in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar){
            if(is_scalar) SetElems_cpu_scal_impl<cytnx_int32,cytnx_int32>(in,out,offj,new_offj,locators,TotalElem);
            else SetElems_cpu_impl<cytnx_int32,cytnx_int32>(in,out,offj,new_offj,locators,TotalElem);
        }
        void SetElems_cpu_i32tu32(void* in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar){
            if(is_scalar) SetElems_cpu_scal_impl<cytnx_int32,cytnx_uint32>(in,out,offj,new_offj,locators,TotalElem);
            else SetElems_cpu_impl<cytnx_int32,cytnx_uint32>(in,out,offj,new_offj,locators,TotalElem);
        }

        //----
        void SetElems_cpu_u32tcd(void* in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar){
            if(is_scalar) SetElems_cpu_scal_impl<cytnx_uint32,cytnx_complex128>(in,out,offj,new_offj,locators,TotalElem);
            else SetElems_cpu_impl<cytnx_uint32,cytnx_complex128>(in,out,offj,new_offj,locators,TotalElem);
        }
        void SetElems_cpu_u32tcf(void* in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar){
            if(is_scalar) SetElems_cpu_scal_impl<cytnx_uint32,cytnx_complex64>(in,out,offj,new_offj,locators,TotalElem);
            else SetElems_cpu_impl<cytnx_uint32,cytnx_complex64>(in,out,offj,new_offj,locators,TotalElem);
        }
        void SetElems_cpu_u32td(void* in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar){
            if(is_scalar) SetElems_cpu_scal_impl<cytnx_uint32,cytnx_double>(in,out,offj,new_offj,locators,TotalElem);
            else SetElems_cpu_impl<cytnx_uint32,cytnx_double>(in,out,offj,new_offj,locators,TotalElem);
        }
        void SetElems_cpu_u32tf(void* in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar){
            if(is_scalar) SetElems_cpu_scal_impl<cytnx_uint32,cytnx_float>(in,out,offj,new_offj,locators,TotalElem);
            else SetElems_cpu_impl<cytnx_uint32,cytnx_float>(in,out,offj,new_offj,locators,TotalElem);
        }
        void SetElems_cpu_u32ti64(void* in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar){
            if(is_scalar) SetElems_cpu_scal_impl<cytnx_uint32,cytnx_int64>(in,out,offj,new_offj,locators,TotalElem);
            else SetElems_cpu_impl<cytnx_uint32,cytnx_int64>(in,out,offj,new_offj,locators,TotalElem);
        }
        void SetElems_cpu_u32tu64(void* in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar){
            if(is_scalar) SetElems_cpu_scal_impl<cytnx_uint32,cytnx_uint64>(in,out,offj,new_offj,locators,TotalElem);
            else SetElems_cpu_impl<cytnx_uint32,cytnx_uint64>(in,out,offj,new_offj,locators,TotalElem);
        }
        void SetElems_cpu_u32ti32(void* in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar){
            if(is_scalar) SetElems_cpu_scal_impl<cytnx_uint32,cytnx_int32>(in,out,offj,new_offj,locators,TotalElem);
            else SetElems_cpu_impl<cytnx_uint32,cytnx_int32>(in,out,offj,new_offj,locators,TotalElem);
        }
        void SetElems_cpu_u32tu32(void* in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar){
            if(is_scalar) SetElems_cpu_scal_impl<cytnx_uint32,cytnx_uint32>(in,out,offj,new_offj,locators,TotalElem);
            else SetElems_cpu_impl<cytnx_uint32,cytnx_uint32>(in,out,offj,new_offj,locators,TotalElem);
        }


    }
}
