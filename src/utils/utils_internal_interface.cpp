#include "utils/utils_internal_interface.hpp"
#include <vector>
using namespace std;
namespace cytnx{
    namespace utils_internal{


        utils_internal_interface::utils_internal_interface(){

            ElemCast = vector<vector<ElemCast_io> >(N_Type,vector<ElemCast_io>(N_Type,NULL));
            ElemCast[Type.ComplexDouble][Type.ComplexDouble] = Cast_cpu_cdtcd;
            ElemCast[Type.ComplexDouble][Type.ComplexFloat ] = Cast_cpu_cdtcf;
            //ElemCast[Type.ComplexDouble][Type.Double       ] = Cast_cpu_invalid;
            //ElemCast[Type.ComplexDouble][Type.Float        ] = Cast_cpu_invalid;
            //ElemCast[Type.ComplexDouble][Type.Int64        ] = Cast_cpu_invalid;
            //ElemCast[Type.ComplexDouble][Type.Uint64       ] = Cast_cpu_invalid;
            //ElemCast[Type.ComplexDouble][Type.Int32        ] = Cast_cpu_invalid;
            //ElemCast[Type.ComplexDouble][Type.Uint32       ] = Cast_cpu_invalid;

            ElemCast[Type.ComplexFloat][Type.ComplexDouble] = Cast_cpu_cftcd;
            ElemCast[Type.ComplexFloat][Type.ComplexFloat ] = Cast_cpu_cftcf;
            //ElemCast[Type.ComplexFloat][Type.Double       ] = Cast_cpu_invalid;
            //ElemCast[Type.ComplexFloat][Type.Float        ] = Cast_cpu_invalid;
            //ElemCast[Type.ComplexFloat][Type.Int64        ] = Cast_cpu_invalid;
            //ElemCast[Type.ComplexFloat][Type.Uint64       ] = Cast_cpu_invalid;
            //ElemCast[Type.ComplexFloat][Type.Int32        ] = Cast_cpu_invalid;
            //ElemCast[Type.ComplexFloat][Type.Uint32       ] = Cast_cpu_invalid;

            ElemCast[Type.Double][Type.ComplexDouble] = Cast_cpu_dtcd;
            ElemCast[Type.Double][Type.ComplexFloat ] = Cast_cpu_dtcf;
            ElemCast[Type.Double][Type.Double       ] = Cast_cpu_dtd;
            ElemCast[Type.Double][Type.Float        ] = Cast_cpu_dtf;
            ElemCast[Type.Double][Type.Int64        ] = Cast_cpu_dti64;
            ElemCast[Type.Double][Type.Uint64       ] = Cast_cpu_dtu64;
            ElemCast[Type.Double][Type.Int32        ] = Cast_cpu_dti32;
            ElemCast[Type.Double][Type.Uint32       ] = Cast_cpu_dtu32;

            ElemCast[Type.Float][Type.ComplexDouble] = Cast_cpu_ftcd;
            ElemCast[Type.Float][Type.ComplexFloat ] = Cast_cpu_ftcf;
            ElemCast[Type.Float][Type.Double       ] = Cast_cpu_ftd;
            ElemCast[Type.Float][Type.Float        ] = Cast_cpu_ftf;
            ElemCast[Type.Float][Type.Int64        ] = Cast_cpu_fti64;
            ElemCast[Type.Float][Type.Uint64       ] = Cast_cpu_ftu64;
            ElemCast[Type.Float][Type.Int32        ] = Cast_cpu_fti32;
            ElemCast[Type.Float][Type.Uint32       ] = Cast_cpu_ftu32;

            ElemCast[Type.Int64][Type.ComplexDouble] = Cast_cpu_i64tcd;
            ElemCast[Type.Int64][Type.ComplexFloat ] = Cast_cpu_i64tcf;
            ElemCast[Type.Int64][Type.Double       ] = Cast_cpu_i64td;
            ElemCast[Type.Int64][Type.Float        ] = Cast_cpu_i64tf;
            ElemCast[Type.Int64][Type.Int64        ] = Cast_cpu_i64ti64;
            ElemCast[Type.Int64][Type.Uint64       ] = Cast_cpu_i64tu64;
            ElemCast[Type.Int64][Type.Int32        ] = Cast_cpu_i64ti32;
            ElemCast[Type.Int64][Type.Uint32       ] = Cast_cpu_i64tu32;

            ElemCast[Type.Uint64][Type.ComplexDouble] = Cast_cpu_u64tcd;
            ElemCast[Type.Uint64][Type.ComplexFloat ] = Cast_cpu_u64tcf;
            ElemCast[Type.Uint64][Type.Double       ] = Cast_cpu_u64td;

            //
            SetArange_ii.resize(N_Type,NULL);
            SetArange_ii[Type.ComplexDouble] = SetArange_cpu_cd;
            SetArange_ii[Type.ComplexFloat ] = SetArange_cpu_cf;
            SetArange_ii[Type.Double       ] = SetArange_cpu_d ;
            SetArange_ii[Type.Float        ] = SetArange_cpu_f ;
            SetArange_ii[Type.Uint64       ] = SetArange_cpu_u64;
            SetArange_ii[Type.Int64        ] = SetArange_cpu_i64;
            SetArange_ii[Type.Uint32       ] = SetArange_cpu_u32;
            SetArange_ii[Type.Int32        ] = SetArange_cpu_i32;

            //
            GetElems_ii.resize(N_Type,NULL);
            GetElems_ii[Type.ComplexDouble] = GetElems_cpu_cd;
            GetElems_ii[Type.ComplexFloat ] = GetElems_cpu_cf;
            GetElems_ii[Type.Double       ] = GetElems_cpu_d ;
            GetElems_ii[Type.Float        ] = GetElems_cpu_f ;
            GetElems_ii[Type.Uint64       ] = GetElems_cpu_u64;
            GetElems_ii[Type.Int64        ] = GetElems_cpu_i64;
            GetElems_ii[Type.Uint32       ] = GetElems_cpu_u32;
            GetElems_ii[Type.Int32        ] = GetElems_cpu_i32;

            #ifdef UNI_GPU
                cuElemCast = vector<vector<ElemCast_io> >(N_Type,vector<ElemCast_io>(N_Type,NULL));

                cuElemCast[Type.ComplexDouble][Type.ComplexDouble] = cuCast_gpu_cdtcd;
                cuElemCast[Type.ComplexDouble][Type.ComplexFloat ] = cuCast_gpu_cdtcf;
                //cuElemCast[Type.ComplexDouble][Type.Double       ] = cuCast_gpu_invalid;
                //cuElemCast[Type.ComplexDouble][Type.Float        ] = cuCast_gpu_invalid;
                //cuElemCast[Type.ComplexDouble][Type.Int64        ] = cuCast_gpu_invalid;
                //cuElemCast[Type.ComplexDouble][Type.Uint64       ] = cuCast_gpu_invalid;
                //cuElemCast[Type.ComplexDouble][Type.Int32        ] = cuCast_gpu_invalid;
                //cuElemCast[Type.ComplexDouble][Type.Uint32       ] = cuCast_gpu_invalid;

                cuElemCast[Type.ComplexFloat][Type.ComplexDouble] = cuCast_gpu_cftcd;
                cuElemCast[Type.ComplexFloat][Type.ComplexFloat ] = cuCast_gpu_cftcf;
                //cuElemCast[Type.ComplexFloat][Type.Double       ] = cuCast_gpu_invalid;
                //cuElemCast[Type.ComplexFloat][Type.Float        ] = cuCast_gpu_invalid;
                //cuElemCast[Type.ComplexFloat][Type.Int64        ] = cuCast_gpu_invalid;
                //cuElemCast[Type.ComplexFloat][Type.Uint64       ] = cuCast_gpu_invalid;
                //cuElemCast[Type.ComplexFloat][Type.Int32        ] = cuCast_gpu_invalid;
                //cuElemCast[Type.ComplexFloat][Type.Uint32       ] = cuCast_gpu_invalid;

                cuElemCast[Type.Double][Type.ComplexDouble] = cuCast_gpu_dtcd;
                cuElemCast[Type.Double][Type.ComplexFloat ] = cuCast_gpu_dtcf;
                cuElemCast[Type.Double][Type.Double       ] = cuCast_gpu_dtd;
                cuElemCast[Type.Double][Type.Float        ] = cuCast_gpu_dtf;
                cuElemCast[Type.Double][Type.Int64        ] = cuCast_gpu_dti64;
                cuElemCast[Type.Double][Type.Uint64       ] = cuCast_gpu_dtu64;
                cuElemCast[Type.Double][Type.Int32        ] = cuCast_gpu_dti32;
                cuElemCast[Type.Double][Type.Uint32       ] = cuCast_gpu_dtu32;

                cuElemCast[Type.Float][Type.ComplexDouble] = cuCast_gpu_ftcd;
                cuElemCast[Type.Float][Type.ComplexFloat ] = cuCast_gpu_ftcf;
                cuElemCast[Type.Float][Type.Double       ] = cuCast_gpu_ftd;
                cuElemCast[Type.Float][Type.Float        ] = cuCast_gpu_ftf;
                cuElemCast[Type.Float][Type.Int64        ] = cuCast_gpu_fti64;
                cuElemCast[Type.Float][Type.Uint64       ] = cuCast_gpu_ftu64;
                cuElemCast[Type.Float][Type.Int32        ] = cuCast_gpu_fti32;
                cuElemCast[Type.Float][Type.Uint32       ] = cuCast_gpu_ftu32;

                cuElemCast[Type.Int64][Type.ComplexDouble] = cuCast_gpu_i64tcd;
                cuElemCast[Type.Int64][Type.ComplexFloat ] = cuCast_gpu_i64tcf;
                cuElemCast[Type.Int64][Type.Double       ] = cuCast_gpu_i64td;
                cuElemCast[Type.Int64][Type.Float        ] = cuCast_gpu_i64tf;
                cuElemCast[Type.Int64][Type.Int64        ] = cuCast_gpu_i64ti64;
                cuElemCast[Type.Int64][Type.Uint64       ] = cuCast_gpu_i64tu64;
                cuElemCast[Type.Int64][Type.Int32        ] = cuCast_gpu_i64ti32;
                cuElemCast[Type.Int64][Type.Uint32       ] = cuCast_gpu_i64tu32;

                cuElemCast[Type.Uint64][Type.ComplexDouble] = cuCast_gpu_u64tcd;
                cuElemCast[Type.Uint64][Type.ComplexFloat ] = cuCast_gpu_u64tcf;
                cuElemCast[Type.Uint64][Type.Double       ] = cuCast_gpu_u64td;


                cuSetArange_ii.resize(N_Type,NULL);
                cuSetArange_ii[Type.ComplexDouble] = cuSetArange_gpu_cd;
                cuSetArange_ii[Type.ComplexFloat ] = cuSetArange_gpu_cf;
                cuSetArange_ii[Type.Double       ] = cuSetArange_gpu_d ;
                cuSetArange_ii[Type.Float        ] = cuSetArange_gpu_f ;
                cuSetArange_ii[Type.Uint64       ] = cuSetArange_gpu_u64;
                cuSetArange_ii[Type.Int64        ] = cuSetArange_gpu_i64;
                cuSetArange_ii[Type.Uint32       ] = cuSetArange_gpu_u32;
                cuSetArange_ii[Type.Int32        ] = cuSetArange_gpu_i32;

                cuGetElems_ii.resize(N_Type,NULL);
                cuGetElems_ii[Type.ComplexDouble] = cuGetElems_gpu_cd;
                cuGetElems_ii[Type.ComplexFloat ] = cuGetElems_gpu_cf;
                cuGetElems_ii[Type.Double       ] = cuGetElems_gpu_d ;
                cuGetElems_ii[Type.Float        ] = cuGetElems_gpu_f ;
                cuGetElems_ii[Type.Uint64       ] = cuGetElems_gpu_u64;
                cuGetElems_ii[Type.Int64        ] = cuGetElems_gpu_i64;
                cuGetElems_ii[Type.Uint32       ] = cuGetElems_gpu_u32;
                cuGetElems_ii[Type.Int32        ] = cuGetElems_gpu_i32;

            #endif

        }

        utils_internal_interface uii;

    }
}


