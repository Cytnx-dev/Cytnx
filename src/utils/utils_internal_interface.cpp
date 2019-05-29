#include "utils/utils_internal_interface.hpp"
#include <vector>
using namespace std;
namespace cytnx{
    namespace utils_internal{


        utils_internal_interface::utils_internal_interface(){

            ElemCast = vector<vector<ElemCast_io> >(N_Type,vector<ElemCast_io>(N_Type,NULL));
            ElemCast[cytnxtype.ComplexDouble][cytnxtype.ComplexDouble] = Cast_cpu_cdtcd;
            ElemCast[cytnxtype.ComplexDouble][cytnxtype.ComplexFloat ] = Cast_cpu_cdtcf;
            //ElemCast[cytnxtype.ComplexDouble][cytnxtype.Double       ] = Cast_cpu_invalid;
            //ElemCast[cytnxtype.ComplexDouble][cytnxtype.Float        ] = Cast_cpu_invalid;
            //ElemCast[cytnxtype.ComplexDouble][cytnxtype.Int64        ] = Cast_cpu_invalid;
            //ElemCast[cytnxtype.ComplexDouble][cytnxtype.Uint64       ] = Cast_cpu_invalid;
            //ElemCast[cytnxtype.ComplexDouble][cytnxtype.Int32        ] = Cast_cpu_invalid;
            //ElemCast[cytnxtype.ComplexDouble][cytnxtype.Uint32       ] = Cast_cpu_invalid;

            ElemCast[cytnxtype.ComplexFloat][cytnxtype.ComplexDouble] = Cast_cpu_cftcd;
            ElemCast[cytnxtype.ComplexFloat][cytnxtype.ComplexFloat ] = Cast_cpu_cftcf;
            //ElemCast[cytnxtype.ComplexFloat][cytnxtype.Double       ] = Cast_cpu_invalid;
            //ElemCast[cytnxtype.ComplexFloat][cytnxtype.Float        ] = Cast_cpu_invalid;
            //ElemCast[cytnxtype.ComplexFloat][cytnxtype.Int64        ] = Cast_cpu_invalid;
            //ElemCast[cytnxtype.ComplexFloat][cytnxtype.Uint64       ] = Cast_cpu_invalid;
            //ElemCast[cytnxtype.ComplexFloat][cytnxtype.Int32        ] = Cast_cpu_invalid;
            //ElemCast[cytnxtype.ComplexFloat][cytnxtype.Uint32       ] = Cast_cpu_invalid;

            ElemCast[cytnxtype.Double][cytnxtype.ComplexDouble] = Cast_cpu_dtcd;
            ElemCast[cytnxtype.Double][cytnxtype.ComplexFloat ] = Cast_cpu_dtcf;
            ElemCast[cytnxtype.Double][cytnxtype.Double       ] = Cast_cpu_dtd;
            ElemCast[cytnxtype.Double][cytnxtype.Float        ] = Cast_cpu_dtf;
            ElemCast[cytnxtype.Double][cytnxtype.Int64        ] = Cast_cpu_dti64;
            ElemCast[cytnxtype.Double][cytnxtype.Uint64       ] = Cast_cpu_dtu64;
            ElemCast[cytnxtype.Double][cytnxtype.Int32        ] = Cast_cpu_dti32;
            ElemCast[cytnxtype.Double][cytnxtype.Uint32       ] = Cast_cpu_dtu32;

            ElemCast[cytnxtype.Float][cytnxtype.ComplexDouble] = Cast_cpu_ftcd;
            ElemCast[cytnxtype.Float][cytnxtype.ComplexFloat ] = Cast_cpu_ftcf;
            ElemCast[cytnxtype.Float][cytnxtype.Double       ] = Cast_cpu_ftd;
            ElemCast[cytnxtype.Float][cytnxtype.Float        ] = Cast_cpu_ftf;
            ElemCast[cytnxtype.Float][cytnxtype.Int64        ] = Cast_cpu_fti64;
            ElemCast[cytnxtype.Float][cytnxtype.Uint64       ] = Cast_cpu_ftu64;
            ElemCast[cytnxtype.Float][cytnxtype.Int32        ] = Cast_cpu_fti32;
            ElemCast[cytnxtype.Float][cytnxtype.Uint32       ] = Cast_cpu_ftu32;

            ElemCast[cytnxtype.Int64][cytnxtype.ComplexDouble] = Cast_cpu_i64tcd;
            ElemCast[cytnxtype.Int64][cytnxtype.ComplexFloat ] = Cast_cpu_i64tcf;
            ElemCast[cytnxtype.Int64][cytnxtype.Double       ] = Cast_cpu_i64td;
            ElemCast[cytnxtype.Int64][cytnxtype.Float        ] = Cast_cpu_i64tf;
            ElemCast[cytnxtype.Int64][cytnxtype.Int64        ] = Cast_cpu_i64ti64;
            ElemCast[cytnxtype.Int64][cytnxtype.Uint64       ] = Cast_cpu_i64tu64;
            ElemCast[cytnxtype.Int64][cytnxtype.Int32        ] = Cast_cpu_i64ti32;
            ElemCast[cytnxtype.Int64][cytnxtype.Uint32       ] = Cast_cpu_i64tu32;

            ElemCast[cytnxtype.Uint64][cytnxtype.ComplexDouble] = Cast_cpu_u64tcd;
            ElemCast[cytnxtype.Uint64][cytnxtype.ComplexFloat ] = Cast_cpu_u64tcf;
            ElemCast[cytnxtype.Uint64][cytnxtype.Double       ] = Cast_cpu_u64td;

            //
            SetArange_ii.resize(N_Type,NULL);
            SetArange_ii[cytnxtype.ComplexDouble] = SetArange_cpu_cd;
            SetArange_ii[cytnxtype.ComplexFloat ] = SetArange_cpu_cf;
            SetArange_ii[cytnxtype.Double       ] = SetArange_cpu_d ;
            SetArange_ii[cytnxtype.Float        ] = SetArange_cpu_f ;
            SetArange_ii[cytnxtype.Uint64       ] = SetArange_cpu_u64;
            SetArange_ii[cytnxtype.Int64        ] = SetArange_cpu_i64;
            SetArange_ii[cytnxtype.Uint32       ] = SetArange_cpu_u32;
            SetArange_ii[cytnxtype.Int32        ] = SetArange_cpu_i32;

            #ifdef UNI_GPU
                cuElemCast = vector<vector<ElemCast_io> >(N_Type,vector<ElemCast_io>(N_Type,NULL));

                cuElemCast[cytnxtype.ComplexDouble][cytnxtype.ComplexDouble] = cuCast_gpu_cdtcd;
                cuElemCast[cytnxtype.ComplexDouble][cytnxtype.ComplexFloat ] = cuCast_gpu_cdtcf;
                //cuElemCast[cytnxtype.ComplexDouble][cytnxtype.Double       ] = cuCast_gpu_invalid;
                //cuElemCast[cytnxtype.ComplexDouble][cytnxtype.Float        ] = cuCast_gpu_invalid;
                //cuElemCast[cytnxtype.ComplexDouble][cytnxtype.Int64        ] = cuCast_gpu_invalid;
                //cuElemCast[cytnxtype.ComplexDouble][cytnxtype.Uint64       ] = cuCast_gpu_invalid;
                //cuElemCast[cytnxtype.ComplexDouble][cytnxtype.Int32        ] = cuCast_gpu_invalid;
                //cuElemCast[cytnxtype.ComplexDouble][cytnxtype.Uint32       ] = cuCast_gpu_invalid;

                cuElemCast[cytnxtype.ComplexFloat][cytnxtype.ComplexDouble] = cuCast_gpu_cftcd;
                cuElemCast[cytnxtype.ComplexFloat][cytnxtype.ComplexFloat ] = cuCast_gpu_cftcf;
                //cuElemCast[cytnxtype.ComplexFloat][cytnxtype.Double       ] = cuCast_gpu_invalid;
                //cuElemCast[cytnxtype.ComplexFloat][cytnxtype.Float        ] = cuCast_gpu_invalid;
                //cuElemCast[cytnxtype.ComplexFloat][cytnxtype.Int64        ] = cuCast_gpu_invalid;
                //cuElemCast[cytnxtype.ComplexFloat][cytnxtype.Uint64       ] = cuCast_gpu_invalid;
                //cuElemCast[cytnxtype.ComplexFloat][cytnxtype.Int32        ] = cuCast_gpu_invalid;
                //cuElemCast[cytnxtype.ComplexFloat][cytnxtype.Uint32       ] = cuCast_gpu_invalid;

                cuElemCast[cytnxtype.Double][cytnxtype.ComplexDouble] = cuCast_gpu_dtcd;
                cuElemCast[cytnxtype.Double][cytnxtype.ComplexFloat ] = cuCast_gpu_dtcf;
                cuElemCast[cytnxtype.Double][cytnxtype.Double       ] = cuCast_gpu_dtd;
                cuElemCast[cytnxtype.Double][cytnxtype.Float        ] = cuCast_gpu_dtf;
                cuElemCast[cytnxtype.Double][cytnxtype.Int64        ] = cuCast_gpu_dti64;
                cuElemCast[cytnxtype.Double][cytnxtype.Uint64       ] = cuCast_gpu_dtu64;
                cuElemCast[cytnxtype.Double][cytnxtype.Int32        ] = cuCast_gpu_dti32;
                cuElemCast[cytnxtype.Double][cytnxtype.Uint32       ] = cuCast_gpu_dtu32;

                cuElemCast[cytnxtype.Float][cytnxtype.ComplexDouble] = cuCast_gpu_ftcd;
                cuElemCast[cytnxtype.Float][cytnxtype.ComplexFloat ] = cuCast_gpu_ftcf;
                cuElemCast[cytnxtype.Float][cytnxtype.Double       ] = cuCast_gpu_ftd;
                cuElemCast[cytnxtype.Float][cytnxtype.Float        ] = cuCast_gpu_ftf;
                cuElemCast[cytnxtype.Float][cytnxtype.Int64        ] = cuCast_gpu_fti64;
                cuElemCast[cytnxtype.Float][cytnxtype.Uint64       ] = cuCast_gpu_ftu64;
                cuElemCast[cytnxtype.Float][cytnxtype.Int32        ] = cuCast_gpu_fti32;
                cuElemCast[cytnxtype.Float][cytnxtype.Uint32       ] = cuCast_gpu_ftu32;

                cuElemCast[cytnxtype.Int64][cytnxtype.ComplexDouble] = cuCast_gpu_i64tcd;
                cuElemCast[cytnxtype.Int64][cytnxtype.ComplexFloat ] = cuCast_gpu_i64tcf;
                cuElemCast[cytnxtype.Int64][cytnxtype.Double       ] = cuCast_gpu_i64td;
                cuElemCast[cytnxtype.Int64][cytnxtype.Float        ] = cuCast_gpu_i64tf;
                cuElemCast[cytnxtype.Int64][cytnxtype.Int64        ] = cuCast_gpu_i64ti64;
                cuElemCast[cytnxtype.Int64][cytnxtype.Uint64       ] = cuCast_gpu_i64tu64;
                cuElemCast[cytnxtype.Int64][cytnxtype.Int32        ] = cuCast_gpu_i64ti32;
                cuElemCast[cytnxtype.Int64][cytnxtype.Uint32       ] = cuCast_gpu_i64tu32;

                cuElemCast[cytnxtype.Uint64][cytnxtype.ComplexDouble] = cuCast_gpu_u64tcd;
                cuElemCast[cytnxtype.Uint64][cytnxtype.ComplexFloat ] = cuCast_gpu_u64tcf;
                cuElemCast[cytnxtype.Uint64][cytnxtype.Double       ] = cuCast_gpu_u64td;


                cuSetArange_ii.resize(N_Type,NULL);
                cuSetArange_ii[cytnxtype.ComplexDouble] = cuSetArange_gpu_cd;
                cuSetArange_ii[cytnxtype.ComplexFloat ] = cuSetArange_gpu_cf;
                cuSetArange_ii[cytnxtype.Double       ] = cuSetArange_gpu_d ;
                cuSetArange_ii[cytnxtype.Float        ] = cuSetArange_gpu_f ;
                cuSetArange_ii[cytnxtype.Uint64       ] = cuSetArange_gpu_u64;
                cuSetArange_ii[cytnxtype.Int64        ] = cuSetArange_gpu_i64;
                cuSetArange_ii[cytnxtype.Uint32       ] = cuSetArange_gpu_u32;
                cuSetArange_ii[cytnxtype.Int32        ] = cuSetArange_gpu_i32;


            #endif

        }

        utils_internal_interface uii;

    }
}


