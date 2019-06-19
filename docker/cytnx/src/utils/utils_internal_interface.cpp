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
            ElemCast[Type.Uint64][Type.Float        ] = Cast_cpu_u64tf;
            ElemCast[Type.Uint64][Type.Int64        ] = Cast_cpu_u64ti64;
            ElemCast[Type.Uint64][Type.Uint64       ] = Cast_cpu_u64tu64;
            ElemCast[Type.Uint64][Type.Int32        ] = Cast_cpu_u64ti32;
            ElemCast[Type.Uint64][Type.Uint32       ] = Cast_cpu_u64tu32;

            ElemCast[Type.Int32][Type.ComplexDouble] = Cast_cpu_i32tcd;
            ElemCast[Type.Int32][Type.ComplexFloat ] = Cast_cpu_i32tcf;
            ElemCast[Type.Int32][Type.Double       ] = Cast_cpu_i32td;
            ElemCast[Type.Int32][Type.Float        ] = Cast_cpu_i32tf;
            ElemCast[Type.Int32][Type.Int64        ] = Cast_cpu_i32ti64;
            ElemCast[Type.Int32][Type.Uint64       ] = Cast_cpu_i32tu64;
            ElemCast[Type.Int32][Type.Int32        ] = Cast_cpu_i32ti32;
            ElemCast[Type.Int32][Type.Uint32       ] = Cast_cpu_i32tu32;

            ElemCast[Type.Uint32][Type.ComplexDouble] = Cast_cpu_u32tcd;
            ElemCast[Type.Uint32][Type.ComplexFloat ] = Cast_cpu_u32tcf;
            ElemCast[Type.Uint32][Type.Double       ] = Cast_cpu_u32td;
            ElemCast[Type.Uint32][Type.Float        ] = Cast_cpu_u32tf;
            ElemCast[Type.Uint32][Type.Int64        ] = Cast_cpu_u32ti64;
            ElemCast[Type.Uint32][Type.Uint64       ] = Cast_cpu_u32tu64;
            ElemCast[Type.Uint32][Type.Int32        ] = Cast_cpu_u32ti32;
            ElemCast[Type.Uint32][Type.Uint32       ] = Cast_cpu_u32tu32;

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

            //
            SetElems_ii = vector< vector<SetElems_io> >(N_Type,vector<SetElems_io>(N_Type,NULL));
            SetElems_ii[Type.ComplexDouble][Type.ComplexDouble] = SetElems_cpu_cdtcd;
            SetElems_ii[Type.ComplexDouble][Type.ComplexFloat ] = SetElems_cpu_cdtcf;
            //SetElems_ii[Type.ComplexDouble][Type.Double       ] = SetElems_cpu_invalid;
            //SetElems_ii[Type.ComplexDouble][Type.Float        ] = SetElems_cpu_invalid;
            //SetElems_ii[Type.ComplexDouble][Type.Int64        ] = SetElems_cpu_invalid;
            //SetElems_ii[Type.ComplexDouble][Type.Uint64       ] = SetElems_cpu_invalid;
            //SetElems_ii[Type.ComplexDouble][Type.Int32        ] = SetElems_cpu_invalid;
            //SetElems_ii[Type.ComplexDouble][Type.Uint32       ] = SetElems_cpu_invalid;

            SetElems_ii[Type.ComplexFloat][Type.ComplexDouble] = SetElems_cpu_cftcd;
            SetElems_ii[Type.ComplexFloat][Type.ComplexFloat ] = SetElems_cpu_cftcf;
            //SetElems_ii[Type.ComplexFloat][Type.Double       ] = SetElems_cpu_invalid;
            //SetElems_ii[Type.ComplexFloat][Type.Float        ] = SetElems_cpu_invalid;
            //SetElems_ii[Type.ComplexFloat][Type.Int64        ] = SetElems_cpu_invalid;
            //SetElems_ii[Type.ComplexFloat][Type.Uint64       ] = SetElems_cpu_invalid;
            //SetElems_ii[Type.ComplexFloat][Type.Int32        ] = SetElems_cpu_invalid;
            //SetElems_ii[Type.ComplexFloat][Type.Uint32       ] = SetElems_cpu_invalid;

            SetElems_ii[Type.Double][Type.ComplexDouble] = SetElems_cpu_dtcd;
            SetElems_ii[Type.Double][Type.ComplexFloat ] = SetElems_cpu_dtcf;
            SetElems_ii[Type.Double][Type.Double       ] = SetElems_cpu_dtd;
            SetElems_ii[Type.Double][Type.Float        ] = SetElems_cpu_dtf;
            SetElems_ii[Type.Double][Type.Int64        ] = SetElems_cpu_dti64;
            SetElems_ii[Type.Double][Type.Uint64       ] = SetElems_cpu_dtu64;
            SetElems_ii[Type.Double][Type.Int32        ] = SetElems_cpu_dti32;
            SetElems_ii[Type.Double][Type.Uint32       ] = SetElems_cpu_dtu32;

            SetElems_ii[Type.Float][Type.ComplexDouble] = SetElems_cpu_ftcd;
            SetElems_ii[Type.Float][Type.ComplexFloat ] = SetElems_cpu_ftcf;
            SetElems_ii[Type.Float][Type.Double       ] = SetElems_cpu_ftd;
            SetElems_ii[Type.Float][Type.Float        ] = SetElems_cpu_ftf;
            SetElems_ii[Type.Float][Type.Int64        ] = SetElems_cpu_fti64;
            SetElems_ii[Type.Float][Type.Uint64       ] = SetElems_cpu_ftu64;
            SetElems_ii[Type.Float][Type.Int32        ] = SetElems_cpu_fti32;
            SetElems_ii[Type.Float][Type.Uint32       ] = SetElems_cpu_ftu32;

            SetElems_ii[Type.Int64][Type.ComplexDouble] = SetElems_cpu_i64tcd;
            SetElems_ii[Type.Int64][Type.ComplexFloat ] = SetElems_cpu_i64tcf;
            SetElems_ii[Type.Int64][Type.Double       ] = SetElems_cpu_i64td;
            SetElems_ii[Type.Int64][Type.Float        ] = SetElems_cpu_i64tf;
            SetElems_ii[Type.Int64][Type.Int64        ] = SetElems_cpu_i64ti64;
            SetElems_ii[Type.Int64][Type.Uint64       ] = SetElems_cpu_i64tu64;
            SetElems_ii[Type.Int64][Type.Int32        ] = SetElems_cpu_i64ti32;
            SetElems_ii[Type.Int64][Type.Uint32       ] = SetElems_cpu_i64tu32;

            SetElems_ii[Type.Uint64][Type.ComplexDouble] = SetElems_cpu_u64tcd;
            SetElems_ii[Type.Uint64][Type.ComplexFloat ] = SetElems_cpu_u64tcf;
            SetElems_ii[Type.Uint64][Type.Double       ] = SetElems_cpu_u64td;
            SetElems_ii[Type.Uint64][Type.Float        ] = SetElems_cpu_u64tf;
            SetElems_ii[Type.Uint64][Type.Int64        ] = SetElems_cpu_u64ti64;
            SetElems_ii[Type.Uint64][Type.Uint64       ] = SetElems_cpu_u64tu64;
            SetElems_ii[Type.Uint64][Type.Int32        ] = SetElems_cpu_u64ti32;
            SetElems_ii[Type.Uint64][Type.Uint32       ] = SetElems_cpu_u64tu32;

            SetElems_ii[Type.Int32][Type.ComplexDouble] = SetElems_cpu_i32tcd;
            SetElems_ii[Type.Int32][Type.ComplexFloat ] = SetElems_cpu_i32tcf;
            SetElems_ii[Type.Int32][Type.Double       ] = SetElems_cpu_i32td;
            SetElems_ii[Type.Int32][Type.Float        ] = SetElems_cpu_i32tf;
            SetElems_ii[Type.Int32][Type.Int64        ] = SetElems_cpu_i32ti64;
            SetElems_ii[Type.Int32][Type.Uint64       ] = SetElems_cpu_i32tu64;
            SetElems_ii[Type.Int32][Type.Int32        ] = SetElems_cpu_i32ti32;
            SetElems_ii[Type.Int32][Type.Uint32       ] = SetElems_cpu_i32tu32;

            SetElems_ii[Type.Uint32][Type.ComplexDouble] = SetElems_cpu_u32tcd;
            SetElems_ii[Type.Uint32][Type.ComplexFloat ] = SetElems_cpu_u32tcf;
            SetElems_ii[Type.Uint32][Type.Double       ] = SetElems_cpu_u32td;
            SetElems_ii[Type.Uint32][Type.Float        ] = SetElems_cpu_u32tf;
            SetElems_ii[Type.Uint32][Type.Int64        ] = SetElems_cpu_u32ti64;
            SetElems_ii[Type.Uint32][Type.Uint64       ] = SetElems_cpu_u32tu64;
            SetElems_ii[Type.Uint32][Type.Int32        ] = SetElems_cpu_u32ti32;
            SetElems_ii[Type.Uint32][Type.Uint32       ] = SetElems_cpu_u32tu32;


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
                cuElemCast[Type.Uint64][Type.Float        ] = cuCast_gpu_u64tf;
                cuElemCast[Type.Uint64][Type.Int64        ] = cuCast_gpu_u64ti64;
                cuElemCast[Type.Uint64][Type.Uint64       ] = cuCast_gpu_u64tu64;
                cuElemCast[Type.Uint64][Type.Int32        ] = cuCast_gpu_u64ti32;
                cuElemCast[Type.Uint64][Type.Uint32       ] = cuCast_gpu_u64tu32;

                cuElemCast[Type.Int32][Type.ComplexDouble] = cuCast_gpu_i32tcd;
                cuElemCast[Type.Int32][Type.ComplexFloat ] = cuCast_gpu_i32tcf;
                cuElemCast[Type.Int32][Type.Double       ] = cuCast_gpu_i32td;
                cuElemCast[Type.Int32][Type.Float        ] = cuCast_gpu_i32tf;
                cuElemCast[Type.Int32][Type.Int64        ] = cuCast_gpu_i32ti64;
                cuElemCast[Type.Int32][Type.Uint64       ] = cuCast_gpu_i32tu64;
                cuElemCast[Type.Int32][Type.Int32        ] = cuCast_gpu_i32ti32;
                cuElemCast[Type.Int32][Type.Uint32       ] = cuCast_gpu_i32tu32;

                cuElemCast[Type.Uint32][Type.ComplexDouble] = cuCast_gpu_u32tcd;
                cuElemCast[Type.Uint32][Type.ComplexFloat ] = cuCast_gpu_u32tcf;
                cuElemCast[Type.Uint32][Type.Double       ] = cuCast_gpu_u32td;
                cuElemCast[Type.Uint32][Type.Float        ] = cuCast_gpu_u32tf;
                cuElemCast[Type.Uint32][Type.Int64        ] = cuCast_gpu_u32ti64;
                cuElemCast[Type.Uint32][Type.Uint64       ] = cuCast_gpu_u32tu64;
                cuElemCast[Type.Uint32][Type.Int32        ] = cuCast_gpu_u32ti32;
                cuElemCast[Type.Uint32][Type.Uint32       ] = cuCast_gpu_u32tu32;


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

                //
                cuSetElems_ii = vector< vector<SetElems_io> >(N_Type,vector<SetElems_io>(N_Type,NULL));
                cuSetElems_ii[Type.ComplexDouble][Type.ComplexDouble] = cuSetElems_gpu_cdtcd;
                cuSetElems_ii[Type.ComplexDouble][Type.ComplexFloat ] = cuSetElems_gpu_cdtcf;
                //cuSetElems_ii[Type.ComplexDouble][Type.Double       ] = cuSetElems_gpu_invalid;
                //cuSetElems_ii[Type.ComplexDouble][Type.Float        ] = cuSetElems_gpu_invalid;
                //cuSetElems_ii[Type.ComplexDouble][Type.Int64        ] = cuSetElems_gpu_invalid;
                //cuSetElems_ii[Type.ComplexDouble][Type.Uint64       ] = cuSetElems_gpu_invalid;
                //cuSetElems_ii[Type.ComplexDouble][Type.Int32        ] = cuSetElems_gpu_invalid;
                //cuSetElems_ii[Type.ComplexDouble][Type.Uint32       ] = cuSetElems_gpu_invalid;

                cuSetElems_ii[Type.ComplexFloat][Type.ComplexDouble] = cuSetElems_gpu_cftcd;
                cuSetElems_ii[Type.ComplexFloat][Type.ComplexFloat ] = cuSetElems_gpu_cftcf;
                //cuSetElems_ii[Type.ComplexFloat][Type.Double       ] = cuSetElems_gpu_invalid;
                //cuSetElems_ii[Type.ComplexFloat][Type.Float        ] = cuSetElems_gpu_invalid;
                //cuSetElems_ii[Type.ComplexFloat][Type.Int64        ] = cuSetElems_gpu_invalid;
                //cuSetElems_ii[Type.ComplexFloat][Type.Uint64       ] = cuSetElems_gpu_invalid;
                //cuSetElems_ii[Type.ComplexFloat][Type.Int32        ] = cuSetElems_gpu_invalid;
                //cuSetElems_ii[Type.ComplexFloat][Type.Uint32       ] = cuSetElems_gpu_invalid;

                cuSetElems_ii[Type.Double][Type.ComplexDouble] = cuSetElems_gpu_dtcd;
                cuSetElems_ii[Type.Double][Type.ComplexFloat ] = cuSetElems_gpu_dtcf;
                cuSetElems_ii[Type.Double][Type.Double       ] = cuSetElems_gpu_dtd;
                cuSetElems_ii[Type.Double][Type.Float        ] = cuSetElems_gpu_dtf;
                cuSetElems_ii[Type.Double][Type.Int64        ] = cuSetElems_gpu_dti64;
                cuSetElems_ii[Type.Double][Type.Uint64       ] = cuSetElems_gpu_dtu64;
                cuSetElems_ii[Type.Double][Type.Int32        ] = cuSetElems_gpu_dti32;
                cuSetElems_ii[Type.Double][Type.Uint32       ] = cuSetElems_gpu_dtu32;

                cuSetElems_ii[Type.Float][Type.ComplexDouble] = cuSetElems_gpu_ftcd;
                cuSetElems_ii[Type.Float][Type.ComplexFloat ] = cuSetElems_gpu_ftcf;
                cuSetElems_ii[Type.Float][Type.Double       ] = cuSetElems_gpu_ftd;
                cuSetElems_ii[Type.Float][Type.Float        ] = cuSetElems_gpu_ftf;
                cuSetElems_ii[Type.Float][Type.Int64        ] = cuSetElems_gpu_fti64;
                cuSetElems_ii[Type.Float][Type.Uint64       ] = cuSetElems_gpu_ftu64;
                cuSetElems_ii[Type.Float][Type.Int32        ] = cuSetElems_gpu_fti32;
                cuSetElems_ii[Type.Float][Type.Uint32       ] = cuSetElems_gpu_ftu32;

                cuSetElems_ii[Type.Int64][Type.ComplexDouble] = cuSetElems_gpu_i64tcd;
                cuSetElems_ii[Type.Int64][Type.ComplexFloat ] = cuSetElems_gpu_i64tcf;
                cuSetElems_ii[Type.Int64][Type.Double       ] = cuSetElems_gpu_i64td;
                cuSetElems_ii[Type.Int64][Type.Float        ] = cuSetElems_gpu_i64tf;
                cuSetElems_ii[Type.Int64][Type.Int64        ] = cuSetElems_gpu_i64ti64;
                cuSetElems_ii[Type.Int64][Type.Uint64       ] = cuSetElems_gpu_i64tu64;
                cuSetElems_ii[Type.Int64][Type.Int32        ] = cuSetElems_gpu_i64ti32;
                cuSetElems_ii[Type.Int64][Type.Uint32       ] = cuSetElems_gpu_i64tu32;

                cuSetElems_ii[Type.Uint64][Type.ComplexDouble] = cuSetElems_gpu_u64tcd;
                cuSetElems_ii[Type.Uint64][Type.ComplexFloat ] = cuSetElems_gpu_u64tcf;
                cuSetElems_ii[Type.Uint64][Type.Double       ] = cuSetElems_gpu_u64td;
                cuSetElems_ii[Type.Uint64][Type.Float        ] = cuSetElems_gpu_u64tf;
                cuSetElems_ii[Type.Uint64][Type.Int64        ] = cuSetElems_gpu_u64ti64;
                cuSetElems_ii[Type.Uint64][Type.Uint64       ] = cuSetElems_gpu_u64tu64;
                cuSetElems_ii[Type.Uint64][Type.Int32        ] = cuSetElems_gpu_u64ti32;
                cuSetElems_ii[Type.Uint64][Type.Uint32       ] = cuSetElems_gpu_u64tu32;

                cuSetElems_ii[Type.Int32][Type.ComplexDouble] = cuSetElems_gpu_i32tcd;
                cuSetElems_ii[Type.Int32][Type.ComplexFloat ] = cuSetElems_gpu_i32tcf;
                cuSetElems_ii[Type.Int32][Type.Double       ] = cuSetElems_gpu_i32td;
                cuSetElems_ii[Type.Int32][Type.Float        ] = cuSetElems_gpu_i32tf;
                cuSetElems_ii[Type.Int32][Type.Int64        ] = cuSetElems_gpu_i32ti64;
                cuSetElems_ii[Type.Int32][Type.Uint64       ] = cuSetElems_gpu_i32tu64;
                cuSetElems_ii[Type.Int32][Type.Int32        ] = cuSetElems_gpu_i32ti32;
                cuSetElems_ii[Type.Int32][Type.Uint32       ] = cuSetElems_gpu_i32tu32;

                cuSetElems_ii[Type.Uint32][Type.ComplexDouble] = cuSetElems_gpu_u32tcd;
                cuSetElems_ii[Type.Uint32][Type.ComplexFloat ] = cuSetElems_gpu_u32tcf;
                cuSetElems_ii[Type.Uint32][Type.Double       ] = cuSetElems_gpu_u32td;
                cuSetElems_ii[Type.Uint32][Type.Float        ] = cuSetElems_gpu_u32tf;
                cuSetElems_ii[Type.Uint32][Type.Int64        ] = cuSetElems_gpu_u32ti64;
                cuSetElems_ii[Type.Uint32][Type.Uint64       ] = cuSetElems_gpu_u32tu64;
                cuSetElems_ii[Type.Uint32][Type.Int32        ] = cuSetElems_gpu_u32ti32;
                cuSetElems_ii[Type.Uint32][Type.Uint32       ] = cuSetElems_gpu_u32tu32;
            #endif

        }

        utils_internal_interface uii;

    }
}


