#include "linalg/linalg_internal_interface.hpp"

using namespace std;

namespace cytnx{
    namespace linalg_internal{


        linalg_internal_interface lii;

        linalg_internal_interface::linalg_internal_interface(){
            Ari_ii = vector<vector<Arithmicfunc_oii> >(N_Type,vector<Arithmicfunc_oii>(N_Type,NULL));

            Ari_ii[Type.ComplexDouble][Type.ComplexDouble] = Arithmic_internal_cdtcd;
            Ari_ii[Type.ComplexDouble][Type.ComplexFloat ] = Arithmic_internal_cdtcf;
            Ari_ii[Type.ComplexDouble][Type.Double       ] = Arithmic_internal_cdtd;
            Ari_ii[Type.ComplexDouble][Type.Float        ] = Arithmic_internal_cdtf;
            Ari_ii[Type.ComplexDouble][Type.Int64        ] = Arithmic_internal_cdti64;
            Ari_ii[Type.ComplexDouble][Type.Uint64       ] = Arithmic_internal_cdtu64;
            Ari_ii[Type.ComplexDouble][Type.Int32        ] = Arithmic_internal_cdti32;
            Ari_ii[Type.ComplexDouble][Type.Uint32       ] = Arithmic_internal_cdtu32;
            
            Ari_ii[Type.ComplexFloat][Type.ComplexDouble] = Arithmic_internal_cftcd;
            Ari_ii[Type.ComplexFloat][Type.ComplexFloat ] = Arithmic_internal_cftcf;
            Ari_ii[Type.ComplexFloat][Type.Double       ] = Arithmic_internal_cftd;
            Ari_ii[Type.ComplexFloat][Type.Float        ] = Arithmic_internal_cftf;
            Ari_ii[Type.ComplexFloat][Type.Int64        ] = Arithmic_internal_cfti64;
            Ari_ii[Type.ComplexFloat][Type.Uint64       ] = Arithmic_internal_cftu64;
            Ari_ii[Type.ComplexFloat][Type.Int32        ] = Arithmic_internal_cfti32;
            Ari_ii[Type.ComplexFloat][Type.Uint32       ] = Arithmic_internal_cftu32;
            
            Ari_ii[Type.Double][Type.ComplexDouble] = Arithmic_internal_dtcd;
            Ari_ii[Type.Double][Type.ComplexFloat ] = Arithmic_internal_dtcf;
            Ari_ii[Type.Double][Type.Double       ] = Arithmic_internal_dtd;
            Ari_ii[Type.Double][Type.Float        ] = Arithmic_internal_dtf;
            Ari_ii[Type.Double][Type.Int64        ] = Arithmic_internal_dti64;
            Ari_ii[Type.Double][Type.Uint64       ] = Arithmic_internal_dtu64;
            Ari_ii[Type.Double][Type.Int32        ] = Arithmic_internal_dti32;
            Ari_ii[Type.Double][Type.Uint32       ] = Arithmic_internal_dtu32;
            
            Ari_ii[Type.Float][Type.ComplexDouble] = Arithmic_internal_ftcd;
            Ari_ii[Type.Float][Type.ComplexFloat ] = Arithmic_internal_ftcf;
            Ari_ii[Type.Float][Type.Double       ] = Arithmic_internal_ftd;
            Ari_ii[Type.Float][Type.Float        ] = Arithmic_internal_ftf;
            Ari_ii[Type.Float][Type.Int64        ] = Arithmic_internal_fti64;
            Ari_ii[Type.Float][Type.Uint64       ] = Arithmic_internal_ftu64;
            Ari_ii[Type.Float][Type.Int32        ] = Arithmic_internal_fti32;
            Ari_ii[Type.Float][Type.Uint32       ] = Arithmic_internal_ftu32;
            
            Ari_ii[Type.Int64][Type.ComplexDouble] = Arithmic_internal_i64tcd;
            Ari_ii[Type.Int64][Type.ComplexFloat ] = Arithmic_internal_i64tcf;
            Ari_ii[Type.Int64][Type.Double       ] = Arithmic_internal_i64td;
            Ari_ii[Type.Int64][Type.Float        ] = Arithmic_internal_i64tf;
            Ari_ii[Type.Int64][Type.Int64        ] = Arithmic_internal_i64ti64;
            Ari_ii[Type.Int64][Type.Uint64       ] = Arithmic_internal_i64tu64;
            Ari_ii[Type.Int64][Type.Int32        ] = Arithmic_internal_i64ti32;
            Ari_ii[Type.Int64][Type.Uint32       ] = Arithmic_internal_i64tu32;
            
            Ari_ii[Type.Uint64][Type.ComplexDouble] = Arithmic_internal_u64tcd;
            Ari_ii[Type.Uint64][Type.ComplexFloat ] = Arithmic_internal_u64tcf;
            Ari_ii[Type.Uint64][Type.Double       ] = Arithmic_internal_u64td;
            Ari_ii[Type.Uint64][Type.Float        ] = Arithmic_internal_u64tf;
            Ari_ii[Type.Uint64][Type.Int64        ] = Arithmic_internal_u64ti64;
            Ari_ii[Type.Uint64][Type.Uint64       ] = Arithmic_internal_u64tu64;
            Ari_ii[Type.Uint64][Type.Int32        ] = Arithmic_internal_u64ti32;
            Ari_ii[Type.Uint64][Type.Uint32       ] = Arithmic_internal_u64tu32;
            
            Ari_ii[Type.Int32][Type.ComplexDouble] = Arithmic_internal_i32tcd;
            Ari_ii[Type.Int32][Type.ComplexFloat ] = Arithmic_internal_i32tcf;
            Ari_ii[Type.Int32][Type.Double       ] = Arithmic_internal_i32td;
            Ari_ii[Type.Int32][Type.Float        ] = Arithmic_internal_i32tf;
            Ari_ii[Type.Int32][Type.Int64        ] = Arithmic_internal_i32ti64;
            Ari_ii[Type.Int32][Type.Uint64       ] = Arithmic_internal_i32tu64;
            Ari_ii[Type.Int32][Type.Int32        ] = Arithmic_internal_i32ti32;
            Ari_ii[Type.Int32][Type.Uint32       ] = Arithmic_internal_i32tu32;
            
            Ari_ii[Type.Uint32][Type.ComplexDouble] = Arithmic_internal_u32tcd;
            Ari_ii[Type.Uint32][Type.ComplexFloat ] = Arithmic_internal_u32tcf;
            Ari_ii[Type.Uint32][Type.Double       ] = Arithmic_internal_u32td;
            Ari_ii[Type.Uint32][Type.Float        ] = Arithmic_internal_u32tf;
            Ari_ii[Type.Uint32][Type.Int64        ] = Arithmic_internal_u32ti64;
            Ari_ii[Type.Uint32][Type.Uint64       ] = Arithmic_internal_u32tu64;
            Ari_ii[Type.Uint32][Type.Int32        ] = Arithmic_internal_u32ti32;
            Ari_ii[Type.Uint32][Type.Uint32       ] = Arithmic_internal_u32tu32;

            //=====================
            Svd_ii = vector<Svdfunc_oii>(5);

            Svd_ii[Type.ComplexDouble] = Svd_internal_cd;
            Svd_ii[Type.ComplexFloat ] = Svd_internal_cf;
            Svd_ii[Type.Double       ] = Svd_internal_d;
            Svd_ii[Type.Float        ] = Svd_internal_f;

            //=====================
            Eigh_ii = vector<Eighfunc_oii>(5);

            Eigh_ii[Type.ComplexDouble] = Eigh_internal_cd;
            Eigh_ii[Type.ComplexFloat ] = Eigh_internal_cf;
            Eigh_ii[Type.Double       ] = Eigh_internal_d;
            Eigh_ii[Type.Float        ] = Eigh_internal_f;

            //=====================
            Exp_ii = vector<Expfunc_oii>(5);

            Exp_ii[Type.ComplexDouble] = Exp_internal_cd;
            Exp_ii[Type.ComplexFloat ] = Exp_internal_cf;
            Exp_ii[Type.Double       ] = Exp_internal_d;
            Exp_ii[Type.Float        ] = Exp_internal_f;

            //=====================
            Diag_ii = vector<Expfunc_oii>(N_Type);

            Diag_ii[Type.ComplexDouble] = Diag_internal_cd;
            Diag_ii[Type.ComplexFloat ] = Diag_internal_cf;
            Diag_ii[Type.Double       ] = Diag_internal_d;
            Diag_ii[Type.Float        ] = Diag_internal_f;
            Diag_ii[Type.Int64        ] = Diag_internal_i64;
            Diag_ii[Type.Uint64       ] = Diag_internal_u64;
            Diag_ii[Type.Int32        ] = Diag_internal_u32;
            Diag_ii[Type.Uint32       ] = Diag_internal_u32;

            //=====================
            Inv_inplace_ii = vector<Invinplacefunc_oii>(5);

            Inv_inplace_ii[Type.ComplexDouble] = Inv_inplace_internal_cd;
            Inv_inplace_ii[Type.ComplexFloat ] = Inv_inplace_internal_cf;
            Inv_inplace_ii[Type.Double       ] = Inv_inplace_internal_d;
            Inv_inplace_ii[Type.Float        ] = Inv_inplace_internal_f;


            //=====================
            Conj_inplace_ii = vector<Conjinplacefunc_oii>(3);

            Conj_inplace_ii[Type.ComplexDouble] = Conj_inplace_internal_cd;
            Conj_inplace_ii[Type.ComplexFloat ] = Conj_inplace_internal_cf;


            //=====================
            Matmul_ii = vector<Matmulfunc_oii>(N_Type);
            Matmul_ii[Type.ComplexDouble] = Matmul_internal_cd ;
            Matmul_ii[Type.ComplexFloat ] = Matmul_internal_cf ;
            Matmul_ii[Type.Double       ] = Matmul_internal_d  ;
            Matmul_ii[Type.Float        ] = Matmul_internal_f  ;
            Matmul_ii[Type.Int64        ] = Matmul_internal_i64;
            Matmul_ii[Type.Uint64       ] = Matmul_internal_u64;
            Matmul_ii[Type.Int32        ] = Matmul_internal_i32;
            Matmul_ii[Type.Uint32       ] = Matmul_internal_u32;

            //================
            Outer_ii = vector<vector<Outerfunc_oii> >(N_Type,vector<Outerfunc_oii>(N_Type,NULL));

            Outer_ii[Type.ComplexDouble][Type.ComplexDouble] = Outer_internal_cdtcd;
            Outer_ii[Type.ComplexDouble][Type.ComplexFloat ] = Outer_internal_cdtcf;
            Outer_ii[Type.ComplexDouble][Type.Double       ] = Outer_internal_cdtd;
            Outer_ii[Type.ComplexDouble][Type.Float        ] = Outer_internal_cdtf;
            Outer_ii[Type.ComplexDouble][Type.Int64        ] = Outer_internal_cdti64;
            Outer_ii[Type.ComplexDouble][Type.Uint64       ] = Outer_internal_cdtu64;
            Outer_ii[Type.ComplexDouble][Type.Int32        ] = Outer_internal_cdti32;
            Outer_ii[Type.ComplexDouble][Type.Uint32       ] = Outer_internal_cdtu32;
            
            Outer_ii[Type.ComplexFloat][Type.ComplexDouble] = Outer_internal_cftcd;
            Outer_ii[Type.ComplexFloat][Type.ComplexFloat ] = Outer_internal_cftcf;
            Outer_ii[Type.ComplexFloat][Type.Double       ] = Outer_internal_cftd;
            Outer_ii[Type.ComplexFloat][Type.Float        ] = Outer_internal_cftf;
            Outer_ii[Type.ComplexFloat][Type.Int64        ] = Outer_internal_cfti64;
            Outer_ii[Type.ComplexFloat][Type.Uint64       ] = Outer_internal_cftu64;
            Outer_ii[Type.ComplexFloat][Type.Int32        ] = Outer_internal_cfti32;
            Outer_ii[Type.ComplexFloat][Type.Uint32       ] = Outer_internal_cftu32;
            
            Outer_ii[Type.Double][Type.ComplexDouble] = Outer_internal_dtcd;
            Outer_ii[Type.Double][Type.ComplexFloat ] = Outer_internal_dtcf;
            Outer_ii[Type.Double][Type.Double       ] = Outer_internal_dtd;
            Outer_ii[Type.Double][Type.Float        ] = Outer_internal_dtf;
            Outer_ii[Type.Double][Type.Int64        ] = Outer_internal_dti64;
            Outer_ii[Type.Double][Type.Uint64       ] = Outer_internal_dtu64;
            Outer_ii[Type.Double][Type.Int32        ] = Outer_internal_dti32;
            Outer_ii[Type.Double][Type.Uint32       ] = Outer_internal_dtu32;
            
            Outer_ii[Type.Float][Type.ComplexDouble] = Outer_internal_ftcd;
            Outer_ii[Type.Float][Type.ComplexFloat ] = Outer_internal_ftcf;
            Outer_ii[Type.Float][Type.Double       ] = Outer_internal_ftd;
            Outer_ii[Type.Float][Type.Float        ] = Outer_internal_ftf;
            Outer_ii[Type.Float][Type.Int64        ] = Outer_internal_fti64;
            Outer_ii[Type.Float][Type.Uint64       ] = Outer_internal_ftu64;
            Outer_ii[Type.Float][Type.Int32        ] = Outer_internal_fti32;
            Outer_ii[Type.Float][Type.Uint32       ] = Outer_internal_ftu32;
            
            Outer_ii[Type.Int64][Type.ComplexDouble] = Outer_internal_i64tcd;
            Outer_ii[Type.Int64][Type.ComplexFloat ] = Outer_internal_i64tcf;
            Outer_ii[Type.Int64][Type.Double       ] = Outer_internal_i64td;
            Outer_ii[Type.Int64][Type.Float        ] = Outer_internal_i64tf;
            Outer_ii[Type.Int64][Type.Int64        ] = Outer_internal_i64ti64;
            Outer_ii[Type.Int64][Type.Uint64       ] = Outer_internal_i64tu64;
            Outer_ii[Type.Int64][Type.Int32        ] = Outer_internal_i64ti32;
            Outer_ii[Type.Int64][Type.Uint32       ] = Outer_internal_i64tu32;
            
            Outer_ii[Type.Uint64][Type.ComplexDouble] = Outer_internal_u64tcd;
            Outer_ii[Type.Uint64][Type.ComplexFloat ] = Outer_internal_u64tcf;
            Outer_ii[Type.Uint64][Type.Double       ] = Outer_internal_u64td;
            Outer_ii[Type.Uint64][Type.Float        ] = Outer_internal_u64tf;
            Outer_ii[Type.Uint64][Type.Int64        ] = Outer_internal_u64ti64;
            Outer_ii[Type.Uint64][Type.Uint64       ] = Outer_internal_u64tu64;
            Outer_ii[Type.Uint64][Type.Int32        ] = Outer_internal_u64ti32;
            Outer_ii[Type.Uint64][Type.Uint32       ] = Outer_internal_u64tu32;
            
            Outer_ii[Type.Int32][Type.ComplexDouble] = Outer_internal_i32tcd;
            Outer_ii[Type.Int32][Type.ComplexFloat ] = Outer_internal_i32tcf;
            Outer_ii[Type.Int32][Type.Double       ] = Outer_internal_i32td;
            Outer_ii[Type.Int32][Type.Float        ] = Outer_internal_i32tf;
            Outer_ii[Type.Int32][Type.Int64        ] = Outer_internal_i32ti64;
            Outer_ii[Type.Int32][Type.Uint64       ] = Outer_internal_i32tu64;
            Outer_ii[Type.Int32][Type.Int32        ] = Outer_internal_i32ti32;
            Outer_ii[Type.Int32][Type.Uint32       ] = Outer_internal_i32tu32;
            
            Outer_ii[Type.Uint32][Type.ComplexDouble] = Outer_internal_u32tcd;
            Outer_ii[Type.Uint32][Type.ComplexFloat ] = Outer_internal_u32tcf;
            Outer_ii[Type.Uint32][Type.Double       ] = Outer_internal_u32td;
            Outer_ii[Type.Uint32][Type.Float        ] = Outer_internal_u32tf;
            Outer_ii[Type.Uint32][Type.Int64        ] = Outer_internal_u32ti64;
            Outer_ii[Type.Uint32][Type.Uint64       ] = Outer_internal_u32tu64;
            Outer_ii[Type.Uint32][Type.Int32        ] = Outer_internal_u32ti32;
            Outer_ii[Type.Uint32][Type.Uint32       ] = Outer_internal_u32tu32;

            #ifdef UNI_GPU
                cuAri_ii = vector<vector<Arithmicfunc_oii> >(N_Type,vector<Arithmicfunc_oii>(N_Type));

                cuAri_ii[Type.ComplexDouble][Type.ComplexDouble] = cuArithmic_internal_cdtcd;
                cuAri_ii[Type.ComplexDouble][Type.ComplexFloat ] = cuArithmic_internal_cdtcf;
                cuAri_ii[Type.ComplexDouble][Type.Double       ] = cuArithmic_internal_cdtd;
                cuAri_ii[Type.ComplexDouble][Type.Float        ] = cuArithmic_internal_cdtf;
                cuAri_ii[Type.ComplexDouble][Type.Int64        ] = cuArithmic_internal_cdti64;
                cuAri_ii[Type.ComplexDouble][Type.Uint64       ] = cuArithmic_internal_cdtu64;
                cuAri_ii[Type.ComplexDouble][Type.Int32        ] = cuArithmic_internal_cdti32;
                cuAri_ii[Type.ComplexDouble][Type.Uint32       ] = cuArithmic_internal_cdtu32;
                
                cuAri_ii[Type.ComplexFloat][Type.ComplexDouble] = cuArithmic_internal_cftcd;
                cuAri_ii[Type.ComplexFloat][Type.ComplexFloat ] = cuArithmic_internal_cftcf;
                cuAri_ii[Type.ComplexFloat][Type.Double       ] = cuArithmic_internal_cftd;
                cuAri_ii[Type.ComplexFloat][Type.Float        ] = cuArithmic_internal_cftf;
                cuAri_ii[Type.ComplexFloat][Type.Int64        ] = cuArithmic_internal_cfti64;
                cuAri_ii[Type.ComplexFloat][Type.Uint64       ] = cuArithmic_internal_cftu64;
                cuAri_ii[Type.ComplexFloat][Type.Int32        ] = cuArithmic_internal_cfti32;
                cuAri_ii[Type.ComplexFloat][Type.Uint32       ] = cuArithmic_internal_cftu32;
                
                cuAri_ii[Type.Double][Type.ComplexDouble] = cuArithmic_internal_dtcd;
                cuAri_ii[Type.Double][Type.ComplexFloat ] = cuArithmic_internal_dtcf;
                cuAri_ii[Type.Double][Type.Double       ] = cuArithmic_internal_dtd;
                cuAri_ii[Type.Double][Type.Float        ] = cuArithmic_internal_dtf;
                cuAri_ii[Type.Double][Type.Int64        ] = cuArithmic_internal_dti64;
                cuAri_ii[Type.Double][Type.Uint64       ] = cuArithmic_internal_dtu64;
                cuAri_ii[Type.Double][Type.Int32        ] = cuArithmic_internal_dti32;
                cuAri_ii[Type.Double][Type.Uint32       ] = cuArithmic_internal_dtu32;
                
                cuAri_ii[Type.Float][Type.ComplexDouble] = cuArithmic_internal_ftcd;
                cuAri_ii[Type.Float][Type.ComplexFloat ] = cuArithmic_internal_ftcf;
                cuAri_ii[Type.Float][Type.Double       ] = cuArithmic_internal_ftd;
                cuAri_ii[Type.Float][Type.Float        ] = cuArithmic_internal_ftf;
                cuAri_ii[Type.Float][Type.Int64        ] = cuArithmic_internal_fti64;
                cuAri_ii[Type.Float][Type.Uint64       ] = cuArithmic_internal_ftu64;
                cuAri_ii[Type.Float][Type.Int32        ] = cuArithmic_internal_fti32;
                cuAri_ii[Type.Float][Type.Uint32       ] = cuArithmic_internal_ftu32;
                
                cuAri_ii[Type.Int64][Type.ComplexDouble] = cuArithmic_internal_i64tcd;
                cuAri_ii[Type.Int64][Type.ComplexFloat ] = cuArithmic_internal_i64tcf;
                cuAri_ii[Type.Int64][Type.Double       ] = cuArithmic_internal_i64td;
                cuAri_ii[Type.Int64][Type.Float        ] = cuArithmic_internal_i64tf;
                cuAri_ii[Type.Int64][Type.Int64        ] = cuArithmic_internal_i64ti64;
                cuAri_ii[Type.Int64][Type.Uint64       ] = cuArithmic_internal_i64tu64;
                cuAri_ii[Type.Int64][Type.Int32        ] = cuArithmic_internal_i64ti32;
                cuAri_ii[Type.Int64][Type.Uint32       ] = cuArithmic_internal_i64tu32;
                
                cuAri_ii[Type.Uint64][Type.ComplexDouble] = cuArithmic_internal_u64tcd;
                cuAri_ii[Type.Uint64][Type.ComplexFloat ] = cuArithmic_internal_u64tcf;
                cuAri_ii[Type.Uint64][Type.Double       ] = cuArithmic_internal_u64td;
                cuAri_ii[Type.Uint64][Type.Float        ] = cuArithmic_internal_u64tf;
                cuAri_ii[Type.Uint64][Type.Int64        ] = cuArithmic_internal_u64ti64;
                cuAri_ii[Type.Uint64][Type.Uint64       ] = cuArithmic_internal_u64tu64;
                cuAri_ii[Type.Uint64][Type.Int32        ] = cuArithmic_internal_u64ti32;
                cuAri_ii[Type.Uint64][Type.Uint32       ] = cuArithmic_internal_u64tu32;
                
                cuAri_ii[Type.Int32][Type.ComplexDouble] = cuArithmic_internal_i32tcd;
                cuAri_ii[Type.Int32][Type.ComplexFloat ] = cuArithmic_internal_i32tcf;
                cuAri_ii[Type.Int32][Type.Double       ] = cuArithmic_internal_i32td;
                cuAri_ii[Type.Int32][Type.Float        ] = cuArithmic_internal_i32tf;
                cuAri_ii[Type.Int32][Type.Int64        ] = cuArithmic_internal_i32ti64;
                cuAri_ii[Type.Int32][Type.Uint64       ] = cuArithmic_internal_i32tu64;
                cuAri_ii[Type.Int32][Type.Int32        ] = cuArithmic_internal_i32ti32;
                cuAri_ii[Type.Int32][Type.Uint32       ] = cuArithmic_internal_i32tu32;
                
                cuAri_ii[Type.Uint32][Type.ComplexDouble] = cuArithmic_internal_u32tcd;
                cuAri_ii[Type.Uint32][Type.ComplexFloat ] = cuArithmic_internal_u32tcf;
                cuAri_ii[Type.Uint32][Type.Double       ] = cuArithmic_internal_u32td;
                cuAri_ii[Type.Uint32][Type.Float        ] = cuArithmic_internal_u32tf;
                cuAri_ii[Type.Uint32][Type.Int64        ] = cuArithmic_internal_u32ti64;
                cuAri_ii[Type.Uint32][Type.Uint64       ] = cuArithmic_internal_u32tu64;
                cuAri_ii[Type.Uint32][Type.Int32        ] = cuArithmic_internal_u32ti32;
                cuAri_ii[Type.Uint32][Type.Uint32       ] = cuArithmic_internal_u32tu32;
            
                // Svd
                cuSvd_ii = vector<Svdfunc_oii>(5);

                cuSvd_ii[Type.ComplexDouble] = cuSvd_internal_cd;
                cuSvd_ii[Type.ComplexFloat ] = cuSvd_internal_cf;
                cuSvd_ii[Type.Double       ] = cuSvd_internal_d;
                cuSvd_ii[Type.Float        ] = cuSvd_internal_f;

                //=====================
                cuEigh_ii = vector<Eighfunc_oii>(5);

                cuEigh_ii[Type.ComplexDouble] = cuEigh_internal_cd;
                cuEigh_ii[Type.ComplexFloat ] = cuEigh_internal_cf;
                cuEigh_ii[Type.Double       ] = cuEigh_internal_d;
                cuEigh_ii[Type.Float        ] = cuEigh_internal_f;

                //=====================
                cuExp_ii = vector<Expfunc_oii>(5);

                cuExp_ii[Type.ComplexDouble] = cuExp_internal_cd;
                cuExp_ii[Type.ComplexFloat ] = cuExp_internal_cf;
                cuExp_ii[Type.Double       ] = cuExp_internal_d;
                cuExp_ii[Type.Float        ] = cuExp_internal_f;

                //=====================
                cuDiag_ii = vector<Expfunc_oii>(N_Type);

                cuDiag_ii[Type.ComplexDouble] = cuDiag_internal_cd;
                cuDiag_ii[Type.ComplexFloat ] = cuDiag_internal_cf;
                cuDiag_ii[Type.Double       ] = cuDiag_internal_d;
                cuDiag_ii[Type.Float        ] = cuDiag_internal_f;
                cuDiag_ii[Type.Int64        ] = cuDiag_internal_i64;
                cuDiag_ii[Type.Uint64       ] = cuDiag_internal_u64;
                cuDiag_ii[Type.Int32        ] = cuDiag_internal_u32;
                cuDiag_ii[Type.Uint32       ] = cuDiag_internal_u32;

                //=====================
                cuInv_inplace_ii = vector<Invinplacefunc_oii>(5);

                cuInv_inplace_ii[Type.ComplexDouble] = cuInv_inplace_internal_cd;
                cuInv_inplace_ii[Type.ComplexFloat ] = cuInv_inplace_internal_cf;
                cuInv_inplace_ii[Type.Double       ] = cuInv_inplace_internal_d;
                cuInv_inplace_ii[Type.Float        ] = cuInv_inplace_internal_f;

                //=====================
                cuConj_inplace_ii = vector<Conjinplacefunc_oii>(3);

                cuConj_inplace_ii[Type.ComplexDouble] = cuConj_inplace_internal_cd;
                cuConj_inplace_ii[Type.ComplexFloat ] = cuConj_inplace_internal_cf;


                //=====================
                cuMatmul_ii = vector<Matmulfunc_oii>(N_Type);
                cuMatmul_ii[Type.ComplexDouble] = cuMatmul_internal_cd ;
                cuMatmul_ii[Type.ComplexFloat ] = cuMatmul_internal_cf ;
                cuMatmul_ii[Type.Double       ] = cuMatmul_internal_d  ;
                cuMatmul_ii[Type.Float        ] = cuMatmul_internal_f  ;
                cuMatmul_ii[Type.Int64        ] = cuMatmul_internal_i64;
                cuMatmul_ii[Type.Uint64       ] = cuMatmul_internal_u64;
                cuMatmul_ii[Type.Int32        ] = cuMatmul_internal_i32;
                cuMatmul_ii[Type.Uint32       ] = cuMatmul_internal_u32;

                //================
                cuOuter_ii = vector<vector<Outerfunc_oii> >(N_Type,vector<Outerfunc_oii>(N_Type,NULL));

                cuOuter_ii[Type.ComplexDouble][Type.ComplexDouble] = cuOuter_internal_cdtcd;
                cuOuter_ii[Type.ComplexDouble][Type.ComplexFloat ] = cuOuter_internal_cdtcf;
                cuOuter_ii[Type.ComplexDouble][Type.Double       ] = cuOuter_internal_cdtd;
                cuOuter_ii[Type.ComplexDouble][Type.Float        ] = cuOuter_internal_cdtf;
                cuOuter_ii[Type.ComplexDouble][Type.Int64        ] = cuOuter_internal_cdti64;
                cuOuter_ii[Type.ComplexDouble][Type.Uint64       ] = cuOuter_internal_cdtu64;
                cuOuter_ii[Type.ComplexDouble][Type.Int32        ] = cuOuter_internal_cdti32;
                cuOuter_ii[Type.ComplexDouble][Type.Uint32       ] = cuOuter_internal_cdtu32;
                
                cuOuter_ii[Type.ComplexFloat][Type.ComplexDouble] = cuOuter_internal_cftcd;
                cuOuter_ii[Type.ComplexFloat][Type.ComplexFloat ] = cuOuter_internal_cftcf;
                cuOuter_ii[Type.ComplexFloat][Type.Double       ] = cuOuter_internal_cftd;
                cuOuter_ii[Type.ComplexFloat][Type.Float        ] = cuOuter_internal_cftf;
                cuOuter_ii[Type.ComplexFloat][Type.Int64        ] = cuOuter_internal_cfti64;
                cuOuter_ii[Type.ComplexFloat][Type.Uint64       ] = cuOuter_internal_cftu64;
                cuOuter_ii[Type.ComplexFloat][Type.Int32        ] = cuOuter_internal_cfti32;
                cuOuter_ii[Type.ComplexFloat][Type.Uint32       ] = cuOuter_internal_cftu32;
                
                cuOuter_ii[Type.Double][Type.ComplexDouble] = cuOuter_internal_dtcd;
                cuOuter_ii[Type.Double][Type.ComplexFloat ] = cuOuter_internal_dtcf;
                cuOuter_ii[Type.Double][Type.Double       ] = cuOuter_internal_dtd;
                cuOuter_ii[Type.Double][Type.Float        ] = cuOuter_internal_dtf;
                cuOuter_ii[Type.Double][Type.Int64        ] = cuOuter_internal_dti64;
                cuOuter_ii[Type.Double][Type.Uint64       ] = cuOuter_internal_dtu64;
                cuOuter_ii[Type.Double][Type.Int32        ] = cuOuter_internal_dti32;
                cuOuter_ii[Type.Double][Type.Uint32       ] = cuOuter_internal_dtu32;
                
                cuOuter_ii[Type.Float][Type.ComplexDouble] = cuOuter_internal_ftcd;
                cuOuter_ii[Type.Float][Type.ComplexFloat ] = cuOuter_internal_ftcf;
                cuOuter_ii[Type.Float][Type.Double       ] = cuOuter_internal_ftd;
                cuOuter_ii[Type.Float][Type.Float        ] = cuOuter_internal_ftf;
                cuOuter_ii[Type.Float][Type.Int64        ] = cuOuter_internal_fti64;
                cuOuter_ii[Type.Float][Type.Uint64       ] = cuOuter_internal_ftu64;
                cuOuter_ii[Type.Float][Type.Int32        ] = cuOuter_internal_fti32;
                cuOuter_ii[Type.Float][Type.Uint32       ] = cuOuter_internal_ftu32;
                
                cuOuter_ii[Type.Int64][Type.ComplexDouble] = cuOuter_internal_i64tcd;
                cuOuter_ii[Type.Int64][Type.ComplexFloat ] = cuOuter_internal_i64tcf;
                cuOuter_ii[Type.Int64][Type.Double       ] = cuOuter_internal_i64td;
                cuOuter_ii[Type.Int64][Type.Float        ] = cuOuter_internal_i64tf;
                cuOuter_ii[Type.Int64][Type.Int64        ] = cuOuter_internal_i64ti64;
                cuOuter_ii[Type.Int64][Type.Uint64       ] = cuOuter_internal_i64tu64;
                cuOuter_ii[Type.Int64][Type.Int32        ] = cuOuter_internal_i64ti32;
                cuOuter_ii[Type.Int64][Type.Uint32       ] = cuOuter_internal_i64tu32;
                
                cuOuter_ii[Type.Uint64][Type.ComplexDouble] = cuOuter_internal_u64tcd;
                cuOuter_ii[Type.Uint64][Type.ComplexFloat ] = cuOuter_internal_u64tcf;
                cuOuter_ii[Type.Uint64][Type.Double       ] = cuOuter_internal_u64td;
                cuOuter_ii[Type.Uint64][Type.Float        ] = cuOuter_internal_u64tf;
                cuOuter_ii[Type.Uint64][Type.Int64        ] = cuOuter_internal_u64ti64;
                cuOuter_ii[Type.Uint64][Type.Uint64       ] = cuOuter_internal_u64tu64;
                cuOuter_ii[Type.Uint64][Type.Int32        ] = cuOuter_internal_u64ti32;
                cuOuter_ii[Type.Uint64][Type.Uint32       ] = cuOuter_internal_u64tu32;
                
                cuOuter_ii[Type.Int32][Type.ComplexDouble] = cuOuter_internal_i32tcd;
                cuOuter_ii[Type.Int32][Type.ComplexFloat ] = cuOuter_internal_i32tcf;
                cuOuter_ii[Type.Int32][Type.Double       ] = cuOuter_internal_i32td;
                cuOuter_ii[Type.Int32][Type.Float        ] = cuOuter_internal_i32tf;
                cuOuter_ii[Type.Int32][Type.Int64        ] = cuOuter_internal_i32ti64;
                cuOuter_ii[Type.Int32][Type.Uint64       ] = cuOuter_internal_i32tu64;
                cuOuter_ii[Type.Int32][Type.Int32        ] = cuOuter_internal_i32ti32;
                cuOuter_ii[Type.Int32][Type.Uint32       ] = cuOuter_internal_i32tu32;
                
                cuOuter_ii[Type.Uint32][Type.ComplexDouble] = cuOuter_internal_u32tcd;
                cuOuter_ii[Type.Uint32][Type.ComplexFloat ] = cuOuter_internal_u32tcf;
                cuOuter_ii[Type.Uint32][Type.Double       ] = cuOuter_internal_u32td;
                cuOuter_ii[Type.Uint32][Type.Float        ] = cuOuter_internal_u32tf;
                cuOuter_ii[Type.Uint32][Type.Int64        ] = cuOuter_internal_u32ti64;
                cuOuter_ii[Type.Uint32][Type.Uint64       ] = cuOuter_internal_u32tu64;
                cuOuter_ii[Type.Uint32][Type.Int32        ] = cuOuter_internal_u32ti32;
                cuOuter_ii[Type.Uint32][Type.Uint32       ] = cuOuter_internal_u32tu32;

            #endif
        }

    }
}
