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
            Inv_inplace_ii = vector<Invinplacefunc_oii>(5);

            Inv_inplace_ii[Type.ComplexDouble] = Inv_inplace_internal_cd;
            Inv_inplace_ii[Type.ComplexFloat ] = Inv_inplace_internal_cf;
            Inv_inplace_ii[Type.Double       ] = Inv_inplace_internal_d;
            Inv_inplace_ii[Type.Float        ] = Inv_inplace_internal_f;


            //=====================
            Conj_inplace_ii = vector<Conjinplacefunc_oii>(3);

            Conj_inplace_ii[Type.ComplexDouble] = Conj_inplace_internal_cd;
            Conj_inplace_ii[Type.ComplexFloat ] = Conj_inplace_internal_cf;

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

                cuInv_inplace_ii = vector<Invinplacefunc_oii>(5);

                cuInv_inplace_ii[Type.ComplexDouble] = cuInv_inplace_internal_cd;
                cuInv_inplace_ii[Type.ComplexFloat ] = cuInv_inplace_internal_cf;
                cuInv_inplace_ii[Type.Double       ] = cuInv_inplace_internal_d;
                cuInv_inplace_ii[Type.Float        ] = cuInv_inplace_internal_f;

                cuConj_inplace_ii = vector<Conjinplacefunc_oii>(3);

                cuConj_inplace_ii[Type.ComplexDouble] = cuConj_inplace_internal_cd;
                cuConj_inplace_ii[Type.ComplexFloat ] = cuConj_inplace_internal_cf;


            #endif
        }

    }
}
