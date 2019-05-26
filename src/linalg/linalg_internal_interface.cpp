#include "linalg/linalg_internal_interface.hpp"

using namespace std;

namespace cytnx{
    namespace linalg_internal{


        linalg_internal_interface lii;

        linalg_internal_interface::linalg_internal_interface(){
            Ari_ii = vector<vector<Arithmicfunc_oii> >(N_Type,vector<Arithmicfunc_oii>(N_Type,NULL));

            Ari_ii[cytnxtype.ComplexDouble][cytnxtype.ComplexDouble] = Arithmic_internal_cdtcd;
            Ari_ii[cytnxtype.ComplexDouble][cytnxtype.ComplexFloat ] = Arithmic_internal_cdtcf;
            Ari_ii[cytnxtype.ComplexDouble][cytnxtype.Double       ] = Arithmic_internal_cdtd;
            Ari_ii[cytnxtype.ComplexDouble][cytnxtype.Float        ] = Arithmic_internal_cdtf;
            Ari_ii[cytnxtype.ComplexDouble][cytnxtype.Int64        ] = Arithmic_internal_cdti64;
            Ari_ii[cytnxtype.ComplexDouble][cytnxtype.Uint64       ] = Arithmic_internal_cdtu64;
            Ari_ii[cytnxtype.ComplexDouble][cytnxtype.Int32        ] = Arithmic_internal_cdti32;
            Ari_ii[cytnxtype.ComplexDouble][cytnxtype.Uint32       ] = Arithmic_internal_cdtu32;
            
            Ari_ii[cytnxtype.ComplexFloat][cytnxtype.ComplexDouble] = Arithmic_internal_cftcd;
            Ari_ii[cytnxtype.ComplexFloat][cytnxtype.ComplexFloat ] = Arithmic_internal_cftcf;
            Ari_ii[cytnxtype.ComplexFloat][cytnxtype.Double       ] = Arithmic_internal_cftd;
            Ari_ii[cytnxtype.ComplexFloat][cytnxtype.Float        ] = Arithmic_internal_cftf;
            Ari_ii[cytnxtype.ComplexFloat][cytnxtype.Int64        ] = Arithmic_internal_cfti64;
            Ari_ii[cytnxtype.ComplexFloat][cytnxtype.Uint64       ] = Arithmic_internal_cftu64;
            Ari_ii[cytnxtype.ComplexFloat][cytnxtype.Int32        ] = Arithmic_internal_cfti32;
            Ari_ii[cytnxtype.ComplexFloat][cytnxtype.Uint32       ] = Arithmic_internal_cftu32;
            
            Ari_ii[cytnxtype.Double][cytnxtype.ComplexDouble] = Arithmic_internal_dtcd;
            Ari_ii[cytnxtype.Double][cytnxtype.ComplexFloat ] = Arithmic_internal_dtcf;
            Ari_ii[cytnxtype.Double][cytnxtype.Double       ] = Arithmic_internal_dtd;
            Ari_ii[cytnxtype.Double][cytnxtype.Float        ] = Arithmic_internal_dtf;
            Ari_ii[cytnxtype.Double][cytnxtype.Int64        ] = Arithmic_internal_dti64;
            Ari_ii[cytnxtype.Double][cytnxtype.Uint64       ] = Arithmic_internal_dtu64;
            Ari_ii[cytnxtype.Double][cytnxtype.Int32        ] = Arithmic_internal_dti32;
            Ari_ii[cytnxtype.Double][cytnxtype.Uint32       ] = Arithmic_internal_dtu32;
            
            Ari_ii[cytnxtype.Float][cytnxtype.ComplexDouble] = Arithmic_internal_ftcd;
            Ari_ii[cytnxtype.Float][cytnxtype.ComplexFloat ] = Arithmic_internal_ftcf;
            Ari_ii[cytnxtype.Float][cytnxtype.Double       ] = Arithmic_internal_ftd;
            Ari_ii[cytnxtype.Float][cytnxtype.Float        ] = Arithmic_internal_ftf;
            Ari_ii[cytnxtype.Float][cytnxtype.Int64        ] = Arithmic_internal_fti64;
            Ari_ii[cytnxtype.Float][cytnxtype.Uint64       ] = Arithmic_internal_ftu64;
            Ari_ii[cytnxtype.Float][cytnxtype.Int32        ] = Arithmic_internal_fti32;
            Ari_ii[cytnxtype.Float][cytnxtype.Uint32       ] = Arithmic_internal_ftu32;
            
            Ari_ii[cytnxtype.Int64][cytnxtype.ComplexDouble] = Arithmic_internal_i64tcd;
            Ari_ii[cytnxtype.Int64][cytnxtype.ComplexFloat ] = Arithmic_internal_i64tcf;
            Ari_ii[cytnxtype.Int64][cytnxtype.Double       ] = Arithmic_internal_i64td;
            Ari_ii[cytnxtype.Int64][cytnxtype.Float        ] = Arithmic_internal_i64tf;
            Ari_ii[cytnxtype.Int64][cytnxtype.Int64        ] = Arithmic_internal_i64ti64;
            Ari_ii[cytnxtype.Int64][cytnxtype.Uint64       ] = Arithmic_internal_i64tu64;
            Ari_ii[cytnxtype.Int64][cytnxtype.Int32        ] = Arithmic_internal_i64ti32;
            Ari_ii[cytnxtype.Int64][cytnxtype.Uint32       ] = Arithmic_internal_i64tu32;
            
            Ari_ii[cytnxtype.Uint64][cytnxtype.ComplexDouble] = Arithmic_internal_u64tcd;
            Ari_ii[cytnxtype.Uint64][cytnxtype.ComplexFloat ] = Arithmic_internal_u64tcf;
            Ari_ii[cytnxtype.Uint64][cytnxtype.Double       ] = Arithmic_internal_u64td;
            Ari_ii[cytnxtype.Uint64][cytnxtype.Float        ] = Arithmic_internal_u64tf;
            Ari_ii[cytnxtype.Uint64][cytnxtype.Int64        ] = Arithmic_internal_u64ti64;
            Ari_ii[cytnxtype.Uint64][cytnxtype.Uint64       ] = Arithmic_internal_u64tu64;
            Ari_ii[cytnxtype.Uint64][cytnxtype.Int32        ] = Arithmic_internal_u64ti32;
            Ari_ii[cytnxtype.Uint64][cytnxtype.Uint32       ] = Arithmic_internal_u64tu32;
            
            Ari_ii[cytnxtype.Int32][cytnxtype.ComplexDouble] = Arithmic_internal_i32tcd;
            Ari_ii[cytnxtype.Int32][cytnxtype.ComplexFloat ] = Arithmic_internal_i32tcf;
            Ari_ii[cytnxtype.Int32][cytnxtype.Double       ] = Arithmic_internal_i32td;
            Ari_ii[cytnxtype.Int32][cytnxtype.Float        ] = Arithmic_internal_i32tf;
            Ari_ii[cytnxtype.Int32][cytnxtype.Int64        ] = Arithmic_internal_i32ti64;
            Ari_ii[cytnxtype.Int32][cytnxtype.Uint64       ] = Arithmic_internal_i32tu64;
            Ari_ii[cytnxtype.Int32][cytnxtype.Int32        ] = Arithmic_internal_i32ti32;
            Ari_ii[cytnxtype.Int32][cytnxtype.Uint32       ] = Arithmic_internal_i32tu32;
            
            Ari_ii[cytnxtype.Uint32][cytnxtype.ComplexDouble] = Arithmic_internal_u32tcd;
            Ari_ii[cytnxtype.Uint32][cytnxtype.ComplexFloat ] = Arithmic_internal_u32tcf;
            Ari_ii[cytnxtype.Uint32][cytnxtype.Double       ] = Arithmic_internal_u32td;
            Ari_ii[cytnxtype.Uint32][cytnxtype.Float        ] = Arithmic_internal_u32tf;
            Ari_ii[cytnxtype.Uint32][cytnxtype.Int64        ] = Arithmic_internal_u32ti64;
            Ari_ii[cytnxtype.Uint32][cytnxtype.Uint64       ] = Arithmic_internal_u32tu64;
            Ari_ii[cytnxtype.Uint32][cytnxtype.Int32        ] = Arithmic_internal_u32ti32;
            Ari_ii[cytnxtype.Uint32][cytnxtype.Uint32       ] = Arithmic_internal_u32tu32;

            //=====================
            Svd_ii = vector<Svdfunc_oii>(5);

            Svd_ii[cytnxtype.ComplexDouble] = Svd_internal_cd;
            Svd_ii[cytnxtype.ComplexFloat ] = Svd_internal_cf;
            Svd_ii[cytnxtype.Double       ] = Svd_internal_d;
            Svd_ii[cytnxtype.Float        ] = Svd_internal_f;

            //=====================
            Inv_inplace_ii = vector<Invinplacefunc_oii>(5);

            Inv_inplace_ii[cytnxtype.ComplexDouble] = Inv_inplace_internal_cd;
            Inv_inplace_ii[cytnxtype.ComplexFloat ] = Inv_inplace_internal_cf;
            Inv_inplace_ii[cytnxtype.Double       ] = Inv_inplace_internal_d;
            Inv_inplace_ii[cytnxtype.Float        ] = Inv_inplace_internal_f;

            #ifdef UNI_GPU
                cuAri_ii = vector<vector<Arithmicfunc_oii> >(N_Type,vector<Arithmicfunc_oii>(N_Type));

                cuAri_ii[cytnxtype.ComplexDouble][cytnxtype.ComplexDouble] = cuArithmic_internal_cdtcd;
                cuAri_ii[cytnxtype.ComplexDouble][cytnxtype.ComplexFloat ] = cuArithmic_internal_cdtcf;
                cuAri_ii[cytnxtype.ComplexDouble][cytnxtype.Double       ] = cuArithmic_internal_cdtd;
                cuAri_ii[cytnxtype.ComplexDouble][cytnxtype.Float        ] = cuArithmic_internal_cdtf;
                cuAri_ii[cytnxtype.ComplexDouble][cytnxtype.Int64        ] = cuArithmic_internal_cdti64;
                cuAri_ii[cytnxtype.ComplexDouble][cytnxtype.Uint64       ] = cuArithmic_internal_cdtu64;
                cuAri_ii[cytnxtype.ComplexDouble][cytnxtype.Int32        ] = cuArithmic_internal_cdti32;
                cuAri_ii[cytnxtype.ComplexDouble][cytnxtype.Uint32       ] = cuArithmic_internal_cdtu32;
                
                cuAri_ii[cytnxtype.ComplexFloat][cytnxtype.ComplexDouble] = cuArithmic_internal_cftcd;
                cuAri_ii[cytnxtype.ComplexFloat][cytnxtype.ComplexFloat ] = cuArithmic_internal_cftcf;
                cuAri_ii[cytnxtype.ComplexFloat][cytnxtype.Double       ] = cuArithmic_internal_cftd;
                cuAri_ii[cytnxtype.ComplexFloat][cytnxtype.Float        ] = cuArithmic_internal_cftf;
                cuAri_ii[cytnxtype.ComplexFloat][cytnxtype.Int64        ] = cuArithmic_internal_cfti64;
                cuAri_ii[cytnxtype.ComplexFloat][cytnxtype.Uint64       ] = cuArithmic_internal_cftu64;
                cuAri_ii[cytnxtype.ComplexFloat][cytnxtype.Int32        ] = cuArithmic_internal_cfti32;
                cuAri_ii[cytnxtype.ComplexFloat][cytnxtype.Uint32       ] = cuArithmic_internal_cftu32;
                
                cuAri_ii[cytnxtype.Double][cytnxtype.ComplexDouble] = cuArithmic_internal_dtcd;
                cuAri_ii[cytnxtype.Double][cytnxtype.ComplexFloat ] = cuArithmic_internal_dtcf;
                cuAri_ii[cytnxtype.Double][cytnxtype.Double       ] = cuArithmic_internal_dtd;
                cuAri_ii[cytnxtype.Double][cytnxtype.Float        ] = cuArithmic_internal_dtf;
                cuAri_ii[cytnxtype.Double][cytnxtype.Int64        ] = cuArithmic_internal_dti64;
                cuAri_ii[cytnxtype.Double][cytnxtype.Uint64       ] = cuArithmic_internal_dtu64;
                cuAri_ii[cytnxtype.Double][cytnxtype.Int32        ] = cuArithmic_internal_dti32;
                cuAri_ii[cytnxtype.Double][cytnxtype.Uint32       ] = cuArithmic_internal_dtu32;
                
                cuAri_ii[cytnxtype.Float][cytnxtype.ComplexDouble] = cuArithmic_internal_ftcd;
                cuAri_ii[cytnxtype.Float][cytnxtype.ComplexFloat ] = cuArithmic_internal_ftcf;
                cuAri_ii[cytnxtype.Float][cytnxtype.Double       ] = cuArithmic_internal_ftd;
                cuAri_ii[cytnxtype.Float][cytnxtype.Float        ] = cuArithmic_internal_ftf;
                cuAri_ii[cytnxtype.Float][cytnxtype.Int64        ] = cuArithmic_internal_fti64;
                cuAri_ii[cytnxtype.Float][cytnxtype.Uint64       ] = cuArithmic_internal_ftu64;
                cuAri_ii[cytnxtype.Float][cytnxtype.Int32        ] = cuArithmic_internal_fti32;
                cuAri_ii[cytnxtype.Float][cytnxtype.Uint32       ] = cuArithmic_internal_ftu32;
                
                cuAri_ii[cytnxtype.Int64][cytnxtype.ComplexDouble] = cuArithmic_internal_i64tcd;
                cuAri_ii[cytnxtype.Int64][cytnxtype.ComplexFloat ] = cuArithmic_internal_i64tcf;
                cuAri_ii[cytnxtype.Int64][cytnxtype.Double       ] = cuArithmic_internal_i64td;
                cuAri_ii[cytnxtype.Int64][cytnxtype.Float        ] = cuArithmic_internal_i64tf;
                cuAri_ii[cytnxtype.Int64][cytnxtype.Int64        ] = cuArithmic_internal_i64ti64;
                cuAri_ii[cytnxtype.Int64][cytnxtype.Uint64       ] = cuArithmic_internal_i64tu64;
                cuAri_ii[cytnxtype.Int64][cytnxtype.Int32        ] = cuArithmic_internal_i64ti32;
                cuAri_ii[cytnxtype.Int64][cytnxtype.Uint32       ] = cuArithmic_internal_i64tu32;
                
                cuAri_ii[cytnxtype.Uint64][cytnxtype.ComplexDouble] = cuArithmic_internal_u64tcd;
                cuAri_ii[cytnxtype.Uint64][cytnxtype.ComplexFloat ] = cuArithmic_internal_u64tcf;
                cuAri_ii[cytnxtype.Uint64][cytnxtype.Double       ] = cuArithmic_internal_u64td;
                cuAri_ii[cytnxtype.Uint64][cytnxtype.Float        ] = cuArithmic_internal_u64tf;
                cuAri_ii[cytnxtype.Uint64][cytnxtype.Int64        ] = cuArithmic_internal_u64ti64;
                cuAri_ii[cytnxtype.Uint64][cytnxtype.Uint64       ] = cuArithmic_internal_u64tu64;
                cuAri_ii[cytnxtype.Uint64][cytnxtype.Int32        ] = cuArithmic_internal_u64ti32;
                cuAri_ii[cytnxtype.Uint64][cytnxtype.Uint32       ] = cuArithmic_internal_u64tu32;
                
                cuAri_ii[cytnxtype.Int32][cytnxtype.ComplexDouble] = cuArithmic_internal_i32tcd;
                cuAri_ii[cytnxtype.Int32][cytnxtype.ComplexFloat ] = cuArithmic_internal_i32tcf;
                cuAri_ii[cytnxtype.Int32][cytnxtype.Double       ] = cuArithmic_internal_i32td;
                cuAri_ii[cytnxtype.Int32][cytnxtype.Float        ] = cuArithmic_internal_i32tf;
                cuAri_ii[cytnxtype.Int32][cytnxtype.Int64        ] = cuArithmic_internal_i32ti64;
                cuAri_ii[cytnxtype.Int32][cytnxtype.Uint64       ] = cuArithmic_internal_i32tu64;
                cuAri_ii[cytnxtype.Int32][cytnxtype.Int32        ] = cuArithmic_internal_i32ti32;
                cuAri_ii[cytnxtype.Int32][cytnxtype.Uint32       ] = cuArithmic_internal_i32tu32;
                
                cuAri_ii[cytnxtype.Uint32][cytnxtype.ComplexDouble] = cuArithmic_internal_u32tcd;
                cuAri_ii[cytnxtype.Uint32][cytnxtype.ComplexFloat ] = cuArithmic_internal_u32tcf;
                cuAri_ii[cytnxtype.Uint32][cytnxtype.Double       ] = cuArithmic_internal_u32td;
                cuAri_ii[cytnxtype.Uint32][cytnxtype.Float        ] = cuArithmic_internal_u32tf;
                cuAri_ii[cytnxtype.Uint32][cytnxtype.Int64        ] = cuArithmic_internal_u32ti64;
                cuAri_ii[cytnxtype.Uint32][cytnxtype.Uint64       ] = cuArithmic_internal_u32tu64;
                cuAri_ii[cytnxtype.Uint32][cytnxtype.Int32        ] = cuArithmic_internal_u32ti32;
                cuAri_ii[cytnxtype.Uint32][cytnxtype.Uint32       ] = cuArithmic_internal_u32tu32;
            
                // Svd
                cuSvd_ii = vector<Svdfunc_oii>(5);

                cuSvd_ii[cytnxtype.ComplexDouble] = cuSvd_internal_cd;
                cuSvd_ii[cytnxtype.ComplexFloat ] = cuSvd_internal_cf;
                cuSvd_ii[cytnxtype.Double       ] = cuSvd_internal_d;
                cuSvd_ii[cytnxtype.Float        ] = cuSvd_internal_f;

                cuInv_inplace_ii = vector<Invinplacefunc_oii>(5);

                cuInv_inplace_ii[cytnxtype.ComplexDouble] = cuInv_inplace_internal_cd;
                cuInv_inplace_ii[cytnxtype.ComplexFloat ] = cuInv_inplace_internal_cf;
                cuInv_inplace_ii[cytnxtype.Double       ] = cuInv_inplace_internal_d;
                cuInv_inplace_ii[cytnxtype.Float        ] = cuInv_inplace_internal_f;


            #endif
        }

    }
}
