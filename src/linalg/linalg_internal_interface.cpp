#include "linalg/linalg_internal_interface.hpp"

using namespace std;

namespace tor10{
    namespace linalg_internal{


        linalg_internal_interface lii;

        linalg_internal_interface::linalg_internal_interface(){
            Ari_iicpu = vector<vector<Arithmicfunc_oii> >(N_Type,vector<Arithmicfunc_oii>(N_Type,NULL));

            Ari_iicpu[tor10type.ComplexDouble][tor10type.ComplexDouble] = Arithmic_internal_cpu_cdtcd;
            Ari_iicpu[tor10type.ComplexDouble][tor10type.ComplexFloat ] = Arithmic_internal_cpu_cdtcf;
            Ari_iicpu[tor10type.ComplexDouble][tor10type.Double       ] = Arithmic_internal_cpu_cdtd;
            Ari_iicpu[tor10type.ComplexDouble][tor10type.Float        ] = Arithmic_internal_cpu_cdtf;
            Ari_iicpu[tor10type.ComplexDouble][tor10type.Int64        ] = Arithmic_internal_cpu_cdti64;
            Ari_iicpu[tor10type.ComplexDouble][tor10type.Uint64       ] = Arithmic_internal_cpu_cdtu64;
            Ari_iicpu[tor10type.ComplexDouble][tor10type.Int32        ] = Arithmic_internal_cpu_cdti32;
            Ari_iicpu[tor10type.ComplexDouble][tor10type.Uint32       ] = Arithmic_internal_cpu_cdtu32;
            
            Ari_iicpu[tor10type.ComplexFloat][tor10type.ComplexDouble] = Arithmic_internal_cpu_cftcd;
            Ari_iicpu[tor10type.ComplexFloat][tor10type.ComplexFloat ] = Arithmic_internal_cpu_cftcf;
            Ari_iicpu[tor10type.ComplexFloat][tor10type.Double       ] = Arithmic_internal_cpu_cftd;
            Ari_iicpu[tor10type.ComplexFloat][tor10type.Float        ] = Arithmic_internal_cpu_cftf;
            Ari_iicpu[tor10type.ComplexFloat][tor10type.Int64        ] = Arithmic_internal_cpu_cfti64;
            Ari_iicpu[tor10type.ComplexFloat][tor10type.Uint64       ] = Arithmic_internal_cpu_cftu64;
            Ari_iicpu[tor10type.ComplexFloat][tor10type.Int32        ] = Arithmic_internal_cpu_cfti32;
            Ari_iicpu[tor10type.ComplexFloat][tor10type.Uint32       ] = Arithmic_internal_cpu_cftu32;
            
            Ari_iicpu[tor10type.Double][tor10type.ComplexDouble] = Arithmic_internal_cpu_dtcd;
            Ari_iicpu[tor10type.Double][tor10type.ComplexFloat ] = Arithmic_internal_cpu_dtcf;
            Ari_iicpu[tor10type.Double][tor10type.Double       ] = Arithmic_internal_cpu_dtd;
            Ari_iicpu[tor10type.Double][tor10type.Float        ] = Arithmic_internal_cpu_dtf;
            Ari_iicpu[tor10type.Double][tor10type.Int64        ] = Arithmic_internal_cpu_dti64;
            Ari_iicpu[tor10type.Double][tor10type.Uint64       ] = Arithmic_internal_cpu_dtu64;
            Ari_iicpu[tor10type.Double][tor10type.Int32        ] = Arithmic_internal_cpu_dti32;
            Ari_iicpu[tor10type.Double][tor10type.Uint32       ] = Arithmic_internal_cpu_dtu32;
            
            Ari_iicpu[tor10type.Float][tor10type.ComplexDouble] = Arithmic_internal_cpu_ftcd;
            Ari_iicpu[tor10type.Float][tor10type.ComplexFloat ] = Arithmic_internal_cpu_ftcf;
            Ari_iicpu[tor10type.Float][tor10type.Double       ] = Arithmic_internal_cpu_ftd;
            Ari_iicpu[tor10type.Float][tor10type.Float        ] = Arithmic_internal_cpu_ftf;
            Ari_iicpu[tor10type.Float][tor10type.Int64        ] = Arithmic_internal_cpu_fti64;
            Ari_iicpu[tor10type.Float][tor10type.Uint64       ] = Arithmic_internal_cpu_ftu64;
            Ari_iicpu[tor10type.Float][tor10type.Int32        ] = Arithmic_internal_cpu_fti32;
            Ari_iicpu[tor10type.Float][tor10type.Uint32       ] = Arithmic_internal_cpu_ftu32;
            
            Ari_iicpu[tor10type.Int64][tor10type.ComplexDouble] = Arithmic_internal_cpu_i64tcd;
            Ari_iicpu[tor10type.Int64][tor10type.ComplexFloat ] = Arithmic_internal_cpu_i64tcf;
            Ari_iicpu[tor10type.Int64][tor10type.Double       ] = Arithmic_internal_cpu_i64td;
            Ari_iicpu[tor10type.Int64][tor10type.Float        ] = Arithmic_internal_cpu_i64tf;
            Ari_iicpu[tor10type.Int64][tor10type.Int64        ] = Arithmic_internal_cpu_i64ti64;
            Ari_iicpu[tor10type.Int64][tor10type.Uint64       ] = Arithmic_internal_cpu_i64tu64;
            Ari_iicpu[tor10type.Int64][tor10type.Int32        ] = Arithmic_internal_cpu_i64ti32;
            Ari_iicpu[tor10type.Int64][tor10type.Uint32       ] = Arithmic_internal_cpu_i64tu32;
            
            Ari_iicpu[tor10type.Uint64][tor10type.ComplexDouble] = Arithmic_internal_cpu_u64tcd;
            Ari_iicpu[tor10type.Uint64][tor10type.ComplexFloat ] = Arithmic_internal_cpu_u64tcf;
            Ari_iicpu[tor10type.Uint64][tor10type.Double       ] = Arithmic_internal_cpu_u64td;
            Ari_iicpu[tor10type.Uint64][tor10type.Float        ] = Arithmic_internal_cpu_u64tf;
            Ari_iicpu[tor10type.Uint64][tor10type.Int64        ] = Arithmic_internal_cpu_u64ti64;
            Ari_iicpu[tor10type.Uint64][tor10type.Uint64       ] = Arithmic_internal_cpu_u64tu64;
            Ari_iicpu[tor10type.Uint64][tor10type.Int32        ] = Arithmic_internal_cpu_u64ti32;
            Ari_iicpu[tor10type.Uint64][tor10type.Uint32       ] = Arithmic_internal_cpu_u64tu32;
            
            Ari_iicpu[tor10type.Int32][tor10type.ComplexDouble] = Arithmic_internal_cpu_i32tcd;
            Ari_iicpu[tor10type.Int32][tor10type.ComplexFloat ] = Arithmic_internal_cpu_i32tcf;
            Ari_iicpu[tor10type.Int32][tor10type.Double       ] = Arithmic_internal_cpu_i32td;
            Ari_iicpu[tor10type.Int32][tor10type.Float        ] = Arithmic_internal_cpu_i32tf;
            Ari_iicpu[tor10type.Int32][tor10type.Int64        ] = Arithmic_internal_cpu_i32ti64;
            Ari_iicpu[tor10type.Int32][tor10type.Uint64       ] = Arithmic_internal_cpu_i32tu64;
            Ari_iicpu[tor10type.Int32][tor10type.Int32        ] = Arithmic_internal_cpu_i32ti32;
            Ari_iicpu[tor10type.Int32][tor10type.Uint32       ] = Arithmic_internal_cpu_i32tu32;
            
            Ari_iicpu[tor10type.Uint32][tor10type.ComplexDouble] = Arithmic_internal_cpu_u32tcd;
            Ari_iicpu[tor10type.Uint32][tor10type.ComplexFloat ] = Arithmic_internal_cpu_u32tcf;
            Ari_iicpu[tor10type.Uint32][tor10type.Double       ] = Arithmic_internal_cpu_u32td;
            Ari_iicpu[tor10type.Uint32][tor10type.Float        ] = Arithmic_internal_cpu_u32tf;
            Ari_iicpu[tor10type.Uint32][tor10type.Int64        ] = Arithmic_internal_cpu_u32ti64;
            Ari_iicpu[tor10type.Uint32][tor10type.Uint64       ] = Arithmic_internal_cpu_u32tu64;
            Ari_iicpu[tor10type.Uint32][tor10type.Int32        ] = Arithmic_internal_cpu_u32ti32;
            Ari_iicpu[tor10type.Uint32][tor10type.Uint32       ] = Arithmic_internal_cpu_u32tu32;

            #ifdef UNI_GPU
                Ari_iigpu = vector<vector<Arithmicfunc_oii> >(N_Type,vector<Arithmicfunc_oii>(N_Type));

                Ari_iigpu[tor10type.ComplexDouble][tor10type.ComplexDouble] = Arithmic_internal_gpu_cdtcd;
                Ari_iigpu[tor10type.ComplexDouble][tor10type.ComplexFloat ] = Arithmic_internal_gpu_cdtcf;
                Ari_iigpu[tor10type.ComplexDouble][tor10type.Double       ] = Arithmic_internal_gpu_cdtd;
                Ari_iigpu[tor10type.ComplexDouble][tor10type.Float        ] = Arithmic_internal_gpu_cdtf;
                Ari_iigpu[tor10type.ComplexDouble][tor10type.Int64        ] = Arithmic_internal_gpu_cdti64;
                Ari_iigpu[tor10type.ComplexDouble][tor10type.Uint64       ] = Arithmic_internal_gpu_cdtu64;
                Ari_iigpu[tor10type.ComplexDouble][tor10type.Int32        ] = Arithmic_internal_gpu_cdti32;
                Ari_iigpu[tor10type.ComplexDouble][tor10type.Uint32       ] = Arithmic_internal_gpu_cdtu32;
                
                Ari_iigpu[tor10type.ComplexFloat][tor10type.ComplexDouble] = Arithmic_internal_gpu_cftcd;
                Ari_iigpu[tor10type.ComplexFloat][tor10type.ComplexFloat ] = Arithmic_internal_gpu_cftcf;
                Ari_iigpu[tor10type.ComplexFloat][tor10type.Double       ] = Arithmic_internal_gpu_cftd;
                Ari_iigpu[tor10type.ComplexFloat][tor10type.Float        ] = Arithmic_internal_gpu_cftf;
                Ari_iigpu[tor10type.ComplexFloat][tor10type.Int64        ] = Arithmic_internal_gpu_cfti64;
                Ari_iigpu[tor10type.ComplexFloat][tor10type.Uint64       ] = Arithmic_internal_gpu_cftu64;
                Ari_iigpu[tor10type.ComplexFloat][tor10type.Int32        ] = Arithmic_internal_gpu_cfti32;
                Ari_iigpu[tor10type.ComplexFloat][tor10type.Uint32       ] = Arithmic_internal_gpu_cftu32;
                
                Ari_iigpu[tor10type.Double][tor10type.ComplexDouble] = Arithmic_internal_gpu_dtcd;
                Ari_iigpu[tor10type.Double][tor10type.ComplexFloat ] = Arithmic_internal_gpu_dtcf;
                Ari_iigpu[tor10type.Double][tor10type.Double       ] = Arithmic_internal_gpu_dtd;
                Ari_iigpu[tor10type.Double][tor10type.Float        ] = Arithmic_internal_gpu_dtf;
                Ari_iigpu[tor10type.Double][tor10type.Int64        ] = Arithmic_internal_gpu_dti64;
                Ari_iigpu[tor10type.Double][tor10type.Uint64       ] = Arithmic_internal_gpu_dtu64;
                Ari_iigpu[tor10type.Double][tor10type.Int32        ] = Arithmic_internal_gpu_dti32;
                Ari_iigpu[tor10type.Double][tor10type.Uint32       ] = Arithmic_internal_gpu_dtu32;
                
                Ari_iigpu[tor10type.Float][tor10type.ComplexDouble] = Arithmic_internal_gpu_ftcd;
                Ari_iigpu[tor10type.Float][tor10type.ComplexFloat ] = Arithmic_internal_gpu_ftcf;
                Ari_iigpu[tor10type.Float][tor10type.Double       ] = Arithmic_internal_gpu_ftd;
                Ari_iigpu[tor10type.Float][tor10type.Float        ] = Arithmic_internal_gpu_ftf;
                Ari_iigpu[tor10type.Float][tor10type.Int64        ] = Arithmic_internal_gpu_fti64;
                Ari_iigpu[tor10type.Float][tor10type.Uint64       ] = Arithmic_internal_gpu_ftu64;
                Ari_iigpu[tor10type.Float][tor10type.Int32        ] = Arithmic_internal_gpu_fti32;
                Ari_iigpu[tor10type.Float][tor10type.Uint32       ] = Arithmic_internal_gpu_ftu32;
                
                Ari_iigpu[tor10type.Int64][tor10type.ComplexDouble] = Arithmic_internal_gpu_i64tcd;
                Ari_iigpu[tor10type.Int64][tor10type.ComplexFloat ] = Arithmic_internal_gpu_i64tcf;
                Ari_iigpu[tor10type.Int64][tor10type.Double       ] = Arithmic_internal_gpu_i64td;
                Ari_iigpu[tor10type.Int64][tor10type.Float        ] = Arithmic_internal_gpu_i64tf;
                Ari_iigpu[tor10type.Int64][tor10type.Int64        ] = Arithmic_internal_gpu_i64ti64;
                Ari_iigpu[tor10type.Int64][tor10type.Uint64       ] = Arithmic_internal_gpu_i64tu64;
                Ari_iigpu[tor10type.Int64][tor10type.Int32        ] = Arithmic_internal_gpu_i64ti32;
                Ari_iigpu[tor10type.Int64][tor10type.Uint32       ] = Arithmic_internal_gpu_i64tu32;
                
                Ari_iigpu[tor10type.Uint64][tor10type.ComplexDouble] = Arithmic_internal_gpu_u64tcd;
                Ari_iigpu[tor10type.Uint64][tor10type.ComplexFloat ] = Arithmic_internal_gpu_u64tcf;
                Ari_iigpu[tor10type.Uint64][tor10type.Double       ] = Arithmic_internal_gpu_u64td;
                Ari_iigpu[tor10type.Uint64][tor10type.Float        ] = Arithmic_internal_gpu_u64tf;
                Ari_iigpu[tor10type.Uint64][tor10type.Int64        ] = Arithmic_internal_gpu_u64ti64;
                Ari_iigpu[tor10type.Uint64][tor10type.Uint64       ] = Arithmic_internal_gpu_u64tu64;
                Ari_iigpu[tor10type.Uint64][tor10type.Int32        ] = Arithmic_internal_gpu_u64ti32;
                Ari_iigpu[tor10type.Uint64][tor10type.Uint32       ] = Arithmic_internal_gpu_u64tu32;
                
                Ari_iigpu[tor10type.Int32][tor10type.ComplexDouble] = Arithmic_internal_gpu_i32tcd;
                Ari_iigpu[tor10type.Int32][tor10type.ComplexFloat ] = Arithmic_internal_gpu_i32tcf;
                Ari_iigpu[tor10type.Int32][tor10type.Double       ] = Arithmic_internal_gpu_i32td;
                Ari_iigpu[tor10type.Int32][tor10type.Float        ] = Arithmic_internal_gpu_i32tf;
                Ari_iigpu[tor10type.Int32][tor10type.Int64        ] = Arithmic_internal_gpu_i32ti64;
                Ari_iigpu[tor10type.Int32][tor10type.Uint64       ] = Arithmic_internal_gpu_i32tu64;
                Ari_iigpu[tor10type.Int32][tor10type.Int32        ] = Arithmic_internal_gpu_i32ti32;
                Ari_iigpu[tor10type.Int32][tor10type.Uint32       ] = Arithmic_internal_gpu_i32tu32;
                
                Ari_iigpu[tor10type.Uint32][tor10type.ComplexDouble] = Arithmic_internal_gpu_u32tcd;
                Ari_iigpu[tor10type.Uint32][tor10type.ComplexFloat ] = Arithmic_internal_gpu_u32tcf;
                Ari_iigpu[tor10type.Uint32][tor10type.Double       ] = Arithmic_internal_gpu_u32td;
                Ari_iigpu[tor10type.Uint32][tor10type.Float        ] = Arithmic_internal_gpu_u32tf;
                Ari_iigpu[tor10type.Uint32][tor10type.Int64        ] = Arithmic_internal_gpu_u32ti64;
                Ari_iigpu[tor10type.Uint32][tor10type.Uint64       ] = Arithmic_internal_gpu_u32tu64;
                Ari_iigpu[tor10type.Uint32][tor10type.Int32        ] = Arithmic_internal_gpu_u32ti32;
                Ari_iigpu[tor10type.Uint32][tor10type.Uint32       ] = Arithmic_internal_gpu_u32tu32;
            #endif
        }

    }
}
