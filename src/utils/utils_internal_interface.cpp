#include "utils_internal_interface.hpp"
#include <vector>
using namespace std;
namespace cytnx {
  namespace utils_internal {

    // this is an internal function for compare.
    //-------------
    bool _fx_compare_vec_inc(const std::vector<cytnx_int64> &v1,
                             const std::vector<cytnx_int64> &v2) {
      std::pair<std::vector<cytnx_int64>, std::vector<cytnx_int64>> p{v1, v2};

      return p.first < p.second;
    }
    bool _fx_compare_vec_dec(const std::vector<cytnx_int64> &v1,
                             const std::vector<cytnx_int64> &v2) {
      std::pair<std::vector<cytnx_int64>, std::vector<cytnx_int64>> p{v1, v2};

      return p.first > p.second;
    }

    utils_internal_interface::utils_internal_interface() {
      blocks_mvelems_ii.resize(N_Type, NULL);
      // blocks_mvelems_ii[Type.ComplexDouble] = blocks_mvelems_cd;
      // blocks_mvelems_ii[Type.ComplexFloat ] = blocks_mvelems_cf;
      blocks_mvelems_ii[Type.Double] = blocks_mvelems_d;
      // blocks_mvelems_ii[Type.Float        ] = blocks_mvelems_f ;
      // blocks_mvelems_ii[Type.Uint64       ] = blocks_mvelems_u64;
      // blocks_mvelems_ii[Type.Int64        ] = blocks_mvelems_i64;
      // blocks_mvelems_ii[Type.Uint32       ] = blocks_mvelems_u32;
      // blocks_mvelems_ii[Type.Int32        ] = blocks_mvelems_i32;
      // blocks_mvelems_ii[Type.Uint16       ] = blocks_mvelems_u16;
      // blocks_mvelems_ii[Type.Int16        ] = blocks_mvelems_i16;
      // blocks_mvelems_ii[Type.Bool         ] = blocks_mvelems_b;
      ElemCast = vector<vector<ElemCast_io>>(N_Type, vector<ElemCast_io>(N_Type, NULL));
      ElemCast[Type.ComplexDouble][Type.ComplexDouble] = Cast_cpu_cdtcd;
      ElemCast[Type.ComplexDouble][Type.ComplexFloat] = Cast_cpu_cdtcf;
      // ElemCast[Type.ComplexDouble][Type.Double       ] = Cast_cpu_invalid;
      // ElemCast[Type.ComplexDouble][Type.Float        ] = Cast_cpu_invalid;
      // ElemCast[Type.ComplexDouble][Type.Int64        ] = Cast_cpu_invalid;
      // ElemCast[Type.ComplexDouble][Type.Uint64       ] = Cast_cpu_invalid;
      // ElemCast[Type.ComplexDouble][Type.Int32        ] = Cast_cpu_invalid;
      // ElemCast[Type.ComplexDouble][Type.Uint32       ] = Cast_cpu_invalid;
      // ElemCast[Type.ComplexDouble][Type.Int16        ] = Cast_cpu_invalid;
      // ElemCast[Type.ComplexDouble][Type.Uint16       ] = Cast_cpu_invalid;
      // ElemCast[Type.ComplexDouble][Type.Bool         ] = Cast_cpu_invalid;

      ElemCast[Type.ComplexFloat][Type.ComplexDouble] = Cast_cpu_cftcd;
      ElemCast[Type.ComplexFloat][Type.ComplexFloat] = Cast_cpu_cftcf;
      // ElemCast[Type.ComplexFloat][Type.Double       ] = Cast_cpu_invalid;
      // ElemCast[Type.ComplexFloat][Type.Float        ] = Cast_cpu_invalid;
      // ElemCast[Type.ComplexFloat][Type.Int64        ] = Cast_cpu_invalid;
      // ElemCast[Type.ComplexFloat][Type.Uint64       ] = Cast_cpu_invalid;
      // ElemCast[Type.ComplexFloat][Type.Int32        ] = Cast_cpu_invalid;
      // ElemCast[Type.ComplexFloat][Type.Uint32       ] = Cast_cpu_invalid;
      // ElemCast[Type.ComplexFloat][Type.Int16        ] = Cast_cpu_invalid;
      // ElemCast[Type.ComplexFloat][Type.Uint16       ] = Cast_cpu_invalid;
      // ElemCast[Type.ComplexFloat][Type.Bool         ] = Cast_cpu_invalid;

      ElemCast[Type.Double][Type.ComplexDouble] = Cast_cpu_dtcd;
      ElemCast[Type.Double][Type.ComplexFloat] = Cast_cpu_dtcf;
      ElemCast[Type.Double][Type.Double] = Cast_cpu_dtd;
      ElemCast[Type.Double][Type.Float] = Cast_cpu_dtf;
      ElemCast[Type.Double][Type.Int64] = Cast_cpu_dti64;
      ElemCast[Type.Double][Type.Uint64] = Cast_cpu_dtu64;
      ElemCast[Type.Double][Type.Int32] = Cast_cpu_dti32;
      ElemCast[Type.Double][Type.Uint32] = Cast_cpu_dtu32;
      ElemCast[Type.Double][Type.Uint16] = Cast_cpu_dtu16;
      ElemCast[Type.Double][Type.Int16] = Cast_cpu_dti16;
      ElemCast[Type.Double][Type.Bool] = Cast_cpu_dtb;

      ElemCast[Type.Float][Type.ComplexDouble] = Cast_cpu_ftcd;
      ElemCast[Type.Float][Type.ComplexFloat] = Cast_cpu_ftcf;
      ElemCast[Type.Float][Type.Double] = Cast_cpu_ftd;
      ElemCast[Type.Float][Type.Float] = Cast_cpu_ftf;
      ElemCast[Type.Float][Type.Int64] = Cast_cpu_fti64;
      ElemCast[Type.Float][Type.Uint64] = Cast_cpu_ftu64;
      ElemCast[Type.Float][Type.Int32] = Cast_cpu_fti32;
      ElemCast[Type.Float][Type.Uint32] = Cast_cpu_ftu32;
      ElemCast[Type.Float][Type.Uint16] = Cast_cpu_ftu16;
      ElemCast[Type.Float][Type.Int16] = Cast_cpu_fti16;
      ElemCast[Type.Float][Type.Bool] = Cast_cpu_ftb;

      ElemCast[Type.Int64][Type.ComplexDouble] = Cast_cpu_i64tcd;
      ElemCast[Type.Int64][Type.ComplexFloat] = Cast_cpu_i64tcf;
      ElemCast[Type.Int64][Type.Double] = Cast_cpu_i64td;
      ElemCast[Type.Int64][Type.Float] = Cast_cpu_i64tf;
      ElemCast[Type.Int64][Type.Int64] = Cast_cpu_i64ti64;
      ElemCast[Type.Int64][Type.Uint64] = Cast_cpu_i64tu64;
      ElemCast[Type.Int64][Type.Int32] = Cast_cpu_i64ti32;
      ElemCast[Type.Int64][Type.Uint32] = Cast_cpu_i64tu32;
      ElemCast[Type.Int64][Type.Uint16] = Cast_cpu_i64tu16;
      ElemCast[Type.Int64][Type.Int16] = Cast_cpu_i64ti16;
      ElemCast[Type.Int64][Type.Bool] = Cast_cpu_i64tb;

      ElemCast[Type.Uint64][Type.ComplexDouble] = Cast_cpu_u64tcd;
      ElemCast[Type.Uint64][Type.ComplexFloat] = Cast_cpu_u64tcf;
      ElemCast[Type.Uint64][Type.Double] = Cast_cpu_u64td;
      ElemCast[Type.Uint64][Type.Float] = Cast_cpu_u64tf;
      ElemCast[Type.Uint64][Type.Int64] = Cast_cpu_u64ti64;
      ElemCast[Type.Uint64][Type.Uint64] = Cast_cpu_u64tu64;
      ElemCast[Type.Uint64][Type.Int32] = Cast_cpu_u64ti32;
      ElemCast[Type.Uint64][Type.Uint32] = Cast_cpu_u64tu32;
      ElemCast[Type.Uint64][Type.Uint16] = Cast_cpu_u64tu16;
      ElemCast[Type.Uint64][Type.Int16] = Cast_cpu_u64ti16;
      ElemCast[Type.Uint64][Type.Bool] = Cast_cpu_u64tb;

      ElemCast[Type.Int32][Type.ComplexDouble] = Cast_cpu_i32tcd;
      ElemCast[Type.Int32][Type.ComplexFloat] = Cast_cpu_i32tcf;
      ElemCast[Type.Int32][Type.Double] = Cast_cpu_i32td;
      ElemCast[Type.Int32][Type.Float] = Cast_cpu_i32tf;
      ElemCast[Type.Int32][Type.Int64] = Cast_cpu_i32ti64;
      ElemCast[Type.Int32][Type.Uint64] = Cast_cpu_i32tu64;
      ElemCast[Type.Int32][Type.Int32] = Cast_cpu_i32ti32;
      ElemCast[Type.Int32][Type.Uint32] = Cast_cpu_i32tu32;
      ElemCast[Type.Int32][Type.Uint16] = Cast_cpu_i32tu16;
      ElemCast[Type.Int32][Type.Int16] = Cast_cpu_i32ti16;
      ElemCast[Type.Int32][Type.Bool] = Cast_cpu_i32tb;

      ElemCast[Type.Uint32][Type.ComplexDouble] = Cast_cpu_u32tcd;
      ElemCast[Type.Uint32][Type.ComplexFloat] = Cast_cpu_u32tcf;
      ElemCast[Type.Uint32][Type.Double] = Cast_cpu_u32td;
      ElemCast[Type.Uint32][Type.Float] = Cast_cpu_u32tf;
      ElemCast[Type.Uint32][Type.Int64] = Cast_cpu_u32ti64;
      ElemCast[Type.Uint32][Type.Uint64] = Cast_cpu_u32tu64;
      ElemCast[Type.Uint32][Type.Int32] = Cast_cpu_u32ti32;
      ElemCast[Type.Uint32][Type.Uint32] = Cast_cpu_u32tu32;
      ElemCast[Type.Uint32][Type.Uint16] = Cast_cpu_u32tu16;
      ElemCast[Type.Uint32][Type.Int16] = Cast_cpu_u32ti16;
      ElemCast[Type.Uint32][Type.Bool] = Cast_cpu_u32tb;

      ElemCast[Type.Int16][Type.ComplexDouble] = Cast_cpu_i16tcd;
      ElemCast[Type.Int16][Type.ComplexFloat] = Cast_cpu_i16tcf;
      ElemCast[Type.Int16][Type.Double] = Cast_cpu_i16td;
      ElemCast[Type.Int16][Type.Float] = Cast_cpu_i16tf;
      ElemCast[Type.Int16][Type.Int64] = Cast_cpu_i16ti64;
      ElemCast[Type.Int16][Type.Uint64] = Cast_cpu_i16tu64;
      ElemCast[Type.Int16][Type.Int32] = Cast_cpu_i16ti32;
      ElemCast[Type.Int16][Type.Uint32] = Cast_cpu_i16tu32;
      ElemCast[Type.Int16][Type.Uint16] = Cast_cpu_i16tu16;
      ElemCast[Type.Int16][Type.Int16] = Cast_cpu_i16ti16;
      ElemCast[Type.Int16][Type.Bool] = Cast_cpu_i16tb;

      ElemCast[Type.Uint16][Type.ComplexDouble] = Cast_cpu_u16tcd;
      ElemCast[Type.Uint16][Type.ComplexFloat] = Cast_cpu_u16tcf;
      ElemCast[Type.Uint16][Type.Double] = Cast_cpu_u16td;
      ElemCast[Type.Uint16][Type.Float] = Cast_cpu_u16tf;
      ElemCast[Type.Uint16][Type.Int64] = Cast_cpu_u16ti64;
      ElemCast[Type.Uint16][Type.Uint64] = Cast_cpu_u16tu64;
      ElemCast[Type.Uint16][Type.Int32] = Cast_cpu_u16ti32;
      ElemCast[Type.Uint16][Type.Uint32] = Cast_cpu_u16tu32;
      ElemCast[Type.Uint16][Type.Uint16] = Cast_cpu_u16tu16;
      ElemCast[Type.Uint16][Type.Int16] = Cast_cpu_u16ti16;
      ElemCast[Type.Uint16][Type.Bool] = Cast_cpu_u16tb;

      ElemCast[Type.Bool][Type.ComplexDouble] = Cast_cpu_btcd;
      ElemCast[Type.Bool][Type.ComplexFloat] = Cast_cpu_btcf;
      ElemCast[Type.Bool][Type.Double] = Cast_cpu_btd;
      ElemCast[Type.Bool][Type.Float] = Cast_cpu_btf;
      ElemCast[Type.Bool][Type.Int64] = Cast_cpu_bti64;
      ElemCast[Type.Bool][Type.Uint64] = Cast_cpu_btu64;
      ElemCast[Type.Bool][Type.Int32] = Cast_cpu_bti32;
      ElemCast[Type.Bool][Type.Uint32] = Cast_cpu_btu32;
      ElemCast[Type.Bool][Type.Uint16] = Cast_cpu_btu16;
      ElemCast[Type.Bool][Type.Int16] = Cast_cpu_bti16;
      ElemCast[Type.Bool][Type.Bool] = Cast_cpu_btb;

      //
      SetArange_ii.resize(N_Type, NULL);
      SetArange_ii[Type.ComplexDouble] = SetArange_cpu_cd;
      SetArange_ii[Type.ComplexFloat] = SetArange_cpu_cf;
      SetArange_ii[Type.Double] = SetArange_cpu_d;
      SetArange_ii[Type.Float] = SetArange_cpu_f;
      SetArange_ii[Type.Uint64] = SetArange_cpu_u64;
      SetArange_ii[Type.Int64] = SetArange_cpu_i64;
      SetArange_ii[Type.Uint32] = SetArange_cpu_u32;
      SetArange_ii[Type.Int32] = SetArange_cpu_i32;
      SetArange_ii[Type.Uint16] = SetArange_cpu_u16;
      SetArange_ii[Type.Int16] = SetArange_cpu_i16;
      SetArange_ii[Type.Bool] = SetArange_cpu_b;

      //
      GetElems_ii.resize(N_Type, NULL);
      GetElems_ii[Type.ComplexDouble] = GetElems_cpu_cd;
      GetElems_ii[Type.ComplexFloat] = GetElems_cpu_cf;
      GetElems_ii[Type.Double] = GetElems_cpu_d;
      GetElems_ii[Type.Float] = GetElems_cpu_f;
      GetElems_ii[Type.Uint64] = GetElems_cpu_u64;
      GetElems_ii[Type.Int64] = GetElems_cpu_i64;
      GetElems_ii[Type.Uint32] = GetElems_cpu_u32;
      GetElems_ii[Type.Int32] = GetElems_cpu_i32;
      GetElems_ii[Type.Uint16] = GetElems_cpu_u16;
      GetElems_ii[Type.Int16] = GetElems_cpu_i16;
      GetElems_ii[Type.Bool] = GetElems_cpu_b;

      //
      GetElems_conti_ii.resize(N_Type, NULL);
      GetElems_conti_ii[Type.ComplexDouble] = GetElems_contiguous_cpu_cd;
      GetElems_conti_ii[Type.ComplexFloat] = GetElems_contiguous_cpu_cf;
      GetElems_conti_ii[Type.Double] = GetElems_contiguous_cpu_d;
      GetElems_conti_ii[Type.Float] = GetElems_contiguous_cpu_f;
      GetElems_conti_ii[Type.Uint64] = GetElems_contiguous_cpu_u64;
      GetElems_conti_ii[Type.Int64] = GetElems_contiguous_cpu_i64;
      GetElems_conti_ii[Type.Uint32] = GetElems_contiguous_cpu_u32;
      GetElems_conti_ii[Type.Int32] = GetElems_contiguous_cpu_i32;
      GetElems_conti_ii[Type.Uint16] = GetElems_contiguous_cpu_u16;
      GetElems_conti_ii[Type.Int16] = GetElems_contiguous_cpu_i16;
      GetElems_conti_ii[Type.Bool] = GetElems_contiguous_cpu_b;

      //
      SetElems_ii = vector<vector<SetElems_io>>(N_Type, vector<SetElems_io>(N_Type, NULL));
      SetElems_ii[Type.ComplexDouble][Type.ComplexDouble] = SetElems_cpu_cdtcd;
      SetElems_ii[Type.ComplexDouble][Type.ComplexFloat] = SetElems_cpu_cdtcf;
      // SetElems_ii[Type.ComplexDouble][Type.Double       ] = SetElems_cpu_invalid;
      // SetElems_ii[Type.ComplexDouble][Type.Float        ] = SetElems_cpu_invalid;
      // SetElems_ii[Type.ComplexDouble][Type.Int64        ] = SetElems_cpu_invalid;
      // SetElems_ii[Type.ComplexDouble][Type.Uint64       ] = SetElems_cpu_invalid;
      // SetElems_ii[Type.ComplexDouble][Type.Int32        ] = SetElems_cpu_invalid;
      // SetElems_ii[Type.ComplexDouble][Type.Uint32       ] = SetElems_cpu_invalid;

      SetElems_ii[Type.ComplexFloat][Type.ComplexDouble] = SetElems_cpu_cftcd;
      SetElems_ii[Type.ComplexFloat][Type.ComplexFloat] = SetElems_cpu_cftcf;
      // SetElems_ii[Type.ComplexFloat][Type.Double       ] = SetElems_cpu_invalid;
      // SetElems_ii[Type.ComplexFloat][Type.Float        ] = SetElems_cpu_invalid;
      // SetElems_ii[Type.ComplexFloat][Type.Int64        ] = SetElems_cpu_invalid;
      // SetElems_ii[Type.ComplexFloat][Type.Uint64       ] = SetElems_cpu_invalid;
      // SetElems_ii[Type.ComplexFloat][Type.Int32        ] = SetElems_cpu_invalid;
      // SetElems_ii[Type.ComplexFloat][Type.Uint32       ] = SetElems_cpu_invalid;

      SetElems_ii[Type.Double][Type.ComplexDouble] = SetElems_cpu_dtcd;
      SetElems_ii[Type.Double][Type.ComplexFloat] = SetElems_cpu_dtcf;
      SetElems_ii[Type.Double][Type.Double] = SetElems_cpu_dtd;
      SetElems_ii[Type.Double][Type.Float] = SetElems_cpu_dtf;
      SetElems_ii[Type.Double][Type.Int64] = SetElems_cpu_dti64;
      SetElems_ii[Type.Double][Type.Uint64] = SetElems_cpu_dtu64;
      SetElems_ii[Type.Double][Type.Int32] = SetElems_cpu_dti32;
      SetElems_ii[Type.Double][Type.Uint32] = SetElems_cpu_dtu32;
      SetElems_ii[Type.Double][Type.Int16] = SetElems_cpu_dti16;
      SetElems_ii[Type.Double][Type.Uint16] = SetElems_cpu_dtu16;
      SetElems_ii[Type.Double][Type.Bool] = SetElems_cpu_dtb;

      SetElems_ii[Type.Float][Type.ComplexDouble] = SetElems_cpu_ftcd;
      SetElems_ii[Type.Float][Type.ComplexFloat] = SetElems_cpu_ftcf;
      SetElems_ii[Type.Float][Type.Double] = SetElems_cpu_ftd;
      SetElems_ii[Type.Float][Type.Float] = SetElems_cpu_ftf;
      SetElems_ii[Type.Float][Type.Int64] = SetElems_cpu_fti64;
      SetElems_ii[Type.Float][Type.Uint64] = SetElems_cpu_ftu64;
      SetElems_ii[Type.Float][Type.Int32] = SetElems_cpu_fti32;
      SetElems_ii[Type.Float][Type.Uint32] = SetElems_cpu_ftu32;
      SetElems_ii[Type.Float][Type.Int16] = SetElems_cpu_fti16;
      SetElems_ii[Type.Float][Type.Uint16] = SetElems_cpu_ftu16;
      SetElems_ii[Type.Float][Type.Bool] = SetElems_cpu_ftb;

      SetElems_ii[Type.Int64][Type.ComplexDouble] = SetElems_cpu_i64tcd;
      SetElems_ii[Type.Int64][Type.ComplexFloat] = SetElems_cpu_i64tcf;
      SetElems_ii[Type.Int64][Type.Double] = SetElems_cpu_i64td;
      SetElems_ii[Type.Int64][Type.Float] = SetElems_cpu_i64tf;
      SetElems_ii[Type.Int64][Type.Int64] = SetElems_cpu_i64ti64;
      SetElems_ii[Type.Int64][Type.Uint64] = SetElems_cpu_i64tu64;
      SetElems_ii[Type.Int64][Type.Int32] = SetElems_cpu_i64ti32;
      SetElems_ii[Type.Int64][Type.Uint32] = SetElems_cpu_i64tu32;
      SetElems_ii[Type.Int64][Type.Int16] = SetElems_cpu_i64ti16;
      SetElems_ii[Type.Int64][Type.Uint16] = SetElems_cpu_i64tu16;
      SetElems_ii[Type.Int64][Type.Bool] = SetElems_cpu_i64tb;

      SetElems_ii[Type.Uint64][Type.ComplexDouble] = SetElems_cpu_u64tcd;
      SetElems_ii[Type.Uint64][Type.ComplexFloat] = SetElems_cpu_u64tcf;
      SetElems_ii[Type.Uint64][Type.Double] = SetElems_cpu_u64td;
      SetElems_ii[Type.Uint64][Type.Float] = SetElems_cpu_u64tf;
      SetElems_ii[Type.Uint64][Type.Int64] = SetElems_cpu_u64ti64;
      SetElems_ii[Type.Uint64][Type.Uint64] = SetElems_cpu_u64tu64;
      SetElems_ii[Type.Uint64][Type.Int32] = SetElems_cpu_u64ti32;
      SetElems_ii[Type.Uint64][Type.Uint32] = SetElems_cpu_u64tu32;
      SetElems_ii[Type.Uint64][Type.Int16] = SetElems_cpu_u64ti16;
      SetElems_ii[Type.Uint64][Type.Uint16] = SetElems_cpu_u64tu16;
      SetElems_ii[Type.Uint64][Type.Bool] = SetElems_cpu_u64tb;

      SetElems_ii[Type.Int32][Type.ComplexDouble] = SetElems_cpu_i32tcd;
      SetElems_ii[Type.Int32][Type.ComplexFloat] = SetElems_cpu_i32tcf;
      SetElems_ii[Type.Int32][Type.Double] = SetElems_cpu_i32td;
      SetElems_ii[Type.Int32][Type.Float] = SetElems_cpu_i32tf;
      SetElems_ii[Type.Int32][Type.Int64] = SetElems_cpu_i32ti64;
      SetElems_ii[Type.Int32][Type.Uint64] = SetElems_cpu_i32tu64;
      SetElems_ii[Type.Int32][Type.Int32] = SetElems_cpu_i32ti32;
      SetElems_ii[Type.Int32][Type.Uint32] = SetElems_cpu_i32tu32;
      SetElems_ii[Type.Int32][Type.Int16] = SetElems_cpu_i32ti16;
      SetElems_ii[Type.Int32][Type.Uint16] = SetElems_cpu_i32tu16;
      SetElems_ii[Type.Int32][Type.Bool] = SetElems_cpu_i32tb;

      SetElems_ii[Type.Uint32][Type.ComplexDouble] = SetElems_cpu_u32tcd;
      SetElems_ii[Type.Uint32][Type.ComplexFloat] = SetElems_cpu_u32tcf;
      SetElems_ii[Type.Uint32][Type.Double] = SetElems_cpu_u32td;
      SetElems_ii[Type.Uint32][Type.Float] = SetElems_cpu_u32tf;
      SetElems_ii[Type.Uint32][Type.Int64] = SetElems_cpu_u32ti64;
      SetElems_ii[Type.Uint32][Type.Uint64] = SetElems_cpu_u32tu64;
      SetElems_ii[Type.Uint32][Type.Int32] = SetElems_cpu_u32ti32;
      SetElems_ii[Type.Uint32][Type.Uint32] = SetElems_cpu_u32tu32;
      SetElems_ii[Type.Uint32][Type.Int16] = SetElems_cpu_u32ti16;
      SetElems_ii[Type.Uint32][Type.Uint16] = SetElems_cpu_u32tu16;
      SetElems_ii[Type.Uint32][Type.Bool] = SetElems_cpu_u32tb;

      SetElems_ii[Type.Uint16][Type.ComplexDouble] = SetElems_cpu_u16tcd;
      SetElems_ii[Type.Uint16][Type.ComplexFloat] = SetElems_cpu_u16tcf;
      SetElems_ii[Type.Uint16][Type.Double] = SetElems_cpu_u16td;
      SetElems_ii[Type.Uint16][Type.Float] = SetElems_cpu_u16tf;
      SetElems_ii[Type.Uint16][Type.Int64] = SetElems_cpu_u16ti64;
      SetElems_ii[Type.Uint16][Type.Uint64] = SetElems_cpu_u16tu64;
      SetElems_ii[Type.Uint16][Type.Int32] = SetElems_cpu_u16ti32;
      SetElems_ii[Type.Uint16][Type.Uint32] = SetElems_cpu_u16tu32;
      SetElems_ii[Type.Uint16][Type.Int16] = SetElems_cpu_u16ti16;
      SetElems_ii[Type.Uint16][Type.Uint16] = SetElems_cpu_u16tu16;
      SetElems_ii[Type.Uint16][Type.Bool] = SetElems_cpu_u16tb;

      SetElems_ii[Type.Int16][Type.ComplexDouble] = SetElems_cpu_i16tcd;
      SetElems_ii[Type.Int16][Type.ComplexFloat] = SetElems_cpu_i16tcf;
      SetElems_ii[Type.Int16][Type.Double] = SetElems_cpu_i16td;
      SetElems_ii[Type.Int16][Type.Float] = SetElems_cpu_i16tf;
      SetElems_ii[Type.Int16][Type.Int64] = SetElems_cpu_i16ti64;
      SetElems_ii[Type.Int16][Type.Uint64] = SetElems_cpu_i16tu64;
      SetElems_ii[Type.Int16][Type.Int32] = SetElems_cpu_i16ti32;
      SetElems_ii[Type.Int16][Type.Uint32] = SetElems_cpu_i16tu32;
      SetElems_ii[Type.Int16][Type.Int16] = SetElems_cpu_i16ti16;
      SetElems_ii[Type.Int16][Type.Uint16] = SetElems_cpu_i16tu16;
      SetElems_ii[Type.Int16][Type.Bool] = SetElems_cpu_i16tb;

      SetElems_ii[Type.Bool][Type.ComplexDouble] = SetElems_cpu_btcd;
      SetElems_ii[Type.Bool][Type.ComplexFloat] = SetElems_cpu_btcf;
      SetElems_ii[Type.Bool][Type.Double] = SetElems_cpu_btd;
      SetElems_ii[Type.Bool][Type.Float] = SetElems_cpu_btf;
      SetElems_ii[Type.Bool][Type.Int64] = SetElems_cpu_bti64;
      SetElems_ii[Type.Bool][Type.Uint64] = SetElems_cpu_btu64;
      SetElems_ii[Type.Bool][Type.Int32] = SetElems_cpu_bti32;
      SetElems_ii[Type.Bool][Type.Uint32] = SetElems_cpu_btu32;
      SetElems_ii[Type.Bool][Type.Int16] = SetElems_cpu_bti16;
      SetElems_ii[Type.Bool][Type.Uint16] = SetElems_cpu_btu16;
      SetElems_ii[Type.Bool][Type.Bool] = SetElems_cpu_btb;

      SetElems_conti_ii =
        vector<vector<SetElems_conti_io>>(N_Type, vector<SetElems_conti_io>(N_Type, NULL));
      SetElems_conti_ii[Type.ComplexDouble][Type.ComplexDouble] = SetElems_conti_cpu_cdtcd;
      SetElems_conti_ii[Type.ComplexDouble][Type.ComplexFloat] = SetElems_conti_cpu_cdtcf;
      // SetElems_conti_ii[Type.ComplexDouble][Type.Double       ] = SetElems_conti_cpu_invalid;
      // SetElems_conti_ii[Type.ComplexDouble][Type.Float        ] = SetElems_conti_cpu_invalid;
      // SetElems_conti_ii[Type.ComplexDouble][Type.Int64        ] = SetElems_conti_cpu_invalid;
      // SetElems_conti_ii[Type.ComplexDouble][Type.Uint64       ] = SetElems_conti_cpu_invalid;
      // SetElems_conti_ii[Type.ComplexDouble][Type.Int32        ] = SetElems_conti_cpu_invalid;
      // SetElems_conti_ii[Type.ComplexDouble][Type.Uint32       ] = SetElems_conti_cpu_invalid;

      SetElems_conti_ii[Type.ComplexFloat][Type.ComplexDouble] = SetElems_conti_cpu_cftcd;
      SetElems_conti_ii[Type.ComplexFloat][Type.ComplexFloat] = SetElems_conti_cpu_cftcf;
      // SetElems_conti_ii[Type.ComplexFloat][Type.Double       ] = SetElems_conti_cpu_invalid;
      // SetElems_conti_ii[Type.ComplexFloat][Type.Float        ] = SetElems_conti_cpu_invalid;
      // SetElems_conti_ii[Type.ComplexFloat][Type.Int64        ] = SetElems_conti_cpu_invalid;
      // SetElems_conti_ii[Type.ComplexFloat][Type.Uint64       ] = SetElems_conti_cpu_invalid;
      // SetElems_conti_ii[Type.ComplexFloat][Type.Int32        ] = SetElems_conti_cpu_invalid;
      // SetElems_conti_ii[Type.ComplexFloat][Type.Uint32       ] = SetElems_conti_cpu_invalid;

      SetElems_conti_ii[Type.Double][Type.ComplexDouble] = SetElems_conti_cpu_dtcd;
      SetElems_conti_ii[Type.Double][Type.ComplexFloat] = SetElems_conti_cpu_dtcf;
      SetElems_conti_ii[Type.Double][Type.Double] = SetElems_conti_cpu_dtd;
      SetElems_conti_ii[Type.Double][Type.Float] = SetElems_conti_cpu_dtf;
      SetElems_conti_ii[Type.Double][Type.Int64] = SetElems_conti_cpu_dti64;
      SetElems_conti_ii[Type.Double][Type.Uint64] = SetElems_conti_cpu_dtu64;
      SetElems_conti_ii[Type.Double][Type.Int32] = SetElems_conti_cpu_dti32;
      SetElems_conti_ii[Type.Double][Type.Uint32] = SetElems_conti_cpu_dtu32;
      SetElems_conti_ii[Type.Double][Type.Int16] = SetElems_conti_cpu_dti16;
      SetElems_conti_ii[Type.Double][Type.Uint16] = SetElems_conti_cpu_dtu16;
      SetElems_conti_ii[Type.Double][Type.Bool] = SetElems_conti_cpu_dtb;

      SetElems_conti_ii[Type.Float][Type.ComplexDouble] = SetElems_conti_cpu_ftcd;
      SetElems_conti_ii[Type.Float][Type.ComplexFloat] = SetElems_conti_cpu_ftcf;
      SetElems_conti_ii[Type.Float][Type.Double] = SetElems_conti_cpu_ftd;
      SetElems_conti_ii[Type.Float][Type.Float] = SetElems_conti_cpu_ftf;
      SetElems_conti_ii[Type.Float][Type.Int64] = SetElems_conti_cpu_fti64;
      SetElems_conti_ii[Type.Float][Type.Uint64] = SetElems_conti_cpu_ftu64;
      SetElems_conti_ii[Type.Float][Type.Int32] = SetElems_conti_cpu_fti32;
      SetElems_conti_ii[Type.Float][Type.Uint32] = SetElems_conti_cpu_ftu32;
      SetElems_conti_ii[Type.Float][Type.Int16] = SetElems_conti_cpu_fti16;
      SetElems_conti_ii[Type.Float][Type.Uint16] = SetElems_conti_cpu_ftu16;
      SetElems_conti_ii[Type.Float][Type.Bool] = SetElems_conti_cpu_ftb;

      SetElems_conti_ii[Type.Int64][Type.ComplexDouble] = SetElems_conti_cpu_i64tcd;
      SetElems_conti_ii[Type.Int64][Type.ComplexFloat] = SetElems_conti_cpu_i64tcf;
      SetElems_conti_ii[Type.Int64][Type.Double] = SetElems_conti_cpu_i64td;
      SetElems_conti_ii[Type.Int64][Type.Float] = SetElems_conti_cpu_i64tf;
      SetElems_conti_ii[Type.Int64][Type.Int64] = SetElems_conti_cpu_i64ti64;
      SetElems_conti_ii[Type.Int64][Type.Uint64] = SetElems_conti_cpu_i64tu64;
      SetElems_conti_ii[Type.Int64][Type.Int32] = SetElems_conti_cpu_i64ti32;
      SetElems_conti_ii[Type.Int64][Type.Uint32] = SetElems_conti_cpu_i64tu32;
      SetElems_conti_ii[Type.Int64][Type.Int16] = SetElems_conti_cpu_i64ti16;
      SetElems_conti_ii[Type.Int64][Type.Uint16] = SetElems_conti_cpu_i64tu16;
      SetElems_conti_ii[Type.Int64][Type.Bool] = SetElems_conti_cpu_i64tb;

      SetElems_conti_ii[Type.Uint64][Type.ComplexDouble] = SetElems_conti_cpu_u64tcd;
      SetElems_conti_ii[Type.Uint64][Type.ComplexFloat] = SetElems_conti_cpu_u64tcf;
      SetElems_conti_ii[Type.Uint64][Type.Double] = SetElems_conti_cpu_u64td;
      SetElems_conti_ii[Type.Uint64][Type.Float] = SetElems_conti_cpu_u64tf;
      SetElems_conti_ii[Type.Uint64][Type.Int64] = SetElems_conti_cpu_u64ti64;
      SetElems_conti_ii[Type.Uint64][Type.Uint64] = SetElems_conti_cpu_u64tu64;
      SetElems_conti_ii[Type.Uint64][Type.Int32] = SetElems_conti_cpu_u64ti32;
      SetElems_conti_ii[Type.Uint64][Type.Uint32] = SetElems_conti_cpu_u64tu32;
      SetElems_conti_ii[Type.Uint64][Type.Int16] = SetElems_conti_cpu_u64ti16;
      SetElems_conti_ii[Type.Uint64][Type.Uint16] = SetElems_conti_cpu_u64tu16;
      SetElems_conti_ii[Type.Uint64][Type.Bool] = SetElems_conti_cpu_u64tb;

      SetElems_conti_ii[Type.Int32][Type.ComplexDouble] = SetElems_conti_cpu_i32tcd;
      SetElems_conti_ii[Type.Int32][Type.ComplexFloat] = SetElems_conti_cpu_i32tcf;
      SetElems_conti_ii[Type.Int32][Type.Double] = SetElems_conti_cpu_i32td;
      SetElems_conti_ii[Type.Int32][Type.Float] = SetElems_conti_cpu_i32tf;
      SetElems_conti_ii[Type.Int32][Type.Int64] = SetElems_conti_cpu_i32ti64;
      SetElems_conti_ii[Type.Int32][Type.Uint64] = SetElems_conti_cpu_i32tu64;
      SetElems_conti_ii[Type.Int32][Type.Int32] = SetElems_conti_cpu_i32ti32;
      SetElems_conti_ii[Type.Int32][Type.Uint32] = SetElems_conti_cpu_i32tu32;
      SetElems_conti_ii[Type.Int32][Type.Int16] = SetElems_conti_cpu_i32ti16;
      SetElems_conti_ii[Type.Int32][Type.Uint16] = SetElems_conti_cpu_i32tu16;
      SetElems_conti_ii[Type.Int32][Type.Bool] = SetElems_conti_cpu_i32tb;

      SetElems_conti_ii[Type.Uint32][Type.ComplexDouble] = SetElems_conti_cpu_u32tcd;
      SetElems_conti_ii[Type.Uint32][Type.ComplexFloat] = SetElems_conti_cpu_u32tcf;
      SetElems_conti_ii[Type.Uint32][Type.Double] = SetElems_conti_cpu_u32td;
      SetElems_conti_ii[Type.Uint32][Type.Float] = SetElems_conti_cpu_u32tf;
      SetElems_conti_ii[Type.Uint32][Type.Int64] = SetElems_conti_cpu_u32ti64;
      SetElems_conti_ii[Type.Uint32][Type.Uint64] = SetElems_conti_cpu_u32tu64;
      SetElems_conti_ii[Type.Uint32][Type.Int32] = SetElems_conti_cpu_u32ti32;
      SetElems_conti_ii[Type.Uint32][Type.Uint32] = SetElems_conti_cpu_u32tu32;
      SetElems_conti_ii[Type.Uint32][Type.Int16] = SetElems_conti_cpu_u32ti16;
      SetElems_conti_ii[Type.Uint32][Type.Uint16] = SetElems_conti_cpu_u32tu16;
      SetElems_conti_ii[Type.Uint32][Type.Bool] = SetElems_conti_cpu_u32tb;

      SetElems_conti_ii[Type.Uint16][Type.ComplexDouble] = SetElems_conti_cpu_u16tcd;
      SetElems_conti_ii[Type.Uint16][Type.ComplexFloat] = SetElems_conti_cpu_u16tcf;
      SetElems_conti_ii[Type.Uint16][Type.Double] = SetElems_conti_cpu_u16td;
      SetElems_conti_ii[Type.Uint16][Type.Float] = SetElems_conti_cpu_u16tf;
      SetElems_conti_ii[Type.Uint16][Type.Int64] = SetElems_conti_cpu_u16ti64;
      SetElems_conti_ii[Type.Uint16][Type.Uint64] = SetElems_conti_cpu_u16tu64;
      SetElems_conti_ii[Type.Uint16][Type.Int32] = SetElems_conti_cpu_u16ti32;
      SetElems_conti_ii[Type.Uint16][Type.Uint32] = SetElems_conti_cpu_u16tu32;
      SetElems_conti_ii[Type.Uint16][Type.Int16] = SetElems_conti_cpu_u16ti16;
      SetElems_conti_ii[Type.Uint16][Type.Uint16] = SetElems_conti_cpu_u16tu16;
      SetElems_conti_ii[Type.Uint16][Type.Bool] = SetElems_conti_cpu_u16tb;

      SetElems_conti_ii[Type.Int16][Type.ComplexDouble] = SetElems_conti_cpu_i16tcd;
      SetElems_conti_ii[Type.Int16][Type.ComplexFloat] = SetElems_conti_cpu_i16tcf;
      SetElems_conti_ii[Type.Int16][Type.Double] = SetElems_conti_cpu_i16td;
      SetElems_conti_ii[Type.Int16][Type.Float] = SetElems_conti_cpu_i16tf;
      SetElems_conti_ii[Type.Int16][Type.Int64] = SetElems_conti_cpu_i16ti64;
      SetElems_conti_ii[Type.Int16][Type.Uint64] = SetElems_conti_cpu_i16tu64;
      SetElems_conti_ii[Type.Int16][Type.Int32] = SetElems_conti_cpu_i16ti32;
      SetElems_conti_ii[Type.Int16][Type.Uint32] = SetElems_conti_cpu_i16tu32;
      SetElems_conti_ii[Type.Int16][Type.Int16] = SetElems_conti_cpu_i16ti16;
      SetElems_conti_ii[Type.Int16][Type.Uint16] = SetElems_conti_cpu_i16tu16;
      SetElems_conti_ii[Type.Int16][Type.Bool] = SetElems_conti_cpu_i16tb;

      SetElems_conti_ii[Type.Bool][Type.ComplexDouble] = SetElems_conti_cpu_btcd;
      SetElems_conti_ii[Type.Bool][Type.ComplexFloat] = SetElems_conti_cpu_btcf;
      SetElems_conti_ii[Type.Bool][Type.Double] = SetElems_conti_cpu_btd;
      SetElems_conti_ii[Type.Bool][Type.Float] = SetElems_conti_cpu_btf;
      SetElems_conti_ii[Type.Bool][Type.Int64] = SetElems_conti_cpu_bti64;
      SetElems_conti_ii[Type.Bool][Type.Uint64] = SetElems_conti_cpu_btu64;
      SetElems_conti_ii[Type.Bool][Type.Int32] = SetElems_conti_cpu_bti32;
      SetElems_conti_ii[Type.Bool][Type.Uint32] = SetElems_conti_cpu_btu32;
      SetElems_conti_ii[Type.Bool][Type.Int16] = SetElems_conti_cpu_bti16;
      SetElems_conti_ii[Type.Bool][Type.Uint16] = SetElems_conti_cpu_btu16;
      SetElems_conti_ii[Type.Bool][Type.Bool] = SetElems_conti_cpu_btb;

#ifdef UNI_GPU
      cuElemCast = vector<vector<ElemCast_io>>(N_Type, vector<ElemCast_io>(N_Type, NULL));

      cuElemCast[Type.ComplexDouble][Type.ComplexDouble] = cuCast_gpu_cdtcd;
      cuElemCast[Type.ComplexDouble][Type.ComplexFloat] = cuCast_gpu_cdtcf;
      // cuElemCast[Type.ComplexDouble][Type.Double       ] = cuCast_gpu_invalid;
      // cuElemCast[Type.ComplexDouble][Type.Float        ] = cuCast_gpu_invalid;
      // cuElemCast[Type.ComplexDouble][Type.Int64        ] = cuCast_gpu_invalid;
      // cuElemCast[Type.ComplexDouble][Type.Uint64       ] = cuCast_gpu_invalid;
      // cuElemCast[Type.ComplexDouble][Type.Int32        ] = cuCast_gpu_invalid;
      // cuElemCast[Type.ComplexDouble][Type.Uint32       ] = cuCast_gpu_invalid;

      cuElemCast[Type.ComplexFloat][Type.ComplexDouble] = cuCast_gpu_cftcd;
      cuElemCast[Type.ComplexFloat][Type.ComplexFloat] = cuCast_gpu_cftcf;
      // cuElemCast[Type.ComplexFloat][Type.Double       ] = cuCast_gpu_invalid;
      // cuElemCast[Type.ComplexFloat][Type.Float        ] = cuCast_gpu_invalid;
      // cuElemCast[Type.ComplexFloat][Type.Int64        ] = cuCast_gpu_invalid;
      // cuElemCast[Type.ComplexFloat][Type.Uint64       ] = cuCast_gpu_invalid;
      // cuElemCast[Type.ComplexFloat][Type.Int32        ] = cuCast_gpu_invalid;
      // cuElemCast[Type.ComplexFloat][Type.Uint32       ] = cuCast_gpu_invalid;

      cuElemCast[Type.Double][Type.ComplexDouble] = cuCast_gpu_dtcd;
      cuElemCast[Type.Double][Type.ComplexFloat] = cuCast_gpu_dtcf;
      cuElemCast[Type.Double][Type.Double] = cuCast_gpu_dtd;
      cuElemCast[Type.Double][Type.Float] = cuCast_gpu_dtf;
      cuElemCast[Type.Double][Type.Int64] = cuCast_gpu_dti64;
      cuElemCast[Type.Double][Type.Uint64] = cuCast_gpu_dtu64;
      cuElemCast[Type.Double][Type.Int32] = cuCast_gpu_dti32;
      cuElemCast[Type.Double][Type.Uint32] = cuCast_gpu_dtu32;
      cuElemCast[Type.Double][Type.Int16] = cuCast_gpu_dti16;
      cuElemCast[Type.Double][Type.Uint16] = cuCast_gpu_dtu16;
      cuElemCast[Type.Double][Type.Bool] = cuCast_gpu_dtb;

      cuElemCast[Type.Float][Type.ComplexDouble] = cuCast_gpu_ftcd;
      cuElemCast[Type.Float][Type.ComplexFloat] = cuCast_gpu_ftcf;
      cuElemCast[Type.Float][Type.Double] = cuCast_gpu_ftd;
      cuElemCast[Type.Float][Type.Float] = cuCast_gpu_ftf;
      cuElemCast[Type.Float][Type.Int64] = cuCast_gpu_fti64;
      cuElemCast[Type.Float][Type.Uint64] = cuCast_gpu_ftu64;
      cuElemCast[Type.Float][Type.Int32] = cuCast_gpu_fti32;
      cuElemCast[Type.Float][Type.Uint32] = cuCast_gpu_ftu32;
      cuElemCast[Type.Float][Type.Uint16] = cuCast_gpu_ftu16;
      cuElemCast[Type.Float][Type.Int16] = cuCast_gpu_fti16;
      cuElemCast[Type.Float][Type.Bool] = cuCast_gpu_ftb;

      cuElemCast[Type.Int64][Type.ComplexDouble] = cuCast_gpu_i64tcd;
      cuElemCast[Type.Int64][Type.ComplexFloat] = cuCast_gpu_i64tcf;
      cuElemCast[Type.Int64][Type.Double] = cuCast_gpu_i64td;
      cuElemCast[Type.Int64][Type.Float] = cuCast_gpu_i64tf;
      cuElemCast[Type.Int64][Type.Int64] = cuCast_gpu_i64ti64;
      cuElemCast[Type.Int64][Type.Uint64] = cuCast_gpu_i64tu64;
      cuElemCast[Type.Int64][Type.Int32] = cuCast_gpu_i64ti32;
      cuElemCast[Type.Int64][Type.Uint32] = cuCast_gpu_i64tu32;
      cuElemCast[Type.Int64][Type.Uint16] = cuCast_gpu_i64tu16;
      cuElemCast[Type.Int64][Type.Int16] = cuCast_gpu_i64ti16;
      cuElemCast[Type.Int64][Type.Bool] = cuCast_gpu_i64tb;

      cuElemCast[Type.Uint64][Type.ComplexDouble] = cuCast_gpu_u64tcd;
      cuElemCast[Type.Uint64][Type.ComplexFloat] = cuCast_gpu_u64tcf;
      cuElemCast[Type.Uint64][Type.Double] = cuCast_gpu_u64td;
      cuElemCast[Type.Uint64][Type.Float] = cuCast_gpu_u64tf;
      cuElemCast[Type.Uint64][Type.Int64] = cuCast_gpu_u64ti64;
      cuElemCast[Type.Uint64][Type.Uint64] = cuCast_gpu_u64tu64;
      cuElemCast[Type.Uint64][Type.Int32] = cuCast_gpu_u64ti32;
      cuElemCast[Type.Uint64][Type.Uint32] = cuCast_gpu_u64tu32;
      cuElemCast[Type.Uint64][Type.Int16] = cuCast_gpu_u64ti16;
      cuElemCast[Type.Uint64][Type.Uint16] = cuCast_gpu_u64tu16;
      cuElemCast[Type.Uint64][Type.Bool] = cuCast_gpu_u64tb;

      cuElemCast[Type.Int32][Type.ComplexDouble] = cuCast_gpu_i32tcd;
      cuElemCast[Type.Int32][Type.ComplexFloat] = cuCast_gpu_i32tcf;
      cuElemCast[Type.Int32][Type.Double] = cuCast_gpu_i32td;
      cuElemCast[Type.Int32][Type.Float] = cuCast_gpu_i32tf;
      cuElemCast[Type.Int32][Type.Int64] = cuCast_gpu_i32ti64;
      cuElemCast[Type.Int32][Type.Uint64] = cuCast_gpu_i32tu64;
      cuElemCast[Type.Int32][Type.Int32] = cuCast_gpu_i32ti32;
      cuElemCast[Type.Int32][Type.Uint32] = cuCast_gpu_i32tu32;
      cuElemCast[Type.Int32][Type.Int16] = cuCast_gpu_i32ti16;
      cuElemCast[Type.Int32][Type.Uint16] = cuCast_gpu_i32tu16;
      cuElemCast[Type.Int32][Type.Bool] = cuCast_gpu_i32tb;

      cuElemCast[Type.Uint32][Type.ComplexDouble] = cuCast_gpu_u32tcd;
      cuElemCast[Type.Uint32][Type.ComplexFloat] = cuCast_gpu_u32tcf;
      cuElemCast[Type.Uint32][Type.Double] = cuCast_gpu_u32td;
      cuElemCast[Type.Uint32][Type.Float] = cuCast_gpu_u32tf;
      cuElemCast[Type.Uint32][Type.Int64] = cuCast_gpu_u32ti64;
      cuElemCast[Type.Uint32][Type.Uint64] = cuCast_gpu_u32tu64;
      cuElemCast[Type.Uint32][Type.Int32] = cuCast_gpu_u32ti32;
      cuElemCast[Type.Uint32][Type.Uint32] = cuCast_gpu_u32tu32;
      cuElemCast[Type.Uint32][Type.Int16] = cuCast_gpu_u32ti16;
      cuElemCast[Type.Uint32][Type.Uint16] = cuCast_gpu_u32tu16;
      cuElemCast[Type.Uint32][Type.Bool] = cuCast_gpu_u32tb;

      cuElemCast[Type.Int16][Type.ComplexDouble] = cuCast_gpu_i16tcd;
      cuElemCast[Type.Int16][Type.ComplexFloat] = cuCast_gpu_i16tcf;
      cuElemCast[Type.Int16][Type.Double] = cuCast_gpu_i16td;
      cuElemCast[Type.Int16][Type.Float] = cuCast_gpu_i16tf;
      cuElemCast[Type.Int16][Type.Int64] = cuCast_gpu_i16ti64;
      cuElemCast[Type.Int16][Type.Uint64] = cuCast_gpu_i16tu64;
      cuElemCast[Type.Int16][Type.Int32] = cuCast_gpu_i16ti32;
      cuElemCast[Type.Int16][Type.Uint32] = cuCast_gpu_i16tu32;
      cuElemCast[Type.Int16][Type.Int16] = cuCast_gpu_i16ti16;
      cuElemCast[Type.Int16][Type.Uint16] = cuCast_gpu_i16tu16;
      cuElemCast[Type.Int16][Type.Bool] = cuCast_gpu_i16tb;

      cuElemCast[Type.Uint16][Type.ComplexDouble] = cuCast_gpu_u16tcd;
      cuElemCast[Type.Uint16][Type.ComplexFloat] = cuCast_gpu_u16tcf;
      cuElemCast[Type.Uint16][Type.Double] = cuCast_gpu_u16td;
      cuElemCast[Type.Uint16][Type.Float] = cuCast_gpu_u16tf;
      cuElemCast[Type.Uint16][Type.Int64] = cuCast_gpu_u16ti64;
      cuElemCast[Type.Uint16][Type.Uint64] = cuCast_gpu_u16tu64;
      cuElemCast[Type.Uint16][Type.Int32] = cuCast_gpu_u16ti32;
      cuElemCast[Type.Uint16][Type.Uint32] = cuCast_gpu_u16tu32;
      cuElemCast[Type.Uint16][Type.Int16] = cuCast_gpu_u16ti16;
      cuElemCast[Type.Uint16][Type.Uint16] = cuCast_gpu_u16tu16;
      cuElemCast[Type.Uint16][Type.Bool] = cuCast_gpu_u16tb;

      cuElemCast[Type.Bool][Type.ComplexDouble] = cuCast_gpu_btcd;
      cuElemCast[Type.Bool][Type.ComplexFloat] = cuCast_gpu_btcf;
      cuElemCast[Type.Bool][Type.Double] = cuCast_gpu_btd;
      cuElemCast[Type.Bool][Type.Float] = cuCast_gpu_btf;
      cuElemCast[Type.Bool][Type.Int64] = cuCast_gpu_bti64;
      cuElemCast[Type.Bool][Type.Uint64] = cuCast_gpu_btu64;
      cuElemCast[Type.Bool][Type.Int32] = cuCast_gpu_bti32;
      cuElemCast[Type.Bool][Type.Uint32] = cuCast_gpu_btu32;
      cuElemCast[Type.Bool][Type.Int16] = cuCast_gpu_bti16;
      cuElemCast[Type.Bool][Type.Uint16] = cuCast_gpu_btu16;
      cuElemCast[Type.Bool][Type.Bool] = cuCast_gpu_btb;

      cuSetArange_ii.resize(N_Type, NULL);
      cuSetArange_ii[Type.ComplexDouble] = cuSetArange_gpu_cd;
      cuSetArange_ii[Type.ComplexFloat] = cuSetArange_gpu_cf;
      cuSetArange_ii[Type.Double] = cuSetArange_gpu_d;
      cuSetArange_ii[Type.Float] = cuSetArange_gpu_f;
      cuSetArange_ii[Type.Uint64] = cuSetArange_gpu_u64;
      cuSetArange_ii[Type.Int64] = cuSetArange_gpu_i64;
      cuSetArange_ii[Type.Uint32] = cuSetArange_gpu_u32;
      cuSetArange_ii[Type.Int32] = cuSetArange_gpu_i32;
      cuSetArange_ii[Type.Uint16] = cuSetArange_gpu_u16;
      cuSetArange_ii[Type.Int16] = cuSetArange_gpu_i16;
      cuSetArange_ii[Type.Bool] = cuSetArange_gpu_b;

      cuGetElems_ii.resize(N_Type, NULL);
      cuGetElems_ii[Type.ComplexDouble] = cuGetElems_gpu_cd;
      cuGetElems_ii[Type.ComplexFloat] = cuGetElems_gpu_cf;
      cuGetElems_ii[Type.Double] = cuGetElems_gpu_d;
      cuGetElems_ii[Type.Float] = cuGetElems_gpu_f;
      cuGetElems_ii[Type.Uint64] = cuGetElems_gpu_u64;
      cuGetElems_ii[Type.Int64] = cuGetElems_gpu_i64;
      cuGetElems_ii[Type.Uint32] = cuGetElems_gpu_u32;
      cuGetElems_ii[Type.Int32] = cuGetElems_gpu_i32;
      cuGetElems_ii[Type.Uint16] = cuGetElems_gpu_u16;
      cuGetElems_ii[Type.Int16] = cuGetElems_gpu_i16;
      cuGetElems_ii[Type.Bool] = cuGetElems_gpu_b;

      //
      cuSetElems_ii = vector<vector<SetElems_io>>(N_Type, vector<SetElems_io>(N_Type, NULL));
      cuSetElems_ii[Type.ComplexDouble][Type.ComplexDouble] = cuSetElems_gpu_cdtcd;
      cuSetElems_ii[Type.ComplexDouble][Type.ComplexFloat] = cuSetElems_gpu_cdtcf;
      // cuSetElems_ii[Type.ComplexDouble][Type.Double       ] = cuSetElems_gpu_invalid;
      // cuSetElems_ii[Type.ComplexDouble][Type.Float        ] = cuSetElems_gpu_invalid;
      // cuSetElems_ii[Type.ComplexDouble][Type.Int64        ] = cuSetElems_gpu_invalid;
      // cuSetElems_ii[Type.ComplexDouble][Type.Uint64       ] = cuSetElems_gpu_invalid;
      // cuSetElems_ii[Type.ComplexDouble][Type.Int32        ] = cuSetElems_gpu_invalid;
      // cuSetElems_ii[Type.ComplexDouble][Type.Uint32       ] = cuSetElems_gpu_invalid;

      cuSetElems_ii[Type.ComplexFloat][Type.ComplexDouble] = cuSetElems_gpu_cftcd;
      cuSetElems_ii[Type.ComplexFloat][Type.ComplexFloat] = cuSetElems_gpu_cftcf;
      // cuSetElems_ii[Type.ComplexFloat][Type.Double       ] = cuSetElems_gpu_invalid;
      // cuSetElems_ii[Type.ComplexFloat][Type.Float        ] = cuSetElems_gpu_invalid;
      // cuSetElems_ii[Type.ComplexFloat][Type.Int64        ] = cuSetElems_gpu_invalid;
      // cuSetElems_ii[Type.ComplexFloat][Type.Uint64       ] = cuSetElems_gpu_invalid;
      // cuSetElems_ii[Type.ComplexFloat][Type.Int32        ] = cuSetElems_gpu_invalid;
      // cuSetElems_ii[Type.ComplexFloat][Type.Uint32       ] = cuSetElems_gpu_invalid;

      cuSetElems_ii[Type.Double][Type.ComplexDouble] = cuSetElems_gpu_dtcd;
      cuSetElems_ii[Type.Double][Type.ComplexFloat] = cuSetElems_gpu_dtcf;
      cuSetElems_ii[Type.Double][Type.Double] = cuSetElems_gpu_dtd;
      cuSetElems_ii[Type.Double][Type.Float] = cuSetElems_gpu_dtf;
      cuSetElems_ii[Type.Double][Type.Int64] = cuSetElems_gpu_dti64;
      cuSetElems_ii[Type.Double][Type.Uint64] = cuSetElems_gpu_dtu64;
      cuSetElems_ii[Type.Double][Type.Int32] = cuSetElems_gpu_dti32;
      cuSetElems_ii[Type.Double][Type.Uint32] = cuSetElems_gpu_dtu32;
      cuSetElems_ii[Type.Double][Type.Int16] = cuSetElems_gpu_dti16;
      cuSetElems_ii[Type.Double][Type.Uint16] = cuSetElems_gpu_dtu16;
      cuSetElems_ii[Type.Double][Type.Bool] = cuSetElems_gpu_dtb;

      cuSetElems_ii[Type.Float][Type.ComplexDouble] = cuSetElems_gpu_ftcd;
      cuSetElems_ii[Type.Float][Type.ComplexFloat] = cuSetElems_gpu_ftcf;
      cuSetElems_ii[Type.Float][Type.Double] = cuSetElems_gpu_ftd;
      cuSetElems_ii[Type.Float][Type.Float] = cuSetElems_gpu_ftf;
      cuSetElems_ii[Type.Float][Type.Int64] = cuSetElems_gpu_fti64;
      cuSetElems_ii[Type.Float][Type.Uint64] = cuSetElems_gpu_ftu64;
      cuSetElems_ii[Type.Float][Type.Int32] = cuSetElems_gpu_fti32;
      cuSetElems_ii[Type.Float][Type.Uint32] = cuSetElems_gpu_ftu32;
      cuSetElems_ii[Type.Float][Type.Uint16] = cuSetElems_gpu_ftu16;
      cuSetElems_ii[Type.Float][Type.Int16] = cuSetElems_gpu_fti16;
      cuSetElems_ii[Type.Float][Type.Bool] = cuSetElems_gpu_ftb;

      cuSetElems_ii[Type.Int64][Type.ComplexDouble] = cuSetElems_gpu_i64tcd;
      cuSetElems_ii[Type.Int64][Type.ComplexFloat] = cuSetElems_gpu_i64tcf;
      cuSetElems_ii[Type.Int64][Type.Double] = cuSetElems_gpu_i64td;
      cuSetElems_ii[Type.Int64][Type.Float] = cuSetElems_gpu_i64tf;
      cuSetElems_ii[Type.Int64][Type.Int64] = cuSetElems_gpu_i64ti64;
      cuSetElems_ii[Type.Int64][Type.Uint64] = cuSetElems_gpu_i64tu64;
      cuSetElems_ii[Type.Int64][Type.Int32] = cuSetElems_gpu_i64ti32;
      cuSetElems_ii[Type.Int64][Type.Uint32] = cuSetElems_gpu_i64tu32;
      cuSetElems_ii[Type.Int64][Type.Uint16] = cuSetElems_gpu_i64tu16;
      cuSetElems_ii[Type.Int64][Type.Int16] = cuSetElems_gpu_i64ti16;
      cuSetElems_ii[Type.Int64][Type.Bool] = cuSetElems_gpu_i64tb;

      cuSetElems_ii[Type.Uint64][Type.ComplexDouble] = cuSetElems_gpu_u64tcd;
      cuSetElems_ii[Type.Uint64][Type.ComplexFloat] = cuSetElems_gpu_u64tcf;
      cuSetElems_ii[Type.Uint64][Type.Double] = cuSetElems_gpu_u64td;
      cuSetElems_ii[Type.Uint64][Type.Float] = cuSetElems_gpu_u64tf;
      cuSetElems_ii[Type.Uint64][Type.Int64] = cuSetElems_gpu_u64ti64;
      cuSetElems_ii[Type.Uint64][Type.Uint64] = cuSetElems_gpu_u64tu64;
      cuSetElems_ii[Type.Uint64][Type.Int32] = cuSetElems_gpu_u64ti32;
      cuSetElems_ii[Type.Uint64][Type.Uint32] = cuSetElems_gpu_u64tu32;
      cuSetElems_ii[Type.Uint64][Type.Int16] = cuSetElems_gpu_u64ti16;
      cuSetElems_ii[Type.Uint64][Type.Uint16] = cuSetElems_gpu_u64tu16;
      cuSetElems_ii[Type.Uint64][Type.Bool] = cuSetElems_gpu_u64tb;

      cuSetElems_ii[Type.Int32][Type.ComplexDouble] = cuSetElems_gpu_i32tcd;
      cuSetElems_ii[Type.Int32][Type.ComplexFloat] = cuSetElems_gpu_i32tcf;
      cuSetElems_ii[Type.Int32][Type.Double] = cuSetElems_gpu_i32td;
      cuSetElems_ii[Type.Int32][Type.Float] = cuSetElems_gpu_i32tf;
      cuSetElems_ii[Type.Int32][Type.Int64] = cuSetElems_gpu_i32ti64;
      cuSetElems_ii[Type.Int32][Type.Uint64] = cuSetElems_gpu_i32tu64;
      cuSetElems_ii[Type.Int32][Type.Int32] = cuSetElems_gpu_i32ti32;
      cuSetElems_ii[Type.Int32][Type.Uint32] = cuSetElems_gpu_i32tu32;
      cuSetElems_ii[Type.Int32][Type.Uint16] = cuSetElems_gpu_i32tu16;
      cuSetElems_ii[Type.Int32][Type.Int16] = cuSetElems_gpu_i32ti16;
      cuSetElems_ii[Type.Int32][Type.Bool] = cuSetElems_gpu_i32tb;

      cuSetElems_ii[Type.Uint32][Type.ComplexDouble] = cuSetElems_gpu_u32tcd;
      cuSetElems_ii[Type.Uint32][Type.ComplexFloat] = cuSetElems_gpu_u32tcf;
      cuSetElems_ii[Type.Uint32][Type.Double] = cuSetElems_gpu_u32td;
      cuSetElems_ii[Type.Uint32][Type.Float] = cuSetElems_gpu_u32tf;
      cuSetElems_ii[Type.Uint32][Type.Int64] = cuSetElems_gpu_u32ti64;
      cuSetElems_ii[Type.Uint32][Type.Uint64] = cuSetElems_gpu_u32tu64;
      cuSetElems_ii[Type.Uint32][Type.Int32] = cuSetElems_gpu_u32ti32;
      cuSetElems_ii[Type.Uint32][Type.Uint32] = cuSetElems_gpu_u32tu32;
      cuSetElems_ii[Type.Uint32][Type.Uint16] = cuSetElems_gpu_u32tu16;
      cuSetElems_ii[Type.Uint32][Type.Int16] = cuSetElems_gpu_u32ti16;
      cuSetElems_ii[Type.Uint32][Type.Bool] = cuSetElems_gpu_u32tb;

      cuSetElems_ii[Type.Int16][Type.ComplexDouble] = cuSetElems_gpu_i16tcd;
      cuSetElems_ii[Type.Int16][Type.ComplexFloat] = cuSetElems_gpu_i16tcf;
      cuSetElems_ii[Type.Int16][Type.Double] = cuSetElems_gpu_i16td;
      cuSetElems_ii[Type.Int16][Type.Float] = cuSetElems_gpu_i16tf;
      cuSetElems_ii[Type.Int16][Type.Int64] = cuSetElems_gpu_i16ti64;
      cuSetElems_ii[Type.Int16][Type.Uint64] = cuSetElems_gpu_i16tu64;
      cuSetElems_ii[Type.Int16][Type.Int32] = cuSetElems_gpu_i16ti32;
      cuSetElems_ii[Type.Int16][Type.Uint32] = cuSetElems_gpu_i16tu32;
      cuSetElems_ii[Type.Int16][Type.Uint16] = cuSetElems_gpu_i16tu16;
      cuSetElems_ii[Type.Int16][Type.Int16] = cuSetElems_gpu_i16ti16;
      cuSetElems_ii[Type.Int16][Type.Bool] = cuSetElems_gpu_i16tb;

      cuSetElems_ii[Type.Uint16][Type.ComplexDouble] = cuSetElems_gpu_u16tcd;
      cuSetElems_ii[Type.Uint16][Type.ComplexFloat] = cuSetElems_gpu_u16tcf;
      cuSetElems_ii[Type.Uint16][Type.Double] = cuSetElems_gpu_u16td;
      cuSetElems_ii[Type.Uint16][Type.Float] = cuSetElems_gpu_u16tf;
      cuSetElems_ii[Type.Uint16][Type.Int64] = cuSetElems_gpu_u16ti64;
      cuSetElems_ii[Type.Uint16][Type.Uint64] = cuSetElems_gpu_u16tu64;
      cuSetElems_ii[Type.Uint16][Type.Int32] = cuSetElems_gpu_u16ti32;
      cuSetElems_ii[Type.Uint16][Type.Uint32] = cuSetElems_gpu_u16tu32;
      cuSetElems_ii[Type.Uint16][Type.Uint16] = cuSetElems_gpu_u16tu16;
      cuSetElems_ii[Type.Uint16][Type.Int16] = cuSetElems_gpu_u16ti16;
      cuSetElems_ii[Type.Uint16][Type.Bool] = cuSetElems_gpu_u16tb;

      cuSetElems_ii[Type.Bool][Type.ComplexDouble] = cuSetElems_gpu_btcd;
      cuSetElems_ii[Type.Bool][Type.ComplexFloat] = cuSetElems_gpu_btcf;
      cuSetElems_ii[Type.Bool][Type.Double] = cuSetElems_gpu_btd;
      cuSetElems_ii[Type.Bool][Type.Float] = cuSetElems_gpu_btf;
      cuSetElems_ii[Type.Bool][Type.Int64] = cuSetElems_gpu_bti64;
      cuSetElems_ii[Type.Bool][Type.Uint64] = cuSetElems_gpu_btu64;
      cuSetElems_ii[Type.Bool][Type.Int32] = cuSetElems_gpu_bti32;
      cuSetElems_ii[Type.Bool][Type.Uint32] = cuSetElems_gpu_btu32;
      cuSetElems_ii[Type.Bool][Type.Uint16] = cuSetElems_gpu_btu16;
      cuSetElems_ii[Type.Bool][Type.Int16] = cuSetElems_gpu_bti16;
      cuSetElems_ii[Type.Bool][Type.Bool] = cuSetElems_gpu_btb;

#endif
    }

    utils_internal_interface uii;

  }  // namespace utils_internal
}  // namespace cytnx
