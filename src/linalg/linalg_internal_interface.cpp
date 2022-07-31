#include "linalg_internal_interface.hpp"

using namespace std;

namespace cytnx {
  namespace linalg_internal {

    linalg_internal_interface lii;

    linalg_internal_interface::linalg_internal_interface() {
      Ari_ii = vector<vector<Arithmeticfunc_oii>>(N_Type, vector<Arithmeticfunc_oii>(N_Type, NULL));

      Ari_ii[Type.ComplexDouble][Type.ComplexDouble] = Arithmetic_internal_cdtcd;
      Ari_ii[Type.ComplexDouble][Type.ComplexFloat] = Arithmetic_internal_cdtcf;
      Ari_ii[Type.ComplexDouble][Type.Double] = Arithmetic_internal_cdtd;
      Ari_ii[Type.ComplexDouble][Type.Float] = Arithmetic_internal_cdtf;
      Ari_ii[Type.ComplexDouble][Type.Int64] = Arithmetic_internal_cdti64;
      Ari_ii[Type.ComplexDouble][Type.Uint64] = Arithmetic_internal_cdtu64;
      Ari_ii[Type.ComplexDouble][Type.Int32] = Arithmetic_internal_cdti32;
      Ari_ii[Type.ComplexDouble][Type.Uint32] = Arithmetic_internal_cdtu32;
      Ari_ii[Type.ComplexDouble][Type.Uint16] = Arithmetic_internal_cdtu16;
      Ari_ii[Type.ComplexDouble][Type.Int16] = Arithmetic_internal_cdti16;
      Ari_ii[Type.ComplexDouble][Type.Bool] = Arithmetic_internal_cdtb;

      Ari_ii[Type.ComplexFloat][Type.ComplexDouble] = Arithmetic_internal_cftcd;
      Ari_ii[Type.ComplexFloat][Type.ComplexFloat] = Arithmetic_internal_cftcf;
      Ari_ii[Type.ComplexFloat][Type.Double] = Arithmetic_internal_cftd;
      Ari_ii[Type.ComplexFloat][Type.Float] = Arithmetic_internal_cftf;
      Ari_ii[Type.ComplexFloat][Type.Int64] = Arithmetic_internal_cfti64;
      Ari_ii[Type.ComplexFloat][Type.Uint64] = Arithmetic_internal_cftu64;
      Ari_ii[Type.ComplexFloat][Type.Int32] = Arithmetic_internal_cfti32;
      Ari_ii[Type.ComplexFloat][Type.Uint32] = Arithmetic_internal_cftu32;
      Ari_ii[Type.ComplexFloat][Type.Uint16] = Arithmetic_internal_cftu16;
      Ari_ii[Type.ComplexFloat][Type.Int16] = Arithmetic_internal_cfti16;
      Ari_ii[Type.ComplexFloat][Type.Bool] = Arithmetic_internal_cftb;

      Ari_ii[Type.Double][Type.ComplexDouble] = Arithmetic_internal_dtcd;
      Ari_ii[Type.Double][Type.ComplexFloat] = Arithmetic_internal_dtcf;
      Ari_ii[Type.Double][Type.Double] = Arithmetic_internal_dtd;
      Ari_ii[Type.Double][Type.Float] = Arithmetic_internal_dtf;
      Ari_ii[Type.Double][Type.Int64] = Arithmetic_internal_dti64;
      Ari_ii[Type.Double][Type.Uint64] = Arithmetic_internal_dtu64;
      Ari_ii[Type.Double][Type.Int32] = Arithmetic_internal_dti32;
      Ari_ii[Type.Double][Type.Uint32] = Arithmetic_internal_dtu32;
      Ari_ii[Type.Double][Type.Uint16] = Arithmetic_internal_dtu16;
      Ari_ii[Type.Double][Type.Int16] = Arithmetic_internal_dti16;
      Ari_ii[Type.Double][Type.Bool] = Arithmetic_internal_dtb;

      Ari_ii[Type.Float][Type.ComplexDouble] = Arithmetic_internal_ftcd;
      Ari_ii[Type.Float][Type.ComplexFloat] = Arithmetic_internal_ftcf;
      Ari_ii[Type.Float][Type.Double] = Arithmetic_internal_ftd;
      Ari_ii[Type.Float][Type.Float] = Arithmetic_internal_ftf;
      Ari_ii[Type.Float][Type.Int64] = Arithmetic_internal_fti64;
      Ari_ii[Type.Float][Type.Uint64] = Arithmetic_internal_ftu64;
      Ari_ii[Type.Float][Type.Int32] = Arithmetic_internal_fti32;
      Ari_ii[Type.Float][Type.Uint32] = Arithmetic_internal_ftu32;
      Ari_ii[Type.Float][Type.Uint16] = Arithmetic_internal_ftu16;
      Ari_ii[Type.Float][Type.Int16] = Arithmetic_internal_fti16;
      Ari_ii[Type.Float][Type.Bool] = Arithmetic_internal_ftb;

      Ari_ii[Type.Int64][Type.ComplexDouble] = Arithmetic_internal_i64tcd;
      Ari_ii[Type.Int64][Type.ComplexFloat] = Arithmetic_internal_i64tcf;
      Ari_ii[Type.Int64][Type.Double] = Arithmetic_internal_i64td;
      Ari_ii[Type.Int64][Type.Float] = Arithmetic_internal_i64tf;
      Ari_ii[Type.Int64][Type.Int64] = Arithmetic_internal_i64ti64;
      Ari_ii[Type.Int64][Type.Uint64] = Arithmetic_internal_i64tu64;
      Ari_ii[Type.Int64][Type.Int32] = Arithmetic_internal_i64ti32;
      Ari_ii[Type.Int64][Type.Uint32] = Arithmetic_internal_i64tu32;
      Ari_ii[Type.Int64][Type.Uint16] = Arithmetic_internal_i64tu16;
      Ari_ii[Type.Int64][Type.Int16] = Arithmetic_internal_i64ti16;
      Ari_ii[Type.Int64][Type.Bool] = Arithmetic_internal_i64tb;

      Ari_ii[Type.Uint64][Type.ComplexDouble] = Arithmetic_internal_u64tcd;
      Ari_ii[Type.Uint64][Type.ComplexFloat] = Arithmetic_internal_u64tcf;
      Ari_ii[Type.Uint64][Type.Double] = Arithmetic_internal_u64td;
      Ari_ii[Type.Uint64][Type.Float] = Arithmetic_internal_u64tf;
      Ari_ii[Type.Uint64][Type.Int64] = Arithmetic_internal_u64ti64;
      Ari_ii[Type.Uint64][Type.Uint64] = Arithmetic_internal_u64tu64;
      Ari_ii[Type.Uint64][Type.Int32] = Arithmetic_internal_u64ti32;
      Ari_ii[Type.Uint64][Type.Uint32] = Arithmetic_internal_u64tu32;
      Ari_ii[Type.Uint64][Type.Uint16] = Arithmetic_internal_u64tu16;
      Ari_ii[Type.Uint64][Type.Int16] = Arithmetic_internal_u64ti16;
      Ari_ii[Type.Uint64][Type.Bool] = Arithmetic_internal_u64tb;

      Ari_ii[Type.Int32][Type.ComplexDouble] = Arithmetic_internal_i32tcd;
      Ari_ii[Type.Int32][Type.ComplexFloat] = Arithmetic_internal_i32tcf;
      Ari_ii[Type.Int32][Type.Double] = Arithmetic_internal_i32td;
      Ari_ii[Type.Int32][Type.Float] = Arithmetic_internal_i32tf;
      Ari_ii[Type.Int32][Type.Int64] = Arithmetic_internal_i32ti64;
      Ari_ii[Type.Int32][Type.Uint64] = Arithmetic_internal_i32tu64;
      Ari_ii[Type.Int32][Type.Int32] = Arithmetic_internal_i32ti32;
      Ari_ii[Type.Int32][Type.Uint32] = Arithmetic_internal_i32tu32;
      Ari_ii[Type.Int32][Type.Uint16] = Arithmetic_internal_i32tu16;
      Ari_ii[Type.Int32][Type.Int16] = Arithmetic_internal_i32ti16;
      Ari_ii[Type.Int32][Type.Bool] = Arithmetic_internal_i32tb;

      Ari_ii[Type.Uint32][Type.ComplexDouble] = Arithmetic_internal_u32tcd;
      Ari_ii[Type.Uint32][Type.ComplexFloat] = Arithmetic_internal_u32tcf;
      Ari_ii[Type.Uint32][Type.Double] = Arithmetic_internal_u32td;
      Ari_ii[Type.Uint32][Type.Float] = Arithmetic_internal_u32tf;
      Ari_ii[Type.Uint32][Type.Int64] = Arithmetic_internal_u32ti64;
      Ari_ii[Type.Uint32][Type.Uint64] = Arithmetic_internal_u32tu64;
      Ari_ii[Type.Uint32][Type.Int32] = Arithmetic_internal_u32ti32;
      Ari_ii[Type.Uint32][Type.Uint32] = Arithmetic_internal_u32tu32;
      Ari_ii[Type.Uint32][Type.Uint16] = Arithmetic_internal_u32tu16;
      Ari_ii[Type.Uint32][Type.Int16] = Arithmetic_internal_u32ti16;
      Ari_ii[Type.Uint32][Type.Bool] = Arithmetic_internal_u32tb;

      Ari_ii[Type.Int16][Type.ComplexDouble] = Arithmetic_internal_i16tcd;
      Ari_ii[Type.Int16][Type.ComplexFloat] = Arithmetic_internal_i16tcf;
      Ari_ii[Type.Int16][Type.Double] = Arithmetic_internal_i16td;
      Ari_ii[Type.Int16][Type.Float] = Arithmetic_internal_i16tf;
      Ari_ii[Type.Int16][Type.Int64] = Arithmetic_internal_i16ti64;
      Ari_ii[Type.Int16][Type.Uint64] = Arithmetic_internal_i16tu64;
      Ari_ii[Type.Int16][Type.Int32] = Arithmetic_internal_i16ti32;
      Ari_ii[Type.Int16][Type.Uint32] = Arithmetic_internal_i16tu32;
      Ari_ii[Type.Int16][Type.Uint16] = Arithmetic_internal_i16tu16;
      Ari_ii[Type.Int16][Type.Int16] = Arithmetic_internal_i16ti16;
      Ari_ii[Type.Int16][Type.Bool] = Arithmetic_internal_i16tb;

      Ari_ii[Type.Uint16][Type.ComplexDouble] = Arithmetic_internal_u16tcd;
      Ari_ii[Type.Uint16][Type.ComplexFloat] = Arithmetic_internal_u16tcf;
      Ari_ii[Type.Uint16][Type.Double] = Arithmetic_internal_u16td;
      Ari_ii[Type.Uint16][Type.Float] = Arithmetic_internal_u16tf;
      Ari_ii[Type.Uint16][Type.Int64] = Arithmetic_internal_u16ti64;
      Ari_ii[Type.Uint16][Type.Uint64] = Arithmetic_internal_u16tu64;
      Ari_ii[Type.Uint16][Type.Int32] = Arithmetic_internal_u16ti32;
      Ari_ii[Type.Uint16][Type.Uint32] = Arithmetic_internal_u16tu32;
      Ari_ii[Type.Uint16][Type.Uint16] = Arithmetic_internal_u16tu16;
      Ari_ii[Type.Uint16][Type.Int16] = Arithmetic_internal_u16ti16;
      Ari_ii[Type.Uint16][Type.Bool] = Arithmetic_internal_u16tb;

      Ari_ii[Type.Bool][Type.ComplexDouble] = Arithmetic_internal_btcd;
      Ari_ii[Type.Bool][Type.ComplexFloat] = Arithmetic_internal_btcf;
      Ari_ii[Type.Bool][Type.Double] = Arithmetic_internal_btd;
      Ari_ii[Type.Bool][Type.Float] = Arithmetic_internal_btf;
      Ari_ii[Type.Bool][Type.Int64] = Arithmetic_internal_bti64;
      Ari_ii[Type.Bool][Type.Uint64] = Arithmetic_internal_btu64;
      Ari_ii[Type.Bool][Type.Int32] = Arithmetic_internal_bti32;
      Ari_ii[Type.Bool][Type.Uint32] = Arithmetic_internal_btu32;
      Ari_ii[Type.Bool][Type.Uint16] = Arithmetic_internal_btu16;
      Ari_ii[Type.Bool][Type.Int16] = Arithmetic_internal_bti16;
      Ari_ii[Type.Bool][Type.Bool] = Arithmetic_internal_btb;

      iAri_ii =
        vector<vector<Arithmeticfunc_oii>>(N_Type, vector<Arithmeticfunc_oii>(N_Type, NULL));

      iAri_ii[Type.ComplexDouble][Type.ComplexDouble] = iArithmetic_internal_cdtcd;
      iAri_ii[Type.ComplexDouble][Type.ComplexFloat] = iArithmetic_internal_cdtcf;
      iAri_ii[Type.ComplexDouble][Type.Double] = iArithmetic_internal_cdtd;
      iAri_ii[Type.ComplexDouble][Type.Float] = iArithmetic_internal_cdtf;
      iAri_ii[Type.ComplexDouble][Type.Int64] = iArithmetic_internal_cdti64;
      iAri_ii[Type.ComplexDouble][Type.Uint64] = iArithmetic_internal_cdtu64;
      iAri_ii[Type.ComplexDouble][Type.Int32] = iArithmetic_internal_cdti32;
      iAri_ii[Type.ComplexDouble][Type.Uint32] = iArithmetic_internal_cdtu32;
      iAri_ii[Type.ComplexDouble][Type.Uint16] = iArithmetic_internal_cdtu16;
      iAri_ii[Type.ComplexDouble][Type.Int16] = iArithmetic_internal_cdti16;
      iAri_ii[Type.ComplexDouble][Type.Bool] = iArithmetic_internal_cdtb;

      iAri_ii[Type.ComplexFloat][Type.ComplexDouble] = iArithmetic_internal_cftcd;
      iAri_ii[Type.ComplexFloat][Type.ComplexFloat] = iArithmetic_internal_cftcf;
      iAri_ii[Type.ComplexFloat][Type.Double] = iArithmetic_internal_cftd;
      iAri_ii[Type.ComplexFloat][Type.Float] = iArithmetic_internal_cftf;
      iAri_ii[Type.ComplexFloat][Type.Int64] = iArithmetic_internal_cfti64;
      iAri_ii[Type.ComplexFloat][Type.Uint64] = iArithmetic_internal_cftu64;
      iAri_ii[Type.ComplexFloat][Type.Int32] = iArithmetic_internal_cfti32;
      iAri_ii[Type.ComplexFloat][Type.Uint32] = iArithmetic_internal_cftu32;
      iAri_ii[Type.ComplexFloat][Type.Uint16] = iArithmetic_internal_cftu16;
      iAri_ii[Type.ComplexFloat][Type.Int16] = iArithmetic_internal_cfti16;
      iAri_ii[Type.ComplexFloat][Type.Bool] = iArithmetic_internal_cftb;

      iAri_ii[Type.Double][Type.ComplexDouble] = iArithmetic_internal_dtcd;
      iAri_ii[Type.Double][Type.ComplexFloat] = iArithmetic_internal_dtcf;
      iAri_ii[Type.Double][Type.Double] = iArithmetic_internal_dtd;
      iAri_ii[Type.Double][Type.Float] = iArithmetic_internal_dtf;
      iAri_ii[Type.Double][Type.Int64] = iArithmetic_internal_dti64;
      iAri_ii[Type.Double][Type.Uint64] = iArithmetic_internal_dtu64;
      iAri_ii[Type.Double][Type.Int32] = iArithmetic_internal_dti32;
      iAri_ii[Type.Double][Type.Uint32] = iArithmetic_internal_dtu32;
      iAri_ii[Type.Double][Type.Uint16] = iArithmetic_internal_dtu16;
      iAri_ii[Type.Double][Type.Int16] = iArithmetic_internal_dti16;
      iAri_ii[Type.Double][Type.Bool] = iArithmetic_internal_dtb;

      iAri_ii[Type.Float][Type.ComplexDouble] = iArithmetic_internal_ftcd;
      iAri_ii[Type.Float][Type.ComplexFloat] = iArithmetic_internal_ftcf;
      iAri_ii[Type.Float][Type.Double] = iArithmetic_internal_ftd;
      iAri_ii[Type.Float][Type.Float] = iArithmetic_internal_ftf;
      iAri_ii[Type.Float][Type.Int64] = iArithmetic_internal_fti64;
      iAri_ii[Type.Float][Type.Uint64] = iArithmetic_internal_ftu64;
      iAri_ii[Type.Float][Type.Int32] = iArithmetic_internal_fti32;
      iAri_ii[Type.Float][Type.Uint32] = iArithmetic_internal_ftu32;
      iAri_ii[Type.Float][Type.Uint16] = iArithmetic_internal_ftu16;
      iAri_ii[Type.Float][Type.Int16] = iArithmetic_internal_fti16;
      iAri_ii[Type.Float][Type.Bool] = iArithmetic_internal_ftb;

      iAri_ii[Type.Int64][Type.ComplexDouble] = iArithmetic_internal_i64tcd;
      iAri_ii[Type.Int64][Type.ComplexFloat] = iArithmetic_internal_i64tcf;
      iAri_ii[Type.Int64][Type.Double] = iArithmetic_internal_i64td;
      iAri_ii[Type.Int64][Type.Float] = iArithmetic_internal_i64tf;
      iAri_ii[Type.Int64][Type.Int64] = iArithmetic_internal_i64ti64;
      iAri_ii[Type.Int64][Type.Uint64] = iArithmetic_internal_i64tu64;
      iAri_ii[Type.Int64][Type.Int32] = iArithmetic_internal_i64ti32;
      iAri_ii[Type.Int64][Type.Uint32] = iArithmetic_internal_i64tu32;
      iAri_ii[Type.Int64][Type.Uint16] = iArithmetic_internal_i64tu16;
      iAri_ii[Type.Int64][Type.Int16] = iArithmetic_internal_i64ti16;
      iAri_ii[Type.Int64][Type.Bool] = iArithmetic_internal_i64tb;

      iAri_ii[Type.Uint64][Type.ComplexDouble] = iArithmetic_internal_u64tcd;
      iAri_ii[Type.Uint64][Type.ComplexFloat] = iArithmetic_internal_u64tcf;
      iAri_ii[Type.Uint64][Type.Double] = iArithmetic_internal_u64td;
      iAri_ii[Type.Uint64][Type.Float] = iArithmetic_internal_u64tf;
      iAri_ii[Type.Uint64][Type.Int64] = iArithmetic_internal_u64ti64;
      iAri_ii[Type.Uint64][Type.Uint64] = iArithmetic_internal_u64tu64;
      iAri_ii[Type.Uint64][Type.Int32] = iArithmetic_internal_u64ti32;
      iAri_ii[Type.Uint64][Type.Uint32] = iArithmetic_internal_u64tu32;
      iAri_ii[Type.Uint64][Type.Uint16] = iArithmetic_internal_u64tu16;
      iAri_ii[Type.Uint64][Type.Int16] = iArithmetic_internal_u64ti16;
      iAri_ii[Type.Uint64][Type.Bool] = iArithmetic_internal_u64tb;

      iAri_ii[Type.Int32][Type.ComplexDouble] = iArithmetic_internal_i32tcd;
      iAri_ii[Type.Int32][Type.ComplexFloat] = iArithmetic_internal_i32tcf;
      iAri_ii[Type.Int32][Type.Double] = iArithmetic_internal_i32td;
      iAri_ii[Type.Int32][Type.Float] = iArithmetic_internal_i32tf;
      iAri_ii[Type.Int32][Type.Int64] = iArithmetic_internal_i32ti64;
      iAri_ii[Type.Int32][Type.Uint64] = iArithmetic_internal_i32tu64;
      iAri_ii[Type.Int32][Type.Int32] = iArithmetic_internal_i32ti32;
      iAri_ii[Type.Int32][Type.Uint32] = iArithmetic_internal_i32tu32;
      iAri_ii[Type.Int32][Type.Uint16] = iArithmetic_internal_i32tu16;
      iAri_ii[Type.Int32][Type.Int16] = iArithmetic_internal_i32ti16;
      iAri_ii[Type.Int32][Type.Bool] = iArithmetic_internal_i32tb;

      iAri_ii[Type.Uint32][Type.ComplexDouble] = iArithmetic_internal_u32tcd;
      iAri_ii[Type.Uint32][Type.ComplexFloat] = iArithmetic_internal_u32tcf;
      iAri_ii[Type.Uint32][Type.Double] = iArithmetic_internal_u32td;
      iAri_ii[Type.Uint32][Type.Float] = iArithmetic_internal_u32tf;
      iAri_ii[Type.Uint32][Type.Int64] = iArithmetic_internal_u32ti64;
      iAri_ii[Type.Uint32][Type.Uint64] = iArithmetic_internal_u32tu64;
      iAri_ii[Type.Uint32][Type.Int32] = iArithmetic_internal_u32ti32;
      iAri_ii[Type.Uint32][Type.Uint32] = iArithmetic_internal_u32tu32;
      iAri_ii[Type.Uint32][Type.Uint16] = iArithmetic_internal_u32tu16;
      iAri_ii[Type.Uint32][Type.Int16] = iArithmetic_internal_u32ti16;
      iAri_ii[Type.Uint32][Type.Bool] = iArithmetic_internal_u32tb;

      iAri_ii[Type.Int16][Type.ComplexDouble] = iArithmetic_internal_i16tcd;
      iAri_ii[Type.Int16][Type.ComplexFloat] = iArithmetic_internal_i16tcf;
      iAri_ii[Type.Int16][Type.Double] = iArithmetic_internal_i16td;
      iAri_ii[Type.Int16][Type.Float] = iArithmetic_internal_i16tf;
      iAri_ii[Type.Int16][Type.Int64] = iArithmetic_internal_i16ti64;
      iAri_ii[Type.Int16][Type.Uint64] = iArithmetic_internal_i16tu64;
      iAri_ii[Type.Int16][Type.Int32] = iArithmetic_internal_i16ti32;
      iAri_ii[Type.Int16][Type.Uint32] = iArithmetic_internal_i16tu32;
      iAri_ii[Type.Int16][Type.Uint16] = iArithmetic_internal_i16tu16;
      iAri_ii[Type.Int16][Type.Int16] = iArithmetic_internal_i16ti16;
      iAri_ii[Type.Int16][Type.Bool] = iArithmetic_internal_i16tb;

      iAri_ii[Type.Uint16][Type.ComplexDouble] = iArithmetic_internal_u16tcd;
      iAri_ii[Type.Uint16][Type.ComplexFloat] = iArithmetic_internal_u16tcf;
      iAri_ii[Type.Uint16][Type.Double] = iArithmetic_internal_u16td;
      iAri_ii[Type.Uint16][Type.Float] = iArithmetic_internal_u16tf;
      iAri_ii[Type.Uint16][Type.Int64] = iArithmetic_internal_u16ti64;
      iAri_ii[Type.Uint16][Type.Uint64] = iArithmetic_internal_u16tu64;
      iAri_ii[Type.Uint16][Type.Int32] = iArithmetic_internal_u16ti32;
      iAri_ii[Type.Uint16][Type.Uint32] = iArithmetic_internal_u16tu32;
      iAri_ii[Type.Uint16][Type.Uint16] = iArithmetic_internal_u16tu16;
      iAri_ii[Type.Uint16][Type.Int16] = iArithmetic_internal_u16ti16;
      iAri_ii[Type.Uint16][Type.Bool] = iArithmetic_internal_u16tb;

      iAri_ii[Type.Bool][Type.ComplexDouble] = iArithmetic_internal_btcd;
      iAri_ii[Type.Bool][Type.ComplexFloat] = iArithmetic_internal_btcf;
      iAri_ii[Type.Bool][Type.Double] = iArithmetic_internal_btd;
      iAri_ii[Type.Bool][Type.Float] = iArithmetic_internal_btf;
      iAri_ii[Type.Bool][Type.Int64] = iArithmetic_internal_bti64;
      iAri_ii[Type.Bool][Type.Uint64] = iArithmetic_internal_btu64;
      iAri_ii[Type.Bool][Type.Int32] = iArithmetic_internal_bti32;
      iAri_ii[Type.Bool][Type.Uint32] = iArithmetic_internal_btu32;
      iAri_ii[Type.Bool][Type.Uint16] = iArithmetic_internal_btu16;
      iAri_ii[Type.Bool][Type.Int16] = iArithmetic_internal_bti16;
      iAri_ii[Type.Bool][Type.Bool] = iArithmetic_internal_btb;

      //=====================
      QR_ii = vector<Qrfunc_oii>(5);

      QR_ii[Type.ComplexDouble] = QR_internal_cd;
      QR_ii[Type.ComplexFloat] = QR_internal_cf;
      QR_ii[Type.Double] = QR_internal_d;
      QR_ii[Type.Float] = QR_internal_f;

      //=====================
      Svd_ii = vector<Svdfunc_oii>(5);

      Svd_ii[Type.ComplexDouble] = Svd_internal_cd;
      Svd_ii[Type.ComplexFloat] = Svd_internal_cf;
      Svd_ii[Type.Double] = Svd_internal_d;
      Svd_ii[Type.Float] = Svd_internal_f;

      //=====================
      Eigh_ii = vector<Eighfunc_oii>(5);

      Eigh_ii[Type.ComplexDouble] = Eigh_internal_cd;
      Eigh_ii[Type.ComplexFloat] = Eigh_internal_cf;
      Eigh_ii[Type.Double] = Eigh_internal_d;
      Eigh_ii[Type.Float] = Eigh_internal_f;

      //=====================
      Eig_ii = vector<Eighfunc_oii>(5);

      Eig_ii[Type.ComplexDouble] = Eig_internal_cd;
      Eig_ii[Type.ComplexFloat] = Eig_internal_cf;
      Eig_ii[Type.Double] = Eig_internal_d;
      Eig_ii[Type.Float] = Eig_internal_f;

      //=====================
      Exp_ii = vector<Expfunc_oii>(5);

      Exp_ii[Type.ComplexDouble] = Exp_internal_cd;
      Exp_ii[Type.ComplexFloat] = Exp_internal_cf;
      Exp_ii[Type.Double] = Exp_internal_d;
      Exp_ii[Type.Float] = Exp_internal_f;

      //=====================
      MM_ii = vector<MaxMinfunc_oii>(N_Type);

      MM_ii[Type.ComplexDouble] = MaxMin_internal_cd;
      MM_ii[Type.ComplexFloat] = MaxMin_internal_cf;
      MM_ii[Type.Double] = MaxMin_internal_d;
      MM_ii[Type.Float] = MaxMin_internal_f;
      MM_ii[Type.Uint64] = MaxMin_internal_u64;
      MM_ii[Type.Int64] = MaxMin_internal_i64;
      MM_ii[Type.Uint32] = MaxMin_internal_u32;
      MM_ii[Type.Int32] = MaxMin_internal_i32;
      MM_ii[Type.Uint16] = MaxMin_internal_u16;
      MM_ii[Type.Int16] = MaxMin_internal_i16;
      MM_ii[Type.Bool] = MaxMin_internal_b;

      //=====================
      Sum_ii = vector<MaxMinfunc_oii>(N_Type);

      Sum_ii[Type.ComplexDouble] = Sum_internal_cd;
      Sum_ii[Type.ComplexFloat] = Sum_internal_cf;
      Sum_ii[Type.Double] = Sum_internal_d;
      Sum_ii[Type.Float] = Sum_internal_f;
      Sum_ii[Type.Uint64] = Sum_internal_u64;
      Sum_ii[Type.Int64] = Sum_internal_i64;
      Sum_ii[Type.Uint32] = Sum_internal_u32;
      Sum_ii[Type.Int32] = Sum_internal_i32;
      Sum_ii[Type.Uint16] = Sum_internal_u16;
      Sum_ii[Type.Int16] = Sum_internal_i16;
      Sum_ii[Type.Bool] = Sum_internal_b;

      //=====================
      Pow_ii = vector<Powfunc_oii>(5);

      Pow_ii[Type.ComplexDouble] = Pow_internal_cd;
      Pow_ii[Type.ComplexFloat] = Pow_internal_cf;
      Pow_ii[Type.Double] = Pow_internal_d;
      Pow_ii[Type.Float] = Pow_internal_f;

      //=====================
      Abs_ii = vector<Absfunc_oii>(N_Type);

      Abs_ii[Type.ComplexDouble] = Abs_internal_cd;
      Abs_ii[Type.ComplexFloat] = Abs_internal_cf;
      Abs_ii[Type.Double] = Abs_internal_d;
      Abs_ii[Type.Float] = Abs_internal_f;
      Abs_ii[Type.Int64] = Abs_internal_i64;
      Abs_ii[Type.Uint64] = Abs_internal_pass;
      Abs_ii[Type.Int32] = Abs_internal_i32;
      Abs_ii[Type.Uint32] = Abs_internal_pass;
      Abs_ii[Type.Int16] = Abs_internal_i16;
      Abs_ii[Type.Uint16] = Abs_internal_pass;
      Abs_ii[Type.Bool] = Abs_internal_pass;

      //=====================
      Diag_ii = vector<Diagfunc_oii>(N_Type);

      Diag_ii[Type.ComplexDouble] = Diag_internal_cd;
      Diag_ii[Type.ComplexFloat] = Diag_internal_cf;
      Diag_ii[Type.Double] = Diag_internal_d;
      Diag_ii[Type.Float] = Diag_internal_f;
      Diag_ii[Type.Int64] = Diag_internal_i64;
      Diag_ii[Type.Uint64] = Diag_internal_u64;
      Diag_ii[Type.Int32] = Diag_internal_i32;
      Diag_ii[Type.Uint32] = Diag_internal_u32;
      Diag_ii[Type.Int16] = Diag_internal_i16;
      Diag_ii[Type.Uint16] = Diag_internal_u16;
      Diag_ii[Type.Bool] = Diag_internal_b;

      //=====================
      InvM_inplace_ii = vector<InvMinplacefunc_oii>(5);

      InvM_inplace_ii[Type.ComplexDouble] = InvM_inplace_internal_cd;
      InvM_inplace_ii[Type.ComplexFloat] = InvM_inplace_internal_cf;
      InvM_inplace_ii[Type.Double] = InvM_inplace_internal_d;
      InvM_inplace_ii[Type.Float] = InvM_inplace_internal_f;

      //=====================
      Inv_inplace_ii = vector<Invinplacefunc_oii>(5);

      Inv_inplace_ii[Type.ComplexDouble] = Inv_inplace_internal_cd;
      Inv_inplace_ii[Type.ComplexFloat] = Inv_inplace_internal_cf;
      Inv_inplace_ii[Type.Double] = Inv_inplace_internal_d;
      Inv_inplace_ii[Type.Float] = Inv_inplace_internal_f;

      //=====================
      Conj_inplace_ii = vector<Conjinplacefunc_oii>(3);

      Conj_inplace_ii[Type.ComplexDouble] = Conj_inplace_internal_cd;
      Conj_inplace_ii[Type.ComplexFloat] = Conj_inplace_internal_cf;

      //=====================
      Matmul_ii = vector<Matmulfunc_oii>(N_Type);
      Matmul_ii[Type.ComplexDouble] = Matmul_internal_cd;
      Matmul_ii[Type.ComplexFloat] = Matmul_internal_cf;
      Matmul_ii[Type.Double] = Matmul_internal_d;
      Matmul_ii[Type.Float] = Matmul_internal_f;
      Matmul_ii[Type.Int64] = Matmul_internal_i64;
      Matmul_ii[Type.Uint64] = Matmul_internal_u64;
      Matmul_ii[Type.Int32] = Matmul_internal_i32;
      Matmul_ii[Type.Uint32] = Matmul_internal_u32;
      Matmul_ii[Type.Int16] = Matmul_internal_i16;
      Matmul_ii[Type.Uint16] = Matmul_internal_u16;
      Matmul_ii[Type.Bool] = Matmul_internal_b;

      //=====================
      Matmul_dg_ii = vector<Matmul_dgfunc_oii>(N_Type);
      Matmul_dg_ii[Type.ComplexDouble] = Matmul_dg_internal_cd;
      Matmul_dg_ii[Type.ComplexFloat] = Matmul_dg_internal_cf;
      Matmul_dg_ii[Type.Double] = Matmul_dg_internal_d;
      Matmul_dg_ii[Type.Float] = Matmul_dg_internal_f;
      Matmul_dg_ii[Type.Int64] = Matmul_dg_internal_i64;
      Matmul_dg_ii[Type.Uint64] = Matmul_dg_internal_u64;
      Matmul_dg_ii[Type.Int32] = Matmul_dg_internal_i32;
      Matmul_dg_ii[Type.Uint32] = Matmul_dg_internal_u32;
      Matmul_dg_ii[Type.Int16] = Matmul_dg_internal_i16;
      Matmul_dg_ii[Type.Uint16] = Matmul_dg_internal_u16;
      Matmul_dg_ii[Type.Bool] = Matmul_dg_internal_b;

      //=====================
      Matvec_ii = vector<Matvecfunc_oii>(N_Type);
      Matvec_ii[Type.ComplexDouble] = Matvec_internal_cd;
      Matvec_ii[Type.ComplexFloat] = Matvec_internal_cf;
      Matvec_ii[Type.Double] = Matvec_internal_d;
      Matvec_ii[Type.Float] = Matvec_internal_f;
      Matvec_ii[Type.Int64] = Matvec_internal_i64;
      Matvec_ii[Type.Uint64] = Matvec_internal_u64;
      Matvec_ii[Type.Int32] = Matvec_internal_i32;
      Matvec_ii[Type.Uint32] = Matvec_internal_u32;
      Matvec_ii[Type.Int16] = Matvec_internal_i16;
      Matvec_ii[Type.Uint16] = Matvec_internal_u16;
      Matvec_ii[Type.Bool] = Matvec_internal_b;

      //===================
      Norm_ii = vector<Normfunc_oii>(5);
      Norm_ii[Type.ComplexDouble] = Norm_internal_cd;
      Norm_ii[Type.ComplexFloat] = Norm_internal_cf;
      Norm_ii[Type.Double] = Norm_internal_d;
      Norm_ii[Type.Float] = Norm_internal_f;

      //===================
      Det_ii = vector<Detfunc_oii>(5);
      Det_ii[Type.ComplexDouble] = Det_internal_cd;
      Det_ii[Type.ComplexFloat] = Det_internal_cf;
      Det_ii[Type.Double] = Det_internal_d;
      Det_ii[Type.Float] = Det_internal_f;

      //====================
      Vd_ii = vector<Vectordotfunc_oii>(N_Type);
      Vd_ii[Type.ComplexDouble] = Vectordot_internal_cd;
      Vd_ii[Type.ComplexFloat] = Vectordot_internal_cf;
      Vd_ii[Type.Double] = Vectordot_internal_d;
      Vd_ii[Type.Float] = Vectordot_internal_f;
      Vd_ii[Type.Int64] = Vectordot_internal_i64;
      Vd_ii[Type.Uint64] = Vectordot_internal_u64;
      Vd_ii[Type.Int32] = Vectordot_internal_i32;
      Vd_ii[Type.Uint32] = Vectordot_internal_u32;
      Vd_ii[Type.Int16] = Vectordot_internal_i16;
      Vd_ii[Type.Uint16] = Vectordot_internal_u16;
      Vd_ii[Type.Bool] = Vectordot_internal_b;

      //====================
      Td_ii = vector<Tdfunc_oii>(N_Type);
      Td_ii[Type.Double] = Tridiag_internal_d;
      Td_ii[Type.Float] = Tridiag_internal_f;

      //================
      Kron_ii = vector<vector<Kronfunc_oii>>(N_Type, vector<Kronfunc_oii>(N_Type, NULL));

      Kron_ii[Type.ComplexDouble][Type.ComplexDouble] = Kron_internal_cdtcd;
      Kron_ii[Type.ComplexDouble][Type.ComplexFloat] = Kron_internal_cdtcf;
      Kron_ii[Type.ComplexDouble][Type.Double] = Kron_internal_cdtd;
      Kron_ii[Type.ComplexDouble][Type.Float] = Kron_internal_cdtf;
      Kron_ii[Type.ComplexDouble][Type.Int64] = Kron_internal_cdti64;
      Kron_ii[Type.ComplexDouble][Type.Uint64] = Kron_internal_cdtu64;
      Kron_ii[Type.ComplexDouble][Type.Int32] = Kron_internal_cdti32;
      Kron_ii[Type.ComplexDouble][Type.Uint32] = Kron_internal_cdtu32;
      Kron_ii[Type.ComplexDouble][Type.Int16] = Kron_internal_cdti16;
      Kron_ii[Type.ComplexDouble][Type.Uint16] = Kron_internal_cdtu16;
      Kron_ii[Type.ComplexDouble][Type.Bool] = Kron_internal_cdtb;

      Kron_ii[Type.ComplexFloat][Type.ComplexDouble] = Kron_internal_cftcd;
      Kron_ii[Type.ComplexFloat][Type.ComplexFloat] = Kron_internal_cftcf;
      Kron_ii[Type.ComplexFloat][Type.Double] = Kron_internal_cftd;
      Kron_ii[Type.ComplexFloat][Type.Float] = Kron_internal_cftf;
      Kron_ii[Type.ComplexFloat][Type.Int64] = Kron_internal_cfti64;
      Kron_ii[Type.ComplexFloat][Type.Uint64] = Kron_internal_cftu64;
      Kron_ii[Type.ComplexFloat][Type.Int32] = Kron_internal_cfti32;
      Kron_ii[Type.ComplexFloat][Type.Uint32] = Kron_internal_cftu32;
      Kron_ii[Type.ComplexFloat][Type.Int16] = Kron_internal_cfti16;
      Kron_ii[Type.ComplexFloat][Type.Uint16] = Kron_internal_cftu16;
      Kron_ii[Type.ComplexFloat][Type.Bool] = Kron_internal_cftb;

      Kron_ii[Type.Double][Type.ComplexDouble] = Kron_internal_dtcd;
      Kron_ii[Type.Double][Type.ComplexFloat] = Kron_internal_dtcf;
      Kron_ii[Type.Double][Type.Double] = Kron_internal_dtd;
      Kron_ii[Type.Double][Type.Float] = Kron_internal_dtf;
      Kron_ii[Type.Double][Type.Int64] = Kron_internal_dti64;
      Kron_ii[Type.Double][Type.Uint64] = Kron_internal_dtu64;
      Kron_ii[Type.Double][Type.Int32] = Kron_internal_dti32;
      Kron_ii[Type.Double][Type.Uint32] = Kron_internal_dtu32;
      Kron_ii[Type.Double][Type.Int16] = Kron_internal_dti16;
      Kron_ii[Type.Double][Type.Uint16] = Kron_internal_dtu16;
      Kron_ii[Type.Double][Type.Bool] = Kron_internal_dtb;

      Kron_ii[Type.Float][Type.ComplexDouble] = Kron_internal_ftcd;
      Kron_ii[Type.Float][Type.ComplexFloat] = Kron_internal_ftcf;
      Kron_ii[Type.Float][Type.Double] = Kron_internal_ftd;
      Kron_ii[Type.Float][Type.Float] = Kron_internal_ftf;
      Kron_ii[Type.Float][Type.Int64] = Kron_internal_fti64;
      Kron_ii[Type.Float][Type.Uint64] = Kron_internal_ftu64;
      Kron_ii[Type.Float][Type.Int32] = Kron_internal_fti32;
      Kron_ii[Type.Float][Type.Uint32] = Kron_internal_ftu32;
      Kron_ii[Type.Float][Type.Uint16] = Kron_internal_ftu16;
      Kron_ii[Type.Float][Type.Int16] = Kron_internal_fti16;
      Kron_ii[Type.Float][Type.Bool] = Kron_internal_ftb;

      Kron_ii[Type.Int64][Type.ComplexDouble] = Kron_internal_i64tcd;
      Kron_ii[Type.Int64][Type.ComplexFloat] = Kron_internal_i64tcf;
      Kron_ii[Type.Int64][Type.Double] = Kron_internal_i64td;
      Kron_ii[Type.Int64][Type.Float] = Kron_internal_i64tf;
      Kron_ii[Type.Int64][Type.Int64] = Kron_internal_i64ti64;
      Kron_ii[Type.Int64][Type.Uint64] = Kron_internal_i64tu64;
      Kron_ii[Type.Int64][Type.Int32] = Kron_internal_i64ti32;
      Kron_ii[Type.Int64][Type.Uint32] = Kron_internal_i64tu32;
      Kron_ii[Type.Int64][Type.Uint16] = Kron_internal_i64tu16;
      Kron_ii[Type.Int64][Type.Int16] = Kron_internal_i64ti16;
      Kron_ii[Type.Int64][Type.Bool] = Kron_internal_i64tb;

      Kron_ii[Type.Uint64][Type.ComplexDouble] = Kron_internal_u64tcd;
      Kron_ii[Type.Uint64][Type.ComplexFloat] = Kron_internal_u64tcf;
      Kron_ii[Type.Uint64][Type.Double] = Kron_internal_u64td;
      Kron_ii[Type.Uint64][Type.Float] = Kron_internal_u64tf;
      Kron_ii[Type.Uint64][Type.Int64] = Kron_internal_u64ti64;
      Kron_ii[Type.Uint64][Type.Uint64] = Kron_internal_u64tu64;
      Kron_ii[Type.Uint64][Type.Int32] = Kron_internal_u64ti32;
      Kron_ii[Type.Uint64][Type.Uint32] = Kron_internal_u64tu32;
      Kron_ii[Type.Uint64][Type.Uint16] = Kron_internal_u64tu16;
      Kron_ii[Type.Uint64][Type.Int16] = Kron_internal_u64ti16;
      Kron_ii[Type.Uint64][Type.Bool] = Kron_internal_u64tb;

      Kron_ii[Type.Int32][Type.ComplexDouble] = Kron_internal_i32tcd;
      Kron_ii[Type.Int32][Type.ComplexFloat] = Kron_internal_i32tcf;
      Kron_ii[Type.Int32][Type.Double] = Kron_internal_i32td;
      Kron_ii[Type.Int32][Type.Float] = Kron_internal_i32tf;
      Kron_ii[Type.Int32][Type.Int64] = Kron_internal_i32ti64;
      Kron_ii[Type.Int32][Type.Uint64] = Kron_internal_i32tu64;
      Kron_ii[Type.Int32][Type.Int32] = Kron_internal_i32ti32;
      Kron_ii[Type.Int32][Type.Uint32] = Kron_internal_i32tu32;
      Kron_ii[Type.Int32][Type.Uint16] = Kron_internal_i32tu16;
      Kron_ii[Type.Int32][Type.Int16] = Kron_internal_i32ti16;
      Kron_ii[Type.Int32][Type.Bool] = Kron_internal_i32tb;

      Kron_ii[Type.Uint32][Type.ComplexDouble] = Kron_internal_u32tcd;
      Kron_ii[Type.Uint32][Type.ComplexFloat] = Kron_internal_u32tcf;
      Kron_ii[Type.Uint32][Type.Double] = Kron_internal_u32td;
      Kron_ii[Type.Uint32][Type.Float] = Kron_internal_u32tf;
      Kron_ii[Type.Uint32][Type.Int64] = Kron_internal_u32ti64;
      Kron_ii[Type.Uint32][Type.Uint64] = Kron_internal_u32tu64;
      Kron_ii[Type.Uint32][Type.Int32] = Kron_internal_u32ti32;
      Kron_ii[Type.Uint32][Type.Uint32] = Kron_internal_u32tu32;
      Kron_ii[Type.Uint32][Type.Uint16] = Kron_internal_u32tu16;
      Kron_ii[Type.Uint32][Type.Int16] = Kron_internal_u32ti16;
      Kron_ii[Type.Uint32][Type.Bool] = Kron_internal_u32tb;

      //================
      Outer_ii = vector<vector<Outerfunc_oii>>(N_Type, vector<Outerfunc_oii>(N_Type, NULL));

      Outer_ii[Type.ComplexDouble][Type.ComplexDouble] = Outer_internal_cdtcd;
      Outer_ii[Type.ComplexDouble][Type.ComplexFloat] = Outer_internal_cdtcf;
      Outer_ii[Type.ComplexDouble][Type.Double] = Outer_internal_cdtd;
      Outer_ii[Type.ComplexDouble][Type.Float] = Outer_internal_cdtf;
      Outer_ii[Type.ComplexDouble][Type.Int64] = Outer_internal_cdti64;
      Outer_ii[Type.ComplexDouble][Type.Uint64] = Outer_internal_cdtu64;
      Outer_ii[Type.ComplexDouble][Type.Int32] = Outer_internal_cdti32;
      Outer_ii[Type.ComplexDouble][Type.Uint32] = Outer_internal_cdtu32;
      Outer_ii[Type.ComplexDouble][Type.Int16] = Outer_internal_cdti16;
      Outer_ii[Type.ComplexDouble][Type.Uint16] = Outer_internal_cdtu16;
      Outer_ii[Type.ComplexDouble][Type.Bool] = Outer_internal_cdtb;

      Outer_ii[Type.ComplexFloat][Type.ComplexDouble] = Outer_internal_cftcd;
      Outer_ii[Type.ComplexFloat][Type.ComplexFloat] = Outer_internal_cftcf;
      Outer_ii[Type.ComplexFloat][Type.Double] = Outer_internal_cftd;
      Outer_ii[Type.ComplexFloat][Type.Float] = Outer_internal_cftf;
      Outer_ii[Type.ComplexFloat][Type.Int64] = Outer_internal_cfti64;
      Outer_ii[Type.ComplexFloat][Type.Uint64] = Outer_internal_cftu64;
      Outer_ii[Type.ComplexFloat][Type.Int32] = Outer_internal_cfti32;
      Outer_ii[Type.ComplexFloat][Type.Uint32] = Outer_internal_cftu32;
      Outer_ii[Type.ComplexFloat][Type.Int16] = Outer_internal_cfti16;
      Outer_ii[Type.ComplexFloat][Type.Uint16] = Outer_internal_cftu16;
      Outer_ii[Type.ComplexFloat][Type.Bool] = Outer_internal_cftb;

      Outer_ii[Type.Double][Type.ComplexDouble] = Outer_internal_dtcd;
      Outer_ii[Type.Double][Type.ComplexFloat] = Outer_internal_dtcf;
      Outer_ii[Type.Double][Type.Double] = Outer_internal_dtd;
      Outer_ii[Type.Double][Type.Float] = Outer_internal_dtf;
      Outer_ii[Type.Double][Type.Int64] = Outer_internal_dti64;
      Outer_ii[Type.Double][Type.Uint64] = Outer_internal_dtu64;
      Outer_ii[Type.Double][Type.Int32] = Outer_internal_dti32;
      Outer_ii[Type.Double][Type.Uint32] = Outer_internal_dtu32;
      Outer_ii[Type.Double][Type.Int16] = Outer_internal_dti16;
      Outer_ii[Type.Double][Type.Uint16] = Outer_internal_dtu16;
      Outer_ii[Type.Double][Type.Bool] = Outer_internal_dtb;

      Outer_ii[Type.Float][Type.ComplexDouble] = Outer_internal_ftcd;
      Outer_ii[Type.Float][Type.ComplexFloat] = Outer_internal_ftcf;
      Outer_ii[Type.Float][Type.Double] = Outer_internal_ftd;
      Outer_ii[Type.Float][Type.Float] = Outer_internal_ftf;
      Outer_ii[Type.Float][Type.Int64] = Outer_internal_fti64;
      Outer_ii[Type.Float][Type.Uint64] = Outer_internal_ftu64;
      Outer_ii[Type.Float][Type.Int32] = Outer_internal_fti32;
      Outer_ii[Type.Float][Type.Uint32] = Outer_internal_ftu32;
      Outer_ii[Type.Float][Type.Uint16] = Outer_internal_ftu16;
      Outer_ii[Type.Float][Type.Int16] = Outer_internal_fti16;
      Outer_ii[Type.Float][Type.Bool] = Outer_internal_ftb;

      Outer_ii[Type.Int64][Type.ComplexDouble] = Outer_internal_i64tcd;
      Outer_ii[Type.Int64][Type.ComplexFloat] = Outer_internal_i64tcf;
      Outer_ii[Type.Int64][Type.Double] = Outer_internal_i64td;
      Outer_ii[Type.Int64][Type.Float] = Outer_internal_i64tf;
      Outer_ii[Type.Int64][Type.Int64] = Outer_internal_i64ti64;
      Outer_ii[Type.Int64][Type.Uint64] = Outer_internal_i64tu64;
      Outer_ii[Type.Int64][Type.Int32] = Outer_internal_i64ti32;
      Outer_ii[Type.Int64][Type.Uint32] = Outer_internal_i64tu32;
      Outer_ii[Type.Int64][Type.Uint16] = Outer_internal_i64tu16;
      Outer_ii[Type.Int64][Type.Int16] = Outer_internal_i64ti16;
      Outer_ii[Type.Int64][Type.Bool] = Outer_internal_i64tb;

      Outer_ii[Type.Uint64][Type.ComplexDouble] = Outer_internal_u64tcd;
      Outer_ii[Type.Uint64][Type.ComplexFloat] = Outer_internal_u64tcf;
      Outer_ii[Type.Uint64][Type.Double] = Outer_internal_u64td;
      Outer_ii[Type.Uint64][Type.Float] = Outer_internal_u64tf;
      Outer_ii[Type.Uint64][Type.Int64] = Outer_internal_u64ti64;
      Outer_ii[Type.Uint64][Type.Uint64] = Outer_internal_u64tu64;
      Outer_ii[Type.Uint64][Type.Int32] = Outer_internal_u64ti32;
      Outer_ii[Type.Uint64][Type.Uint32] = Outer_internal_u64tu32;
      Outer_ii[Type.Uint64][Type.Uint16] = Outer_internal_u64tu16;
      Outer_ii[Type.Uint64][Type.Int16] = Outer_internal_u64ti16;
      Outer_ii[Type.Uint64][Type.Bool] = Outer_internal_u64tb;

      Outer_ii[Type.Int32][Type.ComplexDouble] = Outer_internal_i32tcd;
      Outer_ii[Type.Int32][Type.ComplexFloat] = Outer_internal_i32tcf;
      Outer_ii[Type.Int32][Type.Double] = Outer_internal_i32td;
      Outer_ii[Type.Int32][Type.Float] = Outer_internal_i32tf;
      Outer_ii[Type.Int32][Type.Int64] = Outer_internal_i32ti64;
      Outer_ii[Type.Int32][Type.Uint64] = Outer_internal_i32tu64;
      Outer_ii[Type.Int32][Type.Int32] = Outer_internal_i32ti32;
      Outer_ii[Type.Int32][Type.Uint32] = Outer_internal_i32tu32;
      Outer_ii[Type.Int32][Type.Uint16] = Outer_internal_i32tu16;
      Outer_ii[Type.Int32][Type.Int16] = Outer_internal_i32ti16;
      Outer_ii[Type.Int32][Type.Bool] = Outer_internal_i32tb;

      Outer_ii[Type.Uint32][Type.ComplexDouble] = Outer_internal_u32tcd;
      Outer_ii[Type.Uint32][Type.ComplexFloat] = Outer_internal_u32tcf;
      Outer_ii[Type.Uint32][Type.Double] = Outer_internal_u32td;
      Outer_ii[Type.Uint32][Type.Float] = Outer_internal_u32tf;
      Outer_ii[Type.Uint32][Type.Int64] = Outer_internal_u32ti64;
      Outer_ii[Type.Uint32][Type.Uint64] = Outer_internal_u32tu64;
      Outer_ii[Type.Uint32][Type.Int32] = Outer_internal_u32ti32;
      Outer_ii[Type.Uint32][Type.Uint32] = Outer_internal_u32tu32;
      Outer_ii[Type.Uint32][Type.Uint16] = Outer_internal_u32tu16;
      Outer_ii[Type.Uint32][Type.Int16] = Outer_internal_u32ti16;
      Outer_ii[Type.Uint32][Type.Bool] = Outer_internal_u32tb;

      //================

      Lstsq_ii = std::vector<Lstsqfunc_oii>(5);
      Lstsq_ii[Type.ComplexDouble] = Lstsq_internal_cd;
      Lstsq_ii[Type.ComplexFloat] = Lstsq_internal_cf;
      Lstsq_ii[Type.Double] = Lstsq_internal_d;
      Lstsq_ii[Type.Float] = Lstsq_internal_f;

#ifdef UNI_GPU
      cuAri_ii = vector<vector<Arithmeticfunc_oii>>(N_Type, vector<Arithmeticfunc_oii>(N_Type));

      cuAri_ii[Type.ComplexDouble][Type.ComplexDouble] = cuArithmetic_internal_cdtcd;
      cuAri_ii[Type.ComplexDouble][Type.ComplexFloat] = cuArithmetic_internal_cdtcf;
      cuAri_ii[Type.ComplexDouble][Type.Double] = cuArithmetic_internal_cdtd;
      cuAri_ii[Type.ComplexDouble][Type.Float] = cuArithmetic_internal_cdtf;
      cuAri_ii[Type.ComplexDouble][Type.Int64] = cuArithmetic_internal_cdti64;
      cuAri_ii[Type.ComplexDouble][Type.Uint64] = cuArithmetic_internal_cdtu64;
      cuAri_ii[Type.ComplexDouble][Type.Int32] = cuArithmetic_internal_cdti32;
      cuAri_ii[Type.ComplexDouble][Type.Uint32] = cuArithmetic_internal_cdtu32;
      cuAri_ii[Type.ComplexDouble][Type.Int16] = cuArithmetic_internal_cdti16;
      cuAri_ii[Type.ComplexDouble][Type.Uint16] = cuArithmetic_internal_cdtu16;
      cuAri_ii[Type.ComplexDouble][Type.Bool] = cuArithmetic_internal_cdtb;

      cuAri_ii[Type.ComplexFloat][Type.ComplexDouble] = cuArithmetic_internal_cftcd;
      cuAri_ii[Type.ComplexFloat][Type.ComplexFloat] = cuArithmetic_internal_cftcf;
      cuAri_ii[Type.ComplexFloat][Type.Double] = cuArithmetic_internal_cftd;
      cuAri_ii[Type.ComplexFloat][Type.Float] = cuArithmetic_internal_cftf;
      cuAri_ii[Type.ComplexFloat][Type.Int64] = cuArithmetic_internal_cfti64;
      cuAri_ii[Type.ComplexFloat][Type.Uint64] = cuArithmetic_internal_cftu64;
      cuAri_ii[Type.ComplexFloat][Type.Int32] = cuArithmetic_internal_cfti32;
      cuAri_ii[Type.ComplexFloat][Type.Uint32] = cuArithmetic_internal_cftu32;
      cuAri_ii[Type.ComplexFloat][Type.Int16] = cuArithmetic_internal_cfti16;
      cuAri_ii[Type.ComplexFloat][Type.Uint16] = cuArithmetic_internal_cftu16;
      cuAri_ii[Type.ComplexFloat][Type.Bool] = cuArithmetic_internal_cftb;

      cuAri_ii[Type.Double][Type.ComplexDouble] = cuArithmetic_internal_dtcd;
      cuAri_ii[Type.Double][Type.ComplexFloat] = cuArithmetic_internal_dtcf;
      cuAri_ii[Type.Double][Type.Double] = cuArithmetic_internal_dtd;
      cuAri_ii[Type.Double][Type.Float] = cuArithmetic_internal_dtf;
      cuAri_ii[Type.Double][Type.Int64] = cuArithmetic_internal_dti64;
      cuAri_ii[Type.Double][Type.Uint64] = cuArithmetic_internal_dtu64;
      cuAri_ii[Type.Double][Type.Int32] = cuArithmetic_internal_dti32;
      cuAri_ii[Type.Double][Type.Uint32] = cuArithmetic_internal_dtu32;
      cuAri_ii[Type.Double][Type.Uint16] = cuArithmetic_internal_dtu16;
      cuAri_ii[Type.Double][Type.Int16] = cuArithmetic_internal_dti16;
      cuAri_ii[Type.Double][Type.Bool] = cuArithmetic_internal_dtb;

      cuAri_ii[Type.Float][Type.ComplexDouble] = cuArithmetic_internal_ftcd;
      cuAri_ii[Type.Float][Type.ComplexFloat] = cuArithmetic_internal_ftcf;
      cuAri_ii[Type.Float][Type.Double] = cuArithmetic_internal_ftd;
      cuAri_ii[Type.Float][Type.Float] = cuArithmetic_internal_ftf;
      cuAri_ii[Type.Float][Type.Int64] = cuArithmetic_internal_fti64;
      cuAri_ii[Type.Float][Type.Uint64] = cuArithmetic_internal_ftu64;
      cuAri_ii[Type.Float][Type.Int32] = cuArithmetic_internal_fti32;
      cuAri_ii[Type.Float][Type.Uint32] = cuArithmetic_internal_ftu32;
      cuAri_ii[Type.Float][Type.Uint16] = cuArithmetic_internal_ftu16;
      cuAri_ii[Type.Float][Type.Int16] = cuArithmetic_internal_fti16;
      cuAri_ii[Type.Float][Type.Bool] = cuArithmetic_internal_ftb;

      cuAri_ii[Type.Int64][Type.ComplexDouble] = cuArithmetic_internal_i64tcd;
      cuAri_ii[Type.Int64][Type.ComplexFloat] = cuArithmetic_internal_i64tcf;
      cuAri_ii[Type.Int64][Type.Double] = cuArithmetic_internal_i64td;
      cuAri_ii[Type.Int64][Type.Float] = cuArithmetic_internal_i64tf;
      cuAri_ii[Type.Int64][Type.Int64] = cuArithmetic_internal_i64ti64;
      cuAri_ii[Type.Int64][Type.Uint64] = cuArithmetic_internal_i64tu64;
      cuAri_ii[Type.Int64][Type.Int32] = cuArithmetic_internal_i64ti32;
      cuAri_ii[Type.Int64][Type.Uint32] = cuArithmetic_internal_i64tu32;
      cuAri_ii[Type.Int64][Type.Uint16] = cuArithmetic_internal_i64tu16;
      cuAri_ii[Type.Int64][Type.Int16] = cuArithmetic_internal_i64ti16;
      cuAri_ii[Type.Int64][Type.Bool] = cuArithmetic_internal_i64tb;

      cuAri_ii[Type.Uint64][Type.ComplexDouble] = cuArithmetic_internal_u64tcd;
      cuAri_ii[Type.Uint64][Type.ComplexFloat] = cuArithmetic_internal_u64tcf;
      cuAri_ii[Type.Uint64][Type.Double] = cuArithmetic_internal_u64td;
      cuAri_ii[Type.Uint64][Type.Float] = cuArithmetic_internal_u64tf;
      cuAri_ii[Type.Uint64][Type.Int64] = cuArithmetic_internal_u64ti64;
      cuAri_ii[Type.Uint64][Type.Uint64] = cuArithmetic_internal_u64tu64;
      cuAri_ii[Type.Uint64][Type.Int32] = cuArithmetic_internal_u64ti32;
      cuAri_ii[Type.Uint64][Type.Uint32] = cuArithmetic_internal_u64tu32;
      cuAri_ii[Type.Uint64][Type.Uint16] = cuArithmetic_internal_u64tu16;
      cuAri_ii[Type.Uint64][Type.Int16] = cuArithmetic_internal_u64ti16;
      cuAri_ii[Type.Uint64][Type.Bool] = cuArithmetic_internal_u64tb;

      cuAri_ii[Type.Int32][Type.ComplexDouble] = cuArithmetic_internal_i32tcd;
      cuAri_ii[Type.Int32][Type.ComplexFloat] = cuArithmetic_internal_i32tcf;
      cuAri_ii[Type.Int32][Type.Double] = cuArithmetic_internal_i32td;
      cuAri_ii[Type.Int32][Type.Float] = cuArithmetic_internal_i32tf;
      cuAri_ii[Type.Int32][Type.Int64] = cuArithmetic_internal_i32ti64;
      cuAri_ii[Type.Int32][Type.Uint64] = cuArithmetic_internal_i32tu64;
      cuAri_ii[Type.Int32][Type.Int32] = cuArithmetic_internal_i32ti32;
      cuAri_ii[Type.Int32][Type.Uint32] = cuArithmetic_internal_i32tu32;
      cuAri_ii[Type.Int32][Type.Uint16] = cuArithmetic_internal_i32tu16;
      cuAri_ii[Type.Int32][Type.Int16] = cuArithmetic_internal_i32ti16;
      cuAri_ii[Type.Int32][Type.Bool] = cuArithmetic_internal_i32tb;

      cuAri_ii[Type.Uint32][Type.ComplexDouble] = cuArithmetic_internal_u32tcd;
      cuAri_ii[Type.Uint32][Type.ComplexFloat] = cuArithmetic_internal_u32tcf;
      cuAri_ii[Type.Uint32][Type.Double] = cuArithmetic_internal_u32td;
      cuAri_ii[Type.Uint32][Type.Float] = cuArithmetic_internal_u32tf;
      cuAri_ii[Type.Uint32][Type.Int64] = cuArithmetic_internal_u32ti64;
      cuAri_ii[Type.Uint32][Type.Uint64] = cuArithmetic_internal_u32tu64;
      cuAri_ii[Type.Uint32][Type.Int32] = cuArithmetic_internal_u32ti32;
      cuAri_ii[Type.Uint32][Type.Uint32] = cuArithmetic_internal_u32tu32;
      cuAri_ii[Type.Uint32][Type.Uint16] = cuArithmetic_internal_u32tu16;
      cuAri_ii[Type.Uint32][Type.Int16] = cuArithmetic_internal_u32ti16;
      cuAri_ii[Type.Uint32][Type.Bool] = cuArithmetic_internal_u32tb;

      cuAri_ii[Type.Int16][Type.ComplexDouble] = cuArithmetic_internal_i16tcd;
      cuAri_ii[Type.Int16][Type.ComplexFloat] = cuArithmetic_internal_i16tcf;
      cuAri_ii[Type.Int16][Type.Double] = cuArithmetic_internal_i16td;
      cuAri_ii[Type.Int16][Type.Float] = cuArithmetic_internal_i16tf;
      cuAri_ii[Type.Int16][Type.Int64] = cuArithmetic_internal_i16ti64;
      cuAri_ii[Type.Int16][Type.Uint64] = cuArithmetic_internal_i16tu64;
      cuAri_ii[Type.Int16][Type.Int32] = cuArithmetic_internal_i16ti32;
      cuAri_ii[Type.Int16][Type.Uint32] = cuArithmetic_internal_i16tu32;
      cuAri_ii[Type.Int16][Type.Uint16] = cuArithmetic_internal_i16tu16;
      cuAri_ii[Type.Int16][Type.Int16] = cuArithmetic_internal_i16ti16;
      cuAri_ii[Type.Int16][Type.Bool] = cuArithmetic_internal_i16tb;

      cuAri_ii[Type.Uint16][Type.ComplexDouble] = cuArithmetic_internal_u16tcd;
      cuAri_ii[Type.Uint16][Type.ComplexFloat] = cuArithmetic_internal_u16tcf;
      cuAri_ii[Type.Uint16][Type.Double] = cuArithmetic_internal_u16td;
      cuAri_ii[Type.Uint16][Type.Float] = cuArithmetic_internal_u16tf;
      cuAri_ii[Type.Uint16][Type.Int64] = cuArithmetic_internal_u16ti64;
      cuAri_ii[Type.Uint16][Type.Uint64] = cuArithmetic_internal_u16tu64;
      cuAri_ii[Type.Uint16][Type.Int32] = cuArithmetic_internal_u16ti32;
      cuAri_ii[Type.Uint16][Type.Uint32] = cuArithmetic_internal_u16tu32;
      cuAri_ii[Type.Uint16][Type.Uint16] = cuArithmetic_internal_u16tu16;
      cuAri_ii[Type.Uint16][Type.Int16] = cuArithmetic_internal_u16ti16;
      cuAri_ii[Type.Uint16][Type.Bool] = cuArithmetic_internal_u16tb;

      cuAri_ii[Type.Bool][Type.ComplexDouble] = cuArithmetic_internal_btcd;
      cuAri_ii[Type.Bool][Type.ComplexFloat] = cuArithmetic_internal_btcf;
      cuAri_ii[Type.Bool][Type.Double] = cuArithmetic_internal_btd;
      cuAri_ii[Type.Bool][Type.Float] = cuArithmetic_internal_btf;
      cuAri_ii[Type.Bool][Type.Int64] = cuArithmetic_internal_bti64;
      cuAri_ii[Type.Bool][Type.Uint64] = cuArithmetic_internal_btu64;
      cuAri_ii[Type.Bool][Type.Int32] = cuArithmetic_internal_bti32;
      cuAri_ii[Type.Bool][Type.Uint32] = cuArithmetic_internal_btu32;
      cuAri_ii[Type.Bool][Type.Uint16] = cuArithmetic_internal_btu16;
      cuAri_ii[Type.Bool][Type.Int16] = cuArithmetic_internal_bti16;
      cuAri_ii[Type.Bool][Type.Bool] = cuArithmetic_internal_btb;

      // Pow
      //====================
      cuPow_ii = vector<Powfunc_oii>(5);

      cuPow_ii[Type.ComplexDouble] = cuPow_internal_cd;
      cuPow_ii[Type.ComplexFloat] = cuPow_internal_cf;
      cuPow_ii[Type.Double] = cuPow_internal_d;
      cuPow_ii[Type.Float] = cuPow_internal_f;

      // Norm
      //====================
      cuNorm_ii = vector<Normfunc_oii>(N_Type);
      cuNorm_ii[Type.ComplexDouble] = cuNorm_internal_cd;
      cuNorm_ii[Type.ComplexFloat] = cuNorm_internal_cf;
      cuNorm_ii[Type.Double] = cuNorm_internal_d;
      cuNorm_ii[Type.Float] = cuNorm_internal_f;

      // Svd
      cuSvd_ii = vector<Svdfunc_oii>(5);

      cuSvd_ii[Type.ComplexDouble] = cuSvd_internal_cd;
      cuSvd_ii[Type.ComplexFloat] = cuSvd_internal_cf;
      cuSvd_ii[Type.Double] = cuSvd_internal_d;
      cuSvd_ii[Type.Float] = cuSvd_internal_f;

      //=====================
      cuEigh_ii = vector<Eighfunc_oii>(5);

      cuEigh_ii[Type.ComplexDouble] = cuEigh_internal_cd;
      cuEigh_ii[Type.ComplexFloat] = cuEigh_internal_cf;
      cuEigh_ii[Type.Double] = cuEigh_internal_d;
      cuEigh_ii[Type.Float] = cuEigh_internal_f;

      //=====================
      cuExp_ii = vector<Expfunc_oii>(5);

      cuExp_ii[Type.ComplexDouble] = cuExp_internal_cd;
      cuExp_ii[Type.ComplexFloat] = cuExp_internal_cf;
      cuExp_ii[Type.Double] = cuExp_internal_d;
      cuExp_ii[Type.Float] = cuExp_internal_f;

      //=====================
      cuDiag_ii = vector<Diagfunc_oii>(N_Type);

      cuDiag_ii[Type.ComplexDouble] = cuDiag_internal_cd;
      cuDiag_ii[Type.ComplexFloat] = cuDiag_internal_cf;
      cuDiag_ii[Type.Double] = cuDiag_internal_d;
      cuDiag_ii[Type.Float] = cuDiag_internal_f;
      cuDiag_ii[Type.Int64] = cuDiag_internal_i64;
      cuDiag_ii[Type.Uint64] = cuDiag_internal_u64;
      cuDiag_ii[Type.Int32] = cuDiag_internal_i32;
      cuDiag_ii[Type.Uint32] = cuDiag_internal_u32;
      cuDiag_ii[Type.Uint16] = cuDiag_internal_u16;
      cuDiag_ii[Type.Int16] = cuDiag_internal_i16;
      cuDiag_ii[Type.Bool] = cuDiag_internal_b;
      //=====================
      cuInvM_inplace_ii = vector<InvMinplacefunc_oii>(5);

      cuInvM_inplace_ii[Type.ComplexDouble] = cuInvM_inplace_internal_cd;
      cuInvM_inplace_ii[Type.ComplexFloat] = cuInvM_inplace_internal_cf;
      cuInvM_inplace_ii[Type.Double] = cuInvM_inplace_internal_d;
      cuInvM_inplace_ii[Type.Float] = cuInvM_inplace_internal_f;

      //=====================
      cuInv_inplace_ii = vector<Invinplacefunc_oii>(5);

      cuInv_inplace_ii[Type.ComplexDouble] = cuInv_inplace_internal_cd;
      cuInv_inplace_ii[Type.ComplexFloat] = cuInv_inplace_internal_cf;
      cuInv_inplace_ii[Type.Double] = cuInv_inplace_internal_d;
      cuInv_inplace_ii[Type.Float] = cuInv_inplace_internal_f;

      //=====================
      cuConj_inplace_ii = vector<Conjinplacefunc_oii>(3);

      cuConj_inplace_ii[Type.ComplexDouble] = cuConj_inplace_internal_cd;
      cuConj_inplace_ii[Type.ComplexFloat] = cuConj_inplace_internal_cf;

      //=====================
      cuMatmul_ii = vector<Matmulfunc_oii>(N_Type);
      cuMatmul_ii[Type.ComplexDouble] = cuMatmul_internal_cd;
      cuMatmul_ii[Type.ComplexFloat] = cuMatmul_internal_cf;
      cuMatmul_ii[Type.Double] = cuMatmul_internal_d;
      cuMatmul_ii[Type.Float] = cuMatmul_internal_f;
      cuMatmul_ii[Type.Int64] = cuMatmul_internal_i64;
      cuMatmul_ii[Type.Uint64] = cuMatmul_internal_u64;
      cuMatmul_ii[Type.Int32] = cuMatmul_internal_i32;
      cuMatmul_ii[Type.Uint32] = cuMatmul_internal_u32;
      cuMatmul_ii[Type.Int16] = cuMatmul_internal_i16;
      cuMatmul_ii[Type.Uint16] = cuMatmul_internal_u16;
      cuMatmul_ii[Type.Bool] = cuMatmul_internal_b;

      //=====================
      cuMatmul_dg_ii = vector<Matmul_dgfunc_oii>(N_Type);
      cuMatmul_dg_ii[Type.ComplexDouble] = cuMatmul_dg_internal_cd;
      cuMatmul_dg_ii[Type.ComplexFloat] = cuMatmul_dg_internal_cf;
      cuMatmul_dg_ii[Type.Double] = cuMatmul_dg_internal_d;
      cuMatmul_dg_ii[Type.Float] = cuMatmul_dg_internal_f;
      cuMatmul_dg_ii[Type.Int64] = cuMatmul_dg_internal_i64;
      cuMatmul_dg_ii[Type.Uint64] = cuMatmul_dg_internal_u64;
      cuMatmul_dg_ii[Type.Int32] = cuMatmul_dg_internal_i32;
      cuMatmul_dg_ii[Type.Uint32] = cuMatmul_dg_internal_u32;
      cuMatmul_dg_ii[Type.Int16] = cuMatmul_dg_internal_i16;
      cuMatmul_dg_ii[Type.Uint16] = cuMatmul_dg_internal_u16;
      cuMatmul_dg_ii[Type.Bool] = cuMatmul_dg_internal_b;

      //=====================

      cuMatvec_ii = vector<Matvecfunc_oii>(N_Type);
      cuMatvec_ii[Type.ComplexDouble] = cuMatvec_internal_cd;
      cuMatvec_ii[Type.ComplexFloat] = cuMatvec_internal_cf;
      cuMatvec_ii[Type.Double] = cuMatvec_internal_d;
      cuMatvec_ii[Type.Float] = cuMatvec_internal_f;
      cuMatvec_ii[Type.Int64] = cuMatvec_internal_i64;
      cuMatvec_ii[Type.Uint64] = cuMatvec_internal_u64;
      cuMatvec_ii[Type.Int32] = cuMatvec_internal_i32;
      cuMatvec_ii[Type.Uint32] = cuMatvec_internal_u32;
      cuMatvec_ii[Type.Int16] = cuMatvec_internal_i16;
      cuMatvec_ii[Type.Uint16] = cuMatvec_internal_u16;
      cuMatvec_ii[Type.Bool] = cuMatvec_internal_b;

      //====================
      cuVd_ii = vector<Vectordotfunc_oii>(N_Type);
      cuVd_ii[Type.ComplexDouble] = cuVectordot_internal_cd;
      cuVd_ii[Type.ComplexFloat] = cuVectordot_internal_cf;
      cuVd_ii[Type.Double] = cuVectordot_internal_d;
      cuVd_ii[Type.Float] = cuVectordot_internal_f;
      cuVd_ii[Type.Int64] = cuVectordot_internal_i64;
      cuVd_ii[Type.Uint64] = cuVectordot_internal_u64;
      cuVd_ii[Type.Int32] = cuVectordot_internal_i32;
      cuVd_ii[Type.Uint32] = cuVectordot_internal_u32;
      cuVd_ii[Type.Int16] = cuVectordot_internal_i16;
      cuVd_ii[Type.Uint16] = cuVectordot_internal_u16;
      cuVd_ii[Type.Bool] = cuVectordot_internal_b;

      //================

      cuOuter_ii = vector<vector<Outerfunc_oii>>(N_Type, vector<Outerfunc_oii>(N_Type, NULL));

      cuOuter_ii[Type.ComplexDouble][Type.ComplexDouble] = cuOuter_internal_cdtcd;
      cuOuter_ii[Type.ComplexDouble][Type.ComplexFloat] = cuOuter_internal_cdtcf;
      cuOuter_ii[Type.ComplexDouble][Type.Double] = cuOuter_internal_cdtd;
      cuOuter_ii[Type.ComplexDouble][Type.Float] = cuOuter_internal_cdtf;
      cuOuter_ii[Type.ComplexDouble][Type.Int64] = cuOuter_internal_cdti64;
      cuOuter_ii[Type.ComplexDouble][Type.Uint64] = cuOuter_internal_cdtu64;
      cuOuter_ii[Type.ComplexDouble][Type.Int32] = cuOuter_internal_cdti32;
      cuOuter_ii[Type.ComplexDouble][Type.Uint32] = cuOuter_internal_cdtu32;
      cuOuter_ii[Type.ComplexDouble][Type.Uint16] = cuOuter_internal_cdtu16;
      cuOuter_ii[Type.ComplexDouble][Type.Int16] = cuOuter_internal_cdti16;
      cuOuter_ii[Type.ComplexDouble][Type.Bool] = cuOuter_internal_cdtb;

      cuOuter_ii[Type.ComplexFloat][Type.ComplexDouble] = cuOuter_internal_cftcd;
      cuOuter_ii[Type.ComplexFloat][Type.ComplexFloat] = cuOuter_internal_cftcf;
      cuOuter_ii[Type.ComplexFloat][Type.Double] = cuOuter_internal_cftd;
      cuOuter_ii[Type.ComplexFloat][Type.Float] = cuOuter_internal_cftf;
      cuOuter_ii[Type.ComplexFloat][Type.Int64] = cuOuter_internal_cfti64;
      cuOuter_ii[Type.ComplexFloat][Type.Uint64] = cuOuter_internal_cftu64;
      cuOuter_ii[Type.ComplexFloat][Type.Int32] = cuOuter_internal_cfti32;
      cuOuter_ii[Type.ComplexFloat][Type.Uint32] = cuOuter_internal_cftu32;
      cuOuter_ii[Type.ComplexFloat][Type.Uint16] = cuOuter_internal_cftu16;
      cuOuter_ii[Type.ComplexFloat][Type.Int16] = cuOuter_internal_cfti16;
      cuOuter_ii[Type.ComplexFloat][Type.Bool] = cuOuter_internal_cftb;

      cuOuter_ii[Type.Double][Type.ComplexDouble] = cuOuter_internal_dtcd;
      cuOuter_ii[Type.Double][Type.ComplexFloat] = cuOuter_internal_dtcf;
      cuOuter_ii[Type.Double][Type.Double] = cuOuter_internal_dtd;
      cuOuter_ii[Type.Double][Type.Float] = cuOuter_internal_dtf;
      cuOuter_ii[Type.Double][Type.Int64] = cuOuter_internal_dti64;
      cuOuter_ii[Type.Double][Type.Uint64] = cuOuter_internal_dtu64;
      cuOuter_ii[Type.Double][Type.Int32] = cuOuter_internal_dti32;
      cuOuter_ii[Type.Double][Type.Uint32] = cuOuter_internal_dtu32;
      cuOuter_ii[Type.Double][Type.Uint16] = cuOuter_internal_dtu16;
      cuOuter_ii[Type.Double][Type.Int16] = cuOuter_internal_dti16;
      cuOuter_ii[Type.Double][Type.Bool] = cuOuter_internal_dtb;

      cuOuter_ii[Type.Float][Type.ComplexDouble] = cuOuter_internal_ftcd;
      cuOuter_ii[Type.Float][Type.ComplexFloat] = cuOuter_internal_ftcf;
      cuOuter_ii[Type.Float][Type.Double] = cuOuter_internal_ftd;
      cuOuter_ii[Type.Float][Type.Float] = cuOuter_internal_ftf;
      cuOuter_ii[Type.Float][Type.Int64] = cuOuter_internal_fti64;
      cuOuter_ii[Type.Float][Type.Uint64] = cuOuter_internal_ftu64;
      cuOuter_ii[Type.Float][Type.Int32] = cuOuter_internal_fti32;
      cuOuter_ii[Type.Float][Type.Uint32] = cuOuter_internal_ftu32;
      cuOuter_ii[Type.Float][Type.Uint16] = cuOuter_internal_ftu16;
      cuOuter_ii[Type.Float][Type.Int16] = cuOuter_internal_fti16;
      cuOuter_ii[Type.Float][Type.Bool] = cuOuter_internal_ftb;

      cuOuter_ii[Type.Int64][Type.ComplexDouble] = cuOuter_internal_i64tcd;
      cuOuter_ii[Type.Int64][Type.ComplexFloat] = cuOuter_internal_i64tcf;
      cuOuter_ii[Type.Int64][Type.Double] = cuOuter_internal_i64td;
      cuOuter_ii[Type.Int64][Type.Float] = cuOuter_internal_i64tf;
      cuOuter_ii[Type.Int64][Type.Int64] = cuOuter_internal_i64ti64;
      cuOuter_ii[Type.Int64][Type.Uint64] = cuOuter_internal_i64tu64;
      cuOuter_ii[Type.Int64][Type.Int32] = cuOuter_internal_i64ti32;
      cuOuter_ii[Type.Int64][Type.Uint32] = cuOuter_internal_i64tu32;
      cuOuter_ii[Type.Int64][Type.Uint16] = cuOuter_internal_i64tu16;
      cuOuter_ii[Type.Int64][Type.Int16] = cuOuter_internal_i64ti16;
      cuOuter_ii[Type.Int64][Type.Bool] = cuOuter_internal_i64tb;

      cuOuter_ii[Type.Uint64][Type.ComplexDouble] = cuOuter_internal_u64tcd;
      cuOuter_ii[Type.Uint64][Type.ComplexFloat] = cuOuter_internal_u64tcf;
      cuOuter_ii[Type.Uint64][Type.Double] = cuOuter_internal_u64td;
      cuOuter_ii[Type.Uint64][Type.Float] = cuOuter_internal_u64tf;
      cuOuter_ii[Type.Uint64][Type.Int64] = cuOuter_internal_u64ti64;
      cuOuter_ii[Type.Uint64][Type.Uint64] = cuOuter_internal_u64tu64;
      cuOuter_ii[Type.Uint64][Type.Int32] = cuOuter_internal_u64ti32;
      cuOuter_ii[Type.Uint64][Type.Uint32] = cuOuter_internal_u64tu32;
      cuOuter_ii[Type.Uint64][Type.Uint16] = cuOuter_internal_u64tu16;
      cuOuter_ii[Type.Uint64][Type.Int16] = cuOuter_internal_u64ti16;
      cuOuter_ii[Type.Uint64][Type.Bool] = cuOuter_internal_u64tb;

      cuOuter_ii[Type.Int32][Type.ComplexDouble] = cuOuter_internal_i32tcd;
      cuOuter_ii[Type.Int32][Type.ComplexFloat] = cuOuter_internal_i32tcf;
      cuOuter_ii[Type.Int32][Type.Double] = cuOuter_internal_i32td;
      cuOuter_ii[Type.Int32][Type.Float] = cuOuter_internal_i32tf;
      cuOuter_ii[Type.Int32][Type.Int64] = cuOuter_internal_i32ti64;
      cuOuter_ii[Type.Int32][Type.Uint64] = cuOuter_internal_i32tu64;
      cuOuter_ii[Type.Int32][Type.Int32] = cuOuter_internal_i32ti32;
      cuOuter_ii[Type.Int32][Type.Uint32] = cuOuter_internal_i32tu32;
      cuOuter_ii[Type.Int32][Type.Uint16] = cuOuter_internal_i32tu16;
      cuOuter_ii[Type.Int32][Type.Int16] = cuOuter_internal_i32ti16;
      cuOuter_ii[Type.Int32][Type.Bool] = cuOuter_internal_i32tb;

      cuOuter_ii[Type.Uint32][Type.ComplexDouble] = cuOuter_internal_u32tcd;
      cuOuter_ii[Type.Uint32][Type.ComplexFloat] = cuOuter_internal_u32tcf;
      cuOuter_ii[Type.Uint32][Type.Double] = cuOuter_internal_u32td;
      cuOuter_ii[Type.Uint32][Type.Float] = cuOuter_internal_u32tf;
      cuOuter_ii[Type.Uint32][Type.Int64] = cuOuter_internal_u32ti64;
      cuOuter_ii[Type.Uint32][Type.Uint64] = cuOuter_internal_u32tu64;
      cuOuter_ii[Type.Uint32][Type.Int32] = cuOuter_internal_u32ti32;
      cuOuter_ii[Type.Uint32][Type.Uint32] = cuOuter_internal_u32tu32;
      cuOuter_ii[Type.Uint32][Type.Uint16] = cuOuter_internal_u32tu16;
      cuOuter_ii[Type.Uint32][Type.Int16] = cuOuter_internal_u32ti16;
      cuOuter_ii[Type.Uint32][Type.Bool] = cuOuter_internal_u32tb;

      cuOuter_ii[Type.Int16][Type.ComplexDouble] = cuOuter_internal_i16tcd;
      cuOuter_ii[Type.Int16][Type.ComplexFloat] = cuOuter_internal_i16tcf;
      cuOuter_ii[Type.Int16][Type.Double] = cuOuter_internal_i16td;
      cuOuter_ii[Type.Int16][Type.Float] = cuOuter_internal_i16tf;
      cuOuter_ii[Type.Int16][Type.Int64] = cuOuter_internal_i16ti64;
      cuOuter_ii[Type.Int16][Type.Uint64] = cuOuter_internal_i16tu64;
      cuOuter_ii[Type.Int16][Type.Int32] = cuOuter_internal_i16ti32;
      cuOuter_ii[Type.Int16][Type.Uint32] = cuOuter_internal_i16tu32;
      cuOuter_ii[Type.Int16][Type.Uint16] = cuOuter_internal_i16tu16;
      cuOuter_ii[Type.Int16][Type.Int16] = cuOuter_internal_i16ti16;
      cuOuter_ii[Type.Int16][Type.Bool] = cuOuter_internal_i16tb;

      cuOuter_ii[Type.Uint16][Type.ComplexDouble] = cuOuter_internal_u16tcd;
      cuOuter_ii[Type.Uint16][Type.ComplexFloat] = cuOuter_internal_u16tcf;
      cuOuter_ii[Type.Uint16][Type.Double] = cuOuter_internal_u16td;
      cuOuter_ii[Type.Uint16][Type.Float] = cuOuter_internal_u16tf;
      cuOuter_ii[Type.Uint16][Type.Int64] = cuOuter_internal_u16ti64;
      cuOuter_ii[Type.Uint16][Type.Uint64] = cuOuter_internal_u16tu64;
      cuOuter_ii[Type.Uint16][Type.Int32] = cuOuter_internal_u16ti32;
      cuOuter_ii[Type.Uint16][Type.Uint32] = cuOuter_internal_u16tu32;
      cuOuter_ii[Type.Uint16][Type.Uint16] = cuOuter_internal_u16tu16;
      cuOuter_ii[Type.Uint16][Type.Int16] = cuOuter_internal_u16ti16;
      cuOuter_ii[Type.Uint16][Type.Bool] = cuOuter_internal_u16tb;

      cuOuter_ii[Type.Bool][Type.ComplexDouble] = cuOuter_internal_btcd;
      cuOuter_ii[Type.Bool][Type.ComplexFloat] = cuOuter_internal_btcf;
      cuOuter_ii[Type.Bool][Type.Double] = cuOuter_internal_btd;
      cuOuter_ii[Type.Bool][Type.Float] = cuOuter_internal_btf;
      cuOuter_ii[Type.Bool][Type.Int64] = cuOuter_internal_bti64;
      cuOuter_ii[Type.Bool][Type.Uint64] = cuOuter_internal_btu64;
      cuOuter_ii[Type.Bool][Type.Int32] = cuOuter_internal_bti32;
      cuOuter_ii[Type.Bool][Type.Uint32] = cuOuter_internal_btu32;
      cuOuter_ii[Type.Bool][Type.Uint16] = cuOuter_internal_btu16;
      cuOuter_ii[Type.Bool][Type.Int16] = cuOuter_internal_bti16;
      cuOuter_ii[Type.Bool][Type.Bool] = cuOuter_internal_btb;

#endif
    }

  }  // namespace linalg_internal
}  // namespace cytnx
