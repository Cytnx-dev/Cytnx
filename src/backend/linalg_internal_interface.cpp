#include "linalg_internal_interface.hpp"
#ifdef UNI_MKL
  #include <mkl.h>

#endif

namespace cytnx {
  namespace linalg_internal {

    linalg_internal_interface lii;

    linalg_internal_interface::~linalg_internal_interface() {}
    linalg_internal_interface::linalg_internal_interface() {
      //=====================
      QR_ii = std::vector<Qrfunc_oii>(5);

      QR_ii[Type.ComplexDouble] = QR_internal_cd;
      QR_ii[Type.ComplexFloat] = QR_internal_cf;
      QR_ii[Type.Double] = QR_internal_d;
      QR_ii[Type.Float] = QR_internal_f;

      //=====================
      Sdd_ii = std::vector<Svdfunc_oii>(5);

      Sdd_ii[Type.ComplexDouble] = Sdd_internal_cd;
      Sdd_ii[Type.ComplexFloat] = Sdd_internal_cf;
      Sdd_ii[Type.Double] = Sdd_internal_d;
      Sdd_ii[Type.Float] = Sdd_internal_f;

      //=====================
      Gesvd_ii = std::vector<Svdfunc_oii>(5);

      Gesvd_ii[Type.ComplexDouble] = Gesvd_internal_cd;
      Gesvd_ii[Type.ComplexFloat] = Gesvd_internal_cf;
      Gesvd_ii[Type.Double] = Gesvd_internal_d;
      Gesvd_ii[Type.Float] = Gesvd_internal_f;

      //=====================
      Eigh_ii = std::vector<Eighfunc_oii>(5);

      Eigh_ii[Type.ComplexDouble] = Eigh_internal_cd;
      Eigh_ii[Type.ComplexFloat] = Eigh_internal_cf;
      Eigh_ii[Type.Double] = Eigh_internal_d;
      Eigh_ii[Type.Float] = Eigh_internal_f;

      //=====================
      Eig_ii = std::vector<Eighfunc_oii>(5);

      Eig_ii[Type.ComplexDouble] = Eig_internal_cd;
      Eig_ii[Type.ComplexFloat] = Eig_internal_cf;
      Eig_ii[Type.Double] = Eig_internal_d;
      Eig_ii[Type.Float] = Eig_internal_f;

      //=====================
      Exp_ii = std::vector<Expfunc_oii>(5);

      Exp_ii[Type.ComplexDouble] = Exp_internal_cd;
      Exp_ii[Type.ComplexFloat] = Exp_internal_cf;
      Exp_ii[Type.Double] = Exp_internal_d;
      Exp_ii[Type.Float] = Exp_internal_f;

      //=====================
      MM_ii = std::vector<MaxMinfunc_oii>(N_Type);

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
      Sum_ii = std::vector<Sumfunc_oii>(N_Type);

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
      Pow_ii = std::vector<Powfunc_oii>(5);

      Pow_ii[Type.ComplexDouble] = Pow_internal_cd;
      Pow_ii[Type.ComplexFloat] = Pow_internal_cf;
      Pow_ii[Type.Double] = Pow_internal_d;
      Pow_ii[Type.Float] = Pow_internal_f;

      //=====================
      Diag_ii = std::vector<Diagfunc_oii>(N_Type);

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
      InvM_inplace_ii = std::vector<InvMinplacefunc_oii>(5);

      InvM_inplace_ii[Type.ComplexDouble] = InvM_inplace_internal_cd;
      InvM_inplace_ii[Type.ComplexFloat] = InvM_inplace_internal_cf;
      InvM_inplace_ii[Type.Double] = InvM_inplace_internal_d;
      InvM_inplace_ii[Type.Float] = InvM_inplace_internal_f;

      //=====================
      Inv_inplace_ii = std::vector<Invinplacefunc_oii>(5);

      Inv_inplace_ii[Type.ComplexDouble] = Inv_inplace_internal_cd;
      Inv_inplace_ii[Type.ComplexFloat] = Inv_inplace_internal_cf;
      Inv_inplace_ii[Type.Double] = Inv_inplace_internal_d;
      Inv_inplace_ii[Type.Float] = Inv_inplace_internal_f;

      //=====================
      Conj_inplace_ii = std::vector<Conjinplacefunc_oii>(3);

      Conj_inplace_ii[Type.ComplexDouble] = Conj_inplace_internal_cd;
      Conj_inplace_ii[Type.ComplexFloat] = Conj_inplace_internal_cf;

      //=====================
      Matmul_ii = std::vector<Matmulfunc_oii>(N_Type);
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
      Matmul_dg_ii = std::vector<Matmul_dgfunc_oii>(N_Type);
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
      Matvec_ii = std::vector<Matvecfunc_oii>(N_Type);
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
      Norm_ii = std::vector<Normfunc_oii>(5);
      Norm_ii[Type.ComplexDouble] = Norm_internal_cd;
      Norm_ii[Type.ComplexFloat] = Norm_internal_cf;
      Norm_ii[Type.Double] = Norm_internal_d;
      Norm_ii[Type.Float] = Norm_internal_f;

      //===================
      Det_ii = std::vector<Detfunc_oii>(5);
      Det_ii[Type.ComplexDouble] = Det_internal_cd;
      Det_ii[Type.ComplexFloat] = Det_internal_cf;
      Det_ii[Type.Double] = Det_internal_d;
      Det_ii[Type.Float] = Det_internal_f;

      //====================
      Vd_ii = std::vector<Vectordotfunc_oii>(N_Type);
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
      Td_ii = std::vector<Tdfunc_oii>(N_Type);
      Td_ii[Type.Double] = Tridiag_internal_d;
      Td_ii[Type.Float] = Tridiag_internal_f;

      //=====================
      Trace_ii = std::vector<Tracefunc_oii>(N_Type);

      Trace_ii[Type.ComplexDouble] = Trace_internal_cd;
      Trace_ii[Type.ComplexFloat] = Trace_internal_cf;
      Trace_ii[Type.Double] = Trace_internal_d;
      Trace_ii[Type.Float] = Trace_internal_f;
      Trace_ii[Type.Uint64] = Trace_internal_u64;
      Trace_ii[Type.Int64] = Trace_internal_i64;
      Trace_ii[Type.Uint32] = Trace_internal_u32;
      Trace_ii[Type.Int32] = Trace_internal_i32;
      Trace_ii[Type.Uint16] = Trace_internal_u16;
      Trace_ii[Type.Int16] = Trace_internal_i16;
      Trace_ii[Type.Bool] = Trace_internal_b;

      //================

      Lstsq_ii = std::vector<Lstsqfunc_oii>(5);
      Lstsq_ii[Type.ComplexDouble] = Lstsq_internal_cd;
      Lstsq_ii[Type.ComplexFloat] = Lstsq_internal_cf;
      Lstsq_ii[Type.Double] = Lstsq_internal_d;
      Lstsq_ii[Type.Float] = Lstsq_internal_f;

      //===============
      ger_ii = std::vector<ger_oii>(5);
      ger_ii[Type.ComplexDouble] = Ger_internal_cd;
      ger_ii[Type.ComplexFloat] = Ger_internal_cf;
      ger_ii[Type.Double] = Ger_internal_d;
      ger_ii[Type.Float] = Ger_internal_f;

      //===============
      Gemm_ii = std::vector<Gemmfunc_oii>(5);
      Gemm_ii[Type.ComplexDouble] = Gemm_internal_cd;
      Gemm_ii[Type.ComplexFloat] = Gemm_internal_cf;
      Gemm_ii[Type.Double] = Gemm_internal_d;
      Gemm_ii[Type.Float] = Gemm_internal_f;

#ifdef UNI_GPU
      //=====================
      cuMM_ii = std::vector<MaxMinfunc_oii>(N_Type);

      cuMM_ii[Type.ComplexDouble] = cuMaxMin_internal_cd;
      cuMM_ii[Type.ComplexFloat] = cuMaxMin_internal_cf;
      cuMM_ii[Type.Double] = cuMaxMin_internal_d;
      cuMM_ii[Type.Float] = cuMaxMin_internal_f;
      cuMM_ii[Type.Uint64] = cuMaxMin_internal_u64;
      cuMM_ii[Type.Int64] = cuMaxMin_internal_i64;
      cuMM_ii[Type.Uint32] = cuMaxMin_internal_u32;
      cuMM_ii[Type.Int32] = cuMaxMin_internal_i32;
      cuMM_ii[Type.Uint16] = cuMaxMin_internal_u16;
      cuMM_ii[Type.Int16] = cuMaxMin_internal_i16;
      cuMM_ii[Type.Bool] = cuMaxMin_internal_b;

      //=====================
      cuSum_ii = std::vector<Sumfunc_oii>(N_Type);

      cuSum_ii[Type.ComplexDouble] = cuSum_internal_cd;
      cuSum_ii[Type.ComplexFloat] = cuSum_internal_cf;
      cuSum_ii[Type.Double] = cuSum_internal_d;
      cuSum_ii[Type.Float] = cuSum_internal_f;
      cuSum_ii[Type.Uint64] = cuSum_internal_u64;
      cuSum_ii[Type.Int64] = cuSum_internal_i64;
      cuSum_ii[Type.Uint32] = cuSum_internal_u32;
      cuSum_ii[Type.Int32] = cuSum_internal_i32;
      cuSum_ii[Type.Uint16] = cuSum_internal_u16;
      cuSum_ii[Type.Int16] = cuSum_internal_i16;
      cuSum_ii[Type.Bool] = cuSum_internal_b;

      //=====================
      //===============
      cuGer_ii = std::vector<ger_oii>(5);
      cuGer_ii[Type.ComplexDouble] = cuGer_internal_cd;
      cuGer_ii[Type.ComplexFloat] = cuGer_internal_cf;
      cuGer_ii[Type.Double] = cuGer_internal_d;
      cuGer_ii[Type.Float] = cuGer_internal_f;

      //===================
      cuDet_ii = std::vector<Detfunc_oii>(5);
      cuDet_ii[Type.ComplexDouble] = cuDet_internal_cd;
      cuDet_ii[Type.ComplexFloat] = cuDet_internal_cf;
      cuDet_ii[Type.Double] = cuDet_internal_d;
      cuDet_ii[Type.Float] = cuDet_internal_f;

      // Pow
      //====================
      // Norm
      //====================
      cuNorm_ii = std::vector<Normfunc_oii>(N_Type);
      cuNorm_ii[Type.ComplexDouble] = cuNorm_internal_cd;
      cuNorm_ii[Type.ComplexFloat] = cuNorm_internal_cf;
      cuNorm_ii[Type.Double] = cuNorm_internal_d;
      cuNorm_ii[Type.Float] = cuNorm_internal_f;

      // Svd
      cuSvd_ii = std::vector<Svdfunc_oii>(5);

      cuSvd_ii[Type.ComplexDouble] = cuSvd_internal_cd;
      cuSvd_ii[Type.ComplexFloat] = cuSvd_internal_cf;
      cuSvd_ii[Type.Double] = cuSvd_internal_d;
      cuSvd_ii[Type.Float] = cuSvd_internal_f;

      // GeSvd
      cuGeSvd_ii = std::vector<Svdfunc_oii>(5);

      cuGeSvd_ii[Type.ComplexDouble] = cuGeSvd_internal_cd;
      cuGeSvd_ii[Type.ComplexFloat] = cuGeSvd_internal_cf;
      cuGeSvd_ii[Type.Double] = cuGeSvd_internal_d;
      cuGeSvd_ii[Type.Float] = cuGeSvd_internal_f;

      //=====================
      cuEigh_ii = std::vector<Eighfunc_oii>(5);

      cuEigh_ii[Type.ComplexDouble] = cuEigh_internal_cd;
      cuEigh_ii[Type.ComplexFloat] = cuEigh_internal_cf;
      cuEigh_ii[Type.Double] = cuEigh_internal_d;
      cuEigh_ii[Type.Float] = cuEigh_internal_f;

      //=====================
      //=====================
      cuDiag_ii = std::vector<Diagfunc_oii>(N_Type);

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
      cuInvM_inplace_ii = std::vector<InvMinplacefunc_oii>(5);

      cuInvM_inplace_ii[Type.ComplexDouble] = cuInvM_inplace_internal_cd;
      cuInvM_inplace_ii[Type.ComplexFloat] = cuInvM_inplace_internal_cf;
      cuInvM_inplace_ii[Type.Double] = cuInvM_inplace_internal_d;
      cuInvM_inplace_ii[Type.Float] = cuInvM_inplace_internal_f;

      //=====================
      cuInv_inplace_ii = std::vector<Invinplacefunc_oii>(5);

      cuInv_inplace_ii[Type.ComplexDouble] = cuInv_inplace_internal_cd;
      cuInv_inplace_ii[Type.ComplexFloat] = cuInv_inplace_internal_cf;
      cuInv_inplace_ii[Type.Double] = cuInv_inplace_internal_d;
      cuInv_inplace_ii[Type.Float] = cuInv_inplace_internal_f;

      //=====================
      //=====================
      cuGemm_ii = std::vector<Gemmfunc_oii>(5);
      cuGemm_ii[Type.ComplexDouble] = cuGemm_internal_cd;
      cuGemm_ii[Type.ComplexFloat] = cuGemm_internal_cf;
      cuGemm_ii[Type.Double] = cuGemm_internal_d;
      cuGemm_ii[Type.Float] = cuGemm_internal_f;

      //=====================
      cuGemm_Batch_ii = std::vector<Gemm_Batchfunc_oii>(5);
      cuGemm_Batch_ii[Type.ComplexDouble] = cuGemm_Batch_internal_cd;
      cuGemm_Batch_ii[Type.ComplexFloat] = cuGemm_Batch_internal_cf;
      cuGemm_Batch_ii[Type.Double] = cuGemm_Batch_internal_d;
      cuGemm_Batch_ii[Type.Float] = cuGemm_Batch_internal_f;

      //=====================
      cuMatmul_ii = std::vector<Matmulfunc_oii>(N_Type);
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
      cuMatmul_dg_ii = std::vector<Matmul_dgfunc_oii>(N_Type);
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

      cuMatvec_ii = std::vector<Matvecfunc_oii>(N_Type);
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
      cuVd_ii = std::vector<Vectordotfunc_oii>(N_Type);
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

      //=====================
      cuTrace_ii = std::vector<Tracefunc_oii>(N_Type);

      cuTrace_ii[Type.ComplexDouble] = cuTrace_internal_cd;
      cuTrace_ii[Type.ComplexFloat] = cuTrace_internal_cf;
      cuTrace_ii[Type.Double] = cuTrace_internal_d;
      cuTrace_ii[Type.Float] = cuTrace_internal_f;
      cuTrace_ii[Type.Uint64] = cuTrace_internal_u64;
      cuTrace_ii[Type.Int64] = cuTrace_internal_i64;
      cuTrace_ii[Type.Uint32] = cuTrace_internal_u32;
      cuTrace_ii[Type.Int32] = cuTrace_internal_i32;
      cuTrace_ii[Type.Uint16] = cuTrace_internal_u16;
      cuTrace_ii[Type.Int16] = cuTrace_internal_i16;
      cuTrace_ii[Type.Bool] = cuTrace_internal_b;

  #ifdef UNI_CUQUANTUM
      cuQuantumGeSvd_ii = std::vector<cuQuantumGeSvd_oii>(N_Type);
      cuQuantumGeSvd_ii[Type.ComplexDouble] = cuQuantumGeSvd_internal_cd;
      cuQuantumGeSvd_ii[Type.ComplexFloat] = cuQuantumGeSvd_internal_cf;
      cuQuantumGeSvd_ii[Type.Double] = cuQuantumGeSvd_internal_d;
      cuQuantumGeSvd_ii[Type.Float] = cuQuantumGeSvd_internal_f;

      cuQuantumQr_ii = std::vector<cuQuantumQr_oii>(N_Type);
      cuQuantumQr_ii[Type.ComplexDouble] = cuQuantumQr_internal_cd;
      cuQuantumQr_ii[Type.ComplexFloat] = cuQuantumQr_internal_cf;
      cuQuantumQr_ii[Type.Double] = cuQuantumQr_internal_d;
      cuQuantumQr_ii[Type.Float] = cuQuantumQr_internal_f;
  #endif

  #ifdef UNI_CUTENSOR
      cuTensordot_ii = std::vector<Tensordotfunc_oii>(N_Type);
      cuTensordot_ii[Type.ComplexDouble] = cuTensordot_internal_cd;
      cuTensordot_ii[Type.ComplexFloat] = cuTensordot_internal_cf;
      cuTensordot_ii[Type.Double] = cuTensordot_internal_d;
      cuTensordot_ii[Type.Float] = cuTensordot_internal_f;
  #endif
#endif
    }

  }  // namespace linalg_internal
}  // namespace cytnx
