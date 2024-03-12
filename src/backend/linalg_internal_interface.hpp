#ifndef _H_linalg_internal_interface_
#define _H_linalg_internal_interface_
#include <iostream>
#include <vector>

#include "Type.hpp"
#include "backend/Scalar.hpp"
#include "backend/Storage.hpp"
#include "linalg_internal_cpu/Abs_internal.hpp"
#include "linalg_internal_cpu/Arithmetic_internal.hpp"
#include "linalg_internal_cpu/Axpy_internal.hpp"
#include "linalg_internal_cpu/Conj_inplace_internal.hpp"
#include "linalg_internal_cpu/Det_internal.hpp"
#include "linalg_internal_cpu/Diag_internal.hpp"
#include "linalg_internal_cpu/Eig_internal.hpp"
#include "linalg_internal_cpu/Eigh_internal.hpp"
#include "linalg_internal_cpu/Exp_internal.hpp"
#include "linalg_internal_cpu/Gemm_Batch_internal.hpp"
#include "linalg_internal_cpu/Gemm_internal.hpp"
#include "linalg_internal_cpu/Ger_internal.hpp"
#include "linalg_internal_cpu/Gesvd_internal.hpp"
#include "linalg_internal_cpu/InvM_inplace_internal.hpp"
#include "linalg_internal_cpu/Inv_inplace_internal.hpp"
#include "linalg_internal_cpu/Kron_internal.hpp"
#include "linalg_internal_cpu/Lstsq_internal.hpp"
#include "linalg_internal_cpu/Matmul_dg_internal.hpp"
#include "linalg_internal_cpu/Matmul_internal.hpp"
#include "linalg_internal_cpu/Matvec_internal.hpp"
#include "linalg_internal_cpu/MaxMin_internal.hpp"
#include "linalg_internal_cpu/Norm_internal.hpp"
#include "linalg_internal_cpu/Outer_internal.hpp"
#include "linalg_internal_cpu/Pow_internal.hpp"
#include "linalg_internal_cpu/QR_internal.hpp"
#include "linalg_internal_cpu/Sdd_internal.hpp"
#include "linalg_internal_cpu/Sum_internal.hpp"
#include "linalg_internal_cpu/Trace_internal.hpp"
#include "linalg_internal_cpu/Tridiag_internal.hpp"
#include "linalg_internal_cpu/Vectordot_internal.hpp"
#include "linalg_internal_cpu/iArithmetic_internal.hpp"
#include "linalg_internal_cpu/memcpyTruncation.hpp"

#ifdef UNI_GPU
  #include "linalg_internal_gpu/cuAbs_internal.hpp"
  #include "linalg_internal_gpu/cuArithmetic_internal.hpp"
  #include "linalg_internal_gpu/cuConj_inplace_internal.hpp"
  #include "linalg_internal_gpu/cuDet_internal.hpp"
  #include "linalg_internal_gpu/cuDiag_internal.hpp"
  #include "linalg_internal_gpu/cuEigh_internal.hpp"
  #include "linalg_internal_gpu/cuExp_internal.hpp"
  #include "linalg_internal_gpu/cuGeSvd_internal.hpp"
  #include "linalg_internal_gpu/cuGemm_Batch_internal.hpp"
  #include "linalg_internal_gpu/cuGemm_internal.hpp"
  #include "linalg_internal_gpu/cuGer_internal.hpp"
  #include "linalg_internal_gpu/cuInvM_inplace_internal.hpp"
  #include "linalg_internal_gpu/cuInv_inplace_internal.hpp"
  #include "linalg_internal_gpu/cuKron_internal.hpp"
  #include "linalg_internal_gpu/cuMatmul_dg_internal.hpp"
  #include "linalg_internal_gpu/cuMatmul_internal.hpp"
  #include "linalg_internal_gpu/cuMatvec_internal.hpp"
  #include "linalg_internal_gpu/cuMaxMin_internal.hpp"
  #include "linalg_internal_gpu/cuNorm_internal.hpp"
  #include "linalg_internal_gpu/cuOuter_internal.hpp"
  #include "linalg_internal_gpu/cuPow_internal.hpp"
  #include "linalg_internal_gpu/cuSum_internal.hpp"
  #include "linalg_internal_gpu/cuSvd_internal.hpp"
  #include "linalg_internal_gpu/cuVectordot_internal.hpp"
  #include "linalg_internal_gpu/cudaMemcpyTruncation.hpp"
  #ifdef UNI_CUTENSOR
    #include "linalg_internal_gpu/cuTensordot_internal.hpp"
  #endif

  #ifdef UNI_CUQUANTUM
    #include "linalg_internal_gpu/cuQuantumGeSvd_internal.hpp"
    #include "linalg_internal_gpu/cuQuantumQr_internal.hpp"
  #endif
#endif

namespace cytnx {

  namespace linalg_internal {
    typedef void (*Normfunc_oii)(void *, const boost::intrusive_ptr<Storage_base> &);
    typedef void (*Detfunc_oii)(void *, const boost::intrusive_ptr<Storage_base> &,
                                const cytnx_uint64 &);
    typedef void (*Arithmeticfunc_oii)(
      boost::intrusive_ptr<Storage_base> &, boost::intrusive_ptr<Storage_base> &,
      boost::intrusive_ptr<Storage_base> &, const unsigned long long &len,
      const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L,
      const std::vector<cytnx_uint64> &invmapper_R, const char &type);

    typedef void (*axpy_oii)(const boost::intrusive_ptr<Storage_base> &,
                             boost::intrusive_ptr<Storage_base> &, const Scalar &);
    typedef void (*ger_oii)(boost::intrusive_ptr<Storage_base> &,
                            const boost::intrusive_ptr<Storage_base> &,
                            const boost::intrusive_ptr<Storage_base> &, const Scalar &);

    typedef void (*Gemmfunc_oii)(boost::intrusive_ptr<Storage_base> &,
                                 const boost::intrusive_ptr<Storage_base> &,
                                 const boost::intrusive_ptr<Storage_base> &, const cytnx_int64 &,
                                 const cytnx_int64 &, const cytnx_int64 &, const Scalar &,
                                 const Scalar &);

    typedef void (*Gemm_Batchfunc_oii)(
      const char *transa_array, const char *transb_array, const blas_int *m_array,
      const blas_int *n_array, const blas_int *k_array, const std::vector<Scalar> &alpha_array,
      const void **a_array, const blas_int *lda_array, const void **b_array,
      const blas_int *ldb_array, const std::vector<Scalar> &beta_array, void **c_array,
      const blas_int *ldc_array, const blas_int group_count, const blas_int *group_size);

    typedef void (*Svdfunc_oii)(const boost::intrusive_ptr<Storage_base> &,
                                boost::intrusive_ptr<Storage_base> &,
                                boost::intrusive_ptr<Storage_base> &,
                                boost::intrusive_ptr<Storage_base> &, const cytnx_int64 &,
                                const cytnx_int64 &);
    typedef void (*Qrfunc_oii)(const boost::intrusive_ptr<Storage_base> &,
                               boost::intrusive_ptr<Storage_base> &,
                               boost::intrusive_ptr<Storage_base> &,
                               boost::intrusive_ptr<Storage_base> &,
                               boost::intrusive_ptr<Storage_base> &, const cytnx_int64 &,
                               const cytnx_int64 &, const bool &);
    typedef void (*Eighfunc_oii)(const boost::intrusive_ptr<Storage_base> &,
                                 boost::intrusive_ptr<Storage_base> &,
                                 boost::intrusive_ptr<Storage_base> &, const cytnx_int64 &);
    typedef void (*InvMinplacefunc_oii)(boost::intrusive_ptr<Storage_base> &, const cytnx_int64 &);
    typedef void (*Conjinplacefunc_oii)(boost::intrusive_ptr<Storage_base> &, const cytnx_uint64 &);
    typedef void (*Expfunc_oii)(boost::intrusive_ptr<Storage_base> &,
                                const boost::intrusive_ptr<Storage_base> &, const cytnx_uint64 &);
    typedef void (*Diagfunc_oii)(boost::intrusive_ptr<Storage_base> &,
                                 const boost::intrusive_ptr<Storage_base> &, const cytnx_uint64 &,
                                 const cytnx_bool &);
    typedef void (*Matmulfunc_oii)(boost::intrusive_ptr<Storage_base> &,
                                   const boost::intrusive_ptr<Storage_base> &,
                                   const boost::intrusive_ptr<Storage_base> &, const cytnx_int64 &,
                                   const cytnx_int64 &, const cytnx_int64 &);
    typedef void (*Matmul_dgfunc_oii)(boost::intrusive_ptr<Storage_base> &,
                                      const boost::intrusive_ptr<Storage_base> &,
                                      const boost::intrusive_ptr<Storage_base> &,
                                      const cytnx_int64 &, const cytnx_int64 &, const cytnx_int64 &,
                                      const int &);
    typedef void (*Matvecfunc_oii)(boost::intrusive_ptr<Storage_base> &,
                                   const boost::intrusive_ptr<Storage_base> &,
                                   const boost::intrusive_ptr<Storage_base> &, const cytnx_int64 &,
                                   const cytnx_int64 &);
    typedef void (*Outerfunc_oii)(boost::intrusive_ptr<Storage_base> &,
                                  const boost::intrusive_ptr<Storage_base> &,
                                  const boost::intrusive_ptr<Storage_base> &, const cytnx_uint64 &,
                                  const cytnx_uint64 &);
    typedef void (*Vectordotfunc_oii)(boost::intrusive_ptr<Storage_base> &,
                                      const boost::intrusive_ptr<Storage_base> &,
                                      const boost::intrusive_ptr<Storage_base> &,
                                      const unsigned long long &, const bool &);
    typedef void (*Tdfunc_oii)(const boost::intrusive_ptr<Storage_base> &,
                               const boost::intrusive_ptr<Storage_base> &,
                               boost::intrusive_ptr<Storage_base> &,
                               boost::intrusive_ptr<Storage_base> &, const cytnx_int64 &,
                               bool throw_excp);
    typedef void (*Kronfunc_oii)(boost::intrusive_ptr<Storage_base> &,
                                 const boost::intrusive_ptr<Storage_base> &,
                                 const boost::intrusive_ptr<Storage_base> &,
                                 const std::vector<cytnx_uint64> &,
                                 const std::vector<cytnx_uint64> &);
    typedef void (*Powfunc_oii)(boost::intrusive_ptr<Storage_base> &,
                                const boost::intrusive_ptr<Storage_base> &, const cytnx_uint64 &,
                                const double &);
    typedef void (*Absfunc_oii)(boost::intrusive_ptr<Storage_base> &,
                                const boost::intrusive_ptr<Storage_base> &, const cytnx_uint64 &);
    typedef void (*MaxMinfunc_oii)(boost::intrusive_ptr<Storage_base> &,
                                   const boost::intrusive_ptr<Storage_base> &, const cytnx_uint64 &,
                                   const char &);
    typedef void (*Invinplacefunc_oii)(boost::intrusive_ptr<Storage_base> &ten,
                                       const cytnx_uint64 &Nelem, const double &clip);

    typedef void (*Lstsqfunc_oii)(boost::intrusive_ptr<Storage_base> &in,
                                  boost::intrusive_ptr<Storage_base> &b,
                                  boost::intrusive_ptr<Storage_base> &s,
                                  boost::intrusive_ptr<Storage_base> &r, const cytnx_int64 &M,
                                  const cytnx_int64 &N, const cytnx_int64 &nrhs,
                                  const cytnx_float &rcond);

    typedef void (*Tracefunc_oii)(const bool &, Tensor &, const Tensor &, const cytnx_uint64 &,
                                  const int &, const cytnx_uint64 &,
                                  const std::vector<cytnx_uint64> &,
                                  const std::vector<cytnx_uint64> &,
                                  const std::vector<cytnx_int64> &, const cytnx_uint64 &,
                                  const cytnx_uint64 &);

    typedef void (*Tensordotfunc_oii)(Tensor &out, const Tensor &Lin, const Tensor &Rin,
                                      const std::vector<cytnx_uint64> &idxl,
                                      const std::vector<cytnx_uint64> &idxr);

    typedef void (*memcpyTruncation_oii)(Tensor &U, Tensor &vT, Tensor &S, Tensor &terr,
                                         const cytnx_uint64 &keepdim, const double &err,
                                         const bool &is_U, const bool &is_vT,
                                         const unsigned int &return_err,
                                         const unsigned int &mindim);

#ifdef UNI_GPU

    typedef void (*cudaMemcpyTruncation_oii)(Tensor &U, Tensor &vT, Tensor &S, Tensor &terr,
                                             const cytnx_uint64 &keepdim, const double &err,
                                             const bool &is_U, const bool &is_vT,
                                             const unsigned int &return_err,
                                             const unsigned int &mindim);

  #ifdef UNI_CUQUANTUM
    typedef void (*cuQuantumGeSvd_oii)(const Tensor &Tin, const cytnx_uint64 &keepdim,
                                       const double &err, const unsigned int &return_err, Tensor &U,
                                       Tensor &S, Tensor &vT, Tensor &terr);
    typedef void (*cuQuantumQr_oii)(const boost::intrusive_ptr<Storage_base> &in,
                                    boost::intrusive_ptr<Storage_base> &Q,
                                    boost::intrusive_ptr<Storage_base> &R,
                                    boost::intrusive_ptr<Storage_base> &D,
                                    boost::intrusive_ptr<Storage_base> &tau, const cytnx_int64 &M,
                                    const cytnx_int64 &N, const bool &is_d);
  #endif
#endif
    class linalg_internal_interface {
     public:
      std::vector<std::vector<Arithmeticfunc_oii>> Ari_ii;
      std::vector<std::vector<Arithmeticfunc_oii>> iAri_ii;
      std::vector<Svdfunc_oii> Sdd_ii;
      std::vector<Svdfunc_oii> Gesvd_ii;
      std::vector<Eighfunc_oii> Eigh_ii;
      std::vector<Eighfunc_oii> Eig_ii;
      std::vector<InvMinplacefunc_oii> InvM_inplace_ii;
      std::vector<Invinplacefunc_oii> Inv_inplace_ii;
      std::vector<Conjinplacefunc_oii> Conj_inplace_ii;
      std::vector<Expfunc_oii> Exp_ii;
      std::vector<Powfunc_oii> Pow_ii;
      std::vector<Absfunc_oii> Abs_ii;
      std::vector<Diagfunc_oii> Diag_ii;
      std::vector<Matmulfunc_oii> Matmul_ii;
      std::vector<Gemmfunc_oii> Gemm_ii;
      std::vector<Gemm_Batchfunc_oii> Gemm_Batch_ii;
      std::vector<Matmul_dgfunc_oii> Matmul_dg_ii;
      std::vector<Matvecfunc_oii> Matvec_ii;
      std::vector<std::vector<Outerfunc_oii>> Outer_ii;
      std::vector<std::vector<Kronfunc_oii>> Kron_ii;
      std::vector<Vectordotfunc_oii> Vd_ii;
      std::vector<Tdfunc_oii> Td_ii;
      std::vector<Normfunc_oii> Norm_ii;
      std::vector<Qrfunc_oii> QR_ii;
      std::vector<MaxMinfunc_oii> MM_ii;
      std::vector<MaxMinfunc_oii> Sum_ii;
      std::vector<Detfunc_oii> Det_ii;

      std::vector<Lstsqfunc_oii> Lstsq_ii;
      std::vector<Tracefunc_oii> Trace_ii;

      std::vector<axpy_oii> axpy_ii;
      std::vector<ger_oii> ger_ii;

      std::vector<memcpyTruncation_oii> memcpyTruncation_ii;

      int mkl_code;

#ifdef UNI_GPU
      std::vector<std::vector<Arithmeticfunc_oii>> cuAri_ii;
      std::vector<Svdfunc_oii> cuSvd_ii;
      std::vector<Svdfunc_oii> cuGeSvd_ii;
      std::vector<InvMinplacefunc_oii> cuInvM_inplace_ii;
      std::vector<Invinplacefunc_oii> cuInv_inplace_ii;
      std::vector<Conjinplacefunc_oii> cuConj_inplace_ii;
      std::vector<Expfunc_oii> cuExp_ii;
      std::vector<Diagfunc_oii> cuDiag_ii;
      std::vector<Eighfunc_oii> cuEigh_ii;
      std::vector<Matmulfunc_oii> cuMatmul_ii;
      std::vector<Gemmfunc_oii> cuGemm_ii;
      std::vector<Gemm_Batchfunc_oii> cuGemm_Batch_ii;
      std::vector<Matmul_dgfunc_oii> cuMatmul_dg_ii;
      std::vector<Matvecfunc_oii> cuMatvec_ii;
      std::vector<std::vector<Outerfunc_oii>> cuOuter_ii;
      std::vector<Normfunc_oii> cuNorm_ii;
      std::vector<Vectordotfunc_oii> cuVd_ii;
      std::vector<Powfunc_oii> cuPow_ii;
      std::vector<Absfunc_oii> cuAbs_ii;
      std::vector<ger_oii> cuGer_ii;
      std::vector<Detfunc_oii> cuDet_ii;
      std::vector<MaxMinfunc_oii> cuMM_ii;
      std::vector<MaxMinfunc_oii> cuSum_ii;
      std::vector<std::vector<Kronfunc_oii>> cuKron_ii;
      std::vector<Tensordotfunc_oii> cuTensordot_ii;

      std::vector<cudaMemcpyTruncation_oii> cudaMemcpyTruncation_ii;

  #ifdef UNI_CUQUANTUM
      std::vector<cuQuantumGeSvd_oii> cuQuantumGeSvd_ii;
      std::vector<cuQuantumQr_oii> cuQuantumQr_ii;
  #endif
#endif

      linalg_internal_interface();
      ~linalg_internal_interface();
      int set_mkl_ilp64();
      int get_mkl_code();
    };
    extern linalg_internal_interface lii;
  }  // namespace linalg_internal

}  // namespace cytnx
#endif
