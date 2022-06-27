#ifndef _H_linalg_internal_interface_
#define _H_linalg_internal_interface_
#include <iostream>
#include <vector>

#include "Type.hpp"
#include "Storage.hpp"
#include "linalg/linalg_internal_cpu/Arithmetic_internal.hpp"
#include "linalg/linalg_internal_cpu/iArithmetic_internal.hpp"
#include "linalg/linalg_internal_cpu/Svd_internal.hpp"
#include "linalg/linalg_internal_cpu/Eigh_internal.hpp"
#include "linalg/linalg_internal_cpu/Eig_internal.hpp"
#include "linalg/linalg_internal_cpu/InvM_inplace_internal.hpp"
#include "linalg/linalg_internal_cpu/Inv_inplace_internal.hpp"
#include "linalg/linalg_internal_cpu/Conj_inplace_internal.hpp"
#include "linalg/linalg_internal_cpu/Exp_internal.hpp"
#include "linalg/linalg_internal_cpu/Matmul_internal.hpp"
#include "linalg/linalg_internal_cpu/Matmul_dg_internal.hpp"
#include "linalg/linalg_internal_cpu/Diag_internal.hpp"
#include "linalg/linalg_internal_cpu/Outer_internal.hpp"
#include "linalg/linalg_internal_cpu/Kron_internal.hpp"
#include "linalg/linalg_internal_cpu/Vectordot_internal.hpp"
#include "linalg/linalg_internal_cpu/Tridiag_internal.hpp"
#include "linalg/linalg_internal_cpu/Norm_internal.hpp"
#include "linalg/linalg_internal_cpu/Matvec_internal.hpp"
#include "linalg/linalg_internal_cpu/Pow_internal.hpp"
#include "linalg/linalg_internal_cpu/Abs_internal.hpp"
#include "linalg/linalg_internal_cpu/QR_internal.hpp"
#include "linalg/linalg_internal_cpu/MaxMin_internal.hpp"
#include "linalg/linalg_internal_cpu/Sum_internal.hpp"
#include "linalg/linalg_internal_cpu/Det_internal.hpp"
#include "linalg/linalg_internal_cpu/Lstsq_internal.hpp"
#ifdef UNI_GPU
  #include "linalg/linalg_internal_gpu/cuArithmetic_internal.hpp"
  #include "linalg/linalg_internal_gpu/cuSvd_internal.hpp"
  #include "linalg/linalg_internal_gpu/cuEigh_internal.hpp"
  #include "linalg/linalg_internal_gpu/cuInvM_inplace_internal.hpp"
  #include "linalg/linalg_internal_gpu/cuInv_inplace_internal.hpp"
  #include "linalg/linalg_internal_gpu/cuConj_inplace_internal.hpp"
  #include "linalg/linalg_internal_gpu/cuExp_internal.hpp"
  #include "linalg/linalg_internal_gpu/cuMatmul_internal.hpp"
  #include "linalg/linalg_internal_gpu/cuMatmul_dg_internal.hpp"
  #include "linalg/linalg_internal_gpu/cuDiag_internal.hpp"
  #include "linalg/linalg_internal_gpu/cuOuter_internal.hpp"
  #include "linalg/linalg_internal_gpu/cuNorm_internal.hpp"
  #include "linalg/linalg_internal_gpu/cuMatvec_internal.hpp"
  #include "linalg/linalg_internal_gpu/cuVectordot_internal.hpp"
  #include "linalg/linalg_internal_gpu/cuPow_internal.hpp"
  //#include "linalg/linalg_internal_gpu/cuAbs_internal.hpp"
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
                               boost::intrusive_ptr<Storage_base> &, const cytnx_int64 &);
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

    class linalg_internal_interface {
     public:
      std::vector<std::vector<Arithmeticfunc_oii>> Ari_ii;
      std::vector<std::vector<Arithmeticfunc_oii>> iAri_ii;
      std::vector<Svdfunc_oii> Svd_ii;
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

#ifdef UNI_GPU
      std::vector<std::vector<Arithmeticfunc_oii>> cuAri_ii;
      std::vector<Svdfunc_oii> cuSvd_ii;
      std::vector<InvMinplacefunc_oii> cuInvM_inplace_ii;
      std::vector<Invinplacefunc_oii> cuInv_inplace_ii;
      std::vector<Conjinplacefunc_oii> cuConj_inplace_ii;
      std::vector<Expfunc_oii> cuExp_ii;
      std::vector<Diagfunc_oii> cuDiag_ii;
      std::vector<Eighfunc_oii> cuEigh_ii;
      std::vector<Matmulfunc_oii> cuMatmul_ii;
      std::vector<Matmul_dgfunc_oii> cuMatmul_dg_ii;
      std::vector<Matvecfunc_oii> cuMatvec_ii;
      std::vector<std::vector<Outerfunc_oii>> cuOuter_ii;
      std::vector<Normfunc_oii> cuNorm_ii;
      std::vector<Vectordotfunc_oii> cuVd_ii;
      std::vector<Powfunc_oii> cuPow_ii;
#endif

      linalg_internal_interface();
    };
    extern linalg_internal_interface lii;
  }  // namespace linalg_internal

}  // namespace cytnx
#endif
