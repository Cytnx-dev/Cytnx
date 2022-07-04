#ifndef _H_linalg_internal_interface_
#define _H_linalg_internal_interface_
#include <iostream>
#include <vector>

#include "Type.hpp"
#include "Storage.hpp"
#include "linalg/linalg_internal_cpu/Arithmic_internal.hpp"
#include "linalg/linalg_internal_cpu/Svd_internal.hpp"
#include "linalg/linalg_internal_cpu/Eigh_internal.hpp"
#include "linalg/linalg_internal_cpu/Inv_inplace_internal.hpp"
#include "linalg/linalg_internal_cpu/Conj_inplace_internal.hpp"
#include "linalg/linalg_internal_cpu/Exp_internal.hpp"
#include "linalg/linalg_internal_cpu/Matmul_internal.hpp"
#include "linalg/linalg_internal_cpu/Diag_internal.hpp"
#include "linalg/linalg_internal_cpu/Outer_internal.hpp"
#ifdef UNI_GPU
  #include "linalg/linalg_internal_gpu/cuArithmic_internal.hpp"
  #include "linalg/linalg_internal_gpu/cuSvd_internal.hpp"
  #include "linalg/linalg_internal_gpu/cuEigh_internal.hpp"
  #include "linalg/linalg_internal_gpu/cuInv_inplace_internal.hpp"
  #include "linalg/linalg_internal_gpu/cuConj_inplace_internal.hpp"
  #include "linalg/linalg_internal_gpu/cuExp_internal.hpp"
  #include "linalg/linalg_internal_gpu/cuMatmul_internal.hpp"
  #include "linalg/linalg_internal_gpu/cuDiag_internal.hpp"
  #include "linalg/linalg_internal_gpu/cuOuter_internal.hpp"
#endif

namespace cytnx {

  namespace linalg_internal {
    typedef void (*Arithmicfunc_oii)(
      boost::intrusive_ptr<Storage_base> &, boost::intrusive_ptr<Storage_base> &,
      boost::intrusive_ptr<Storage_base> &, const unsigned long long &len,
      const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L,
      const std::vector<cytnx_uint64> &invmapper_R, const char &type);
    typedef void (*Svdfunc_oii)(const boost::intrusive_ptr<Storage_base> &,
                                boost::intrusive_ptr<Storage_base> &,
                                boost::intrusive_ptr<Storage_base> &,
                                boost::intrusive_ptr<Storage_base> &, const cytnx_int32 &,
                                const cytnx_int32 &);
    typedef void (*Eighfunc_oii)(const boost::intrusive_ptr<Storage_base> &,
                                 boost::intrusive_ptr<Storage_base> &,
                                 boost::intrusive_ptr<Storage_base> &, const cytnx_int32 &);
    typedef void (*Invinplacefunc_oii)(boost::intrusive_ptr<Storage_base> &, const cytnx_int32 &);
    typedef void (*Conjinplacefunc_oii)(boost::intrusive_ptr<Storage_base> &, const cytnx_uint64 &);
    typedef void (*Expfunc_oii)(boost::intrusive_ptr<Storage_base> &,
                                const boost::intrusive_ptr<Storage_base> &, const cytnx_uint64 &);
    typedef void (*Diagfunc_oii)(boost::intrusive_ptr<Storage_base> &,
                                 const boost::intrusive_ptr<Storage_base> &, const cytnx_uint64 &);
    typedef void (*Matmulfunc_oii)(boost::intrusive_ptr<Storage_base> &,
                                   const boost::intrusive_ptr<Storage_base> &,
                                   const boost::intrusive_ptr<Storage_base> &, const cytnx_int32 &,
                                   const cytnx_int32 &, const cytnx_int32 &);
    typedef void (*Outerfunc_oii)(boost::intrusive_ptr<Storage_base> &,
                                  const boost::intrusive_ptr<Storage_base> &,
                                  const boost::intrusive_ptr<Storage_base> &);

    class linalg_internal_interface {
     public:
      std::vector<std::vector<Arithmicfunc_oii>> Ari_ii;
      std::vector<Svdfunc_oii> Svd_ii;
      std::vector<Eighfunc_oii> Eigh_ii;
      std::vector<Invinplacefunc_oii> Inv_inplace_ii;
      std::vector<Conjinplacefunc_oii> Conj_inplace_ii;
      std::vector<Expfunc_oii> Exp_ii;
      std::vector<Diagfunc_oii> Diag_ii;
      std::vector<Matmulfunc_oii> Matmul_ii;
      std::vector<std::vector<Outerfunc_oii>> Outer_ii;

#ifdef UNI_GPU
      std::vector<std::vector<Arithmicfunc_oii>> cuAri_ii;
      std::vector<Svdfunc_oii> cuSvd_ii;
      std::vector<Invinplacefunc_oii> cuInv_inplace_ii;
      std::vector<Conjinplacefunc_oii> cuConj_inplace_ii;
      std::vector<Expfunc_oii> cuExp_ii;
      std::vector<Diagfunc_oii> cuDiag_ii;
      std::vector<Eighfunc_oii> cuEigh_ii;
      std::vector<Matmulfunc_oii> cuMatmul_ii;
      std::vector<std::vector<Outerfunc_oii>> cuOuter_ii;
#endif

      linalg_internal_interface();
    };
    extern linalg_internal_interface lii;
  }  // namespace linalg_internal

}  // namespace cytnx
#endif
