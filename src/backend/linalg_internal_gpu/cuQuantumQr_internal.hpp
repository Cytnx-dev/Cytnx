#ifndef __cuQuantumQr_internal_H__
#define __cuQuantumQr_internal_H__

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "backend/Storage.hpp"
#include "Type.hpp"
#include "cytnx_error.hpp"
#include "backend/lapack_wrapper.hpp"
#include "../linalg_internal_interface.hpp"

#ifdef UNI_GPU
  #ifdef UNI_CUQUANTUM
    #include <cutensornet.h>
    #include <cuda_runtime.h>
  #endif
#endif

namespace cytnx {
  namespace linalg_internal {

#ifdef UNI_GPU
  #ifdef UNI_CUQUANTUM
    /// cuSvd
    void cuQuantumQr_internal_cd(const boost::intrusive_ptr<Storage_base> &in,
                                 boost::intrusive_ptr<Storage_base> &Q,
                                 boost::intrusive_ptr<Storage_base> &R,
                                 boost::intrusive_ptr<Storage_base> &D,
                                 boost::intrusive_ptr<Storage_base> &tau, const cytnx_int64 &M,
                                 const cytnx_int64 &N, const bool &is_d);
    void cuQuantumQr_internal_cf(const boost::intrusive_ptr<Storage_base> &in,
                                 boost::intrusive_ptr<Storage_base> &Q,
                                 boost::intrusive_ptr<Storage_base> &R,
                                 boost::intrusive_ptr<Storage_base> &D,
                                 boost::intrusive_ptr<Storage_base> &tau, const cytnx_int64 &M,
                                 const cytnx_int64 &N, const bool &is_d);
    void cuQuantumQr_internal_d(const boost::intrusive_ptr<Storage_base> &in,
                                boost::intrusive_ptr<Storage_base> &Q,
                                boost::intrusive_ptr<Storage_base> &R,
                                boost::intrusive_ptr<Storage_base> &D,
                                boost::intrusive_ptr<Storage_base> &tau, const cytnx_int64 &M,
                                const cytnx_int64 &N, const bool &is_d);
    void cuQuantumQr_internal_f(const boost::intrusive_ptr<Storage_base> &in,
                                boost::intrusive_ptr<Storage_base> &Q,
                                boost::intrusive_ptr<Storage_base> &R,
                                boost::intrusive_ptr<Storage_base> &D,
                                boost::intrusive_ptr<Storage_base> &tau, const cytnx_int64 &M,
                                const cytnx_int64 &N, const bool &is_d);
  #endif
#endif

  }  // namespace linalg_internal
}  // namespace cytnx

#endif
