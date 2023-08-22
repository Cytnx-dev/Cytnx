#ifndef __cuQuantumGeSvd_internal_H__
#define __cuQuantumGeSvd_internal_H__

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "Storage.hpp"
#include "Type.hpp"
#include "cytnx_error.hpp"
#include "lapack_wrapper.hpp"
#include "linalg/linalg_internal_interface.hpp"


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
    void cuQuantumGeSvd_internal_cd(const Tensor &Tin, const cytnx_uint64 &keepdim,
                                       const double &err, const unsigned int &return_err, Tensor U, Tensor S, Tensor vT);
    #endif
  #endif

  }  // namespace linalg_internal
}  // namespace cytnx

#endif
