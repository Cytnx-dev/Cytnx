#include "Generator.hpp"

#include "utils/utils.hpp"

#include "linalg.hpp"
#include <cfloat>
#include <iostream>

#ifdef BACKEND_TORCH
#else
  #include "backend/Storage.hpp"
  #include "backend/utils_internal_interface.hpp"

namespace cytnx {

  Tensor zeros(const std::vector<cytnx_uint64> &shape, unsigned int dtype, int device) {
    Tensor out(shape, dtype, device, true);
    // out._impl->storage().set_zeros();
    return out;
  }
  Tensor zeros(std::initializer_list<cytnx_uint64> shape, unsigned int dtype, int device) {
    return zeros(std::vector<cytnx_uint64>(shape), dtype, device);
  }
  //-----------------
  Tensor ones(const std::vector<cytnx_uint64> &shape, unsigned int dtype, int device) {
    Tensor out(shape, dtype, device);
    out._impl->storage().fill(1);
    return out;
  }
  Tensor ones(std::initializer_list<cytnx_uint64> shape, unsigned int dtype, int device) {
    return ones(std::vector<cytnx_uint64>(shape), dtype, device);
  }

  Tensor identity(cytnx_uint64 Dim, unsigned int dtype, int device) {
    Tensor out = ones({Dim}, dtype, device);
    return linalg::Diag(out);
  }
  Tensor eye(cytnx_uint64 Dim, unsigned int dtype, int device) {
    return identity(Dim, dtype, device);
  }
  //-----------------
  Tensor arange(cytnx_double start, cytnx_double end, cytnx_double step, unsigned int dtype,
                int device) {
    cytnx_error_msg((end - start) / step <= 0,
                    "[ERROR] arange(start=%f,end=%f,step=%f) "
                    "No values in the specified range.\n",
                    start, end, step);
    cytnx_uint64 Nelem;
    Tensor out;
    if (start < end) {
      Nelem = (end - start) / step;
      if (fmod((end - start), step) > 1.0e-14) Nelem += 1;
    } else {
      Nelem = (start - end) / (-step);
      if (fmod((start - end), (-step)) > 1.0e-14) Nelem += 1;
    }
    cytnx_error_msg(Nelem == 0, "[ERROR] arange(start,end,step)%s",
                    "Nelem cannot be zero! check the range!\n");
    out.Init({Nelem}, dtype, device);

    if (device == Device.cpu) {
      utils_internal::uii.SetArange_ii[dtype](out._impl->storage()._impl, start, end, step, Nelem);
    } else {
  #ifdef UNI_GPU
      checkCudaErrors(cudaSetDevice(out.device()));
      utils_internal::uii.cuSetArange_ii[dtype](out._impl->storage()._impl, start, end, step,
                                                Nelem);
  #else
      cytnx_error_msg(true, "[ERROR] fatal internal, %s",
                      " [arange] the container is on gpu without CUDA support!");
  #endif
    }

    return out;
  }
  Tensor arange(cytnx_int64 Nelem) {
    cytnx_error_msg(Nelem <= 0, "[ERROR] arange(Nelem) , %s", "Nelem must be integer > 0");
    return arange(0, Nelem, 1);
  }

  Tensor linspace(cytnx_double start, cytnx_double end, cytnx_uint64 Nelem, bool endpoint,
                  unsigned int dtype, int device) {
    Tensor out;
    cytnx_error_msg(Nelem == 0, "[ERROR] linspace(start,end,Nelem)%s", "Nelem cannot be zero!\n");
    out.Init({Nelem}, dtype, device);
    cytnx_double step;
    if (Nelem == 1)
      step = end - start;
    else {
      if (endpoint)
        step = (end - start) / (Nelem - 1);
      else
        step = (end - start) / (Nelem);
    }

    if (device == Device.cpu) {
      utils_internal::uii.SetArange_ii[dtype](out._impl->storage()._impl, start, end, step, Nelem);
    } else {
  #ifdef UNI_GPU
      checkCudaErrors(cudaSetDevice(out.device()));
      utils_internal::uii.cuSetArange_ii[dtype](out._impl->storage()._impl, start, end, step,
                                                Nelem);
  #else
      cytnx_error_msg(true, "[ERROR] fatal internal, %s",
                      " [arange] the container is on gpu without CUDA support!");
  #endif
    }
    return out;
  }

  //--------------

}  // namespace cytnx
#endif
