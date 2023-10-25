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

  Tensor zeros(const cytnx_uint64 &Nelem, const unsigned int &dtype, const int &device) {
    Tensor out({Nelem}, dtype, device, true);  // init_zero
    // out is already init as zeros
    // out._impl->storage().set_zeros();
    return out;
  }
  Tensor zeros(const std::vector<cytnx_uint64> &shape, const unsigned int &dtype,
               const int &device) {
    // std::cout << "OK" << std::endl;
    // std::cout << Nelem << std::endl;
    Tensor out(shape, dtype, device, true);
    // out._impl->storage().set_zeros();
    return out;
  }
  //-----------------
  Tensor ones(const cytnx_uint64 &Nelem, const unsigned int &dtype, const int &device) {
    Tensor out({Nelem}, dtype, device);  // the default
    out._impl->storage().fill(1);
    return out;
  }
  Tensor ones(const std::vector<cytnx_uint64> &shape, const unsigned int &dtype,
              const int &device) {
    Tensor out(shape, dtype, device);
    out._impl->storage().fill(1);
    return out;
  }

  Tensor identity(const cytnx_uint64 &Dim, const unsigned int &dtype, const int &device) {
    Tensor out = ones(Dim, dtype, device);
    return linalg::Diag(out);
  }
  Tensor eye(const cytnx_uint64 &Dim, const unsigned int &dtype, const int &device) {
    return identity(Dim, dtype, device);
  }
  //-----------------
  Tensor arange(const cytnx_double &start, const cytnx_double &end, const cytnx_double &step,
                const unsigned int &dtype, const int &device) {
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
                      " [arange] the container is on gpu without CUDA support!%s", "\n")
  #endif
    }

    return out;
  }
  Tensor arange(const std::vector<cytnx_uint64> &shape, const cytnx_double &start,
                const cytnx_double &end, const cytnx_double &step = 1,
                const unsigned int &dtype = Type.Double, const int &device = Device.cpu) {
    return arange(start, end, step, dtype, device).reshape(shape);
  }
  Tensor arange(const cytnx_int64 &Nelem) {
    cytnx_error_msg(Nelem <= 0, "[ERROR] arange(Nelem) , %s", "Nelem must be integer > 0");
    return arange(0, Nelem, 1);
  }
  Tensor arange(const std::vector<cytnx_uint64> &shape) {
    return arange(std::reduce(
                    shape.begin(), shape.end(), 1,
                    [](cytnx_uint64 lhs, cytnx_uint64 rhs) -> cytnx_uint64 { return lhs * rhs; }))
      .reshape(shape);
  }
  Tensor linspace(const cytnx_double &start, const cytnx_double &end, const cytnx_uint64 &Nelem,
                  const bool &endpoint, const unsigned int &dtype, const int &device) {
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
                      " [arange] the container is on gpu without CUDA support!%s", "\n")
  #endif
    }
    return out;
  }
  Tensor linspace(const std::vector<cytnx_uint64> &shape, const cytnx_double &start,
                  const cytnx_double &end, const cytnx_uint64 &Nelem, const bool &endpoint = true,
                  const unsigned int &dtype = Type.Double, const int &device = Device.cpu) {
    return linspace(start, end, Nelem, endpoint, dtype, device).reshape(shape);
  }

  //--------------

}  // namespace cytnx
#endif
