#include "Generator.hpp"

#include "utils/utils.hpp"

#include "linalg.hpp"
#include <cfloat>
#include <cmath>
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
    cytnx_error_msg(step == 0, "[ERROR] arange(start=%f,end=%f,step=%f): step cannot be zero.\n",
                    start, end, step);
    cytnx_error_msg(
      !std::isfinite(start) || !std::isfinite(end) || !std::isfinite(step),
      "[ERROR] arange(start=%f,end=%f,step=%f): start, end, and step must all be finite.\n", start,
      end, step);

    // The element count is ceil((end - start) / step), matching numpy.arange. The range is
    // nominally half-open [start, end), but -- exactly as with numpy -- floating-point rounding
    // can make the final element equal or slightly exceed `end` (e.g. arange(0.5, 0.8, 0.1) ->
    // [0.5, 0.6, 0.7, 0.8]). A non-positive count is an empty or direction-mismatched range and
    // yields a zero-extent tensor rather than an error (zero-extent tensors are supported). This
    // replaced the old integer-truncation + `fmod(...) > 1e-14` test, which mishandled the
    // endpoint and miscounted at small scales. See #1076, #1083.
    const cytnx_double count = (end - start) / step;
    cytnx_uint64 Nelem = 0;
    if (count > 0) {
      const cytnx_double nelem = std::ceil(count);
      // Guard the double -> uint64 cast below: `count` can overflow to +inf even for finite
      // start/end/step (a huge span with a tiny step), and casting a double whose truncated value
      // is >= 2^64 to uint64_t is undefined behavior. The threshold is 2^64 (0x1p64), NOT
      // UINT64_MAX: 2^64 - 1 is not representable as a double and rounds up to 2^64, so a `>`
      // test against (double)UINT64_MAX would let nelem == 2^64 slip through into the UB cast --
      // `>=` is required. std::ceil(+inf) == +inf, which this also catches.
      cytnx_error_msg(nelem >= 0x1p64,
                      "[ERROR] arange(start=%f,end=%f,step=%f): the requested number of elements "
                      "exceeds the maximum representable size.\n",
                      start, end, step);
      Nelem = static_cast<cytnx_uint64>(nelem);
    }

    Tensor out;
    out.Init({Nelem}, dtype, device);
    if (Nelem == 0) return out;  // zero-extent range: nothing to fill

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
    cytnx_error_msg(Nelem < 0, "[ERROR] arange(Nelem): Nelem must be a non-negative integer.%s",
                    "\n");
    // Nelem == 0 yields a zero-extent tensor (consistent with the empty-range case of the
    // start/end/step overload).
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
