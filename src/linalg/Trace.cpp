#include "linalg.hpp"
#include "utils/utils.hpp"
#include "Device.hpp"
#include "Tensor.hpp"
#include "UniTensor.hpp"
// #include "cytnx.hpp"

#ifdef BACKEND_TORCH
#else
  #include "backend/linalg_internal_interface.hpp"

using namespace std;
namespace cytnx {
  namespace linalg {
    cytnx::UniTensor Trace(const cytnx::UniTensor &Tin, const cytnx_int64 &a,
                           const cytnx_int64 &b) {
      return Tin.Trace(a, b);
    }
    cytnx::UniTensor Trace(const cytnx::UniTensor &Tin, const std::string &a,
                           const std::string &b) {
      return Tin.Trace(a, b);
    }

  }  // namespace linalg
}  // namespace cytnx

namespace cytnx {

  namespace linalg {
    // dtype -1: default
    // device -2: default.
    Tensor Trace(const Tensor &Tn, const cytnx_uint64 &axisA, const cytnx_uint64 &axisB) {
      // checking:
      cytnx_error_msg(Tn.shape().size() < 2, "[ERROR] Tensor must have at least rank-2.%s", "\n");
      cytnx_error_msg(axisA >= Tn.shape().size(), "[ERROR] axisA out of bound.%s", "\n");
      cytnx_error_msg(axisB >= Tn.shape().size(), "[ERROR] axisB out of bound.%s", "\n");
      cytnx_error_msg(axisA == axisB, "[ERROR] axisB cannot be the same as axisA.%s", "\n");
      cytnx_error_msg(Tn.dtype() == Type.Void, "[ERROR] Cannot have output type to be Type.Void.%s",
                      "\n");
      cytnx_error_msg(
        Tn.dtype() == Type.Bool,
        "[ERROR][Trace] Bool type cannot perform Trace, use .astype() to promote first.%s", "\n");
      cytnx_error_msg(Tn.shape()[axisA] != Tn.shape()[axisB],
                      "[ERROR][Trace] Index size of traced indices needs to match, but "
                      "shape(%d) = %ld and shape(%d) = %ld do not match.%s",
                      axisA, Tn.shape()[axisA], axisB, Tn.shape()[axisB], "\n");

      if (Tn.device() == Device.cpu) {
        return linalg_internal::lii.Trace_ii[Tn.dtype()](Tn, axisA, axisB);
      }
  #ifdef UNI_GPU
      checkCudaErrors(cudaSetDevice(Tn.device()));
      return linalg_internal::lii.cuTrace_ii[Tn.dtype()](Tn, axisA, axisB);
  #else
      cytnx_error_msg(true, "[Trace] fatal error,%s",
                      "try to call the gpu section without CUDA support.\n");
      return Tensor();
  #endif
    }

  }  // namespace linalg
}  // namespace cytnx

#endif  // BACKEND_TORCH
