#include "linalg.hpp"

#include "utils/utils.hpp"
#include "Tensor.hpp"
#include "backend/Scalar.hpp"
#include "Generator.hpp"

#ifdef BACKEND_TORCH
#else
  #include "backend/linalg_internal_interface.hpp"

namespace cytnx {

  namespace linalg {
    Tensor Ger(const Tensor &x, const Tensor &y, const Scalar &a) {
      // checking
      cytnx_error_msg(x.shape().size() != 1, "[ERROR][Ger] x must be rank-1 (vector).%s", "\n");
      cytnx_error_msg(y.shape().size() != 1, "[ERROR][Ger] y must be rank-1 (vector).%s", "\n");
      cytnx_error_msg(x.device() != y.device(), "[ERROR][Ger] x and y must on same device!%s",
                      "\n");

      // find the promoted dtype (a participates only when explicitly given):
      unsigned int fin_dtype = Type.type_promote(x.dtype(), y.dtype());
      if (a.dtype() != Type.Void) fin_dtype = Type.type_promote(fin_dtype, a.dtype());

      // the ger kernels only cover the four float types; floor integer/bool to Double
      if (fin_dtype > Type.Float) fin_dtype = Type.Double;

      Scalar alph = (a.dtype() == Type.Void) ? Scalar(1, fin_dtype) : a.astype(fin_dtype);

      // convert dtype (astype is a no-op when the dtype already matches):
      Tensor px = x.astype(fin_dtype);
      Tensor py = y.astype(fin_dtype);

      Tensor out = zeros({x.shape()[0], y.shape()[0]}, fin_dtype, x.device());

      if (x.device() == Device.cpu) {
        linalg_internal::lii.ger_ii[fin_dtype](out.storage()._impl, px.storage()._impl,
                                               py.storage()._impl, alph);
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(px.device()));
        linalg_internal::lii.cuGer_ii[fin_dtype](out.storage()._impl, px.storage()._impl,
                                                 py.storage()._impl, alph);
  #else
        cytnx_error_msg(true, "[Ger] fatal error,%s",
                        "try to call the gpu section without CUDA support.\n");

  #endif
        // cytnx_error_msg(true,"[Developing ger.gpu]%s","\n");
      }

      return out;
    };

  }  // namespace linalg
}  // namespace cytnx

#endif  // BACKEND_TORCH
