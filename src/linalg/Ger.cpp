#include "linalg.hpp"

#include "utils/utils.hpp"
#include "Tensor.hpp"
#include "backend/Scalar.hpp"
#include "Generator.hpp"

#ifdef BACKEND_TORCH
#else
  #include "../backend/linalg_internal_interface.hpp"

namespace cytnx {

  namespace linalg {
    using namespace std;
    Tensor Ger(const Tensor &x, const Tensor &y, const Scalar &a) {
      // checking
      cytnx_error_msg(x.shape().size() != 1, "[ERROR][Ger] x must be rank-1 (vector).%s", "\n");
      cytnx_error_msg(y.shape().size() != 1, "[ERROR][Ger] y must be rank-1 (vector).%s", "\n");
      cytnx_error_msg(x.device() != y.device(), "[ERROR][Ger] x and y must on same device!%s",
                      "\n");

      // checking the largest dtype!
      int fin_dtype = x.dtype();
      if (y.dtype() < fin_dtype) fin_dtype = y.dtype();

      Scalar alph;
      if (a.dtype() == Type.Void)
        alph = Scalar(1, fin_dtype);
      else {
        if (a.dtype() <= fin_dtype) fin_dtype = a.dtype();
        alph = a;
      }
      if (fin_dtype < Type.Float) fin_dtype = Type.Double;

      // convert dtype:
      Tensor px;
      if (x.dtype() > fin_dtype)
        px = x.astype(fin_dtype);
      else
        px = x;

      Tensor py;
      if (y.dtype() > fin_dtype)
        py = y.astype(fin_dtype);
      else
        py = y;

      if (alph.dtype() > fin_dtype) alph = a.astype(fin_dtype);

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
