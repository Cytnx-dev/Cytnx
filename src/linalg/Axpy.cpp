#include "linalg.hpp"
#include "utils/utils.hpp"
#include "Tensor.hpp"
#include "Generator.hpp"

#ifdef BACKEND_TORCH
#else

  #include "backend/linalg_internal_interface.hpp"
  #include "backend/Scalar.hpp"

namespace cytnx {

  namespace linalg {
    Tensor Axpy(const Scalar &a, const Tensor &x, const Tensor &y) {
      bool no_y = false;
      // find the promoted dtype:
      unsigned int fin_dtype = Type.type_promote(a.dtype(), x.dtype());

      if (y.shape().size() == 0)
        no_y = true;
      else {
        //   a*x + y -> return
        cytnx_error_msg(x.shape() != y.shape(),
                        "[ERROR][Axpy] x.shape() and y.shape() must be the same!%s", "\n");
        cytnx_error_msg(x.device() != y.device(), "[ERROR][Axpy] x and y must be on same device!%s",
                        "\n");
        fin_dtype = Type.type_promote(fin_dtype, y.dtype());
      }

      // the axpy kernels only cover the four float types; floor integer/bool to Double
      if (fin_dtype > Type.Float) fin_dtype = Type.Double;

      // convert dtype (astype is a no-op when the dtype already matches):
      Tensor px = x.astype(fin_dtype);

      Tensor out;
      if (no_y) {
        out = cytnx::zeros(x.shape(), fin_dtype, x.device());
      } else {
        // astype aliases y when the dtype already matches; clone to keep y intact
        out = (y.dtype() == fin_dtype) ? y.clone() : y.astype(fin_dtype);
      }

      Scalar pa = a.astype(fin_dtype);

      if (x.device() == Device.cpu) {
        linalg_internal::lii.axpy_ii[fin_dtype](px.storage()._impl, out.storage()._impl, pa);
      } else {
        cytnx_error_msg(true, "[Developing axpy.gpu]%s", "\n");
      }

      return out;
    };

    void Axpy_(const Scalar &a, const Tensor &x, Tensor &y) {
      //   y = a*x + y

      cytnx_error_msg(x.shape() != y.shape(),
                      "[ERROR][Axpy] x.shape() and y.shape() must be the same!%s", "\n");
      cytnx_error_msg(x.device() != y.device(), "[ERROR][Axpy] x and y must be on same device!%s",
                      "\n");

      // find the promoted dtype:
      unsigned int fin_dtype = Type.type_promote(x.dtype(), y.dtype());
      fin_dtype = Type.type_promote(fin_dtype, a.dtype());

      // the axpy kernels only cover the four float types; floor integer/bool to Double
      if (fin_dtype > Type.Float) fin_dtype = Type.Double;

      // convert dtype (astype is a no-op when the dtype already matches):
      Tensor px = x.astype(fin_dtype);

      if (y.dtype() != fin_dtype) y = y.astype(fin_dtype);

      Scalar pa = a.astype(fin_dtype);

      if (x.device() == Device.cpu) {
        linalg_internal::lii.axpy_ii[fin_dtype](px.storage()._impl, y.storage()._impl, pa);
      } else {
        cytnx_error_msg(true, "[Developing axpy.gpu]%s", "\n");
      }
    };

  }  // namespace linalg
}  // namespace cytnx

#endif  // BACKEND_TORCH
