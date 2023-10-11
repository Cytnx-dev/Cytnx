#include "linalg.hpp"
#include "utils/utils.hpp"
#include "Tensor.hpp"
#include "Generator.hpp"

#ifdef BACKEND_TORCH
#else

  #include "../backend/linalg_internal_interface.hpp"
  #include "backend/Scalar.hpp"

namespace cytnx {

  namespace linalg {
    using namespace std;
    Tensor Axpy(const Scalar &a, const Tensor &x, const Tensor &y) {
      bool no_y = false;
      // checking the largest dtype!
      int fin_dtype = x.dtype();
      if (a.dtype() < fin_dtype) fin_dtype = a.dtype();

      if (y.shape().size() == 0)
        no_y = true;
      else {
        //   a*x + y -> return
        cytnx_error_msg(x.shape() != y.shape(),
                        "[ERROR][Axpy] x.shape() and y.shape() must be the same!%s", "\n");
        cytnx_error_msg(x.device() != y.device(), "[ERROR][Axpy] x and y must be on same device!%s",
                        "\n");
        if (y.dtype() < fin_dtype) fin_dtype = y.dtype();
      }

      if (fin_dtype < Type.Float) fin_dtype = Type.Double;

      // convert dtype:
      Tensor px;
      if (x.dtype() > fin_dtype)
        px = x.astype(fin_dtype);
      else
        px = x;

      Tensor out;
      if (no_y) {
        out = cytnx::zeros(x.shape(), fin_dtype, x.device());
      } else {
        if (y.dtype() > fin_dtype)
          out = y.astype(fin_dtype);
        else
          out = y.clone();
      }

      Scalar pa;
      if (a.dtype() > fin_dtype)
        pa = a.astype(fin_dtype);
      else
        pa = a;

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

      // checking the largest dtype!
      int fin_dtype = x.dtype();
      if (y.dtype() < fin_dtype) fin_dtype = y.dtype();
      if (a.dtype() < fin_dtype) fin_dtype = a.dtype();

      if (fin_dtype < Type.Float) fin_dtype = Type.Double;

      // convert dtype:
      Tensor px;
      if (x.dtype() > fin_dtype)
        px = x.astype(fin_dtype);
      else
        px = x;

      if (y.dtype() > fin_dtype) y = y.astype(fin_dtype);

      Scalar pa;
      if (a.dtype() > fin_dtype)
        pa = a.astype(fin_dtype);
      else
        pa = a;

      if (x.device() == Device.cpu) {
        linalg_internal::lii.axpy_ii[fin_dtype](px.storage()._impl, y.storage()._impl, pa);
      } else {
        cytnx_error_msg(true, "[Developing axpy.gpu]%s", "\n");
      }
    };

  }  // namespace linalg
}  // namespace cytnx

#endif  // BACKEND_TORCH
