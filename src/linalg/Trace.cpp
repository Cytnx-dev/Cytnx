#include "linalg.hpp"
#include "utils/utils.hpp"
#include "Tensor.hpp"
#include "UniTensor.hpp"
// #include "cytnx.hpp"

#ifdef BACKEND_TORCH
#else
  #include "../backend/linalg_internal_interface.hpp"

  #ifdef UNI_OMP
    #include <omp.h>
  #endif

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
      cytnx_error_msg(Tn.dtype() == Type.Void, "[ERROR] cannot have output type to be Type.Void.%s",
                      "\n");
      cytnx_error_msg(
        Tn.dtype() == Type.Bool,
        "[ERROR][Trace] Bool type cannot perform Trace, use .astype() to promote first.%s", "\n");

      cytnx_uint64 ax1, ax2;
      if (axisA < axisB) {
        ax1 = axisA;
        ax2 = axisB;
      } else {
        ax1 = axisB;
        ax2 = axisA;
      }

      // 1) get redundant rank:
      vector<cytnx_int64> shape(Tn.shape().begin(), Tn.shape().end());
      vector<cytnx_uint64> accu;
      shape.erase(shape.begin() + ax2);
      shape.erase(shape.begin() + ax1);
      // 2) get out put elementsize.
      cytnx_uint64 Nelem = 1;
      for (int i = 0; i < shape.size(); i++) Nelem *= shape[i];
      // 3) get diagonal element numbers:
      cytnx_uint64 Ndiag = Tn.shape()[ax1] < Tn.shape()[ax2] ? Tn.shape()[ax1] : Tn.shape()[ax2];

      Tensor out = Tensor({Nelem}, Tn.dtype(), Tn.device());
      out.storage().set_zeros();

      int Nomp = 1;
  #ifdef UNI_OMP
    #pragma omp parallel
      {
        if (omp_get_thread_num() == 0) Nomp = omp_get_num_threads();
      }
  #endif

      if (shape.size() == 0) {
        // 2d
        if (Tn.device() == Device.cpu)
          linalg_internal::lii.Trace_ii[Tn.dtype()](true, out, Tn, Ndiag, Nomp, 0, {}, {}, {}, 0,
                                                    0);  // only the first 4 args will be used.
        else {
          cytnx_error_msg(true, "[ERROR][Trace] GPU is under developing.%s", "\n");
        }
      } else {
        // nd
        vector<cytnx_uint64> remain_rank_id;
        vector<cytnx_uint64> accu(shape.size());
        accu.back() = 1;
        for (int i = shape.size() - 1; i > 0; i--) accu[i - 1] = accu[i] * shape[i];

        for (cytnx_uint64 i = 0; i < Tn.shape().size(); i++) {
          if (i != ax1 && i != ax2) remain_rank_id.push_back(i);
        }
        // std::cout << "entry Trace" << std::endl;
        if (Tn.device() == Device.cpu)
          linalg_internal::lii.Trace_ii[Tn.dtype()](false, out, Tn, Ndiag, Nelem, Nomp, accu,
                                                    remain_rank_id, shape, ax1, ax2);
        else {
          cytnx_error_msg(true, "[ERROR][Trace] GPU is under developing.%s", "\n");
        }
        out.reshape_(shape);
      }

      return out;
    }

  }  // namespace linalg
}  // namespace cytnx

#endif  // BACKEND_TORCH
