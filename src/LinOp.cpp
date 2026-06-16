#include "LinOp.hpp"
#include "Tensor.hpp"

#ifdef BACKEND_TORCH
#else

namespace cytnx {

  Tensor LinOp::matvec(const Tensor &Tin) {
    cytnx_error_msg(
      true,
      "[ERROR][LinOp] matvec(const Tensor&) must be overridden in a derived class before use.%s",
      "\n");
    return Tensor();
  }

  UniTensor LinOp::matvec(const UniTensor &Tin) {
    cytnx_error_msg(
      true,
      "[ERROR][LinOp] matvec(const UniTensor&) must be overridden in a derived class before use.%s",
      "\n");
    return UniTensor();
  }

}  // namespace cytnx
#endif
