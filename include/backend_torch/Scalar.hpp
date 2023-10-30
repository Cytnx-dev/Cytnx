#ifndef _H_Scalar_
#define _H_Scalar_
#ifdef BACKEND_TORCH
  #include <torch/torch.h>

namespace cytnx {
  class Scalar : public torch::Scalar {};
}  // namespace cytnx
#endif
#endif
