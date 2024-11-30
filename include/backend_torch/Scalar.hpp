#ifndef CYTNX_BACKEND_TORCH_SCALAR_H_
#define CYTNX_BACKEND_TORCH_SCALAR_H_

#ifdef BACKEND_TORCH
  #include <torch/torch.h>

namespace cytnx {
  class Scalar : public torch::Scalar {};
}  // namespace cytnx
#endif

#endif  // CYTNX_BACKEND_TORCH_SCALAR_H_
