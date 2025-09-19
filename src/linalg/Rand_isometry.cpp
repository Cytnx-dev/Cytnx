#include <iostream>
#include <string>
#include <vector>

#include "Tensor.hpp"
#include "UniTensor.hpp"
#include "algo.hpp"
#include "linalg.hpp"
#include "random.hpp"

#ifdef BACKEND_TORCH
#else
  #include "backend/linalg_internal_interface.hpp"

namespace cytnx {
  namespace linalg {
    Tensor Rand_isometry(const Tensor& Tin, const cytnx_uint64& keepdim,
                         const cytnx_uint64& power_iteration, const unsigned int& seed) {
      std::vector<cytnx_uint64> shape = Tin.shape();
      cytnx_int64 truncdim = std::min({keepdim, shape[0], shape[1]});
      shape[0] = shape[1];
      shape[1] = truncdim;
      Tensor randmat = random::normal(shape[0] * shape[1], 0., 1., Tin.device(), seed, Tin.dtype());
      randmat.reshape_(shape);
      randmat = Matmul(Tin, randmat);
      std::vector<Tensor> Q = Qr(randmat, false);
      if (power_iteration > 0) {
        Tensor dag = Tin.Conj().permute_({1, 0});
        for (int pit = 0; pit < power_iteration; pit++) {
          randmat = Matmul(dag, Q[0]);
          Q = Qr(randmat, false);
          randmat = Matmul(Tin, Q[0]);
          Q = Qr(randmat, false);
        }
      }
      return Q[0];
    }
  }  // namespace linalg
}  // namespace cytnx

#endif  // BACKEND_TORCH
