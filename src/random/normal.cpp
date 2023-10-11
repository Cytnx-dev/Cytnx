#include "random.hpp"
#include "Type.hpp"

#ifdef BACKEND_TORCH
#else
namespace cytnx {
  namespace random {
    Tensor normal(const cytnx_uint64 &Nelem, const double &mean, const double &std,
                  const int &device, const unsigned int &seed, const unsigned int &dtype) {
      Tensor out({Nelem}, dtype, device);
      normal_(out, mean, std, seed);
      return out;
    }
    Tensor normal(const std::vector<cytnx_uint64> &Nelem, const double &mean, const double &std,
                  const int &device, const unsigned int &seed, const unsigned int &dtype) {
      Tensor out(Nelem, dtype, device);
      normal_(out, mean, std, seed);
      return out;
    }

  }  // namespace random
}  // namespace cytnx
#endif
