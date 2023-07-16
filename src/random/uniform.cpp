#include "random.hpp"
#include "Type.hpp"
namespace cytnx {
  namespace random {
    Tensor uniform(const cytnx_uint64 &Nelem, const double &low, const double &high,
                   const int &device, const unsigned int &seed, const unsigned int &dtype) {
      Tensor out({Nelem}, dtype, device);
      Make_uniform(out, low, high, seed);
      return out;
    }
    Tensor uniform(const std::vector<cytnx_uint64> &Nelem, const double &low, const double &high,
                   const int &device, const unsigned int &seed, const unsigned int &dtype) {
      Tensor out(Nelem, dtype, device);
      Make_uniform(out, low, high, seed);
      return out;
    }

  }  // namespace random
}  // namespace cytnx
