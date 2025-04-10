#include "Normal_internal.hpp"

#include <random>
using namespace std;
namespace cytnx {
  namespace random_internal {

    void Rng_normal_cpu_cd(boost::intrusive_ptr<Storage_base> &in, const double &a, const double &b,
                           const unsigned int &seed) {
      mt19937 eng(seed);
      std::normal_distribution<double> distro(a, b);

      double *rptr = static_cast<double *>(in->data());
      for (cytnx_uint64 i = 0; i < in->size() * 2; i++) {
        rptr[i] = distro(eng);
      }
    }
    void Rng_normal_cpu_cf(boost::intrusive_ptr<Storage_base> &in, const double &a, const double &b,
                           const unsigned int &seed) {
      mt19937 eng(seed);
      std::normal_distribution<float> distro(a, b);

      float *rptr = static_cast<float *>(in->data());
      for (cytnx_uint64 i = 0; i < in->size() * 2; i++) {
        rptr[i] = distro(eng);
      }
    }
    void Rng_normal_cpu_d(boost::intrusive_ptr<Storage_base> &in, const double &a, const double &b,
                          const unsigned int &seed) {
      mt19937 eng(seed);
      std::normal_distribution<double> distro(a, b);
      double *rptr = static_cast<double *>(in->data());
      for (cytnx_uint64 i = 0; i < in->size(); i++) {
        rptr[i] = distro(eng);
      }
    }
    void Rng_normal_cpu_f(boost::intrusive_ptr<Storage_base> &in, const double &a, const double &b,
                          const unsigned int &seed) {
      mt19937 eng(seed);
      std::normal_distribution<float> distro(a, b);
      float *rptr = static_cast<float *>(in->data());
      for (cytnx_uint64 i = 0; i < in->size(); i++) {
        rptr[i] = distro(eng);
      }
    }

  }  // namespace random_internal
}  // namespace cytnx
