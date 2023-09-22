#include "cuUniform_internal.hpp"

namespace cytnx {
  namespace random_internal {

    void cuRng_uniform_cd(boost::intrusive_ptr<Storage_base> &in, const double &a, const double &b,
                          const unsigned int &seed) {
      double *rptr = static_cast<double *>(in->Mem);

      curandGenerator_t gen;
      curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937);

      // seed:
      curandSetPseudoRandomGeneratorSeed(gen, seed);

      // generate:
      curandGenerateUniformDouble(gen, rptr, in->len * 2);

      curandDestroyGenerator(gen);
    }
    void cuRng_uniform_cf(boost::intrusive_ptr<Storage_base> &in, const double &a, const double &b,
                          const unsigned int &seed) {
      float *rptr = static_cast<float *>(in->Mem);
      curandGenerator_t gen;
      curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937);

      // seed:
      curandSetPseudoRandomGeneratorSeed(gen, seed);

      // generate:
      curandGenerateUniform(gen, rptr, in->len * 2);

      curandDestroyGenerator(gen);
    }
    void cuRng_uniform_d(boost::intrusive_ptr<Storage_base> &in, const double &a, const double &b,
                         const unsigned int &seed) {
      double *rptr = static_cast<double *>(in->Mem);
      curandGenerator_t gen;
      curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937);

      // seed:
      curandSetPseudoRandomGeneratorSeed(gen, seed);

      // generate:
      curandGenerateUniformDouble(gen, rptr, in->len);

      curandDestroyGenerator(gen);
    }
    void cuRng_uniform_f(boost::intrusive_ptr<Storage_base> &in, const double &a, const double &b,
                         const unsigned int &seed) {
      float *rptr = static_cast<float *>(in->Mem);
      curandGenerator_t gen;
      curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937);

      // seed:
      curandSetPseudoRandomGeneratorSeed(gen, seed);

      // generate:
      curandGenerateUniform(gen, rptr, in->len);

      curandDestroyGenerator(gen);
    }

  }  // namespace random_internal
}  // namespace cytnx
