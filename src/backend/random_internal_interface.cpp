#include "random_internal_interface.hpp"
#include <vector>

namespace cytnx {
  namespace random_internal {

    random_internal_interface::random_internal_interface() {
      /// function signature.
      Normal = std::vector<Rnd_io>(N_fType, nullptr);
      Normal[Type.ComplexDouble] = Rng_normal_cpu_cd;
      Normal[Type.ComplexFloat] = Rng_normal_cpu_cf;
      Normal[Type.Double] = Rng_normal_cpu_d;
      Normal[Type.Float] = Rng_normal_cpu_f;

      /// function signature.
      Uniform = std::vector<Rnd_io>(N_fType, nullptr);
      Uniform[Type.ComplexDouble] = Rng_uniform_cpu_cd;
      Uniform[Type.ComplexFloat] = Rng_uniform_cpu_cf;
      Uniform[Type.Double] = Rng_uniform_cpu_d;
      Uniform[Type.Float] = Rng_uniform_cpu_f;

#ifdef UNI_GPU
      cuNormal = std::vector<Rnd_io>(N_fType, nullptr);
      cuNormal[Type.ComplexDouble] = cuRng_normal_cd;
      cuNormal[Type.ComplexFloat] = cuRng_normal_cf;
      cuNormal[Type.Double] = cuRng_normal_d;
      cuNormal[Type.Float] = cuRng_normal_f;

      cuUniform = std::vector<Rnd_io>(N_fType, nullptr);
      cuUniform[Type.ComplexDouble] = cuRng_uniform_cd;
      cuUniform[Type.ComplexFloat] = cuRng_uniform_cf;
      cuUniform[Type.Double] = cuRng_uniform_d;
      cuUniform[Type.Float] = cuRng_uniform_f;
#endif
    }

    random_internal_interface rii;

  }  // namespace random_internal
}  // namespace cytnx
