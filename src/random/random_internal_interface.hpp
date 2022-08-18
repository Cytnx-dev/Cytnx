#ifndef _H_random_internal_
#define _H_random_internal_

#include "random_internal_cpu/Normal_internal.hpp"
#include "random_internal_cpu/Uniform_internal.hpp"
#ifdef UNI_GPU
  #include "random_internal_gpu/cuNormal_internal.hpp"
  #include "random_internal_gpu/cuUniform_internal.hpp"
#endif

#include "Type.hpp"
#include "Storage.hpp"
namespace cytnx {
  namespace random_internal {

    // func signature
    typedef void (*Rnd_io)(boost::intrusive_ptr<Storage_base> &, const double &a, const double &b,
                           const unsigned int &seed);
    class random_internal_interface {
     public:
      // fucntion signature:
      //-------------------------
      std::vector<Rnd_io> Normal;
      std::vector<Rnd_io> Uniform;
#ifdef UNI_GPU
      std::vector<Rnd_io> cuNormal;
      std::vector<Rnd_io> cuUniform;
#endif

      //--------------------------

      random_internal_interface();
    };
    extern random_internal_interface rii;

  }  // namespace random_internal
}  // namespace cytnx

#endif
