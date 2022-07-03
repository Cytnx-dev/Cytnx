#ifndef _H_Uniform_cpu_
#define _H_Uniform_cpu_

#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <climits>
#include "Type.hpp"
#include "Storage.hpp"
#include "cytnx_error.hpp"

namespace cytnx {
  namespace random_internal {

    void Rng_uniform_cpu_cd(boost::intrusive_ptr<Storage_base> &in, const double &a,
                            const double &b, const unsigned int &seed);
    void Rng_uniform_cpu_cf(boost::intrusive_ptr<Storage_base> &in, const double &a,
                            const double &b, const unsigned int &seed);
    void Rng_uniform_cpu_d(boost::intrusive_ptr<Storage_base> &in, const double &a, const double &b,
                           const unsigned int &seed);
    void Rng_uniform_cpu_f(boost::intrusive_ptr<Storage_base> &in, const double &a, const double &b,
                           const unsigned int &seed);

  }  // namespace random_internal
}  // namespace cytnx
#endif
