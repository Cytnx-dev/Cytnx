#ifndef _H_Normal_cpu_
#define _H_Normal_cpu_

#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <climits>
#include "Type.hpp"
#include "backend/Storage.hpp"
#include "cytnx_error.hpp"

namespace cytnx {
  namespace random_internal {

    void Rng_normal_cpu_cd(boost::intrusive_ptr<Storage_base> &in, const double &a, const double &b,
                           const unsigned int &seed);
    void Rng_normal_cpu_cf(boost::intrusive_ptr<Storage_base> &in, const double &a, const double &b,
                           const unsigned int &seed);
    void Rng_normal_cpu_d(boost::intrusive_ptr<Storage_base> &in, const double &a, const double &b,
                          const unsigned int &seed);
    void Rng_normal_cpu_f(boost::intrusive_ptr<Storage_base> &in, const double &a, const double &b,
                          const unsigned int &seed);

  }  // namespace random_internal
}  // namespace cytnx
#endif
