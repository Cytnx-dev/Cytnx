#ifndef _H_Normal_gpu_
#define _H_Normal_gpu_

#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <climits>
#include "Type.hpp"
#include "Storage.hpp"
#include "cytnx_error.hpp"

namespace cytnx {
  namespace random_internal {

    void cuRng_normal_cd(boost::intrusive_ptr<Storage_base> &in, const double &a, const double &b,
                         const unsigned int &seed);
    void cuRng_normal_cf(boost::intrusive_ptr<Storage_base> &in, const double &a, const double &b,
                         const unsigned int &seed);
    void cuRng_normal_d(boost::intrusive_ptr<Storage_base> &in, const double &a, const double &b,
                        const unsigned int &seed);
    void cuRng_normal_f(boost::intrusive_ptr<Storage_base> &in, const double &a, const double &b,
                        const unsigned int &seed);

  }  // namespace random_internal
}  // namespace cytnx
#endif
