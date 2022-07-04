#ifndef _H_algo_internal_interface_
#define _H_algo_internal_interface_
#include <iostream>
#include <vector>

#include "Type.hpp"
#include "Storage.hpp"
#include "algo/algo_internal_cpu/Sort_internal.hpp"
#ifdef UNI_GPU
  //#include "algo/algo_internal_gpu/cuSort_internal.hpp"
#endif

namespace cytnx {

  namespace algo_internal {
    typedef void (*Sort_internal_ii)(boost::intrusive_ptr<Storage_base> &, const cytnx_uint64 &,
                                     const cytnx_uint64 &);

    class algo_internal_interface {
     public:
      std::vector<Sort_internal_ii> Sort_ii;

#ifdef UNI_GPU
        // std::vector<Sort_internal_ii> cuSort_ii;
#endif

      algo_internal_interface();
    };
    extern algo_internal_interface aii;
  }  // namespace algo_internal

}  // namespace cytnx
#endif
