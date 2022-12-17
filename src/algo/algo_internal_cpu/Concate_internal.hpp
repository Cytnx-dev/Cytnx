#ifndef __Concate_internal_H__
#define __Concate_internal_H__

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "Storage.hpp"
#include "Type.hpp"

namespace cytnx {

  namespace algo_internal {

    void Concate_internal(boost::intrusive_ptr<Storage_base> &out, std::vector<boost::intrusive_ptr<Storage_base> > &ins);
       


  }  // namespace algo_internal

}  // namespace cytnx

#endif
