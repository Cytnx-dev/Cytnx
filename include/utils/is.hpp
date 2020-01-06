#ifndef __is__H_
#define __is__H_

#include "Type.hpp"
#include "Tensor.hpp"
#include "Storage.hpp"

namespace cytnx{
   
    bool is(const Tensor&L, const Tensor&R);
    bool is(const Storage&L, const Storage&R);

}// namespace cytnx

#endif
