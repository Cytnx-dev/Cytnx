#ifndef __is__H_
#define __is__H_

#include "Type.hpp"
#include "Tensor.hpp"
#include "Storage.hpp"
#include "Bond.hpp"
#include "Symmetry.hpp"


namespace cytnx{
   
    bool is(const Tensor&L, const Tensor&R);
    bool is(const Storage&L, const Storage&R);

}// namespace cytnx

namespace cytnx_extension{
    bool is(const cytnx_extension::Bond &L, const cytnx_extension::Bond &R);
    bool is(const cytnx_extension::Symmetry &L, const cytnx_extension::Symmetry& R); 
}

#endif
