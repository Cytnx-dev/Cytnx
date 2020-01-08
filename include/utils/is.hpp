#ifndef __is__H_
#define __is__H_

#include "Type.hpp"
#include "Tensor.hpp"
#include "Storage.hpp"
#ifdef EXT_Enable
    #include "extension/Bond.hpp"
    #include "extension/Symmetry.hpp"
#endif


namespace cytnx{
   
    bool is(const Tensor&L, const Tensor&R);
    bool is(const Storage&L, const Storage&R);

    #ifdef EXT_Enable
    bool is(const Bond &L, const Bond& R);
    bool is(const Symmetry &L, const Symmetry& R); 
    #endif

}// namespace cytnx

#endif
