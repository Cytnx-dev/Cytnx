#ifndef __Accessor_H_
#define __Accessor_H_

#include "Type.hpp"
#include "cytnx_error.hpp"
#include <vector>
#include <cstring>
#include <string>
namespace cytnx {
  /**
   * @brief object that mimic the python slice to access elements in C++ [this is for c++ API only].
   */
  class Accessor {
   private:
    cytnx_int64 type;
    cytnx_int64 min, max, step;
    cytnx_int64 loc;

    // if type is singl, min/max/step     are not used
    // if type is all  , min/max/step/loc are not used
    // if type is range, loc              are not used.

   public:
    ///@cond
    enum : cytnx_int64 { none, Singl, All, Range };

    Accessor() : type(Accessor::none){};
    ///@endcond

    // singul constr.
    /**
    @brief access the specific index at the assigned rank in Tensor.
    @param loc the specify index

    ## Example:
    ### c++ API:
    \include example/Accessor/example.cpp
    #### output>
    \verbinclude example/Accessor/example.cpp.out
    ### python API:
    \include example/Accessor/example.py
    #### output>
    \verbinclude example/Accessor/example.py.out
    */
    Accessor(const cytnx_int64 &loc);

    ///@cond
    // all constr. ( use string to dispatch )
    Accessor(const std::string &str);

    // range constr.
    Accessor(const cytnx_int64 &min, const cytnx_int64 &max, const cytnx_int64 &step);

    // copy constructor:
    Accessor(const Accessor &rhs);
    // copy assignment:
    Accessor &operator=(const Accessor &rhs);
    ///@endcond

    // handy generator function :
    /**
    @brief access the whole rank, this is similar to [:] in python

    ## Example:
    ### c++ API:
    \include example/Accessor/example.cpp
    #### output>
    \verbinclude example/Accessor/example.cpp.out
    ### python API:
    \include example/Accessor/example.py
    #### output>
    \verbinclude example/Accessor/example.py.out
    */
    static Accessor all() { return Accessor(std::string("s")); };
    /**
    @brief access the range at assigned rank, this is similar to [min:max:step] in python
    @param min
    @param max
    @param step

    ## Example:
    ### c++ API:
    \include example/Accessor/example.cpp
    #### output>
    \verbinclude example/Accessor/example.cpp.out
    ### python API:
    \include example/Accessor/example.py
    #### output>
    \verbinclude example/Accessor/example.py.out
    */
    static Accessor range(const cytnx_int64 &min, const cytnx_int64 &max,
                          const cytnx_int64 &step = 1) {
      return Accessor(min, max, step);
    };

    ///@cond
    // get the real len from dim
    // if type is all, pos will be null, and len == dim
    // if type is range, pos will be the locator, and len == len(pos)
    // if type is singl, pos will be pos, and len == 0
    void get_len_pos(const cytnx_uint64 &dim, cytnx_uint64 &len,
                     std::vector<cytnx_uint64> &pos) const;
    ///@endcond
  };

}  // namespace cytnx

#endif
