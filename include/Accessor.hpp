#ifndef __Accessor_H_
#define __Accessor_H_

#include "Type.hpp"
#include "cytnx_error.hpp"
//#include "Tensor.hpp"
#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <initializer_list>
namespace cytnx {
  /**
   * @brief object that mimic the python slice to access elements in C++ [this is for c++ API only].
   */
  class Accessor {
   private:
    cytnx_int64 _type;

   public:
    ///@cond
    cytnx_int64 _min{}, _max{}, _step{};
    cytnx_int64 loc{};
    std::vector<cytnx_int64> idx_list;

    std::vector<std::vector<cytnx_int64>> qns_list;

    // if type is singl, _min/_max/_step     are not used
    // if type is all  , _min/_max/_step/loc are not used
    // if type is range, loc              are not used.
    // if type is tilend, loc/_max are not used.
    // if type is Qns, only qns_list are used.

    enum : cytnx_int64 { none, Singl, All, Range, Tilend, Step, Tn, list, Qns };

    Accessor() : _type(Accessor::none){};
    ///@endcond

    // single constructor
    /**
    @brief access the specific index at the assigned rank in Tensor.
    @param loc the specify index

    See also \link cytnx::Tensor.get() cytnx::Tensor.get() \endlink for how to using them.

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
    explicit Accessor(const cytnx_int64 &loc);
    // explicit Accessor(const Tensor &tn);// construct from Tensor, should be 1d with dtype
    // integer.

    template <class T>
    explicit Accessor(const std::initializer_list<T> &list) {
      std::vector<T> tmp = list;
      this->_type = this->list;
      this->idx_list = std::vector<cytnx_int64>(tmp.begin(), tmp.end());
      // std::cout << "VV" << this->idx_list.size() << std::endl;
    };  // construct from vector/list, should be 1d with dtype integer.

    template <class T>
    explicit Accessor(const std::vector<T> &list) {
      this->_type = this->list;
      this->idx_list = std::vector<cytnx_int64>(list.begin(), list.end());
    };  // construct from vector/list, should be 1d with dtype integer.

    ///@cond

    // all constr. ( use string to dispatch )
    explicit Accessor(const std::string &str);

    // range constr.
    Accessor(const cytnx_int64 &min, const cytnx_int64 &max, const cytnx_int64 &step);

    // copy constructor:
    Accessor(const Accessor &rhs);
    // copy assignment:
    Accessor &operator=(const Accessor &rhs);
    ///@endcond

    int type() const { return this->_type; }

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
    static Accessor all() { return Accessor(std::string(":")); };

    /**
    @brief access the range at assigned rank, this is similar to min:max:step in python
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

    static Accessor tilend(const cytnx_int64 &min, const cytnx_int64 &step = 1) {
      cytnx_error_msg(step == 0, "[ERROR] cannot have _step=0 for tilend%s", "\n");
      Accessor out;
      out._type = Accessor::Tilend;
      out._min = min;
      out._step = step;
      return out;
    };

    static Accessor step(const cytnx_int64 &step) {
      cytnx_error_msg(step == 0, "[ERROR] cannot have _step=0 for _step%s", "\n");
      Accessor out;
      out._type = Accessor::Step;
      // out._min = 0;
      out._step = step;
      return out;
    };

    static Accessor qns(const std::vector<std::vector<cytnx_int64>> &qns) {
      cytnx_error_msg(qns.size() == 0, "[ERROR] cannot have empty qnums.%s", "\n");
      Accessor out;

      out._type = Accessor::Qns;
      out.qns_list = qns;
      return out;
    }

    ///@cond
    // get the real len from dim
    // if type is all, pos will be null, and len == dim
    // if type is range, pos will be the locator, and len == len(pos)
    // if type is singl, pos will be pos, and len == 0
    void get_len_pos(const cytnx_uint64 &dim, cytnx_uint64 &len,
                     std::vector<cytnx_uint64> &pos) const;
    ///@endcond
  };  // class Accessor

  /// @cond
  // layout:
  std::ostream &operator<<(std::ostream &os, const Accessor &in);

  // elements resolver
  template <class T>
  void _resolve_elems(std::vector<cytnx::Accessor> &cool, const T &a) {
    cool.push_back(cytnx::Accessor(a));
  }

  template <class T, class... Ts>
  void _resolve_elems(std::vector<cytnx::Accessor> &cool, const T &a, const Ts &...args) {
    cool.push_back(cytnx::Accessor(a));
    _resolve_elems(cool, args...);
  }

  template <class T, class... Ts>
  std::vector<cytnx::Accessor> Indices_resolver(const T &a, const Ts &...args) {
    // std::cout << a << std::endl;;
    std::vector<cytnx::Accessor> idxs;
    _resolve_elems(idxs, a, args...);
    // cout << idxs << endl;
    return idxs;
  }
  ///@endcond

}  // namespace cytnx

#endif
