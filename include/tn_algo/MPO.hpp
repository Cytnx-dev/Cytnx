#ifndef _H_MPO_
#define _H_MPO_

#include "cytnx_error.hpp"
#include "Device.hpp"
#include "intrusive_ptr_base.hpp"
#include "UniTensor.hpp"
#include <iostream>
#include <fstream>
#include "utils/vec_clone.hpp"
#include "Accessor.hpp"
#include <vector>
#include <initializer_list>
#include <string>
#include "Scalar.hpp"
#include "tn_algo/MPS.hpp"

namespace cytnx {
  namespace tn_algo {
    ///@cond
    class MPO_impl : public intrusive_ptr_base<MPO_impl> {
     private:
     public:
      std::vector<UniTensor> _TNs;

      friend class MPS;
      friend class MPO;

      boost::intrusive_ptr<MPO_impl> clone() const {
        boost::intrusive_ptr<MPO_impl> out(new MPO_impl());
        out->_TNs = vec_clone(this->_TNs);
        return out;
      }

      virtual std::ostream &Print(std::ostream &os);
      virtual cytnx_uint64 size() { return 0; };
      virtual UniTensor get_op(const cytnx_uint64 &site_idx);
    };

    class RegularMPO : public MPO_impl {
     public:
      std::ostream &Print(std::ostream &os);
      cytnx_uint64 size() { return this->_TNs.size(); };
      UniTensor get_op(const cytnx_uint64 &site_idx);
    };
    ///@endcond

    // API
    class MPO {
     private:
     public:
      ///@cond
      boost::intrusive_ptr<MPO_impl> _impl;
      MPO()
          : _impl(new RegularMPO()){
              // currently default init is RegularMPO;
            };

      MPO(const MPO &rhs) { _impl = rhs._impl; }
      ///@endcond

      MPO &operator=(const MPO &rhs) {
        _impl = rhs._impl;
        return *this;
      }

      cytnx_uint64 size() { return this->_impl->size(); }

      void append(const UniTensor &rc) { this->_impl->_TNs.push_back(rc); }

      void assign(const cytnx_uint64 &N, const UniTensor &rc) { this->_impl->_TNs.assign(N, rc); }

      std::vector<UniTensor> &get_all() { return this->_impl->_TNs; }

      const std::vector<UniTensor> &get_all() const { return this->_impl->_TNs; }

      // expose to user:
      virtual UniTensor get_op(const cytnx_uint64 &site_idx) {
        return this->_impl->get_op(site_idx);
      };
    };

    std::ostream &operator<<(std::ostream &os, const MPO &in);

  }  // namespace tn_algo
}  // namespace cytnx

#endif
